"""Step 8 - train the deployable virtual-cell model on ALL measured perturbations.

Produces models/k562/:
  perturbformer_{0,1,2}.pt   ensemble of KernelPerturbFormer
  flow.pt                    conditional flow-matching generator (trained on all perturbations)
  meta.npz                   genes, perturbations, responses, PCA basis, control cells, ...
  gene_universe_embeddings.npz   prior knowledge for ~21k genes (lets users query any gene)
  config.json
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import ensure_dir, train_val_split  # noqa: E402
from vcp.embeddings import response_embedding  # noqa: E402
from vcp.flow import FlowModel  # noqa: E402
from vcp.models import predict_perturbformer_loo, train_perturbformer  # noqa: E402
from vcp.pipeline import PROC, ROOT, load_bundle  # noqa: E402

torch.set_num_threads(4)
DS = "replogle_k562_essential"
N_ENS = 3
FORMER_KW = {"d_model": 128, "n_heads": 2, "topk": 20, "fusion": "attn", "dropout": 0.1}
FLOW_KW = {"hidden": 256, "n_blocks": 3, "sigma": 0.05}


def main():
    t0 = time.time()
    out = ensure_dir(os.path.join(ROOT, "models", "k562"))
    b = load_bundle(DS)
    cells = np.load(os.path.join(PROC, f"{DS}_cells.npz"), allow_pickle=True)
    Z, lab, ctrl_label = cells["Z"], cells["label"], int(cells["ctrl_label"])
    comps, ctrl_mean = cells["components"], cells["ctrl_mean"]
    allp = np.arange(len(b.pdat.perts))
    tr, va = train_val_split(allp, 0.1, 0)
    feats, masks = b.fold_features(allp)
    resp_gene, _ = response_embedding(b.pdat.genes, b.pdat.genes, b.Y, dim=64)
    kd = knockdown_fraction(b.pdat.pert_mean, ctrl_mean, b.target_idx)

    loo = []
    for e in range(N_ENS):
        m = train_perturbformer(feats, masks, b.Y, b.W, b.S, tr, va, seed=e, **{k: v for k, v in FORMER_KW.items()})
        torch.save(m.state_dict(), os.path.join(out, f"perturbformer_{e}.pt"))
        loo.append(predict_perturbformer_loo(m, feats, masks, b.Y, allp))
        print(f"PerturbFormer {e} trained ({time.time() - t0:.0f}s)")
    loo = knockdown_rule(np.mean(loo, 0), b.pdat.perts, b.target_idx, ctrl_mean, kd)
    cond = {int(p): loo[p] @ comps.T for p in allp}
    flow = FlowModel(d=Z.shape[1], c_dim=Z.shape[1], seed=0, **FLOW_KW)
    flow.fit(Z, lab, ctrl_label, cond, steps=8000)
    torch.save(flow.state_dict(), os.path.join(out, "flow.pt"))
    print(f"flow trained ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(0)
    ctrl_idx = np.nonzero(lab == ctrl_label)[0]
    np.savez_compressed(
        os.path.join(out, "meta.npz"), genes=b.pdat.genes, perts=b.pdat.perts, Y=b.Y,
        ctrl_mean=ctrl_mean, ctrl_var=b.pdat.ctrl_var, components=comps, kd_frac=kd,
        resp_gene_emb=resp_gene, n_cells=b.pdat.n_cells, loo_delta=loo.astype(np.float32),
        ctrl_latent=Z[rng.choice(ctrl_idx, 3000, replace=False)],
    )
    shutil.copy(os.path.join(PROC, "gene_universe_embeddings.npz"), os.path.join(out, "gene_universe_embeddings.npz"))
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump({"dataset": DS, "n_ens": N_ENS, "basis_k": 32,
                   "dims": {k: int(v.shape[1]) for k, v in feats.items()},
                   "former_kwargs": FORMER_KW, "flow_kwargs": FLOW_KW}, fh, indent=2)
    print(f"saved to {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
