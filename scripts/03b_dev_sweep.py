"""Step 3b - small, transparent hyper-parameter sweep for PerturbFormer.

Only the *development split* is used: an inner validation set carved out of the training
perturbations of CV fold 0.  The five test folds used in 03_benchmark.py are never touched,
so the reported benchmark numbers are not optimistically biased by model selection.

Output: results/{dataset}/dev_sweep.csv
"""
import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import ensure_dir, kfold_perturbation_split, train_val_split  # noqa: E402
from vcp.metrics import evaluate, summarise  # noqa: E402
from vcp.models import predict_perturbformer, train_perturbformer  # noqa: E402
from vcp.pipeline import ROOT, load_bundle  # noqa: E402

torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replogle_k562_essential")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    b = load_bundle(args.dataset)
    fold = kfold_perturbation_split(b.pdat.perts, 5, 0)
    tr_all0 = np.nonzero(fold != 0)[0]
    tr_outer, dev = train_val_split(tr_all0, 0.1, 0)          # dev = held-out validation perts
    tr, va = train_val_split(tr_outer, 0.1, 77)               # early-stopping split
    feats, masks = b.fold_features(tr_outer)
    kd = knockdown_fraction(b.pdat.pert_mean[tr_outer], b.pdat.ctrl_mean, b.target_idx[tr_outer])
    ref = b.Y[tr_outer].mean(0)

    grid = {
        "topk": [10, 20, 50],
        "n_heads": [1, 2, 4],
        "lam_dir": [0.0, 0.5, 2.0],
    }
    default = {"topk": 20, "n_heads": 2, "lam_dir": 0.5}
    configs = [dict(default)]
    for k, vals in grid.items():               # one-factor-at-a-time around the default
        for v in vals:
            c = dict(default, **{k: v})
            if c not in configs:
                configs.append(c)
    configs += [dict(default, d_model=64), dict(default, dropout=0.3), dict(default, basis_k=64)]

    rows = []
    t0 = time.time()
    for c in configs:
        ms = []
        for sd in range(args.seeds):
            m = train_perturbformer(feats, masks, b.Y, b.W, b.S, tr, va, seed=sd, **c)
            p = predict_perturbformer(m, feats, masks, b.Y, tr_outer, dev)
            p = knockdown_rule(p, b.pdat.perts[dev], b.target_idx[dev], b.pdat.ctrl_mean, kd)
            ms.append(summarise(evaluate(p, b.Y[dev], b.pdat.tstat[dev], b.pdat.fdr[dev], ref, b.ctrl_std)))
        r = pd.DataFrame(ms).mean().to_dict()
        r.update({f"sd_{k}": v for k, v in pd.DataFrame(ms).std().to_dict().items() if k in ("pearson_de20", "pearson_sys")})
        r.update({"config": str(c)})
        rows.append(r)
        print(f"{str(c):60s} de20 {r['pearson_de20']:.3f}  sys {r['pearson_sys']:.3f}  all {r['pearson_all']:.3f} ({time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows).set_index("config")
    out.to_csv(os.path.join(ensure_dir(os.path.join(ROOT, "results", args.dataset)), "dev_sweep.csv"))


if __name__ == "__main__":
    main()
