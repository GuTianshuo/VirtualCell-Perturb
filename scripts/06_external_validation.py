"""Step 6 - external validation: train on Replogle 2022, predict Adamson 2016.

Different laboratory, different library, different year - same cell line (K562) and same
modality (CRISPRi).  All 86 Adamson target genes are removed from the Replogle training set,
so every Adamson perturbation is an *unseen gene* for the model.

Evaluation on the 1,910 genes measured in both screens.
Reference: for the 41 genes perturbed in both screens, the Replogle *measured* response is
used as a predictor of the Adamson response (cross-laboratory reproducibility ceiling).

Output: results/external_adamson/{summary.csv, per_perturbation.csv, predictions.npz}
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import KNNEmbedding, LinearEmbedding, knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import ensure_dir, train_val_split  # noqa: E402
from vcp.embeddings import response_embedding  # noqa: E402
from vcp.metrics import evaluate  # noqa: E402
from vcp.models import predict_perturbformer, train_perturbformer  # noqa: E402
from vcp.pipeline import ROOT, load_bundle, load_embeddings  # noqa: E402

torch.set_num_threads(4)


def main():
    src = load_bundle("replogle_k562_essential")
    tgt = load_bundle("adamson")
    out_dir = ensure_dir(os.path.join(ROOT, "results", "external_adamson"))
    sp, tp = src.pdat, tgt.pdat

    shared = [g for g in sp.genes if g in set(tp.genes)]
    si = np.array([list(sp.genes).index(g) for g in shared])
    ti = np.array([list(tp.genes).index(g) for g in shared])
    tr_all = np.array([i for i, p in enumerate(sp.perts) if p not in set(tp.perts)])
    tr, va = train_val_split(tr_all, 0.1, 0)
    print(f"shared genes {len(shared)}; Replogle training perts {len(tr_all)} (Adamson genes removed)")

    # features for source perts + target perts (query)
    q_static, q_mask = load_embeddings(tp.perts)
    resp_all, resp_m = response_embedding(np.concatenate([sp.perts, tp.perts]), sp.genes, src.Y[tr_all], dim=64)
    feats = {k: np.concatenate([src.static[k], q_static[k]]) for k in src.static}
    masks = {k: np.concatenate([src.static_m[k], q_mask[k]]) for k in src.static}
    feats["resp"], masks["resp"] = resp_all, resp_m.astype(np.float32)
    P = len(sp.perts)
    q = np.arange(P, P + len(tp.perts))
    Yext = np.concatenate([src.Y, np.zeros((len(tp.perts), src.Y.shape[1]), np.float32)])
    Wext = np.concatenate([src.W, np.ones((len(tp.perts), src.W.shape[1]), np.float32)])
    Sext = np.concatenate([src.S, np.zeros((len(tp.perts), src.S.shape[1]), np.float32)])

    # target-gene index in source gene space for the KD rule
    gl = {g: i for i, g in enumerate(sp.genes)}
    t_idx = np.array([gl.get(p, -1) for p in tp.perts])
    kd = knockdown_fraction(sp.pert_mean[tr_all], sp.ctrl_mean, src.target_idx[tr_all])

    preds = {"Mean-shift": np.repeat(src.Y[tr_all].mean(0)[None], len(tp.perts), 0)}
    Ecat = np.concatenate([feats[k] for k in feats], 1)
    preds["Linear-concat"] = LinearEmbedding(32, 0.1).fit(src.Y[tr_all], Ecat[tr_all]).predict(Ecat[q])
    preds["kNN-concat"] = KNNEmbedding(10, 0.1).fit(src.Y[tr_all], Ecat[tr_all]).predict(Ecat[q])
    outs = []
    for e in range(3):
        m = train_perturbformer(feats, masks, Yext, Wext, Sext, tr, va, seed=e)
        outs.append(predict_perturbformer(m, feats, masks, Yext, tr_all, q))
    preds["PerturbFormer"] = np.mean(outs, 0)
    for k in ("Linear-concat", "kNN-concat", "PerturbFormer"):
        preds[k + "+KD"] = knockdown_rule(preds[k], tp.perts, t_idx, sp.ctrl_mean, kd)

    true = tp.delta[:, ti]
    tstat, fdr = tgt.pdat.tstat[:, ti], tgt.pdat.fdr[:, ti]
    ref = src.Y[tr_all].mean(0)[si]
    rows, per = [], []
    overlap = np.array([p in set(sp.perts) for p in tp.perts])
    # cross-lab reference: Replogle measured delta for the same gene
    meas = np.zeros_like(true)
    pl = {p: i for i, p in enumerate(sp.perts)}
    for i, p in enumerate(tp.perts):
        if p in pl:
            meas[i] = src.Y[pl[p]][si]
    for name, Pm in list(preds.items()) + [("Replogle-measured (same gene)", None)]:
        if Pm is None:
            sub = overlap
            r = evaluate(meas[sub], true[sub], tstat[sub], fdr[sub], ref)
        else:
            sub = np.ones(len(tp.perts), bool)
            r = evaluate(Pm[:, si], true, tstat, fdr, ref)
        s = {k: float(np.nanmean(v)) for k, v in r.items()}
        s.update({"model": name, "n_perts": int(sub.sum())})
        if Pm is not None:   # same metrics restricted to the 41 overlapping genes for a fair comparison
            r2 = evaluate(Pm[overlap][:, si], true[overlap], tstat[overlap], fdr[overlap], ref)
            s.update({f"{k}_overlap41": float(np.nanmean(v)) for k, v in r2.items() if k in ("pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20")})
        rows.append(s)
        for i, p in enumerate(np.nonzero(sub)[0]):
            per.append({"model": name, "pert": tp.perts[p], **{k: float(v[i]) for k, v in r.items()}})
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(os.path.join(out_dir, "summary.csv"))
    pd.DataFrame(per).to_csv(os.path.join(out_dir, "per_perturbation.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "predictions.npz"), shared_genes=np.array(shared), perts=tp.perts,
                        true=true, measured_replogle=meas, overlap=overlap,
                        **{k.replace("-", "_").replace("+", "_plus_"): v[:, si] for k, v in preds.items()})
    pd.set_option("display.width", 220)
    print(df[["n_perts", "pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20", "rank",
              "pearson_de20_overlap41", "pearson_sys_overlap41"]].round(3))


if __name__ == "__main__":
    main()
