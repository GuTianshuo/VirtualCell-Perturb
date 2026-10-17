"""Step 4 - technical noise ceiling (split-half reproducibility).

Cells of every perturbation and of the control population are randomly split into two halves.
The response measured in half A is used as a "prediction" of the response measured in half B.
No model can be expected to beat this (up to the Spearman-Brown correction for using half
of the cells), so it calibrates how good the benchmark numbers really are.

Output: results/{dataset}/noise_ceiling.csv
"""
import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.data import CTRL, PerturbData, bh_fdr, ensure_dir, group_mean_var, parse_condition, welch_t  # noqa: E402
from vcp.metrics import evaluate  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replogle_k562_essential")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pdat = PerturbData.load(os.path.join(ROOT, "data", "processed", f"{args.dataset}_pseudobulk.npz"))
    adata = ad.read_h5ad(os.path.join(ROOT, "data", "raw", args.dataset, "perturb_processed.h5ad"), backed="r")
    cond = adata.obs["condition"].astype(str).map(parse_condition).values
    P = len(pdat.perts)
    lut = {p: i for i, p in enumerate(pdat.perts)}
    lab = np.array([lut.get(c, P if c == CTRL else -1) for c in cond])
    rng = np.random.default_rng(args.seed)
    half = rng.integers(0, 2, len(lab))
    lab2 = np.where(lab >= 0, lab + half * (P + 1), -1)
    mean, var, cnt = group_mean_var(adata.X, lab2, 2 * (P + 1))
    mA, vA, nA = mean[: P + 1], var[: P + 1], cnt[: P + 1]
    mB, vB, nB = mean[P + 1:], var[P + 1:], cnt[P + 1:]
    dA = mA[:P] - mA[P][None]
    dB = mB[:P] - mB[P][None]
    tB, pB = welch_t(mB[:P], vB[:P], nB[:P, None], mB[P][None], vB[P][None], nB[P])
    fB = bh_fdr(pB, axis=1)
    ref = dA.mean(0)
    r = evaluate(dA, dB, tB, fB, ref, np.sqrt(pdat.ctrl_var))
    df = pd.DataFrame({k: v for k, v in r.items()})
    df.insert(0, "pert", pdat.perts)
    out_dir = ensure_dir(os.path.join(ROOT, "results", args.dataset))
    df.to_csv(os.path.join(out_dir, "noise_ceiling_per_pert.csv"), index=False)
    summ = df.drop(columns="pert").mean()
    sb = {}
    for k in ("pearson_all", "pearson_de20", "pearson_sys"):
        rr = summ[k]
        sb[k + "_spearman_brown"] = 2 * rr / (1 + rr)
        # attainable ceiling for "prediction vs full-data measurement": a perfect predictor of the
        # true mean response correlates with the (noisy) full-data estimate at sqrt(reliability)
        ri = np.clip(df[k].values, 0, None)
        sb[k + "_pred_ceiling"] = float(np.nanmean(np.sqrt(2 * ri / (1 + ri))))
    summ = pd.concat([summ, pd.Series(sb)])
    summ.to_csv(os.path.join(out_dir, "noise_ceiling.csv"), header=["value"])
    print(summ.round(3))


if __name__ == "__main__":
    main()
