"""Step 1 - preprocessing of a GEARS-format Perturb-seq h5ad.

Outputs (data/processed/):
  {name}_pseudobulk.npz : per-perturbation mean / variance / Welch-t / FDR vs control
  {name}_cells.npz      : PCA latent of every single cell + perturbation label
                          (used by the single-cell conditional flow-matching model)

Usage:
  python scripts/01_preprocess.py --dataset replogle_k562_essential
  python scripts/01_preprocess.py --dataset adamson
"""
import argparse
import os
import sys
import time

import anndata as ad
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.data import CTRL, PerturbData, bh_fdr, ensure_dir, group_mean_var, parse_condition, welch_t  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replogle_k562_essential")
    ap.add_argument("--min_cells", type=int, default=15)
    ap.add_argument("--n_pcs", type=int, default=100)
    ap.add_argument("--pca_cells_per_pert", type=int, default=40)
    ap.add_argument("--pca_ctrl_cells", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    path = os.path.join(ROOT, "data", "raw", args.dataset, "perturb_processed.h5ad")
    out_dir = ensure_dir(os.path.join(ROOT, "data", "processed"))
    adata = ad.read_h5ad(path, backed="r")
    genes = adata.var["gene_name"].astype(str).values
    cond = adata.obs["condition"].astype(str).map(parse_condition).values
    print(f"[{args.dataset}] {adata.shape[0]} cells x {adata.shape[1]} genes")

    # ---- perturbation labels (single-gene perturbations only) --------------------------
    is_ctrl = cond == CTRL
    single = np.array([(c != CTRL) and ("+" not in c) for c in cond])
    uniq, counts = np.unique(cond[single], return_counts=True)
    perts = uniq[counts >= args.min_cells]
    pert_to_idx = {p: i for i, p in enumerate(perts)}
    labels = np.full(len(cond), -1, dtype=np.int64)
    for i, c in enumerate(cond):
        if c in pert_to_idx:
            labels[i] = pert_to_idx[c]
    ctrl_label = len(perts)
    labels[is_ctrl] = ctrl_label
    print(f"  perturbations kept: {len(perts)} (min_cells={args.min_cells}); control cells: {is_ctrl.sum()}")

    # ---- pseudobulk statistics ------------------------------------------------------------
    mean, var, cnt = group_mean_var(adata.X, labels, len(perts) + 1)
    ctrl_mean, ctrl_var, n_ctrl = mean[ctrl_label], var[ctrl_label], int(cnt[ctrl_label])
    pert_mean, pert_var, n_cells = mean[:-1], var[:-1], cnt[:-1]
    tstat, pval = welch_t(pert_mean, pert_var, n_cells[:, None], ctrl_mean[None], ctrl_var[None], n_ctrl)
    fdr = bh_fdr(pval, axis=1)
    print(f"  pseudobulk done ({time.time() - t0:.0f}s)")

    gene_idx = {g: i for i, g in enumerate(genes)}
    target_idx = np.array([gene_idx.get(p, -1) for p in perts])
    pd_obj = PerturbData(
        name=args.dataset, genes=genes, perts=perts, ctrl_mean=ctrl_mean, ctrl_var=ctrl_var,
        n_ctrl=n_ctrl, pert_mean=pert_mean, pert_var=pert_var, n_cells=n_cells,
        tstat=tstat, fdr=fdr, extra={"target_idx": target_idx},
    )

    # ---- PCA latent space of single cells --------------------------------------------------
    rng = np.random.default_rng(args.seed)
    chosen = []
    for k in range(len(perts) + 1):
        idx = np.nonzero(labels == k)[0]
        m = args.pca_ctrl_cells if k == ctrl_label else args.pca_cells_per_pert
        chosen.append(rng.choice(idx, size=min(m, len(idx)), replace=False))
    chosen = np.sort(np.concatenate(chosen))
    rows, chunk = [], 20000
    for start in range(0, adata.shape[0], chunk):
        end = min(start + chunk, adata.shape[0])
        sel = chosen[(chosen >= start) & (chosen < end)] - start
        if len(sel):
            rows.append(sp.csr_matrix(adata.X[start:end])[sel].toarray())
    Xs = np.vstack(rows).astype(np.float32) - ctrl_mean[None]
    pca = PCA(n_components=args.n_pcs, svd_solver="randomized", random_state=args.seed).fit(Xs)
    del Xs
    comps = pca.components_.astype(np.float32)          # (k, G)
    print(f"  PCA({args.n_pcs}) explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    Z = np.zeros((adata.shape[0], args.n_pcs), dtype=np.float32)
    offset = (ctrl_mean[None] @ comps.T).astype(np.float32)
    for start in range(0, adata.shape[0], chunk):
        end = min(start + chunk, adata.shape[0])
        xb = sp.csr_matrix(adata.X[start:end])
        Z[start:end] = np.asarray(xb @ comps.T) - offset

    keep = labels >= 0
    pd_obj.extra["pca_explained_var"] = pca.explained_variance_ratio_.astype(np.float32)
    pd_obj.save(os.path.join(out_dir, f"{args.dataset}_pseudobulk.npz"))
    np.savez_compressed(
        os.path.join(out_dir, f"{args.dataset}_cells.npz"),
        Z=Z[keep], label=labels[keep], ctrl_label=ctrl_label, components=comps,
        ctrl_mean=ctrl_mean, explained_var=pca.explained_variance_ratio_.astype(np.float32),
        barcodes=np.asarray(adata.obs_names)[keep],
    )
    print(f"  saved to {out_dir} ({time.time() - t0:.0f}s total)")

    # ---- quick summary ------------------------------------------------------------------------
    sig = (fdr < 0.05).sum(1)
    print(f"  #DEG (FDR<0.05) per perturbation: median {np.median(sig):.0f}, "
          f"mean {sig.mean():.0f}, max {sig.max()}")
    measured = target_idx >= 0
    kd = pert_mean[measured, target_idx[measured]] - ctrl_mean[target_idx[measured]]
    print(f"  target gene measured for {measured.sum()} perts; median target log-FC {np.median(kd):.3f}")


if __name__ == "__main__":
    main()
