"""Virtual cell: input "cells + gene knockdown" -> predicted post-perturbation cell state.

Examples
  # 300 random K562 control cells, knock down GATA1 (treated as unseen: removed from memory)
  python scripts/predict.py --gene GATA1

  # your own cells (h5ad with log-normalised expression; genes matched by symbol)
  python scripts/predict.py --gene MED12 --cells my_cells.h5ad --out results/predictions/MED12

  # a gene that was never perturbed in the screen (any gene with GO / STRING / GenePT knowledge)
  python scripts/predict.py --gene TP53

Outputs in --out:
  report.md                  human-readable summary (top genes, pathways, evidence)
  pseudobulk_prediction.csv  per-gene control mean, predicted mean, predicted delta (+ measured delta)
  predicted_cells.h5ad       predicted single cells (X = log-normalised expression, obsm["X_pca"])
  prediction.png             PCA view of control / predicted (/ measured) cells + top genes
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp import plotting as vp  # noqa: E402
from vcp.data import ensure_dir  # noqa: E402
from vcp.predictor import VirtualCell  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_cells(path, genes):
    import anndata as ad
    import scipy.sparse as sp
    a = ad.read_h5ad(path)
    names = a.var["gene_name"].astype(str).values if "gene_name" in a.var else a.var_names.astype(str).values
    idx = {g: i for i, g in enumerate(names)}
    X = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    out = np.zeros((a.n_obs, len(genes)), np.float32)
    found = [(j, idx[g]) for j, g in enumerate(genes) if g in idx]
    for j, i in found:
        out[:, j] = X[:, i]
    print(f"loaded {a.n_obs} cells; {len(found)}/{len(genes)} model genes found")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gene", required=True)
    ap.add_argument("--cells", default=None, help="h5ad / npy of log-normalised control cells")
    ap.add_argument("--n_cells", type=int, default=300)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "k562"))
    ap.add_argument("--seen", action="store_true", help="allow using the measured response of this gene")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    vc = VirtualCell.load(args.model, gmt=os.path.join(ROOT, "data", "raw", "genesets", "MSigDB_Hallmark_2020.gmt"))
    if not vc.known_gene(args.gene):
        sys.exit(f"{args.gene}: no prior knowledge (GO/STRING/GenePT) available - cannot encode this perturbation")
    cells = None
    if args.cells:
        cells = np.load(args.cells) if args.cells.endswith(".npy") else load_cells(args.cells, vc.genes)
    out = vc.predict(args.gene, cells=cells, n_cells=args.n_cells, exclude_self=not args.seen)
    rep = vc.report(out)
    print(rep)

    od = ensure_dir(args.out or os.path.join(ROOT, "results", "predictions", args.gene))
    with open(os.path.join(od, "report.md"), "w") as fh:
        fh.write(rep + "\n")
    df = pd.DataFrame({"gene": vc.genes, "control_mean": vc.m["ctrl_mean"],
                       "predicted_mean": vc.m["ctrl_mean"] + out["delta"], "predicted_delta": out["delta"]})
    if "measured_delta" in out:
        df["measured_delta"] = out["measured_delta"]
    df.sort_values("predicted_delta", key=np.abs, ascending=False).to_csv(
        os.path.join(od, "pseudobulk_prediction.csv"), index=False)
    try:
        import anndata as ad
        a = ad.AnnData(X=out["cells"].astype(np.float32))
        a.var_names = vc.genes
        a.obsm["X_pca"] = out["z_pred"]
        a.obsm["X_pca_control"] = out["z_control"]
        a.uns["perturbation"] = args.gene
        a.write_h5ad(os.path.join(od, "predicted_cells.h5ad"))
    except Exception as e:   # pragma: no cover
        print("could not write h5ad:", e)

    # figure
    vp.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    z0, z1 = out["z_control"], out["z_pred"]
    ax[0].scatter(z0[:, 0], z0[:, 1], s=6, color=vp.GRAY_LIGHT, label="对照细胞 (输入)", lw=0)
    ax[0].scatter(z1[:, 0], z1[:, 1], s=6, color=vp.BLUE, label=f"预测: {args.gene} 敲低后", lw=0)
    for i in range(0, len(z0), max(1, len(z0) // 40)):
        ax[0].annotate("", xy=z1[i, :2], xytext=z0[i, :2],
                       arrowprops={"arrowstyle": "-", "color": vp.AXIS, "lw": 0.5})
    ax[0].set_xlabel("PC1")
    ax[0].set_ylabel("PC2")
    ax[0].set_title("单细胞状态转移 (流匹配)")
    ax[0].legend(fontsize=8, loc="best")
    d = out["delta"]
    top = np.argsort(-np.abs(d))[:15][::-1]
    cols = [vp.DIV_POS if d[i] > 0 else vp.DIV_NEG for i in top]
    ax[1].barh([vc.genes[i] for i in top], d[top], color=cols, height=0.6)
    if "measured_delta" in out:
        ax[1].scatter(out["measured_delta"][top], range(len(top)), color=vp.INK, s=14, zorder=3, label="实测")
        ax[1].legend(fontsize=8, loc="lower right")
    ax[1].axvline(0, color=vp.AXIS, lw=1)
    ax[1].set_xlabel("预测 Δ log-expression")
    ax[1].set_title("变化最大的基因 (红=上调, 蓝=下调)")
    ax[1].grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(od, "prediction.png"))
    print(f"\nsaved to {od}")


if __name__ == "__main__":
    main()
