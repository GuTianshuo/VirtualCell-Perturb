"""Step 11 - control / perturbed cell representation.

  A. single-cell state map: UMAP of the 100-d PCA latent; control cells vs cells of three
     well-defined perturbation classes (ribosome, proteasome, Mediator)
  B. perturbation response map: UMAP of the 1,092 pseudobulk response profiles; do knockdowns
     of the same complex land together?  (the structure PerturbFormer's retrieval relies on)
  C. how much of each perturbation's mean response lies inside the 100-d latent space, when the
     space is fitted WITHOUT that perturbation (from 05_flow.py, fold-specific PCA)

Outputs: figures/fig_repr.png, results/interpretation/representation.md
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp import plotting as vp  # noqa: E402
from vcp.data import ensure_dir  # noqa: E402
from vcp.interpret import gene_class  # noqa: E402
from vcp.pipeline import PROC, ROOT, load_bundle  # noqa: E402

DS = "replogle_k562_essential"
CLASSES = [("胞质核糖体 Ribosome", vp.BLUE, "核糖体"), ("蛋白酶体 Proteasome", vp.ORANGE, "蛋白酶体"),
           ("Mediator 复合体", vp.AQUA, "Mediator")]


def knn_purity(E, cls, named, k=10):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    m = np.isin(cls, named)
    return float((cls[idx[m]] == cls[m][:, None]).mean())


def main():
    vp.setup()
    out = ensure_dir(os.path.join(ROOT, "results", "interpretation"))
    b = load_bundle(DS)
    perts = np.array(b.pdat.perts)
    cls = np.array([gene_class(p) for p in perts])
    cells = np.load(os.path.join(PROC, f"{DS}_cells.npz"), allow_pickle=True)
    Z, lab, ctrl_label = cells["Z"], cells["label"], int(cells["ctrl_label"])
    rng = np.random.default_rng(0)
    md = ["# 细胞与扰动的表征 (自动生成)", ""]

    # ---- A. single-cell map -------------------------------------------------------------------
    sel, grp = [rng.choice(np.nonzero(lab == ctrl_label)[0], 2500, replace=False)], ["对照"] * 2500
    for name, _, short in CLASSES:
        pidx = np.nonzero(cls == name)[0]
        c = np.concatenate([rng.choice(np.nonzero(lab == p)[0], min(25, (lab == p).sum()), replace=False) for p in pidx])
        sel.append(c)
        grp += [short] * len(c)
    sel = np.concatenate(sel)
    grp = np.array(grp)
    U = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z[sel][:, :50])
    # how far does each class move from control in latent space (mean shift / control spread)?
    ctrl_sd = np.sqrt(Z[lab == ctrl_label][:, :50].var(0).sum())
    shift = {short: float(np.linalg.norm(Z[sel][grp == short][:, :50].mean(0) - Z[sel][grp == "对照"][:, :50].mean(0)) / ctrl_sd)
             for _, _, short in CLASSES}

    # ---- B. response map ----------------------------------------------------------------------
    Yc = b.Y - b.Y.mean(0)
    Uy, Sy, _ = np.linalg.svd(Yc, full_matrices=False)
    E = Uy[:, :30] * Sy[:30]
    R = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=0).fit_transform(E)
    named = [c for c, _, _ in CLASSES]
    pur = knn_purity(E, cls, named)
    pur_null = np.mean([knn_purity(E, rng.permutation(cls), named) for _ in range(20)])

    # ---- C. latent-space capture of unseen responses -----------------------------------------
    cap_p = os.path.join(ROOT, "results", DS, "flow_pca_capture.csv")
    cap = pd.read_csv(cap_p) if os.path.exists(cap_p) else None

    fig, ax = plt.subplots(1, 3 if cap is not None else 2, figsize=(15 if cap is not None else 10.5, 4.6))
    a = ax[0]
    a.scatter(U[grp == "对照", 0], U[grp == "对照", 1], s=3, color=vp.GRAY_LIGHT, lw=0, label="对照细胞")
    for _, col, short in CLASSES:
        m = grp == short
        a.scatter(U[m, 0], U[m, 1], s=4, color=col, lw=0, alpha=0.8, label=f"{short}敲低 (位移 {shift[short]:.2f}σ)")
    a.set_title("A  单细胞状态: 对照 vs 扰动细胞 (UMAP)")
    a.set_xticks([])
    a.set_yticks([])
    a.legend(fontsize=7.5, markerscale=3, loc="best")
    a = ax[1]
    other = ~np.isin(cls, named)
    a.scatter(R[other, 0], R[other, 1], s=6, color=vp.GRAY_LIGHT, lw=0, label="其他基因")
    for name, col, short in CLASSES:
        m = cls == name
        a.scatter(R[m, 0], R[m, 1], s=16, color=col, edgecolor="white", lw=0.5, label=f"{short} (n={m.sum()})")
    a.set_title(f"B  1,092 个扰动的应答谱 (UMAP)\n10 近邻同类比例 {pur:.0%} (随机 {pur_null:.0%})", fontsize=10.5)
    a.set_xticks([])
    a.set_yticks([])
    a.legend(fontsize=7.5, loc="best")
    if cap is not None:
        a = ax[2]
        te = cap[cap.split == "test"].frac_delta_captured
        tr = cap[cap.split == "train"].frac_delta_captured
        bins = np.linspace(0, 1, 26)
        a.hist(tr, bins=bins, color=vp.GRAY_LIGHT, density=True, label=f"训练扰动 (中位数 {tr.median():.0%})")
        a.hist(te, bins=bins, color=vp.BLUE, alpha=0.75, density=True, label=f"未见扰动 (中位数 {te.median():.0%})")
        a.set_xlabel("平均响应 Δ 落在 100 维潜空间内的比例")
        a.set_ylabel("密度")
        a.set_title("C  潜空间保留了未见扰动的大部分效应", fontsize=10.5)
        a.legend(fontsize=8, loc="upper left")
        md.append(f"- C. 每折只用训练细胞拟合的 100 维 PCA: 未见扰动的平均响应 Δ 落在潜空间内的比例中位数 {te.median():.1%} "
                  f"(训练扰动 {tr.median():.1%}); 而单细胞总方差只有约 15.5% 落在其中——其余主要是测量噪声")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figures", "fig_repr.png"))
    plt.close(fig)

    md.insert(2, "- A. 相对对照细胞的潜空间平均位移 (以对照细胞总标准差为单位): "
              + ", ".join(f"{k} {v:.2f}" for k, v in shift.items()))
    md.insert(3, f"- B. 应答谱空间中, 三类复合体基因的 10 近邻同类比例 {pur:.1%}, 随机打乱类别后 {pur_null:.1%}")
    import json
    with open(os.path.join(out, "representation.json"), "w") as fh:
        json.dump({"shift_sd": shift, "knn_purity": pur, "knn_purity_null": float(pur_null),
                   "capture_test_median": None if cap is None else float(te.median()),
                   "capture_train_median": None if cap is None else float(tr.median())}, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "representation.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
