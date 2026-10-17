"""Step 7 - biological interpretation of the out-of-fold (unseen-gene) predictions.

  A. CRISPRi sanity check: is the targeted gene itself knocked down?
  B. Which functional classes of genes are predictable?
  C. Do predicted responses preserve protein-complex structure (CORUM)?
  D. Pathway-level consistency (MSigDB Hallmark)
  E. When does the model fail? (novelty of the gene / effect size)
  F. Which knowledge source does PerturbFormer trust for which class?
  G. Case studies (GATA1, RPS19, PSMB5, MED12): predicted vs measured, enrichment of
     predicted DEGs, retrieved "look-alike" perturbations

Outputs: results/interpretation/*.csv|md, figures/fig_interp_*.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp import plotting as vp  # noqa: E402
from vcp.data import ensure_dir  # noqa: E402
from vcp.interpret import enrichment, gene_class, pathway_scores, read_gmt  # noqa: E402
from vcp.pipeline import ROOT, load_bundle  # noqa: E402

DS = "replogle_k562_essential"
RES = os.path.join(ROOT, "results", DS)
OUT = ensure_dir(os.path.join(ROOT, "results", "interpretation"))
FIG = ensure_dir(os.path.join(ROOT, "figures"))
GS = os.path.join(ROOT, "data", "raw", "genesets")
CASES = ["RPS19", "MED12", "PSMB5", "GATA1"]   # three successes, one instructive failure
MAIN, CMP = "PerturbFormer+KD", ["kNN-concat+KD", "Linear-concat+KD", "Mean-shift"]
key = lambda m: m.replace("-", "_").replace("+", "_plus_")  # noqa: E731


def cosine_rows(A, B):
    return normalize(A) @ normalize(B).T


def main():
    vp.setup()
    b = load_bundle(DS)
    pdat, Y = b.pdat, b.Y
    genes, perts = list(pdat.genes), list(pdat.perts)
    oof = np.load(os.path.join(RES, "oof_predictions.npz"), allow_pickle=True)
    met = pd.read_csv(os.path.join(RES, "per_perturbation_metrics.csv"))
    fold = np.load(os.path.join(RES, "folds.npy"))
    pred = oof[key(MAIN)]
    cls = np.array([gene_class(p) for p in perts])
    report = []

    # ---------------- A. knockdown validation ----------------------------------------------
    t = b.target_idx
    m = t >= 0
    kd_lfc = np.array([Y[i, t[i]] for i in np.nonzero(m)[0]])
    kd_rel = np.array([pdat.pert_mean[i, t[i]] / max(pdat.ctrl_mean[t[i]], 1e-6) for i in np.nonzero(m)[0]])
    kd_sig = np.array([(pdat.fdr[i, t[i]] < 0.05) and (Y[i, t[i]] < 0) for i in np.nonzero(m)[0]])
    order = np.argsort(-np.abs(pdat.tstat), 1)
    kd_rank = np.array([int(np.nonzero(order[i] == t[i])[0][0]) + 1 for i in np.nonzero(m)[0]])
    pred_kd = np.array([pred[i, t[i]] for i in np.nonzero(m)[0]])
    report.append(f"## A. CRISPRi 敲低验证\n- 靶基因在测量基因中: {m.sum()} / {len(perts)}\n"
                  f"- 靶基因显著下调 (FDR<0.05): {kd_sig.mean():.1%}\n"
                  f"- 中位残余表达 (扰动/对照, log 空间): {np.median(kd_rel):.2f}\n"
                  f"- 靶基因是该扰动 |t| 排名第 1 的 DEG: {(kd_rank == 1).mean():.1%}; 排名前 20: {(kd_rank <= 20).mean():.1%}\n"
                  f"- KD 规则预测 vs 实测靶基因变化 Pearson r = {np.corrcoef(pred_kd, kd_lfc)[0, 1]:.3f}\n")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(kd_lfc, bins=40, color=vp.BLUE, edgecolor="white", linewidth=0.5)
    ax[0].axvline(0, color=vp.INK2, lw=1)
    ax[0].set_xlabel("靶基因 Δ log-expression (扰动 − 对照)")
    ax[0].set_ylabel("扰动数")
    ax[0].set_title(f"A  靶基因被显著敲低: {kd_sig.mean():.0%}")
    bins = [1, 2, 3, 6, 21, 101, 5001]
    lab = ["1", "2", "3-5", "6-20", "21-100", ">100"]
    cnt = np.histogram(kd_rank, bins=bins)[0]
    ax[1].bar(lab, cnt / cnt.sum(), color=vp.BLUE, width=0.6)
    ax[1].set_xlabel("靶基因在该扰动 DEG 中的排名 (|t|)")
    ax[1].set_ylabel("比例")
    ax[1].set_title("B  靶基因通常就是最强的 DEG")
    ax[1].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_interp_A_knockdown.png"))
    plt.close(fig)

    # ---------------- B. functional classes -------------------------------------------------
    rows = []
    for c in sorted(set(cls)):
        ps = set(np.array(perts)[cls == c])
        r = {"class": c, "n": len(ps), "median_nDEG": float(np.median((pdat.fdr[cls == c] < 0.05).sum(1)))}
        for mdl in [MAIN] + CMP:
            sub = met[(met.model == mdl) & met.pert.isin(ps)]
            r[f"{mdl}|pearson_sys"] = sub.pearson_sys.mean()
            r[f"{mdl}|pearson_de20"] = sub.pearson_de20.mean()
            r[f"{mdl}|sys_se"] = sub.pearson_sys.std() / np.sqrt(len(sub))
        rows.append(r)
    cdf = pd.DataFrame(rows).sort_values(f"{MAIN}|pearson_sys", ascending=False)
    cdf.to_csv(os.path.join(OUT, "class_performance.csv"), index=False)
    cdf_plot = cdf[cdf.n >= 5]
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(cdf_plot) + 1.2))
    yy = np.arange(len(cdf_plot))
    for mdl, off, col in [(MAIN, -0.18, vp.BLUE), ("kNN-concat+KD", 0.0, vp.ORANGE), ("Linear-concat+KD", 0.18, vp.AQUA)]:
        ax.errorbar(cdf_plot[f"{mdl}|pearson_sys"], yy + off, xerr=cdf_plot[f"{mdl}|sys_se"] if mdl == MAIN else None,
                    fmt="o", ms=5, color=col, ecolor=col, elinewidth=1, label=mdl, mec="white", mew=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(cdf_plot["class"], cdf_plot.n)])
    ax.invert_yaxis()
    ax.axvline(0, color=vp.AXIS, lw=1)
    ax.set_xlabel("去系统偏差 Pearson Δ (Systema), 未见基因")
    ax.set_title("不同功能类别基因的可预测性")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_interp_B_classes.png"))
    plt.close(fig)
    report.append("## B. 功能类别可预测性 (PerturbFormer+KD, pearson_sys)\n" +
                  cdf[["class", "n", "median_nDEG", f"{MAIN}|pearson_sys", f"{MAIN}|pearson_de20"]].round(3).to_markdown(index=False) + "\n")

    # ---------------- C. complex structure (CORUM) -----------------------------------------
    corum = read_gmt(os.path.join(GS, "CORUM.gmt"), min_size=2, universe=perts)
    g2c = {}
    for cname, gs in corum.items():
        for g in gs:
            g2c.setdefault(g, set()).add(cname)
    inC = np.array([p in g2c and any(len(corum[c]) > 1 for c in g2c[p]) for p in perts])
    share = np.zeros((len(perts), len(perts)), dtype=bool)
    for i, p in enumerate(perts):
        if p in g2c:
            for j, q in enumerate(perts):
                if i != j and q in g2c and g2c[p] & g2c[q]:
                    share[i, j] = True

    def nn_share(S, k=5):
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        nn = np.argsort(-S, 1)[:, :k]
        return np.array([share[i, nn[i]].mean() for i in range(len(perts))])

    E_prior = np.concatenate([normalize(b.static[s]) for s in b.static], 1)
    neigh = oof["neigh_idx"][:, :5]
    attn_share = np.array([share[i, neigh[i][neigh[i] >= 0]].mean() if (neigh[i] >= 0).any() else np.nan
                           for i in range(len(perts))])
    rand = share.sum(1) / (len(perts) - 1)
    cc = {
        "随机": rand,
        "实测应答谱之间": nn_share(cosine_rows(Y, Y)),
        "先验知识相似度 (GO+PPI+GenePT)": nn_share(cosine_rows(E_prior, E_prior)),
        "PerturbFormer 注意力检索邻居": attn_share,
        "预测应答谱 → 最近的实测应答谱": nn_share(cosine_rows(pred, Y)),
    }
    cres = pd.DataFrame({k: [np.nanmean(v[inC])] for k, v in cc.items()}).T.rename(columns={0: "frac_top5_same_complex"})
    cres.to_csv(os.path.join(OUT, "complex_recovery.csv"))
    fig, ax = plt.subplots(figsize=(7, 2.8))
    cols = [vp.GRAY_LIGHT, vp.INK2, vp.MUTED, vp.BLUE, vp.BLUE]
    vp.bar_h(ax, list(cres.index), list(cres.frac_top5_same_complex), cols, fmt="{:.1%}")
    ax.set_xlabel("Top-5 邻居与查询基因属于同一 CORUM 复合体的比例")
    ax.set_title(f"预测的扰动应答保留了蛋白复合体结构 (n={inC.sum()} 个复合体成员基因)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_interp_C_complex.png"))
    plt.close(fig)
    report.append("## C. CORUM 复合体结构保留\n" + cres.round(3).to_markdown() + "\n")

    # ---------------- D. pathway consistency -----------------------------------------------
    hall = read_gmt(os.path.join(GS, "MSigDB_Hallmark_2020.gmt"))
    Ts, names = pathway_scores(Y, genes, hall)
    rows = []
    for mdl in [MAIN] + CMP:
        Ps, _ = pathway_scores(oof[key(mdl)], genes, hall)
        r_path = [np.corrcoef(Ps[:, j], Ts[:, j])[0, 1] if Ps[:, j].std() > 0 else 0 for j in range(len(names))]
        r_pert = [spearmanr(Ps[i], Ts[i])[0] if Ps[i].std() > 0 else 0 for i in range(len(perts))]
        sign = (np.sign(Ps) == np.sign(Ts))[np.abs(Ts) > np.percentile(np.abs(Ts), 75)].mean()
        rows.append({"model": mdl, "mean_r_per_pathway": np.nanmean(r_path), "mean_spearman_per_pert": np.nanmean(r_pert),
                     "pathway_direction_acc_top25pct": sign})
        if mdl == MAIN:
            pr = pd.DataFrame({"pathway": names, "r_pred_vs_true": r_path, "true_sd": Ts.std(0)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "pathway_consistency_summary.csv"), index=False)
    pr = pr.sort_values("true_sd", ascending=False)
    pr.to_csv(os.path.join(OUT, "pathway_consistency_per_pathway.csv"), index=False)
    top = pr.head(20).sort_values("r_pred_vs_true", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    vp.bar_h(ax, list(top.pathway), list(top.r_pred_vs_true), [vp.BLUE] * len(top), fmt="{:.2f}", xlim=(0, 1))
    ax.set_xlabel("Pearson r (预测 vs 实测通路活性, 跨 1,092 个未见扰动)")
    ax.set_title("Hallmark 通路层面一致性 (变化最大的 20 条通路)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_interp_D_pathways.png"))
    plt.close(fig)
    report.append("## D. 通路一致性 (Hallmark)\n" + pd.DataFrame(rows).round(3).to_markdown(index=False) + "\n\n"
                  + top.round(3).to_markdown(index=False) + "\n")

    # ---------------- E. failure modes -------------------------------------------------------
    nov = np.zeros(len(perts))
    for f in np.unique(fold):
        te, tr = np.nonzero(fold == f)[0], np.nonzero(fold != f)[0]
        nov[te] = cosine_rows(E_prior[te], E_prior[tr]).max(1)
    mm = met[met.model == MAIN].set_index("pert").loc[perts]
    ndeg = (pdat.fdr < 0.05).sum(1)
    fdf = pd.DataFrame({"pert": perts, "max_prior_sim_to_train": nov, "n_deg": ndeg, "n_cells": pdat.n_cells,
                        "pearson_sys": mm.pearson_sys.values, "pearson_de20": mm.pearson_de20.values, "class": cls})
    fdf.to_csv(os.path.join(OUT, "failure_analysis.csv"), index=False)
    rho_nov = spearmanr(nov, fdf.pearson_sys)[0]
    rho_deg = spearmanr(ndeg, fdf.pearson_sys)[0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, x, xl, rho in [(ax[0], nov, "与训练集中最相似基因的先验相似度", rho_nov),
                          (ax[1], np.log10(ndeg + 1), "log10(#DEG + 1)  (扰动效应强度)", rho_deg)]:
        a.scatter(x, fdf.pearson_sys, s=6, color=vp.BLUE, alpha=0.35, lw=0)
        qs = np.quantile(x, np.linspace(0, 1, 9))
        mid = [(qs[i] + qs[i + 1]) / 2 for i in range(8)]
        mu = [fdf.pearson_sys[(x >= qs[i]) & (x <= qs[i + 1])].mean() for i in range(8)]
        a.plot(mid, mu, color=vp.INK, lw=2, marker="o", ms=4)
        a.set_xlabel(xl)
        a.set_ylabel("pearson_sys")
        a.set_title(f"Spearman ρ = {rho:.2f}")
    fig.suptitle("模型何时失败? — 缺少相似的已测基因 / 扰动效应弱", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_interp_E_failure.png"))
    plt.close(fig)
    report.append(f"## E. 失败模式\n- pearson_sys 与先验相似度 Spearman ρ = {rho_nov:.3f}\n"
                  f"- pearson_sys 与 #DEG Spearman ρ = {rho_deg:.3f}\n"
                  f"- 最差 10 个: {', '.join(fdf.sort_values('pearson_sys').pert.head(10))}\n"
                  f"- 最好 10 个: {', '.join(fdf.sort_values('pearson_sys', ascending=False).pert.head(10))}\n")

    # ---------------- F. knowledge-source weights -----------------------------------------
    gate = oof["gate"]
    src_names = list(b.static) + ["resp"]
    H = gate.shape[1] - len(src_names) - 1 if np.isfinite(gate).any() else 0
    srcw = gate[:, 3:3 + len(src_names)]                 # [head1, head2, param, src...]
    if np.isfinite(srcw).all():
        sdf = pd.DataFrame(srcw, columns=src_names)
        sdf["class"] = cls
        sdf["param_gate"] = gate[:, 2]
        g = sdf.groupby("class").mean()
        g = g.loc[[c for c in cdf_plot["class"]]]
        g.to_csv(os.path.join(OUT, "source_weights_by_class.csv"))
        fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(g) + 1.2))
        left = np.zeros(len(g))
        labels = {"go": "GO", "ppi": "STRING PPI", "genept": "GenePT (LLM)", "resp": "应答谱 Resp"}
        for s, col in zip(src_names, [vp.BLUE, vp.ORANGE, vp.AQUA, vp.YELLOW]):
            ax.barh(np.arange(len(g)), g[s], left=left, height=0.6, color=col, label=labels.get(s, s),
                    edgecolor="white", linewidth=2)
            left += g[s].values
        ax.set_yticks(np.arange(len(g)))
        ax.set_yticklabels(g.index)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("PerturbFormer 对各知识来源的平均注意力权重")
        ax.set_title("模型在不同基因类别上信任哪种先验知识")
        ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=8)
        ax.grid(axis="y", visible=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_interp_F_sources.png"))
        plt.close(fig)
        report.append("## F. 知识来源权重\n" + g.round(3).to_markdown() + "\n")

    # ---------------- G. case studies ------------------------------------------------------
    hall_u = read_gmt(os.path.join(GS, "MSigDB_Hallmark_2020.gmt"), universe=genes)
    gobp = read_gmt(os.path.join(GS, "GO_Biological_Process_2023.gmt"), universe=genes)
    sets = {**{f"[H] {k}": v for k, v in hall_u.items()}, **{f"[GO] {k}": v for k, v in gobp.items()}}
    fig, axes = plt.subplots(1, len(CASES), figsize=(4.0 * len(CASES), 3.9))
    case_md = ["## G. 案例分析 (全部为未见基因的 out-of-fold 预测)\n"]
    for ax, gname in zip(axes, CASES):
        i = perts.index(gname)
        yt, yp = Y[i], pred[i]
        top20 = order[i, :20]
        ax.scatter(yt, yp, s=4, color=vp.GRAY_LIGHT, lw=0)
        ax.scatter(yt[top20], yp[top20], s=16, color=vp.BLUE, edgecolor="white", lw=0.6, zorder=3)
        lim = np.abs(np.concatenate([yt, yp])).max() * 1.08
        ax.plot([-lim, lim], [-lim, lim], color=vp.AXIS, lw=1, zorder=1)
        ax.axhline(0, color=vp.GRID, lw=0.8)
        ax.axvline(0, color=vp.GRID, lw=0.8)
        special = [g for g in ("HBG1", "HBG2", "HBZ", "GATA1") if gname == "GATA1" and g in genes]
        cand = [genes.index(g) for g in special] or list(top20)
        placed = []
        for j in cand:   # label at most 4 well-separated genes (no overlapping labels)
            if len(placed) >= 4:
                break
            if all(abs(yt[j] - yt[k]) + abs(yp[j] - yp[k]) > 0.18 * lim for k in placed):
                placed.append(j)
                ax.annotate(genes[j], (yt[j], yp[j]), fontsize=7.5, color=vp.INK, xytext=(5, -10 if yp[j] > 0 else 6),
                            textcoords="offset points", fontstyle="italic")
                if special:
                    ax.scatter([yt[j]], [yp[j]], s=26, color=vp.RED, edgecolor="white", lw=0.8, zorder=4)
        r = np.corrcoef(yt, yp)[0, 1]
        rde = np.corrcoef(yt[top20], yp[top20])[0, 1]
        ax.set_title(f"{gname} 敲低  (r={r:.2f}, DE20 r={rde:.2f})", fontsize=10,
                     color=vp.INK if r > 0.6 else vp.RED)
        ax.set_xlabel("实测 Δ expression")
        ax.set_ylabel("预测 Δ expression" if gname == CASES[0] else "")
        # retrieval + enrichment
        nb = [perts[j] for j in oof["neigh_idx"][i, :5] if j >= 0]
        w = oof["neigh_w"][i, :5]
        case_md.append(f"### {gname}  (类别: {cls[i]}, #DEG={ndeg[i]}, 细胞数={pdat.n_cells[i]})\n"
                       f"- 预测 vs 实测: Pearson(全部基因) = {r:.3f}, Pearson(Top-20 DEG) = {rde:.3f}\n"
                       f"- PerturbFormer 检索到的相似扰动 (权重): " + ", ".join(f"{n} ({x:.2f})" for n, x in zip(nb, w)) + "\n"
                       f"- 实测 Top-10 下调: {', '.join(genes[j] for j in np.argsort(yt)[:10])}\n"
                       f"- 预测 Top-10 下调: {', '.join(genes[j] for j in np.argsort(yp)[:10])}\n"
                       f"- 实测 Top-10 上调: {', '.join(genes[j] for j in np.argsort(-yt)[:10])}\n"
                       f"- 预测 Top-10 上调: {', '.join(genes[j] for j in np.argsort(-yp)[:10])}\n")
        for direction, sgn in (("下调", 1), ("上调", -1)):
            hits_p = [genes[j] for j in np.argsort(sgn * yp)[:50]]
            hits_t = [genes[j] for j in np.argsort(sgn * yt)[:50]]
            ep = enrichment(hits_p, genes, sets, top=4)
            et = enrichment(hits_t, genes, sets, top=4)
            case_md.append(f"- 预测 Top-50 {direction}基因富集: " + "; ".join(f"{e[0]} (q={e[4]:.1e})" for e in ep) + "\n")
            case_md.append(f"- 实测 Top-50 {direction}基因富集: " + "; ".join(f"{e[0]} (q={e[4]:.1e})" for e in et) + "\n")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_interp_G_cases.png"))
    plt.close(fig)
    report += case_md

    with open(os.path.join(OUT, "interpretation_report.md"), "w") as fh:
        fh.write("# 生物学解释结果 (自动生成)\n\n" + "\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
