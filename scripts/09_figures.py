"""Step 9 - benchmark figures and statistics.

  fig1_data_overview.png        dataset, effect sizes, noise ceiling
  fig2_benchmark.png            main comparison (mean ± 95 % bootstrap CI) vs noise ceiling
  fig3_priors.png               which prior knowledge generalises (Linear / kNN per source)
  fig4_ablation.png             PerturbFormer ablations
  fig5_paired.png               per-perturbation paired comparison
  fig6_strata.png               performance vs effect size
  fig7_flow.png                 single-cell generation (energy distance, variance) + examples
  fig8_external.png             Replogle -> Adamson external validation
  results/<ds>/stats_tests.csv  paired Wilcoxon tests (PerturbFormer+KD vs every model)
  results/<ds>/main_table.md    publication table
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp import plotting as vp  # noqa: E402
from vcp.data import PerturbData, ensure_dir  # noqa: E402
from vcp.pipeline import PROC, ROOT  # noqa: E402

DS = "replogle_k562_essential"
RES = os.path.join(ROOT, "results", DS)
FIG = ensure_dir(os.path.join(ROOT, "figures"))
MAIN = "PerturbFormer+KD"
SHOW = ["No-change", "Mean-shift", "Linear-concat+KD", "kNN-concat+KD", "PerturbNet-attn+KD", MAIN]
METRICS = [("pearson_all", "Pearson Δ (全部基因)"), ("pearson_de20", "Pearson Δ (Top-20 DEG)"),
           ("pearson_sys", "去系统偏差 Pearson (Systema)"), ("dir_acc_de20", "DEG 方向准确率 (Top-20)"),
           ("de_overlap20", "Top-20 DEG 重合率"), ("rank", "扰动识别排名 (越低越好)")]
LABEL = {"No-change": "No-change (不变)", "Mean-shift": "Mean-shift (均值平移)",
         "Linear-concat+KD": "Linear (Ahlmann-Eltze)", "kNN-concat+KD": "kNN (多源先验)",
         "PerturbNet-attn+KD": "PerturbNet (MLP/Transformer)", MAIN: "PerturbFormer (本文)"}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    vp.setup()
    met = pd.read_csv(os.path.join(RES, "per_perturbation_metrics.csv"))
    nc = pd.read_csv(os.path.join(RES, "noise_ceiling.csv"), index_col=0)["value"]
    pdat = PerturbData.load(os.path.join(PROC, f"{DS}_pseudobulk.npz"))
    models = list(met.model.unique())

    # ---------------- stats: paired Wilcoxon + bootstrap CI ------------------------------
    piv = {m: met.pivot(index="pert", columns="model", values=m) for m, _ in METRICS}
    rows = []
    for mdl in models:
        r = {"model": mdl}
        for m, _ in METRICS:
            mu, lo, hi = boot_ci(piv[m][mdl].values)
            r[m] = mu
            r[m + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            if mdl != MAIN:
                a, b_ = piv[m][MAIN], piv[m][mdl]
                ok = ~(a.isna() | b_.isna())
                diff = (a[ok] - b_[ok])
                r[m + "_p"] = wilcoxon(a[ok], b_[ok]).pvalue if (diff != 0).any() else 1.0
                better = diff < 0 if m == "rank" else diff > 0
                r[m + "_win"] = float(better.mean())
        rows.append(r)
    st = pd.DataFrame(rows).set_index("model").sort_values("pearson_de20", ascending=False)
    st.to_csv(os.path.join(RES, "stats_tests.csv"))

    # publication table
    lines = ["| 模型 | " + " | ".join(lbl for _, lbl in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for mdl in [m for m in SHOW[::-1]] + [m for m in st.index if m not in SHOW]:
        if mdl not in st.index:
            continue
        cells = []
        for m, _ in METRICS:
            v = st.loc[mdl, m]
            p = st.loc[mdl].get(m + "_p", np.nan)
            star = "" if mdl == MAIN or np.isnan(p) else ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else " (n.s.)")
            cells.append(f"{v:.3f}{star}")
        lines.append(f"| {mdl} | " + " | ".join(cells) + " |")
    lines.append(f"| *细胞拆半一致性 (参考)* | {nc['pearson_all']:.3f} | {nc['pearson_de20']:.3f} | "
                 f"{nc['pearson_sys']:.3f} | {nc['dir_acc_de20']:.3f} | {nc['de_overlap20']:.3f} | {nc['rank']:.3f} |")
    lines.append(f"| *预测可达上限 √(SB 校正一致性)* | {nc['pearson_all_pred_ceiling']:.3f} | {nc['pearson_de20_pred_ceiling']:.3f} | "
                 f"{nc['pearson_sys_pred_ceiling']:.3f} | — | — | — |")
    with open(os.path.join(RES, "main_table.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n\n星号: 与 PerturbFormer+KD 的配对 Wilcoxon 检验 (*p<0.05, **p<0.01, ***p<0.001); 星号只表示差异显著, 不表示方向, 请比较数值; "
                 f"n = {met.pert.nunique()} 个未见扰动 (5 折交叉验证 out-of-fold)。\n")
    print("\n".join(lines))

    # ---------------- fig1 data overview -------------------------------------------------
    ndeg = (pdat.fdr < 0.05).sum(1)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].hist(pdat.n_cells, bins=np.logspace(np.log10(15), np.log10(pdat.n_cells.max()), 40), color=vp.BLUE,
               edgecolor="white", linewidth=0.4)
    ax[0].set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator
    ax[0].xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
    ax[0].xaxis.set_minor_locator(NullLocator())
    ax[0].set_xticklabels(["20", "50", "100", "200", "500", "1000"])
    ax[0].set_xlabel("每个扰动的细胞数")
    ax[0].set_ylabel("扰动数")
    ax[0].set_title(f"A  {len(pdat.perts)} 个 CRISPRi 敲低 + {pdat.n_ctrl:,} 对照细胞")
    ax[1].hist(np.log10(ndeg + 1), bins=40, color=vp.BLUE, edgecolor="white", linewidth=0.4)
    ax[1].set_xlabel("log10(#DEG + 1)  (FDR < 0.05)")
    ax[1].set_ylabel("扰动数")
    ax[1].set_title(f"B  效应强度差异巨大 (中位数 {np.median(ndeg):.0f} 个 DEG)")
    keys = ["pearson_all", "pearson_de20", "pearson_sys"]
    lbl = ["Pearson\n全部基因", "Pearson\nTop-20 DEG", "去偏\nPearson"]
    xx = np.arange(len(keys))
    a1 = [nc[k] for k in keys]
    a2 = [nc[k + "_pred_ceiling"] for k in keys]
    ax[2].bar(xx - 0.18, a1, width=0.34, color=vp.GRAY_LIGHT, label="细胞拆半一致性")
    ax[2].bar(xx + 0.18, a2, width=0.34, color=vp.INK2, label="预测可达上限")
    for i in range(len(keys)):
        ax[2].text(xx[i] - 0.18, a1[i] + 0.02, f"{a1[i]:.2f}", ha="center", fontsize=8.5, color=vp.INK2)
        ax[2].text(xx[i] + 0.18, a2[i] + 0.02, f"{a2[i]:.2f}", ha="center", fontsize=8.5, color=vp.INK2)
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels(lbl)
    ax[2].set_ylim(0, 1.12)
    ax[2].legend(fontsize=8, loc="upper left")
    ax[2].set_title("C  测量噪声决定的上限")
    ax[2].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_data_overview.png"))
    plt.close(fig)

    # ---------------- fig2 main benchmark ------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for ax, (m, lbl) in zip(axes.ravel(), METRICS):
        vals, errs, cols, names = [], [], [], []
        for mdl in SHOW:
            mu, lo, hi = boot_ci(piv[m][mdl].values)
            vals.append(mu)
            errs.append([mu - lo, hi - mu])
            cols.append(vp.model_color(mdl))
            names.append(LABEL[mdl])
        y = np.arange(len(SHOW))
        ax.barh(y, vals, height=0.6, color=cols, xerr=np.array(errs).T, error_kw={"ecolor": vp.INK2, "elinewidth": 0.8})
        for i, v in enumerate(vals):
            ax.text(max(v, 0) + 0.015, i, f"{v:.3f}", va="center", fontsize=8, color=vp.INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax in axes[:, 0] else [""] * len(y), fontsize=8.5)
        ax.invert_yaxis()
        ce = nc.get(m + "_pred_ceiling")
        if ce is not None and not np.isnan(ce):
            ax.axvline(ce, color=vp.INK, lw=1.2)
            ax.text(ce - 0.012, len(SHOW) - 0.62, f"可达上限 {ce:.2f}", fontsize=7.5, color=vp.INK, ha="right")
        ax.set_title(lbl, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 1.0 if m != "rank" else 0.6)
    fig.suptitle(f"未见基因扰动预测 (Replogle K562, {met.pert.nunique()} 个扰动, 5 折 CV, 误差线 = 95% bootstrap CI; "
                 "竖线 = 测量噪声决定的可达上限)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_benchmark.png"))
    plt.close(fig)

    # ---------------- fig3 priors ---------------------------------------------------------
    srcs = [("go", "GO 注释"), ("ppi", "STRING PPI"), ("genept", "GenePT\n(LLM 文本)"), ("resp", "应答谱\n(数据驱动)"), ("concat", "多源融合")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a, m in zip(ax, ["pearson_de20", "pearson_sys"]):
        x = np.arange(len(srcs))
        for k, (fam, col, off) in enumerate([("Linear", vp.AQUA, -0.2), ("kNN", vp.ORANGE, 0.2)]):
            v = [piv[m].get(f"{fam}-{s}", pd.Series([np.nan])).mean() for s, _ in srcs]
            a.bar(x + off, v, width=0.36, color=col, label=f"{fam} + 该先验", edgecolor="white", linewidth=2)
        ms = piv[m]["Mean-shift"].mean()
        if ms > 0.01:
            a.axhline(ms, color=vp.MUTED, lw=1.2)
            a.text(len(srcs) - 0.55, ms + 0.012, f"Mean-shift {ms:.2f}", fontsize=8, color=vp.INK2, ha="right")
        a.set_xticks(x)
        a.set_xticklabels([l for _, l in srcs], fontsize=8.5)
        a.set_title(dict(METRICS)[m])
        a.grid(axis="x", visible=False)
        a.set_ylim(0, max(0.8, a.get_ylim()[1]) if m == "pearson_de20" else 0.55)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.93), fontsize=9)
    fig.suptitle("哪种先验知识能泛化到未见基因?", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(FIG, "fig3_priors.png"))
    plt.close(fig)

    # ---------------- fig4 ablation ---------------------------------------------------------
    abl = [("PerturbFormer+KD", "完整模型"), ("PerturbFormer", "− KD 敲低规则"),
           ("Ablation-no-go+KD", "− GO"), ("Ablation-no-ppi+KD", "− PPI"), ("Ablation-no-genept+KD", "− GenePT"),
           ("Ablation-no-resp+KD", "− 应答谱"), ("Ablation-no-dirloss+KD", "− 方向损失"),
           ("Ablation-concat-fusion+KD", "拼接融合 (无 Transformer)"),
           ("Ablation-free-attention+KD", "自由注意力 (无知识核)"), ("Ablation-no-topk+KD", "− top-k 稀疏")]
    abl = [(k, l) for k, l in abl if k in piv["pearson_sys"].columns]
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.36 * len(abl) + 1.4))
    for a, m in zip(ax, ["pearson_de20", "pearson_sys"]):
        full = piv[m][MAIN]
        d = [(piv[m][k] - full).mean() for k, _ in abl]
        se = [(piv[m][k] - full).std() / np.sqrt(len(full)) for k, _ in abl]
        cols = [vp.BLUE if k == MAIN else (vp.RED if v < 0 else vp.GREEN) for (k, _), v in zip(abl, d)]
        a.barh(range(len(abl)), d, xerr=se, color=cols, height=0.6, error_kw={"ecolor": vp.INK2, "elinewidth": 0.8})
        a.set_yticks(range(len(abl)))
        a.set_yticklabels([l for _, l in abl] if a is ax[0] else [""] * len(abl))
        a.invert_yaxis()
        a.axvline(0, color=vp.INK2, lw=1)
        a.set_title(f"Δ {dict(METRICS)[m]} (相对完整模型)", fontsize=10)
        a.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_ablation.png"))
    plt.close(fig)

    # ---------------- fig5 paired ------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for a, other in zip(ax, ["kNN-concat+KD", "Linear-concat+KD"]):
        x, y = piv["pearson_sys"][other], piv["pearson_sys"][MAIN]
        a.scatter(x, y, s=7, color=vp.BLUE, alpha=0.45, lw=0)
        lim = [min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05]
        a.plot(lim, lim, color=vp.INK2, lw=1)
        win = (y > x).mean()
        p = wilcoxon(y, x).pvalue
        a.set_title(f"vs {LABEL[other]}: 胜率 {win:.0%}, Wilcoxon p={p:.1e}", fontsize=9.5)
        a.set_xlabel(f"{LABEL[other]}  去偏 Pearson")
        a.set_ylabel("PerturbFormer  去偏 Pearson")
        a.set_xlim(lim)
        a.set_ylim(lim)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_paired.png"))
    plt.close(fig)

    # ---------------- fig6 effect-size strata --------------------------------------------
    nd = pd.Series(ndeg, index=pdat.perts)
    q = pd.qcut(nd, 3, labels=["弱效应", "中等效应", "强效应"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for a, m in zip(ax, ["pearson_de20", "pearson_sys"]):
        x = np.arange(3)
        sel = ["Mean-shift", "Linear-concat+KD", "kNN-concat+KD", MAIN]
        for k, mdl in enumerate(sel):
            v = [piv[m][mdl][q[piv[m].index] == s].mean() for s in ["弱效应", "中等效应", "强效应"]]
            a.bar(x + (k - 1.5) * 0.2, v, width=0.18, color=vp.model_color(mdl), label=LABEL[mdl],
                  edgecolor="white", linewidth=1.5)
        a.set_xticks(x)
        a.set_xticklabels([f"{s}\n(#DEG {int(nd[q == s].min())}–{int(nd[q == s].max())})" for s in ["弱效应", "中等效应", "强效应"]], fontsize=8.5)
        a.set_title(dict(METRICS)[m])
        a.grid(axis="x", visible=False)
    ax[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("按扰动效应强度分层", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_strata.png"))
    plt.close(fig)

    # ---------------- fig7 flow --------------------------------------------------------------
    fp = os.path.join(RES, "flow_singlecell_metrics.csv")
    if os.path.exists(fp):
        fl = pd.read_csv(fp)
        order = ["No-change", "Mean-shift", "PerturbFormer-shift", "CFM (ours)", "Oracle-shift"]
        lab = {"No-change": "不变", "Mean-shift": "均值平移", "PerturbFormer-shift": "PerturbFormer 平移",
               "CFM (ours)": "条件流匹配 (本文)", "Oracle-shift": "Oracle 平移 (真实均值)"}
        colm = {"No-change": vp.GRAY_LIGHT, "Mean-shift": vp.MUTED, "PerturbFormer-shift": vp.ORANGE,
                "CFM (ours)": vp.BLUE, "Oracle-shift": vp.INK2}
        ex_path = os.path.join(RES, "flow_examples.npz")
        ex = np.load(ex_path) if os.path.exists(ex_path) else None
        avail = {k.split("|")[0] for k in ex.files} if ex is not None else set()
        ex_genes = [g for g in ("RPS19", "MED12", "PSMB5", "EIF2S1", "POLR2A", "GATA1") if g in avail][:3]
        fig = plt.figure(figsize=(13, 6.6))
        gs = fig.add_gridspec(2, 3)
        for j, (m, t) in enumerate([("energy_distance", "能量距离 (越低越好)"), ("mmd", "MMD (越低越好)"),
                                    ("logvar_err_top20pc", "方差误差 |log σ²| (越低越好)")]):
            a = fig.add_subplot(gs[0, j])
            v = [fl[fl.model == o][m].mean() for o in order]
            se = [fl[fl.model == o][m].std() / np.sqrt((fl.model == o).sum()) for o in order]
            a.barh(range(len(order)), v, xerr=se, color=[colm[o] for o in order], height=0.6,
                   error_kw={"ecolor": vp.INK2, "elinewidth": 0.8})
            for i, x in enumerate(v):
                a.text(x * 1.02 + se[i], i, f"{x:.3f}", va="center", fontsize=8, color=vp.INK2)
            a.set_yticks(range(len(order)))
            a.set_yticklabels([lab[o] for o in order] if j == 0 else [""] * len(order), fontsize=8.5)
            a.invert_yaxis()
            a.set_title(t, fontsize=10)
            a.grid(axis="y", visible=False)
        from scipy.stats import gaussian_kde

        def contour(a, P, color, label, filled=False):
            """density contours enclosing ~50 % and ~85 % of the cells"""
            k = gaussian_kde(P.T)
            xs = np.linspace(lims[0], lims[1], 120)
            ys = np.linspace(lims[2], lims[3], 120)
            XX, YY = np.meshgrid(xs, ys)
            Z = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            lv = np.sort(np.quantile(k(P.T), [0.15, 0.5]))
            if filled:
                a.contourf(XX, YY, Z, levels=[lv[0], lv[1], Z.max()], colors=[color, color], alpha=0.18)
            a.contour(XX, YY, Z, levels=lv, colors=[color], linewidths=[1.0, 2.0])
            a.plot([], [], color=color, lw=2, label=label)

        for j, gname in enumerate(ex_genes):
            a = fig.add_subplot(gs[1, j])
            ctrl, true, gen = ex[f"{gname}|No-change"], ex[f"{gname}|True"], ex[f"{gname}|CFM (ours)"]
            sh = ex[f"{gname}|PerturbFormer-shift"]
            u = true.mean(0) - ctrl.mean(0)
            u /= np.linalg.norm(u) + 1e-9
            R = (true - true.mean(0)) - np.outer((true - true.mean(0)) @ u, u)
            v = np.linalg.svd(R, full_matrices=False)[2][0]
            proj = lambda X: np.stack([X @ u, X @ v], 1)  # noqa: E731
            C, T, G, Sh = proj(ctrl), proj(true), proj(gen), proj(sh)
            allp = np.vstack([C, T, G])
            lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0)
            pad = (hi - lo) * 0.15
            lims = [lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]]
            a.scatter(C[:, 0], C[:, 1], s=4, color=vp.GRAY_LIGHT, lw=0, label="对照细胞 (输入)")
            contour(a, T, vp.INK, "真实扰动细胞", filled=True)
            contour(a, Sh, vp.ORANGE, "仅平移均值")
            contour(a, G, vp.BLUE, "流匹配生成")
            a.set_xlim(lims[0], lims[1])
            a.set_ylim(lims[2], lims[3])
            a.set_title(f"{gname} 敲低 (训练时未见)", fontsize=10)
            a.set_xlabel("沿真实扰动方向的投影")
            a.set_ylabel("正交方向主变化" if j == 0 else "")
            if j == 0:
                a.legend(fontsize=7.5, loc="lower left", markerscale=2.5)
        fig.suptitle("单细胞层面: 从对照细胞生成未见扰动后的细胞群体", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig7_flow.png"))
        plt.close(fig)
        if ex_genes:   # compact 2x2 version for the slides
            fig, axs = plt.subplots(2, 2, figsize=(8.6, 7.2))
            for a, (m, t) in zip(axs[0], [("energy_distance", "能量距离 (越低越好)"), ("logvar_err_top20pc", "方差误差 |log σ²| (越低越好)")]):
                v = [fl[fl.model == o][m].mean() for o in order]
                a.barh(range(len(order)), v, color=[colm[o] for o in order], height=0.6)
                for i, x in enumerate(v):
                    a.text(x * 1.02, i, f"{x:.3f}", va="center", fontsize=9.5, color=vp.INK2)
                a.set_yticks(range(len(order)))
                a.set_yticklabels([lab[o] for o in order] if a is axs[0, 0] else [""] * len(order), fontsize=10)
                a.invert_yaxis()
                a.set_title(t, fontsize=11.5)
                a.grid(axis="y", visible=False)
                a.set_xlim(0, max(v) * 1.25)
            for a, gname in zip(axs[1], ex_genes[:2]):
                ctrl, true, gen = ex[f"{gname}|No-change"], ex[f"{gname}|True"], ex[f"{gname}|CFM (ours)"]
                sh = ex[f"{gname}|PerturbFormer-shift"]
                u = true.mean(0) - ctrl.mean(0)
                u /= np.linalg.norm(u) + 1e-9
                R = (true - true.mean(0)) - np.outer((true - true.mean(0)) @ u, u)
                v = np.linalg.svd(R, full_matrices=False)[2][0]
                C, T, G, Sh = [np.stack([X @ u, X @ v], 1) for X in (ctrl, true, gen, sh)]
                allp = np.vstack([C, T, G])
                lo, hi = np.percentile(allp, 1, 0), np.percentile(allp, 99, 0)
                pad = (hi - lo) * 0.15
                lims = [lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]]
                a.scatter(C[:, 0], C[:, 1], s=4, color=vp.GRAY_LIGHT, lw=0, label="对照细胞 (输入)")
                contour(a, T, vp.INK, "真实扰动细胞", filled=True)
                contour(a, Sh, vp.ORANGE, "仅平移均值")
                contour(a, G, vp.BLUE, "流匹配生成")
                a.set_xlim(lims[0], lims[1])
                a.set_ylim(lims[2], lims[3])
                a.set_title(f"{gname} 敲低 (训练时未见)", fontsize=11.5)
                a.set_xlabel("沿真实扰动方向的投影")
            axs[1, 0].legend(fontsize=8.5, loc="lower left", markerscale=2.5)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, "fig7_flow_slide.png"))
            plt.close(fig)
        if ex_genes:   # title-slide hero: control cells flowing to generated knockdown cells (transparent bg)
            gname = ex_genes[0]
            ctrl, gen = ex[f"{gname}|No-change"], ex[f"{gname}|CFM (ours)"]
            fig, a = plt.subplots(figsize=(5.4, 5.6))
            fig.patch.set_alpha(0)
            a.set_facecolor("none")
            a.axis("off")
            for k in range(0, len(ctrl), 6):
                a.annotate("", xy=gen[k, :2], xytext=ctrl[k, :2],
                           arrowprops={"arrowstyle": "-", "color": "#5f7a92", "lw": 0.5, "alpha": 0.6})
            a.scatter(ctrl[:, 0], ctrl[:, 1], s=9, color="#9fb7cc", lw=0, alpha=0.8)
            a.scatter(gen[:, 0], gen[:, 1], s=9, color="#4f95ea", lw=0, alpha=0.95)
            a.text(0.02, 0.02, f"对照细胞 → {gname} 敲低后 (流匹配生成, 训练时未见该基因)", transform=a.transAxes,
                   color="#9fb7cc", fontsize=8.5)
            fig.savefig(os.path.join(FIG, "hero_cells.png"), transparent=True)
            plt.close(fig)

    # ---------------- fig8 external ------------------------------------------------------------
    ep = os.path.join(ROOT, "results", "external_adamson", "summary.csv")
    if os.path.exists(ep):
        ext = pd.read_csv(ep, index_col=0)
        sel = ["Mean-shift", "Linear-concat+KD", "kNN-concat+KD", "PerturbFormer+KD"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.0))
        for a, m in zip(ax, ["pearson_all", "pearson_de20", "pearson_sys"]):
            v = [ext.loc[s, m + "_overlap41"] for s in sel]
            a.barh(range(len(sel)), v, color=[vp.model_color(s) for s in sel], height=0.6)
            for i, x in enumerate(v):
                a.text(max(x, 0) + 0.01, i, f"{x:.3f}", va="center", fontsize=8, color=vp.INK2)
            ceil = ext.loc["Replogle-measured (same gene)", m]
            a.axvline(ceil, color=vp.INK, lw=1.2)
            a.text(ceil + 0.012, len(sel) - 0.62, f"跨实验室上限 {ceil:.2f}", fontsize=7.5, ha="left", color=vp.INK)
            a.set_yticks(range(len(sel)))
            a.set_yticklabels([LABEL[s] for s in sel] if a is ax[0] else [""] * len(sel), fontsize=8.5)
            a.invert_yaxis()
            a.set_title(dict(METRICS)[m], fontsize=10)
            a.grid(axis="y", visible=False)
            a.set_xlim(0, 0.8)
        fig.suptitle("外部验证: Replogle 训练 → Adamson (另一实验室) 未见基因, 41 个可对照扰动", fontsize=11,
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig8_external.png"))
        plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
