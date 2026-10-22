"""Step 3d - give the baselines the same tuning budget as PerturbFormer.

PerturbFormer's hyper-parameters were checked on the development split (03b_dev_sweep.py);
kNN and Linear were run with fixed settings.  To make the comparison fair, both baselines are
tuned here on exactly the same development split (inner validation set of CV fold 0), and the
tuned versions are then re-evaluated on the test folds of all three CV splits and compared, per
perturbation, with PerturbFormer.

Output: results/<ds>/baseline_tuning.csv (dev grid), results/<ds>/baseline_tuned_vs_pf.md
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import KNNEmbedding, LinearEmbedding, knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import kfold_perturbation_split, train_val_split  # noqa: E402
from vcp.metrics import evaluate, summarise  # noqa: E402
from vcp.pipeline import ROOT, load_bundle  # noqa: E402

DS = "replogle_k562_essential"
KEYS = ["pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20", "de_overlap20", "rank"]


def fit_predict(kind, params, E, Y, tr, te):
    if kind == "kNN":
        m = KNNEmbedding(k=params["k"], tau=params["tau"])
    else:
        m = LinearEmbedding(k=params["k"], lam=params["lam"])
    return m.fit(Y[tr], E[tr]).predict(E[te])


def main():
    b = load_bundle(DS)
    Y, perts = b.Y, b.pdat.perts
    grids = {"kNN": [{"k": k, "tau": t} for k, t in itertools.product([3, 5, 10, 20, 40], [0.02, 0.05, 0.1, 0.2, 0.5])],
             "Linear": [{"k": k, "lam": l} for k, l in itertools.product([8, 16, 32, 64, 128], [0.01, 0.1, 1.0, 10.0])]}

    # ---- 1. tune on the development split (identical to 03b_dev_sweep.py) ---------------------
    fold = kfold_perturbation_split(perts, 5, 0)
    tr_outer, dev = train_val_split(np.nonzero(fold != 0)[0], 0.1, 0)
    feats, _ = b.fold_features(tr_outer)
    E = np.concatenate([feats[s] for s in feats], 1)
    kd = knockdown_fraction(b.pdat.pert_mean[tr_outer], b.pdat.ctrl_mean, b.target_idx[tr_outer])
    ref = Y[tr_outer].mean(0)
    rows = []
    for kind, grid in grids.items():
        for prm in grid:
            P = knockdown_rule(fit_predict(kind, prm, E, Y, tr_outer, dev), perts[dev], b.target_idx[dev], b.pdat.ctrl_mean, kd)
            r = summarise(evaluate(P, Y[dev], b.pdat.tstat[dev], b.pdat.fdr[dev], ref, b.ctrl_std))
            rows.append({"model": kind, **prm, **r, "select": np.mean([r["pearson_all"], r["pearson_de20"], r["pearson_sys"]])})
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(os.path.join(ROOT, "results", DS, "baseline_tuning.csv"), index=False)
    best = {k: grid_df[grid_df.model == k].sort_values("select", ascending=False).iloc[0] for k in grids}
    pf_dev = pd.read_csv(os.path.join(ROOT, "results", DS, "dev_sweep.csv")).iloc[0]   # default PF config
    lines = ["## 基线调参（与 PerturbFormer 相同的开发集）", "",
             "选择标准：开发集上 Pearson 全部 / Top-20 DEG / 去偏 三项的平均。", "",
             "| 模型 | 选中的参数 | 开发集 Pearson 全部 | Top-20 DEG | 去偏 |", "|---|---|---|---|---|"]
    for k, r in best.items():
        prm = f"k={int(r.k)}, τ={r.tau}" if k == "kNN" else f"k={int(r.k)}, λ={r.lam}"
        lines.append(f"| {k}（调参后） | {prm} | {r.pearson_all:.3f} | {r.pearson_de20:.3f} | {r.pearson_sys:.3f} |")
    lines.append(f"| PerturbFormer（默认配置，未按开发集挑选） | — | {pf_dev.pearson_all:.3f} | {pf_dev.pearson_de20:.3f} | {pf_dev.pearson_sys:.3f} |")
    print("\n".join(lines))

    # ---- 2. re-evaluate the tuned baselines on every test fold of all three splits ---------------
    lines += ["", "## 调参后的基线在 3 种划分测试集上的表现（PerturbFormer 减去基线，配对 Wilcoxon）", "",
              "| 基线 | 划分 | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 2) + "|"]
    summary = []
    for seed, tag in [(0, ""), (1, "_seed1"), (2, "_seed2")]:
        pf = pd.read_csv(os.path.join(ROOT, "results", DS + tag, "per_perturbation_metrics.csv"))
        pf = pf[pf.model == "PerturbFormer+KD"].set_index("pert")
        fold = kfold_perturbation_split(perts, 5, seed)
        per = {k: [] for k in best}
        for f in range(5):
            te, tr_all = np.nonzero(fold == f)[0], np.nonzero(fold != f)[0]
            feats, _ = b.fold_features(tr_all)
            E = np.concatenate([feats[s] for s in feats], 1)
            kd = knockdown_fraction(b.pdat.pert_mean[tr_all], b.pdat.ctrl_mean, b.target_idx[tr_all])
            ref = Y[tr_all].mean(0)
            for kind, r in best.items():
                prm = {"k": int(r.k), "tau": r.tau} if kind == "kNN" else {"k": int(r.k), "lam": r.lam}
                P = knockdown_rule(fit_predict(kind, prm, E, Y, tr_all, te), perts[te], b.target_idx[te], b.pdat.ctrl_mean, kd)
                ev = evaluate(P, Y[te], b.pdat.tstat[te], b.pdat.fdr[te], ref, b.ctrl_std)
                per[kind].append(pd.DataFrame({k: ev[k] for k in KEYS}, index=perts[te]))
        for kind in best:
            d = pd.concat(per[kind])
            cells = []
            for k in KEYS:
                a, c = pf[k].reindex(d.index), d[k]
                ok = ~(a.isna() | c.isna())
                diff = a[ok].mean() - c[ok].mean()
                pv = wilcoxon(a[ok], c[ok]).pvalue
                cells.append(f"{diff:+.3f} (p={pv:.0e})")
                summary.append({"split": seed, "baseline": kind, "metric": k, "baseline_value": c[ok].mean(),
                                "pf_value": a[ok].mean(), "diff": diff, "p": pv})
            lines.append(f"| {kind}（调参后） | {seed} | " + " | ".join(cells) + " |")
    s = pd.DataFrame(summary)
    s["p_holm"] = np.nan
    for kind in best:
        idx = s.baseline == kind
        s.loc[idx, "p_holm"] = multipletests(s.loc[idx, "p"], method="holm")[1]
    s.to_csv(os.path.join(ROOT, "results", DS, "baseline_tuned_vs_pf.csv"), index=False)
    for kind in best:
        t = s[s.baseline == kind]
        eff = (t["diff"] > 0) ^ (t.metric == "rank")
        raw_better = int(((t.p < 0.05) & eff).sum())
        raw_worse = int(((t.p < 0.05) & ~eff).sum())
        holm_better = int(((t.p_holm < 0.05) & eff).sum())
        holm_worse = int(((t.p_holm < 0.05) & ~eff).sum())
        lines.append(f"\n{kind}（调参后）：{len(t)} 次比较中，未校正 p<0.05 时 PerturbFormer 显著更好 {raw_better} 次、"
                     f"显著更差 {raw_worse} 次；Holm 校正后显著更好 {holm_better} 次、显著更差 {holm_worse} 次、"
                     f"无显著差异 {len(t) - holm_better - holm_worse} 次。")
    with open(os.path.join(ROOT, "results", DS, "baseline_tuned_vs_pf.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()
