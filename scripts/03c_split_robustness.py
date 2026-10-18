"""Step 3c - robustness of the benchmark to the random cross-validation split.

The 5-fold perturbation split is re-drawn with seeds 0, 1, 2 (results/<ds>, <ds>_seed1, <ds>_seed2);
for each split we report the main metrics and the paired PerturbFormer-vs-baseline comparison.

Output: results/<ds>/split_robustness.csv, results/<ds>/split_robustness.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.pipeline import ROOT  # noqa: E402

DS = "replogle_k562_essential"
MAIN = "PerturbFormer+KD"
MODELS = ["Mean-shift", "Linear-concat+KD", "PerturbNet-attn+KD", "kNN-concat+KD", MAIN]
METRICS = ["pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20", "de_overlap20", "rank"]
NAMES = {"pearson_all": "Pearson 全部", "pearson_de20": "Pearson Top-20 DEG", "pearson_sys": "去偏 Pearson",
         "dir_acc_de20": "DEG 方向准确率", "de_overlap20": "Top-20 DEG 重合", "rank": "识别排名↓"}


def main():
    runs = {0: DS, 1: DS + "_seed1", 2: DS + "_seed2"}
    per = {}
    for sd, d in runs.items():
        p = os.path.join(ROOT, "results", d, "per_perturbation_metrics.csv")
        if os.path.exists(p):
            per[sd] = pd.read_csv(p)
    print("splits found:", sorted(per))
    rows = []
    for sd, m in per.items():
        for mdl in MODELS:
            sub = m[m.model == mdl]
            rows.append({"split_seed": sd, "model": mdl, **{k: sub[k].mean() for k in METRICS}})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", DS, "split_robustness.csv"), index=False)

    agg = df.groupby("model")[METRICS].agg(["mean", "std"]).loc[MODELS]
    lines = [f"## 不同交叉验证划分下的稳健性（{len(per)} 种随机划分，均值 ± 标准差）", "",
             "| 模型 | " + " | ".join(NAMES[k] for k in METRICS) + " |", "|---" * (len(METRICS) + 1) + "|"]
    for mdl in MODELS:
        lines.append(f"| {mdl} | " + " | ".join(f"{agg.loc[mdl, (k, 'mean')]:.3f} ± {agg.loc[mdl, (k, 'std')]:.3f}"
                                               for k in METRICS) + " |")
    lines += ["", "## PerturbFormer+KD 减去对照模型（每种划分内的配对 Wilcoxon）", "",
              "| 对照 | 划分 | " + " | ".join(NAMES[k] for k in METRICS) + " |", "|---" * (len(METRICS) + 2) + "|"]
    for other in ["kNN-concat+KD", "Linear-concat+KD", "PerturbNet-attn+KD"]:
        for sd, m in per.items():
            cells = []
            for k in METRICS:
                q = m.pivot(index="pert", columns="model", values=k).dropna(subset=[MAIN, other])
                d = q[MAIN].mean() - q[other].mean()
                pv = wilcoxon(q[MAIN], q[other]).pvalue if (q[MAIN] != q[other]).any() else 1.0
                cells.append(f"{d:+.3f} (p={pv:.0e})")
            lines.append(f"| {other} | {sd} | " + " | ".join(cells) + " |")
    lines.append("\n识别排名越低越好，其差值为负表示 PerturbFormer 更好。")
    with open(os.path.join(ROOT, "results", DS, "split_robustness.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
