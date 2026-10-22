# 预测示例

每个子目录对应一个基因，包含：

- `report.md`：人类可读的预测报告（检索证据、知识源权重、通路变化、可靠性评级）
- `pseudobulk_prediction.csv`：每个基因的对照均值、预测均值和预测 Δ
- `prediction.png`：PCA 状态转移图与变化最大的基因

`predicted_cells.h5ad`（每个约 6–7 MB）没有纳入 git；运行以下命令可重新生成：

```bash
python scripts/predict.py --gene MYC
python scripts/predict.py --gene GATA1
python scripts/predict.py --gene TP53
python scripts/predict.py --gene RPS19
```

这四个基因中 MYC、TP53 在筛选中未被扰动（全新基因），GATA1、RPS19 是已测基因；后者报告中的“与实测比较”仅作参考，严格的未见基因评估见报告第 5 节。
