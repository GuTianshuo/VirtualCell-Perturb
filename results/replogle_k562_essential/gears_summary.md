## GEARS（已发表方法）与本项目模型对比：第 0 折，217 个未见基因

GEARS 设置：每个扰动 15 个细胞 + 1500 个对照细胞，10 个 epoch，其余为默认参数；训练用时 53 分钟（4 核 CPU）。

| 模型 | pearson_all | pearson_de20 | pearson_sys | dir_acc_de20 | de_overlap20 | rank |
|---|---|---|---|---|---|---|
| Mean-shift | 0.385 | 0.502 | 0.000 | 0.667 | 0.082 | 0.498 |
| Linear-concat+KD | 0.541 | 0.703 | 0.451 | 0.801 | 0.262 | 0.203 |
| PerturbNet-attn+KD | 0.568 | 0.750 | 0.473 | 0.861 | 0.280 | 0.177 |
| kNN-concat+KD | 0.582 | 0.757 | 0.492 | 0.835 | 0.290 | 0.168 |
| GEARS | 0.375 | 0.541 | 0.232 | 0.693 | 0.108 | 0.340 |
| GEARS+KD | 0.387 | 0.608 | 0.249 | 0.702 | 0.124 | 0.321 |
| PerturbFormer+KD | 0.597 | 0.770 | 0.504 | 0.849 | 0.311 | 0.155 |

PerturbFormer+KD 减去 GEARS+KD（配对 Wilcoxon）：pearson_all +0.210 (p=3e-32)；pearson_de20 +0.162 (p=2e-21)；pearson_sys +0.255 (p=1e-26)；dir_acc_de20 +0.147 (p=3e-22)；de_overlap20 +0.188 (p=5e-26)；rank -0.165 (p=1e-23)
