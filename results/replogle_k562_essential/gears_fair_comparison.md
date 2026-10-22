# GEARS 公平重跑对比

同一 CV 划分（seed 0, fold 0, 217 个未见基因），只改变 GEARS 的训练预算。

| 设置 | 模型 | pearson_all | pearson_de20 | pearson_sys | dir_acc_de20 | de_overlap20 | rank |
|---|---|---|---|---|---|---|---|
| original 15 cells / 10 ep | GEARS | 0.375 | 0.541 | 0.232 | 0.693 | 0.108 | 0.340 |
| original 15 cells / 10 ep | GEARS+KD | 0.387 | 0.608 | 0.249 | 0.702 | 0.124 | 0.321 |
| fair 30 cells / 20 ep | GEARS | 0.347 | 0.499 | 0.209 | 0.666 | 0.096 | 0.390 |
| fair 30 cells / 20 ep | GEARS+KD | 0.360 | 0.564 | 0.227 | 0.675 | 0.113 | 0.369 |
| fair 50 cells / 20 ep | GEARS | 0.357 | 0.480 | 0.213 | 0.656 | 0.112 | 0.385 |
| fair 50 cells / 20 ep | GEARS+KD | 0.370 | 0.550 | 0.231 | 0.665 | 0.129 | 0.365 |
| same fold | PerturbFormer+KD | 0.597 | 0.770 | 0.504 | 0.849 | 0.311 | 0.155 |

说明：30/50 细胞版本使用 5,000 个对照细胞和 20 epoch，GPU 训练；原版为 15 细胞、1,500 对照、10 epoch、CPU。
两版公平重跑中 GEARS+KD 仍未超过原版，也仍显著低于 PerturbFormer+KD；单折、单数据子集，结论方向与原文一致。
