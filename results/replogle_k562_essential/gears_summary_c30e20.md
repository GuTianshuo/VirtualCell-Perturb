# GEARS 公平重跑（30 细胞/扰动，20 epoch）

设置：每个扰动 30 个细胞、5000 个对照细胞、20 个 epoch；第 0 折、217 个未见基因；GPU 训练约 32 分钟。

| 模型 | pearson_all | pearson_de20 | pearson_sys | dir_acc_de20 | de_overlap20 | rank |
|---|---|---|---|---|---|---|---|
| GEARS | 0.347 | 0.499 | 0.209 | 0.666 | 0.096 | 0.390 |
| GEARS+KD | 0.360 | 0.564 | 0.227 | 0.675 | 0.113 | 0.369 |
| PerturbFormer+KD（同折） | 0.597 | 0.770 | 0.504 | 0.849 | 0.311 | 0.155 |

结论：即使把细胞数和 epoch 都提高，GEARS(+KD) 仍未超过原来的 15 细胞/10 epoch 设置，也仍显著低于 PerturbFormer+KD。单折、单数据子集。
