# 训练好的 K562 虚拟细胞模型

本目录包含 `scripts/08_train_final.py` 在全部 1,092 个 Replogle K562 扰动上训练后可部署模型，`scripts/predict.py` 和 `vcp.predictor.VirtualCell` 直接读取这里。

| 文件 | 内容 |
|---|---|
| `config.json` | 模型维度、集成数量、PerturbFormer / flow 的超参数 |
| `meta.npz` | 基因列表、扰动响应矩阵、对照均值/方差、PCA 基、对照细胞潜空间、应答谱、敲低比例 |
| `gene_universe_embeddings.npz` | 约 2.1 万个人类基因的 GO / STRING PPI / GenePT 先验嵌入及缺失掩码 |
| `perturbformer_0/1/2.pt` | PerturbFormer 三模型集成 |
| `flow.pt` | 条件流匹配（OT-CFM）单细胞生成器 |
| `reliability.json` | 按先验相似度校准的“高/中/低”可靠性分档 |

## 使用

```bash
python scripts/predict.py --gene GATA1
python scripts/predict.py --gene MYC --cells my_cells.h5ad --out results/predictions/MYC
```

## 重建

```bash
python scripts/08_train_final.py
python scripts/predict.py --gene MYC
```

> 注意：最终模型在全部 1,092 个扰动上训练过。对筛选内基因做“与实测比较”会偏乐观；严格的未见基因评估见报告第 5 节的 5 折交叉验证。

## 校验和（SHA-256）

```
5d67ee9d9b847c1a8f7014ed5b565c8e982598b397deb8bf275c6c38c9d65ab2  config.json
6bd495762ef5aa5ac4daa11d20adc4fcd2132bc39b9b90c306809bf626e281e9  flow.pt
487891df04f0ea8798f6b55f3322a401e8763c230cd28355a22dd7d1923ecd94  gene_universe_embeddings.npz
c4a36aff7adaa53f823f444ffa633c521e581a1b06330e80d1f8ef813f4fd0db  meta.npz
38d2adace79fc7bd34188545ad38d8afcbe6744a69b4a760ada57760dfec1ebf  perturbformer_0.pt
3590f969d0d702463a0014c222b3b6b8a37890ca8456410875875e9beda6ebd9  perturbformer_1.pt
af7f0f2cd20dc7f0029ec601a7e788c4e63ebcccb46c98077a5ef6c2cc2482ca  perturbformer_2.pt
e6abe3397c424945004e0587125cfe19c3ee1fbd123d0cb3e9803fbdad4d4d70  reliability.json
```
