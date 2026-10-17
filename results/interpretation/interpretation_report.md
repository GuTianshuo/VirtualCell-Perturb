# 生物学解释结果 (自动生成)

## A. CRISPRi 敲低验证
- 靶基因在测量基因中: 412 / 1092
- 靶基因显著下调 (FDR<0.05): 100.0%
- 中位残余表达 (扰动/对照, log 空间): 0.00
- 靶基因是该扰动 |t| 排名第 1 的 DEG: 94.9%; 排名前 20: 99.3%
- KD 规则预测 vs 实测靶基因变化 Pearson r = 0.973

## B. 功能类别可预测性 (PerturbFormer+KD, pearson_sys)
| class                 |   n |   median_nDEG |   PerturbFormer+KD|pearson_sys |   PerturbFormer+KD|pearson_de20 |
|:----------------------|----:|--------------:|-------------------------------:|--------------------------------:|
| 伴侣蛋白 CCT/TRiC         |   8 |         897   |                          0.829 |                           0.974 |
| 胞质核糖体 Ribosome        |  70 |        1755.5 |                          0.821 |                           0.926 |
| RNA 外切体 Exosome       |  10 |         341.5 |                          0.74  |                           0.954 |
| Mediator 复合体          |  20 |        1241.5 |                          0.732 |                           0.932 |
| 蛋白酶体 Proteasome       |  26 |         593.5 |                          0.706 |                           0.781 |
| 核糖体生物合成 (UTP/NOP/WDR) |  28 |         292.5 |                          0.705 |                           0.909 |
| DNA 复制                |  22 |         362.5 |                          0.69  |                           0.946 |
| 线粒体核糖体 Mito-ribosome  |  44 |         103   |                          0.648 |                           0.877 |
| 核孔 Nuclear pore       |   8 |         917   |                          0.637 |                           0.964 |
| tRNA 合成酶              |  19 |         483   |                          0.635 |                           0.843 |
| RNA Pol I/III         |   9 |         122   |                          0.607 |                           0.947 |
| 剪接体 Spliceosome       |  50 |        1060   |                          0.583 |                           0.839 |
| 翻译起始 eIF              |  26 |         676   |                          0.514 |                           0.901 |
| RNA Pol II / 转录起始     |  46 |         425.5 |                          0.468 |                           0.807 |
| 其他 Other              | 706 |         219   |                          0.381 |                           0.671 |

## C. CORUM 复合体结构保留
|                         |   frac_top5_same_complex |
|:------------------------|-------------------------:|
| 随机                      |                    0.027 |
| 实测应答谱之间                 |                    0.427 |
| 先验知识相似度 (GO+PPI+GenePT) |                    0.603 |
| PerturbFormer 注意力检索邻居   |                    0.561 |
| 预测应答谱 → 最近的实测应答谱        |                    0.486 |

## D. 通路一致性 (Hallmark)
| model            |   mean_r_per_pathway |   mean_spearman_per_pert |   pathway_direction_acc_top25pct |
|:-----------------|---------------------:|-------------------------:|---------------------------------:|
| PerturbFormer+KD |                0.587 |                    0.72  |                            0.938 |
| kNN-concat+KD    |                0.579 |                    0.717 |                            0.939 |
| Linear-concat+KD |                0.494 |                    0.658 |                            0.908 |
| Mean-shift       |               -0.057 |                    0.594 |                            0.854 |

| pathway                           |   r_pred_vs_true |   true_sd |
|:----------------------------------|-----------------:|----------:|
| DNA Repair                        |            0.744 |     0.032 |
| Oxidative Phosphorylation         |            0.709 |     0.044 |
| Interferon Alpha Response         |            0.698 |     0.036 |
| Myogenesis                        |            0.682 |     0.03  |
| Spermatogenesis                   |            0.676 |     0.036 |
| Pperoxisome                       |            0.668 |     0.027 |
| Cholesterol Homeostasis           |            0.657 |     0.029 |
| Bile Acid Metabolism              |            0.654 |     0.029 |
| heme Metabolism                   |            0.639 |     0.04  |
| Interferon Gamma Response         |            0.637 |     0.028 |
| Estrogen Response Late            |            0.634 |     0.029 |
| Reactive Oxygen Species Pathway   |            0.607 |     0.037 |
| Epithelial Mesenchymal Transition |            0.592 |     0.036 |
| Unfolded Protein Response         |            0.589 |     0.033 |
| Myc Targets V2                    |            0.577 |     0.103 |
| Myc Targets V1                    |            0.561 |     0.088 |
| TNF-alpha Signaling via NF-kB     |            0.544 |     0.03  |
| mTORC1 Signaling                  |            0.533 |     0.031 |
| E2F Targets                       |            0.529 |     0.053 |
| G2-M Checkpoint                   |            0.528 |     0.042 |

## E. 失败模式
- pearson_sys 与先验相似度 Spearman ρ = 0.518
- pearson_sys 与 #DEG Spearman ρ = 0.485
- 最差 10 个: RPL13, RUVBL1, PPRC1, FTSJ3, GATA1, NDUFA2, MCMBP, STRAP, DLD, OPA1
- 最好 10 个: RPL6, RPL31, PHB, RPL8, RPS19, RPL36, RPL37A, RPL34, EIF2B2, RPL4

## F. 知识来源权重
| class                 |    go |   ppi |   genept |   resp |   param_gate |
|:----------------------|------:|------:|---------:|-------:|-------------:|
| 伴侣蛋白 CCT/TRiC         | 0.385 | 0.268 |    0.237 |  0.11  |        0.03  |
| 胞质核糖体 Ribosome        | 0.302 | 0.291 |    0.303 |  0.103 |        0.019 |
| RNA 外切体 Exosome       | 0.399 | 0.294 |    0.238 |  0.069 |        0.023 |
| Mediator 复合体          | 0.365 | 0.413 |    0.157 |  0.065 |        0.035 |
| 蛋白酶体 Proteasome       | 0.405 | 0.295 |    0.228 |  0.073 |        0.028 |
| 核糖体生物合成 (UTP/NOP/WDR) | 0.325 | 0.345 |    0.236 |  0.093 |        0.027 |
| DNA 复制                | 0.383 | 0.295 |    0.22  |  0.102 |        0.035 |
| 线粒体核糖体 Mito-ribosome  | 0.348 | 0.314 |    0.257 |  0.08  |        0.022 |
| 核孔 Nuclear pore       | 0.383 | 0.239 |    0.217 |  0.16  |        0.033 |
| tRNA 合成酶              | 0.314 | 0.278 |    0.276 |  0.132 |        0.046 |
| RNA Pol I/III         | 0.313 | 0.289 |    0.265 |  0.133 |        0.034 |
| 剪接体 Spliceosome       | 0.359 | 0.279 |    0.253 |  0.11  |        0.052 |
| 翻译起始 eIF              | 0.276 | 0.249 |    0.31  |  0.166 |        0.035 |
| RNA Pol II / 转录起始     | 0.366 | 0.338 |    0.202 |  0.094 |        0.048 |
| 其他 Other              | 0.344 | 0.309 |    0.231 |  0.116 |        0.054 |

## G. 案例分析 (全部为未见基因的 out-of-fold 预测)

### RPS19  (类别: 胞质核糖体 Ribosome, #DEG=2680, 细胞数=393)
- 预测 vs 实测: Pearson(全部基因) = 0.965, Pearson(Top-20 DEG) = 0.992
- PerturbFormer 检索到的相似扰动 (权重): RPS17 (0.08), RPS16 (0.07), RPS24 (0.07), RPS15 (0.06), RPS14 (0.06)
- 实测 Top-10 下调: SNHG5, MT-ATP6, HBG2, HIST1H1C, RPL3, RPS3, RPL39, SLC25A6, MT-ND6, RPS9
- 预测 Top-10 下调: SNHG5, HIST1H1C, MT-ATP6, RPS9, RPS3, MT-ND4L, MT-ND6, SLC25A6, RPS27, RPL3
- 实测 Top-10 上调: BTG1, TXNIP, MALAT1, SNHG1, SNHG7, DGCR6L, TBX1, MED10, ATP5IF1, ORMDL1
- 预测 Top-10 上调: TXNIP, SNHG1, MALAT1, BTG1, SNHG7, MED10, DGCR6L, DDIT3, HBZ, TBX1

- 预测 Top-50 下调基因富集: [GO] Cytoplasmic Translation (GO:0002181) (q=2.7e-29); [GO] Peptide Biosynthetic Process (GO:0043043) (q=2.8e-26); [GO] Macromolecule Biosynthetic Process (GO:0009059) (q=7.4e-26); [GO] Translation (GO:0006412) (q=7.7e-25)

- 实测 Top-50 下调基因富集: [GO] Cytoplasmic Translation (GO:0002181) (q=9.8e-25); [GO] Peptide Biosynthetic Process (GO:0043043) (q=3.4e-22); [GO] Macromolecule Biosynthetic Process (GO:0009059) (q=5.2e-22); [GO] Translation (GO:0006412) (q=4.0e-21)

- 预测 Top-50 上调基因富集: [GO] Mitochondrial Gene Expression (GO:0140053) (q=2.9e-02); [GO] Regulation Of Mononuclear Cell Migration (GO:0071675) (q=2.9e-02); [GO] Mitochondrial Translation (GO:0032543) (q=2.9e-02); [GO] Regulation Of Cell Communication (GO:0010646) (q=5.3e-02)

- 实测 Top-50 上调基因富集: [GO] Mitochondrial Gene Expression (GO:0140053) (q=3.2e-02); [GO] Regulation Of Mononuclear Cell Migration (GO:0071675) (q=3.2e-02); [GO] Mitochondrial Translation (GO:0032543) (q=3.2e-02); [GO] Regulation Of Cell Communication (GO:0010646) (q=5.7e-02)

### MED12  (类别: Mediator 复合体, #DEG=2860, 细胞数=567)
- 预测 vs 实测: Pearson(全部基因) = 0.911, Pearson(Top-20 DEG) = 0.981
- PerturbFormer 检索到的相似扰动 (权重): C7orf26 (0.11), MED30 (0.08), MED17 (0.07), MED4 (0.07), MED8 (0.07)
- 实测 Top-10 下调: GAL, ODC1, DANCR, PTMA, PPA1, SRM, SNHG5, NME1, MYC, LDHA
- 预测 Top-10 下调: SNHG5, PPA1, NPM1, SRM, ODC1, GAL, DANCR, ZFAS1, LDHA, HSPD1
- 实测 Top-10 上调: GYPB, SH3BGRL3, BNIP3L, LGALS1, CYBA, B2M, ASAH1, AIF1, TIMP1, S100A11
- 预测 Top-10 上调: GYPB, B2M, TRAPPC6A, HIST1H1C, CD63, PSAP, MYL6, STMN1, ASAH1, S100A11

- 预测 Top-50 下调基因富集: [H] Myc Targets V1 (q=5.2e-15); [H] Myc Targets V2 (q=1.3e-09); [GO] Chaperone-Mediated Protein Complex Assembly (GO:0051131) (q=3.4e-05); [GO] Protein Targeting To Mitochondrion (GO:0006626) (q=5.1e-04)

- 实测 Top-50 下调基因富集: [H] Myc Targets V1 (q=4.6e-20); [H] Myc Targets V2 (q=1.6e-11); [H] Unfolded Protein Response (q=9.5e-05); [H] E2F Targets (q=9.5e-05)

- 预测 Top-50 上调基因富集: [H] heme Metabolism (q=1.1e-06); [GO] Positive Regulation Of Endocytosis (GO:0045807) (q=2.7e-03); [GO] Negative Regulation Of Endopeptidase Activity (GO:0010951) (q=2.7e-03); [GO] Negative Regulation Of Peptidase Activity (GO:0010466) (q=5.2e-03)

- 实测 Top-50 上调基因富集: [H] heme Metabolism (q=2.3e-04); [GO] Positive Regulation Of Endocytosis (GO:0045807) (q=2.3e-04); [GO] Negative Regulation Of Peptidase Activity (GO:0010466) (q=2.7e-04); [GO] Positive Regulation Of Interleukin-6 Production (GO:0032755) (q=7.2e-04)

### PSMB5  (类别: 蛋白酶体 Proteasome, #DEG=1144, 细胞数=188)
- 预测 vs 实测: Pearson(全部基因) = 0.853, Pearson(Top-20 DEG) = 0.979
- PerturbFormer 检索到的相似扰动 (权重): PSMB2 (0.18), PSMB3 (0.17), PSMB1 (0.11), PSMA6 (0.09), PSMA7 (0.09)
- 实测 Top-10 下调: GP1BB, DANCR, CKB, PTMA, HBZ, NME1, HIST1H4C, SLC39A3, RANBP1, CYCS
- 预测 Top-10 下调: GP1BB, CKB, DANCR, HMGN2, PTMA, H2AFZ, PRKAR2B, GYPC, CYCS, GAL
- 实测 Top-10 上调: HSPA1A, SQSTM1, VIM, UBC, LGALS1, ANXA2, MALAT1, NEAT1, FTL, MLLT11
- 预测 Top-10 上调: ANXA2, MALAT1, SQSTM1, VIM, SH3BGRL3, LGALS1, UBC, MLLT11, KRT18, BLVRB

- 预测 Top-50 下调基因富集: [H] Myc Targets V1 (q=2.9e-12); [H] Myc Targets V2 (q=2.1e-03); [H] mTORC1 Signaling (q=5.1e-03); [GO] Positive Regulation Of Cysteine-Type Endopeptidase Activity Involved In Apoptotic Process (GO:0043280) (q=2.3e-02)

- 实测 Top-50 下调基因富集: [H] Myc Targets V1 (q=4.5e-08); [H] Oxidative Phosphorylation (q=8.3e-04); [GO] Aerobic Electron Transport Chain (GO:0019646) (q=3.9e-03); [GO] Mitochondrial ATP Synthesis Coupled Electron Transport (GO:0042775) (q=3.9e-03)

- 预测 Top-50 上调基因富集: [GO] Retrograde Axonal Transport (GO:0008090) (q=4.8e-05); [H] Epithelial Mesenchymal Transition (q=1.5e-04); [GO] Regulation Of Apoptotic Process (GO:0042981) (q=4.0e-04); [GO] Positive Regulation Of Programmed Cell Death (GO:0043068) (q=1.5e-03)

- 实测 Top-50 上调基因富集: [GO] Cellular Response To Heat (GO:0034605) (q=3.2e-05); [H] Epithelial Mesenchymal Transition (q=2.0e-04); [GO] Regulation Of Protein Modification By Small Protein Conjugation Or Removal (GO:1903320) (q=2.9e-04); [GO] Regulation Of Apoptotic Process (GO:0042981) (q=3.3e-04)

### GATA1  (类别: 其他 Other, #DEG=1598, 细胞数=95)
- 预测 vs 实测: Pearson(全部基因) = 0.326, Pearson(Top-20 DEG) = 0.439
- PerturbFormer 检索到的相似扰动 (权重): ATF4 (0.15), MAX (0.11), TFDP1 (0.11), ZBTB17 (0.08), HSF1 (0.07)
- 实测 Top-10 下调: GP1BB, APOC1, HBG2, HBZ, SNHG5, HSPD1, NME1, EIF5A, ODC1, GYPC
- 预测 Top-10 下调: GATA1, ATF4, GAL, LDHA, YARS, COMT, FABP5, TOMM40, NOLC1, SRM
- 实测 Top-10 上调: CTSC, LTB, TMSB4X, LGALS1, ATF7IP2, LST1, S100A13, PLD3, TXNIP, ARL6IP5
- 预测 Top-10 上调: HBZ, HBG2, BLVRB, HIST1H1C, HBG1, MALAT1, GSTP1, BTG1, GYPB, SLC25A37

- 预测 Top-50 下调基因富集: [H] Myc Targets V1 (q=7.1e-17); [H] Myc Targets V2 (q=1.0e-11); [H] E2F Targets (q=6.3e-07); [H] mTORC1 Signaling (q=7.4e-05)

- 实测 Top-50 下调基因富集: [H] Myc Targets V1 (q=1.3e-13); [H] Myc Targets V2 (q=8.7e-10); [GO] Establishment Of Protein Localization To Mitochondrion (GO:0072655) (q=2.5e-04); [GO] Oxygen Transport (GO:0015671) (q=2.5e-04)

- 预测 Top-50 上调基因富集: [GO] Oxygen Transport (GO:0015671) (q=7.8e-09); [GO] Carbon Dioxide Transport (GO:0015670) (q=1.5e-08); [GO] Gas Transport (GO:0015669) (q=1.5e-08); [GO] One-Carbon Compound Transport (GO:0019755) (q=4.7e-07)

- 实测 Top-50 上调基因富集: [H] IL-2/STAT5 Signaling (q=5.3e-04); [GO] Positive Regulation Of Cytokine Production (GO:0001819) (q=6.6e-04); [GO] Positive Regulation Of Interleukin-6 Production (GO:0032755) (q=6.6e-04); [GO] Positive Regulation Of Response To External Stimulus (GO:0032103) (q=6.6e-04)
