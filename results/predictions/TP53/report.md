# 虚拟细胞预测: 敲低 TP53 (CRISPRi, K562)

- 训练筛选中是否测过该扰动: 否 (全新基因)
- 依据的相似扰动 (注意力权重): PHB 0.10, ATF4 0.07, GATA1 0.07, TELO2 0.06, ZBTB17 0.06, TFDP1 0.06
- 知识来源权重: go=0.30, ppi=0.37, genept=0.24, resp=0.09
- 预测下调基因: PTMA(-0.39), HSPE1(-0.35), LDHA(-0.33), HSPA8(-0.32), GP1BB(-0.30), SRM(-0.30), HSPD1(-0.29), FABP5(-0.29), NME1(-0.28), NCL(-0.28), TOMM40(-0.28), GAL(-0.27)
- 预测上调基因: MALAT1(+0.36), GAS5(+0.34), TXNIP(+0.31), BTG1(+0.29), SH3BGRL3(+0.24), LGALS1(+0.24), TPT1(+0.21), PIM1(+0.21), PSAP(+0.20), DDIT3(+0.18), CCPG1(+0.18), FTH1(+0.18)
- 通路活性下降: Myc Targets V2 (-0.176); Myc Targets V1 (-0.151); E2F Targets (-0.094); G2-M Checkpoint (-0.075)
- 通路活性上升: IL-6/JAK/STAT3 Signaling (+0.037); Interferon Gamma Response (+0.031); Epithelial Mesenchymal Transition (+0.028); Interferon Alpha Response (+0.027)
- 单细胞: 生成 300 个扰动后细胞; 潜空间平均位移 5.22 (细胞间差异 sd=3.79)
