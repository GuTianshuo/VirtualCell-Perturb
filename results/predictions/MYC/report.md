# 虚拟细胞预测: 敲低 MYC (CRISPRi, K562)

- 训练筛选中是否测过该扰动: 否 (全新基因)
- 靶基因 MYC 预测变化: -1.187 (log 表达)
- 预测可靠性: 中 (与已测扰动的最大先验相似度 0.71, 最相似: MAX; 交叉验证中同档基因的平均去偏 Pearson 0.47, Top-20 DEG Pearson 0.75)
- 依据的相似扰动 (注意力权重): GATA1 0.10, ATF4 0.07, ZBTB17 0.07, MAX 0.06, ZNF335 0.06, TFDP1 0.05
- 知识来源权重: go=0.32, ppi=0.43, genept=0.21, resp=0.04
- 预测下调基因: MYC(-1.19), PTMA(-0.29), SRM(-0.29), HSPD1(-0.29), LDHA(-0.28), GAL(-0.28), HSPE1(-0.27), FABP5(-0.27), DDX21(-0.26), EIF5A(-0.25), DANCR(-0.25), HSPA8(-0.25)
- 预测上调基因: MALAT1(+0.33), TXNIP(+0.30), LGALS1(+0.26), SH3BGRL3(+0.25), BTG1(+0.24), PLD3(+0.20), NEAT1(+0.20), PSAP(+0.18), PIM1(+0.17), LTB(+0.17), S100A13(+0.16), CTSC(+0.16)
- 通路活性下降: Myc Targets V2 (-0.202); Myc Targets V1 (-0.154); Wnt-beta Catenin Signaling (-0.102); E2F Targets (-0.075)
- 通路活性上升: IL-6/JAK/STAT3 Signaling (+0.034); Epithelial Mesenchymal Transition (+0.023); Interferon Gamma Response (+0.023); TGF-beta Signaling (+0.021)
- 单细胞: 生成 300 个扰动后细胞; 潜空间平均位移 6.14 (细胞间差异 sd=4.32)
