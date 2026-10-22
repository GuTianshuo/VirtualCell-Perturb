# 虚拟细胞预测: 敲低 RPS19 (CRISPRi, K562)

- 训练筛选中是否测过该扰动: 是。已将其从检索记忆中移除，但部署模型的参数训练时见过它，下方与实测的比较偏乐观；严格的未见基因评估见 5 折交叉验证结果
- 预测可靠性: 高 (与已测扰动的最大先验相似度 0.94, 最相似: RPS17; 交叉验证中同档基因的平均去偏 Pearson 0.65, Top-20 DEG Pearson 0.88)
- 依据的相似扰动 (注意力权重): RPS17 0.07, RPS16 0.06, RPS24 0.06, RPS15 0.06, RPS9 0.05, RPS14 0.05
- 知识来源权重: go=0.25, ppi=0.37, genept=0.28, resp=0.10
- 预测下调基因: SNHG5(-0.86), MT-ATP6(-0.72), HIST1H1C(-0.69), RPS9(-0.66), MT-ND6(-0.59), MT-ND4L(-0.58), SLC25A6(-0.56), RPS3(-0.56), RPS11(-0.55), RPL3(-0.54), MT-ND2(-0.53), RPS27(-0.52)
- 预测上调基因: TXNIP(+0.70), SNHG1(+0.67), BTG1(+0.66), MALAT1(+0.65), SNHG7(+0.54), MED10(+0.50), HBZ(+0.49), DDIT3(+0.48), DGCR6L(+0.48), EIF5(+0.44), AC139493.2(+0.43), TBX1(+0.42)
- 通路活性下降: E2F Targets (-0.033); Cholesterol Homeostasis (-0.033); mTORC1 Signaling (-0.022); G2-M Checkpoint (-0.022)
- 通路活性上升: Interferon Alpha Response (+0.097); Interferon Gamma Response (+0.072); heme Metabolism (+0.065); DNA Repair (+0.064)
- 与实测比较 (偏乐观, 仅供参考): Pearson(全部基因) = 0.965; Pearson(实测 Top-20 DEG) = 0.994; 方向一致率(Top-20) = 100%
- 单细胞: 生成 300 个扰动后细胞; 潜空间平均位移 8.98 (细胞间差异 sd=2.32)
