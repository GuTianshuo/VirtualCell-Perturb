# 虚拟细胞预测: 敲低 GATA1 (CRISPRi, K562)

- 训练筛选中是否测过该扰动: 是。已将其从检索记忆中移除，但部署模型的参数训练时见过它，下方与实测的比较偏乐观；严格的未见基因评估见 5 折交叉验证结果
- 靶基因 GATA1 预测变化: -0.683 (log 表达)
- 预测可靠性: 低 (与已测扰动的最大先验相似度 0.66, 最相似: ATF4; 交叉验证中同档基因的平均去偏 Pearson 0.32, Top-20 DEG Pearson 0.61)
- 依据的相似扰动 (注意力权重): FAM136A 0.17, ATF4 0.12, ZBTB17 0.08, PHB 0.07, ATF5 0.07, E4F1 0.05
- 知识来源权重: go=0.25, ppi=0.39, genept=0.21, resp=0.15
- 预测下调基因: GATA1(-0.68), PTMA(-0.39), LDHA(-0.35), NME1(-0.34), PPA1(-0.34), FABP5(-0.34), SRM(-0.34), GAL(-0.33), TOMM40(-0.33), NOP56(-0.33), NPM1(-0.33), NCL(-0.32)
- 预测上调基因: MALAT1(+0.32), BTG1(+0.30), PSAP(+0.27), SH3BGRL3(+0.26), GYPB(+0.24), B2M(+0.24), CD63(+0.22), FTH1(+0.21), ASAH1(+0.21), LGALS1(+0.21), ALAS2(+0.21), MDK(+0.18)
- 通路活性下降: Myc Targets V2 (-0.194); Myc Targets V1 (-0.165); E2F Targets (-0.089); G2-M Checkpoint (-0.071)
- 通路活性上升: Interferon Gamma Response (+0.032); Interferon Alpha Response (+0.031); IL-6/JAK/STAT3 Signaling (+0.030); Epithelial Mesenchymal Transition (+0.028)
- 与实测比较 (偏乐观, 仅供参考): Pearson(全部基因) = 0.636; Pearson(实测 Top-20 DEG) = 0.735; 方向一致率(Top-20) = 90%
- 单细胞: 生成 300 个扰动后细胞; 潜空间平均位移 14.19 (细胞间差异 sd=4.39)
