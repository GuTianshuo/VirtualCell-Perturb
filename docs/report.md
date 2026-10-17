# 虚拟细胞扰动预测：预测基因敲除后的细胞响应
### 生物医学信息学课程设计 · 选题九

---

## 摘要

**问题**：如果敲低一个从未做过实验的基因，细胞的转录组会如何变化？
**数据**：Replogle 2022 K562 必需基因 CRISPRi Perturb-seq（1,092 个基因敲低，162,751 个细胞），Adamson 2016（另一实验室，86 个敲低）作外部验证；GO、STRING、GenePT（大语言模型基因嵌入）作为先验知识。
**模型**：提出 **PerturbFormer**，一种在"已测扰动记忆库"上做**知识核交叉注意力**的检索增强 Transformer：用多源先验知识找到功能相似且已经测过的基因，组合它们的真实响应，并加入 CRISPRi 敲低规则与 DEG 方向多任务损失；在此基础上用**条件流匹配（OT-CFM）**把预测展开为每个细胞的反事实状态，实现"输入细胞 + 基因扰动 → 输出扰动后细胞状态"。
**评测**：在 5 折未见基因交叉验证（1,092 个基因都作为未见基因被预测一次）中，PerturbFormer 的去系统偏差 Pearson 达到 0.480（测量噪声可达上限的 59%），在全部 7 项指标上显著优于 Ahlmann-Eltze 线性基线（p < 1e-21），去偏 Pearson 比标准深度模型 PerturbNet 高 0.037（但 PerturbNet 的 DEG 方向准确率更高），并小幅优于本文同时提出的多源 kNN（6 项指标中 4 优、1 平、1 劣）。流匹配相对仅平移均值，使单细胞能量距离下降 8%、方差误差下降 19%（p < 1e-38）。跨实验室验证中，对未见基因的预测达到"直接重测同一基因"上限的 79–86%。
**解释**：预测保留了 CORUM 复合体与 Hallmark 通路结构（通路相关 r = 0.59）；每个预测都可追溯到注意力最高的已测扰动；核糖体、蛋白酶体、Mediator 等复合体成员最可预测，而 GATA1 这类谱系特异的主调控因子是明确的失败边界。

## 1 Problem：问题定义与价值

**核心问题**：如果敲低（knock down）某个基因，细胞的转录组会发生什么变化？能否对**从未做过实验的基因**给出可靠预测？

**为什么重要**
- 人类约 2 万个蛋白编码基因，两两组合约 2 亿种，再乘以细胞类型和剂量，实验无法穷举。Perturb-seq 单次实验成本高，一个全基因组筛选需要数十万到数百万个细胞。
- "虚拟细胞（virtual cell）"被认为是 AI 驱动生物学的核心目标之一（Bunne et al., *Cell* 2024；CZI、Arc Institute 的 Virtual Cell Challenge 2025）：在计算机中先筛选，再把实验预算花在最有希望的扰动上。
- 应用：靶点发现（预测敲低某靶点后是否逆转疾病表达谱）、遗传相互作用、毒性/脱靶预警、实验设计优化。

**形式化**：给定对照细胞状态 $x_0$ 与扰动基因 $g$，预测
1. 伪批量（pseudobulk）层面：平均表达变化 $\Delta_g = \mathbb{E}[x \mid g] - \mathbb{E}[x \mid \text{ctrl}] \in \mathbb{R}^{G}$；
2. 单细胞层面：扰动后细胞的分布 $p(x \mid g)$，以及**每个对照细胞的反事实状态** $x_0 \mapsto x_1$；
3. 每个差异表达基因（DEG）的方向（上调/下调）。

**关键难点（本项目明确针对）**
- **未见扰动泛化**：测试基因在训练时从未被扰动过，模型不能"记住"答案，只能依靠先验知识迁移。
- **系统性偏差陷阱**：近期研究（Ahlmann-Eltze et al., *Nat Methods* 2025；Viñas Torné et al., *Nat Biotech* 2025）发现，许多深度模型并不优于"所有扰动的平均效应"这一平凡基线——因为大多数扰动共享一个系统性的"受扰动"状态。必须用能剔除系统偏差的指标来评估。
- **技术噪声**：单个扰动只有几十到几百个细胞，测量本身就有噪声，需要估计"天花板"。

## 2 Data：数据

| 数据集 | 来源 | 细胞系 / 技术 | 规模 | 用途 |
|---|---|---|---|---|
| Replogle 2022 K562 essential | Replogle et al., *Cell* 2022 | K562，CRISPRi 敲低 | 162,751 细胞 × 5,000 HVG；**1,092 个基因扰动** + 10,691 对照细胞 | 主数据集：训练与 5 折未见扰动评测 |
| Adamson 2016 UPR | Adamson et al., *Cell* 2016 | K562，CRISPRi 敲低 | 68,603 细胞 × 5,060 基因；86 个扰动 + 24,263 对照 | 外部验证（另一实验室） + 数据集内评测 |

两个数据集均使用 GEARS（Roohani et al., *Nat Biotech* 2024）发布的预处理版本（log-normalized，Harvard Dataverse），保证与文献可比。

**先验知识（用于把"基因"编码成向量）**

| 来源 | 内容 | 覆盖率（1,137 个扰动基因） |
|---|---|---|
| Gene Ontology | 基因的 BP/CC/MF 注释（GEARS gene2go） | 99.6% |
| STRING v12 | 人类蛋白互作网络（combined score ≥ 700） | 97.6% |
| GenePT | 用 GPT-3.5 嵌入 NCBI 基因功能摘要（Chen & Zou 2024），即"生物医学语言模型"表示 | 99.5% |
| 应答谱（数据驱动） | 被扰动基因自身在训练扰动中的表达响应模式（仅当该基因被测量） | 37.7% |

**解释用数据**：MSigDB Hallmark、GO BP 2023、KEGG、Reactome、CORUM 蛋白复合体（Enrichr 基因集库）。

**预处理**（`scripts/01_preprocess.py`）
- 按扰动聚合得到伪批量均值/方差（分块稀疏矩阵运算，峰值内存 3.8 GB）；
- 每个扰动相对对照做 Welch t 检验 + BH 校正，得到 DEG；Top-20 DEG 按 |t| 排序；
- 单细胞：在分层抽样细胞上拟合 100 维 PCA，所有细胞投影到该潜空间（流模型用）。

## 3 Model：模型

### 3.1 扰动编码（Perturbation encoding）
把"敲低基因 g"表示为 4 个知识 token：GO（TF-IDF + SVD，128 维）、PPI（STRING 扩散谱嵌入 $P+P^2$，128 维）、GenePT（PCA 128 维）、应答谱（训练响应矩阵 SVD，64 维，**每折只用训练扰动计算**，无信息泄露）。缺失的知识源用可学习的"缺失 token"代替。

### 3.2 基线（必须打败的对手）
| 基线 | 说明 |
|---|---|
| No-change | 预测扰动无影响 |
| Mean-shift | 预测为所有训练扰动的平均变化（文献中最强的平凡基线） |
| Linear | $\Delta \approx b + G W p$，Ahlmann-Eltze et al. 2025 提出的线性基线（G 为训练响应主成分，p 为扰动嵌入） |
| kNN | 在先验知识空间中找最相似的 10 个已测基因，相似度加权平均其响应 |
| PerturbNet | 知识 token → Transformer 融合 → MLP 解码（标准监督深度模型） |

### 3.3 PerturbFormer（本文主模型，监督版本）
**思想**：预测一个未见基因的效应 ≈ "注意"那些在生物学上与它相似、且已经测过的基因。

$$\hat\Delta_i = \sum_h g_h(i)\,\mathrm{softmax}_j\!\Big(\tfrac{1}{\tau_h}\,\frac{\sum_s w_{hs}(i)\, m_s(i) m_s(j)\cos(A_s e_s(i), A_s e_s(j))}{\sum_s w_{hs}(i)\,m_s(i)m_s(j)}\Big)_{\text{top-}k}\Delta_j \;+\; g_0(i)\,\big(\mu + \mathrm{dec}(h_i)B^\top\big)$$

- **Query** = 待预测基因的知识 token（经 Transformer 编码器融合得到 $h_i$）；**Key** = 已测基因的知识 token；**Value** = 已测基因的真实响应 $\Delta_j$。
- **知识核注意力**：注意力 logits 由每个知识源的余弦相似度构成；$A_s$ 初始化为单位阵，所以初始模型正好等价于多源 kNN，训练只需学习"偏差修正"，在只有约 870 个训练扰动的情况下避免过拟合。
- **查询相关的知识源权重** $w_{hs}(i)$：模型自己决定对这个基因更该相信 GO、PPI、语言模型还是数据驱动的先验——可解释。
- **top-k 稀疏注意力**（k = 20）与多头（不同温度）。
- **参数化分支**：均值 + 训练响应低秩基底，经门控与检索分支混合。
- **CRISPRi 敲低规则（KD）**：被靶向基因本身的表达按训练集中位残余比例下调（领域知识先验）。
- **多任务损失**：DE 加权 MSE + DEG 方向的 logistic 损失（表达变化预测 + DEG 方向预测）；训练时随机丢弃知识源（source dropout），增强对缺失先验的鲁棒性。
- 训练：AdamW，早停（训练扰动内部 10% 作为验证），3 个随机种子集成。

### 3.4 条件流匹配（高级版本：生成模型）
目标：输入"**细胞 + 基因扰动**"，输出**这个细胞**扰动后的状态，并生成整个扰动后细胞群体。

- 在 100 维 PCA 潜空间中学习速度场 $v_\theta(x_t, t, c)$，通过 ODE 把对照细胞分布连续地输运到扰动后细胞分布（Lipman et al. 2023；Tong et al. 2024 OT-CFM）。
- 训练配对：每个扰动内部用 mini-batch **最优传输**（匈牙利算法）配对对照细胞与扰动细胞；$x_t = (1-t)x_0 + t x_1 + \sigma\epsilon$，回归 $u = x_1 - x_0$。
- **条件** $c$ = PerturbFormer 预测的伪批量变化在 PCA 上的投影。训练扰动使用"**排除自身的检索预测**"作为条件，使训练与测试时的条件分布一致（无标签泄露）。
- 网络：FiLM 调制的残差 MLP（时间 + 条件嵌入），中点法 20 步积分。
- 解码：PCA 子空间内的分布来自流模型；PCA 正交补上的均值（如靶基因自身的敲低）由 PerturbFormer 补回。

## 4 Benchmark：评测设计
- **5 折未见扰动交叉验证**：按扰动基因划分，每折 20% 的基因完全不出现在训练中；每个扰动恰好得到一次 out-of-fold 预测（全部 1,092 个用于生物学解释）。
- **开发集与测试集严格分离**：所有设计选择和超参数扫描只在 fold 0 训练扰动内部划出的开发集上进行（`03b_dev_sweep.py`），5 个测试折从未用于调参。
- **指标**（每个扰动计算后取平均）
  - Pearson Δ（全部基因 / Top-20 DEG）、Top-20 DEG MSE（GEARS 标准）；
  - **DEG 方向准确率**（Top-20 / Top-100 显著 DEG 的上下调方向）；
  - **去系统偏差 Pearson（Systema）**：减去训练扰动平均响应后再算相关，Mean-shift 恒为 0，只奖励"扰动特异"的信号；
  - Top-20 DEG 重合率；
  - **扰动识别排名**：用预测去匹配所有真实扰动，真实扰动的归一化排名（0 = 完美，0.5 = 随机）；
  - 单细胞：能量距离（scPerturb E-distance）、RBF-MMD、方差误差。
- **噪声天花板**：把每个扰动和对照细胞随机拆成两半，用一半预测另一半。
- **统计检验**：配对 Wilcoxon 符号秩检验 + 95% bootstrap 置信区间；按效应强度分层。
- **消融**：逐一去掉每个知识源、方向损失、KD 规则、Transformer 融合、知识核注意力、top-k 稀疏。
- **外部验证**：Replogle 训练（剔除所有 Adamson 靶基因）→ 预测 Adamson；以"同一基因在 Replogle 的实测响应"作为跨实验室可重复性上限。

## 5 Benchmark：结果

### 5.1 主结果（1,092 个未见基因，5 折 out-of-fold）

| 模型 | Pearson Δ 全部基因 | Pearson Δ Top-20 DEG | 去偏 Pearson (Systema) | DEG 方向准确率 | Top-20 DEG 重合率 | 识别排名 ↓ |
|---|---|---|---|---|---|---|
| No-change | 0.000 | 0.000 | 0.078 | 0.000 | 0.013 | 0.500 |
| Mean-shift（均值平移） | 0.382 | 0.464 | 0.000 | 0.643 | 0.091 | 0.500 |
| Linear（Ahlmann-Eltze 2025）+KD | 0.526 | 0.696 | 0.430 | 0.793 | 0.237 | 0.222 |
| PerturbNet（Transformer 融合 + MLP）+KD | 0.552 | 0.738 | 0.443 | **0.847** | 0.262 | 0.191 |
| kNN 多源先验 +KD | 0.567 | **0.749** | 0.473 | 0.828 | 0.278 | 0.175 |
| **PerturbFormer +KD（本文）** | **0.573** | 0.746 | **0.480** | 0.833 | **0.285** | **0.174** |
| *细胞拆半一致性（参考）* | 0.544 | 0.817 | 0.526 | 0.806 | 0.270 | 0.038 |
| *测量噪声决定的可达上限* | 0.813 | 0.900 | 0.808 | — | — | — |

（完整表格、95% bootstrap 置信区间与配对检验见 `results/replogle_k562_essential/main_table.md`、`stats_tests.csv`；图见 `figures/fig2_benchmark.png`。）

**结论 1：均值平移这一"平凡基线"看似不差，实则没有任何扰动特异信息。** 它的全基因 Pearson 达到 0.38，DEG 方向准确率 0.64，但去偏 Pearson 恰好为 0、识别排名为随机水平 0.5。这验证了 Systema 等工作的观点：如果只看传统相关性指标，会高估模型。

**结论 2：多源知识检索大幅超越文献中的强基线。** PerturbFormer 相对线性基线，在全部 7 项指标上都显著更好（配对 Wilcoxon p < 1e-21）：去偏 Pearson +0.049，Top-20 DEG Pearson +0.050，方向准确率 +0.040，识别排名 0.222 → 0.174。相对标准深度模型 PerturbNet，去偏 Pearson +0.037（p = 2e-31）。

**结论 3：可学习的 PerturbFormer 只比精心设计的多源 kNN 小幅领先。** 在全基因 Pearson（+0.006，p = 3e-16）、去偏 Pearson（+0.006，p = 4e-8）、方向准确率（+0.005，p = 0.03）、DEG 重合率（+0.008，p = 0.007）和 Top-20 MSE（p = 3e-6）上显著更好；在 Top-20 DEG Pearson 上略差（−0.003，p = 0.004）；识别排名无显著差异。真正起决定作用的是"用多种先验知识找相似的已测基因"这一思路，PerturbFormer 的价值在于稳定的小幅增益与可解释性（知识源权重、检索邻居）。

**结论 4：如实说明一项劣势。** PerturbNet 的 DEG 方向准确率最高（0.847 vs 0.833）。一个可能的原因是：检索式模型的输出是多个已测响应的加权平均，对弱 DEG 的方向存在"平均化"效应，而参数化解码器可以单独调整每个基因的符号。

**与上限的距离。** 用细胞拆半估计每个扰动的测量信度，完美预测真实效应时与实测值的相关上限为 √(Spearman-Brown 校正信度)：全基因 0.81、Top-20 DEG 0.90、去偏 0.81。PerturbFormer 分别达到其 70%、83%、59%。

### 5.2 哪种先验知识有用？消融实验

单独使用一种先验时（Linear / kNN 平均，去偏 Pearson）：**GO（0.42）> STRING PPI（0.37）> GenePT（0.33）≫ 应答谱（0.08）**。纯数据驱动的应答谱只对 38% 被测量的基因可用，而且几乎不能泛化；多源融合（kNN 0.46，Linear 0.42）优于任何单源（图 `fig3_priors.png`）。

| 消融（相对完整模型） | Δ 去偏 Pearson | Δ Top-20 DEG Pearson |
|---|---|---|
| 去掉 GO | −0.029 | −0.009 |
| 去掉 GenePT（语言模型） | −0.024 | −0.005 |
| 自由注意力（不用知识核） | −0.021 | −0.016 |
| 去掉 STRING PPI | −0.020 | −0.006 |
| 拼接代替 Transformer 融合 | −0.009 | −0.002 |
| 去掉应答谱 | −0.006 | +0.003 |
| 去掉 top-k 稀疏 | −0.003 | 0.000 |
| 去掉方向损失 | −0.003 | 0.000 |
| 去掉 CRISPRi 敲低规则 | −0.011 | **−0.072** |

三点发现：（1）三种外部知识互补，去掉任何一种都显著下降；（2）"知识核"是关键设计，把注意力换成从零学习的投影会下降 0.021，说明在约 870 个训练样本下，以多源 kNN 为起点只学修正量是必要的；（3）敲低规则对 Top-20 DEG 指标贡献最大，因为在 412 个靶基因被测量的扰动中，靶基因本身在 94.9% 的情况下就是排名第一的 DEG。

### 5.3 按效应强度分层

把扰动按 DEG 数分成三组（弱：3–152；中：153–577；强：578–3045 个 DEG），PerturbFormer 的去偏 Pearson 分别为 0.35 / 0.45 / 0.64，在三组中都领先线性基线（0.31 / 0.39 / 0.59）。弱效应扰动的信号接近噪声水平，是所有方法共同的难点（`fig6_strata.png`）。

### 5.4 单细胞生成：条件流匹配（高级版本）

对全部 1,092 个未见扰动，用同一批 300 个对照细胞作为输入，比较生成的细胞群与真实扰动细胞（PCA 100 维空间，`fig7_flow.png`）：

| 方法 | 能量距离 ↓ | MMD ↓ | 方差误差 \|log σ²\| ↓ |
|---|---|---|---|
| 不变（对照细胞） | 2.098 | 0.089 | 0.263 |
| 均值平移 | 1.772 | 0.079 | 0.263 |
| PerturbFormer 平移（只移动均值） | 1.111 | 0.052 | 0.263 |
| **条件流匹配（本文）** | **1.020** | **0.043** | **0.215** |
| *Oracle 平移（使用真实均值，参考）* | 0.237 | 0.015 | 0.263 |

- 流匹配相对"只平移均值"：能量距离下降 8%（72% 的扰动更好，配对 Wilcoxon p = 1e-40），方差误差下降 19%（69% 更好，p = 2e-39）。平移操作不改变方差，所以即使使用真实均值的 Oracle 平移，方差误差也停在 0.263，而流匹配能够学到扰动对细胞异质性的影响。例如 MED12 敲低后真实细胞呈双峰（部分细胞几乎不变，可能对应敲低不完全），流匹配部分复现了第二个峰，而平移只能给出一个窄团。
- Oracle 平移的能量距离仅 0.237，远低于所有可实现的方法。**单细胞层面的主要误差来源是对未见基因平均效应的预测，而不是分布形状**；改进伪批量预测仍是第一优先级。
- 解码回基因空间后，生成细胞的平均值与 PerturbFormer 的伪批量预测基本一致（全基因 Pearson 0.557 vs 0.568），说明流模型没有破坏均值预测。



### 5.5 外部验证：Replogle → Adamson（另一实验室）

训练时剔除全部 86 个 Adamson 靶基因，在两数据集共有的 1,910 个基因上评估（`fig8_external.png`）。对 41 个在两个筛选中都被扰动的基因，用"Replogle 中同一基因的实测响应"预测 Adamson，作为跨实验室可重复性上限。

| 模型（41 个可对照基因） | Pearson 全部基因 | Pearson Top-20 DEG | 去偏 Pearson |
|---|---|---|---|
| 均值平移 | 0.299 | 0.451 | 0.000 |
| Linear +KD | 0.358 | 0.472 | 0.234 |
| kNN 多源 +KD | 0.391 | **0.506** | **0.277** |
| PerturbFormer +KD | **0.394** | 0.492 | 0.265 |
| *Replogle 实测同一基因（跨实验室上限）* | 0.458 | 0.624 | 0.398 |

- 跨实验室差异本身很大：即使直接用另一实验室对同一基因的实测结果，全基因相关也只有 0.46。两种知识检索方法对完全未见的基因达到这一上限的 79–86%（全基因、Top-20 DEG）和约 67–70%（去偏）。
- 在外部数据上，kNN 在 Top-20 DEG 和去偏 Pearson 上略好于 PerturbFormer，PerturbFormer 在全基因 Pearson 和识别排名（全部 86 个扰动：0.339 vs 0.353）上更好。可学习模型在训练数据分布内获得的小幅优势，在跨实验室场景中没有保持下来。
- Adamson 筛选聚焦于内质网应激，所有扰动共享很强的 UPR 系统性响应，均值平移的 DEG 方向准确率（0.75）反而高于所有模型，但它的去偏相关为 0。这再次说明需要同时报告去偏指标。



### 5.6 第二个数据集内部的未见基因评测：Adamson（86 个扰动）

用同样的 5 折未见基因设定在 Adamson 数据集内部训练和评估（`results/adamson/`）：

| 模型 | Pearson 全部基因 | Pearson Top-20 DEG | 去偏 Pearson | DEG 方向准确率 | 识别排名 ↓ |
|---|---|---|---|---|---|
| 均值平移 | 0.754 | 0.799 | 0.000 | 0.892 | 0.500 |
| Linear +KD | 0.812 | 0.930 | 0.476 | 0.942 | 0.194 |
| PerturbNet +KD | 0.813 | 0.910 | 0.474 | 0.945 | 0.228 |
| kNN 多源 +KD | 0.835 | 0.945 | 0.498 | 0.954 | 0.189 |
| **PerturbFormer +KD** | **0.857** | **0.950** | **0.544** | **0.960** | **0.172** |
| *可达上限* | 0.986 | 0.976 | 0.946 | — | — |

在这个更小、更同质（均为内质网应激相关基因）的数据集上，PerturbFormer 在所有指标上最优。相对 kNN，去偏 Pearson +0.045（p = 0.02）、全基因 Pearson +0.022（p = 0.007）、Top-20 DEG Pearson +0.005（p = 8e-4）；方向准确率与识别排名差异不显著。Adamson 的测量信度远高于 K562 全基因组筛选（每个扰动约 500 个细胞），可达上限接近 1；均值平移的全基因相关高达 0.75，反映了 UPR 筛选中极强的共同响应，去偏指标再次揭示它不含扰动特异信息。

## 6 Biological interpretation：生物学解释

以下分析全部基于 out-of-fold 预测，即每个基因的预测都来自它被完全留出的那一折。

### 6.1 CRISPRi 自检
412 个靶基因在测量范围内的扰动中，靶基因 **100% 显著下调**，残余表达中位数接近 0；94.9% 的扰动中靶基因是排名第一的差异基因。敲低规则预测的靶基因变化与实测的相关 r = 0.973。这说明数据质量可靠，同时解释了为什么一个简单的领域知识规则能大幅提升 Top-20 DEG 指标。

### 6.2 哪些基因可预测？
按功能类别统计去偏 Pearson（`fig_interp_B_classes.png`）：伴侣蛋白 CCT/TRiC（0.83）、胞质核糖体（0.82）、RNA 外切体（0.74）、Mediator（0.73）、蛋白酶体（0.71）最可预测；RNA Pol II / 转录起始（0.47）、未归类基因（0.38）最难。规律很清楚：**同一复合体成员敲低后效应高度一致**，只要有一个成员被测过，其余成员就能被准确预测。也有例外：在 Mediator 和 RNA Pol II 类别上，kNN（0.77、0.57）优于 PerturbFormer（0.73、0.47），说明学到的知识源权重并非处处最优。

### 6.3 复合体结构
以 CORUM 复合体为金标准，计算每个复合体成员基因 Top-5 最相似扰动中同复合体成员的比例（`fig_interp_C_complex.png`）：随机 2.7%；实测应答谱之间 42.7%；先验知识相似度 60.3%；PerturbFormer 注意力检索邻居 56.1%；用预测应答谱去检索实测应答谱 48.6%。实测数据中"同复合体 → 相似响应"的规律真实存在，这是模型能够"借鉴"的生物学基础；预测应答谱也保留了这一结构。

### 6.4 通路一致性
对 50 条 MSigDB Hallmark 通路，计算每个扰动的通路平均变化，再比较预测与实测：每条通路跨扰动的平均相关 r = 0.587（kNN 0.579，Linear 0.494，均值平移 −0.057）；对实测变化最大的 25% 通路，方向预测正确率 93.8%。在变化最大的 20 条通路中，DNA 修复（0.74）、氧化磷酸化（0.71）、干扰素 α 应答（0.70）预测最准，MYC 靶基因 V1/V2（0.56/0.58）、mTORC1（0.53）居中（`fig_interp_D_pathways.png`）。

### 6.5 案例分析（全部为未见基因）

| 基因 | 功能 | Pearson（全部 / Top-20 DEG） | 模型检索到的已测扰动 | 预测 vs 实测富集 |
|---|---|---|---|---|
| RPS19 | 40S 核糖体蛋白，Diamond-Blackfan 贫血致病基因 | 0.97 / 0.99 | RPS17、RPS16、RPS24、RPS15、RPS14 | 下调：胞质翻译（预测 q=3e-29；实测 q=1e-24） |
| MED12 | Mediator 激酶模块亚基 | 0.91 / 0.98 | C7orf26、MED30、MED17、MED4、MED8 | 下调：MYC 靶基因 V1（预测 q=5e-15；实测 q=5e-20）；上调：血红素代谢（两者均显著） |
| PSMB5 | 20S 蛋白酶体 β5 催化亚基（硼替佐米靶点） | 0.85 / 0.98 | PSMB2、PSMB3、PSMB1、PSMA6、PSMA7 | 下调：MYC 靶基因；上调：EMT、SQSTM1、UBC（蛋白稳态应激） |
| **GATA1**（失败） | 红系主转录因子 | 0.33 / 0.44 | ATF4、MAX、TFDP1、ZBTB17、HSF1 | MYC 靶基因下调预测正确；**"氧运输"（HBG1/HBG2/HBZ）预测上调，实测下调** |

GATA1 是一个有启发的失败。K562 是红白血病细胞系，GATA1 维持其红系身份，敲低后血红蛋白基因下调。训练集中没有效应与 GATA1 相同的基因，模型从 GO/PPI/GenePT 中只能看出"它是一个转录因子"，于是借鉴了 ATF4、MAX 等一般转录因子：共性的 MYC 靶基因下调预测对了，红系特异的部分方向却预测反了。**谱系特异的主调控因子无法靠"相似基因"外推，这正是最需要实验的地方。**

### 6.6 模型如何决策，何时失败
- **知识源权重**（`fig_interp_F_sources.png`）：整体上 GO 权重最高；Mediator 复合体更依赖 PPI（0.41），翻译起始因子更依赖 GenePT（0.31），胞质核糖体则对三种外部知识几乎平均分配（0.30 / 0.29 / 0.30）。模型对不同基因自适应地分配信任。
- **失败模式**（`fig_interp_E_failure.png`）：去偏 Pearson 与"该基因和训练集中最相似基因的先验相似度"的 Spearman ρ = 0.52，与扰动效应强度（DEG 数）的 ρ = 0.49。最差的基因包括 RPL13、RUVBL1、PPRC1、FTSJ3、GATA1；最好的几乎都是核糖体蛋白。
- **启示**：预测可靠性可以事先估计（先验相似度），下一轮实验应优先测知识空间中离已测基因最远的基因（主动学习）。

## 7 最终成果：虚拟细胞原型

1. **命令行**：`python scripts/predict.py --gene <基因> [--cells 细胞.h5ad]`，输出预测的单细胞表达（h5ad）、伪批量变化表、证据报告（检索到的已测扰动与权重、知识源权重、Hallmark 通路变化；若该基因测过，则在将其移出记忆后与实测比较）和可视化。
2. **Python API**：`VirtualCell.load(...).predict(gene, cells)`。
3. **任意基因**：先验嵌入覆盖约 2.1 万个人类基因（GO 88%、STRING 76%、GenePT 91%），包括从未被扰动过的基因。
4. **交互式浏览器** `docs/explorer.html`：浏览全部 1,092 个未见基因的预测与实测对比、检索证据和通路变化。

## 8 讨论

**实验严谨性与可复现性**
- 测试折从未参与任何设计选择；超参数扫描只在开发集上进行（`dev_sweep.csv`）：10 组配置的 Top-20 DEG Pearson 极差 0.007、去偏 Pearson 极差 0.015（同配置不同种子的标准差约 0.004）。最终使用的是默认配置，并未按开发集最优挑选，结论不依赖调参。
- 所有指标同时报告配对检验、95% bootstrap 置信区间和测量噪声上限。
- 开发过程中发现并修复了两个会影响结论可信度的问题，修复后全部结果重新计算：（1）GO 嵌入在不同进程间不可复现（Python 字符串哈希随机化改变了 GO 术语顺序，进而影响随机化 SVD），已改为确定性排序；（2）Welch t 检验的自由度计算中误加了绝对 ε，低表达基因的 p 值被高估，现已与 `scipy.stats.ttest_ind(equal_var=False)` 逐位一致。
- 全流程 `run_all.sh` 一键复现，CPU 即可运行。

**局限**
- 只使用一个细胞系（K562）和单基因敲低；外部验证也限于 K562。
- 模型的泛化能力本质上受限于"是否存在功能相似的已测基因"；对谱系特异的调控因子会失败。
- 流模型在 100 维 PCA 潜空间中工作，单细胞层面的基因细节由 PCA 截断限制。
- 先验知识本身存在研究偏倚：研究充分的基因注释更多、预测更准。

**展望**
- 跨细胞系迁移（Replogle RPE1 / HepG2 / Jurkat），把"细胞背景"作为显式输入；
- 组合扰动（Norman 2019），研究遗传相互作用；
- 用预测不确定性与先验新颖度驱动主动学习，选择下一轮最有信息量的实验。

## 参考文献
1. Replogle J.M. et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575 (2022).
2. Adamson B. et al. A multiplexed single-cell CRISPR screening platform enables systematic dissection of the unfolded protein response. *Cell* 167, 1867–1882 (2016).
3. Roohani Y., Huang K., Leskovec J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nat Biotechnol* 42, 927–935 (2024).
4. Ahlmann-Eltze C., Huber W., Anders S. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. *Nat Methods* (2025).
5. Viñas Torné R. et al. Systema: a framework for evaluating genetic perturbation response prediction beyond systematic variation. *Nat Biotechnol* (2025).
6. Lipman Y. et al. Flow matching for generative modeling. *ICLR* (2023).
7. Tong A. et al. Improving and generalizing flow-based generative models with minibatch optimal transport. *TMLR* (2024).
8. Chen Y., Zou J. GenePT: a simple but hard-to-beat foundation model for genes and cells built from ChatGPT. *bioRxiv* (2024).
9. Peidli S. et al. scPerturb: harmonized single-cell perturbation data. *Nat Methods* 21, 531–540 (2024).
10. Bunne C. et al. How to build the virtual cell with artificial intelligence: priorities and opportunities. *Cell* 187, 7045–7063 (2024).
11. Szklarczyk D. et al. The STRING database in 2023. *Nucleic Acids Res* 51, D638–D646 (2023).
12. Tsitsiridis G. et al. CORUM: the comprehensive resource of mammalian protein complexes—2022. *Nucleic Acids Res* 51, D539–D545 (2023).
