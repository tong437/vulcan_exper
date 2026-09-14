# 神经元重要性与功能分类：方法谱系、适用性筛选与实验建议

更新日期：2026-09-12
适用范围：`llava-hf/llava-1.5-7b-hf`、`Qwen/Qwen3.5-2B`、`google/gemma-4-E2B-it` 的语言解码器 FFN 中间维度。

## 结论摘要

这项研究不应寻找单一的“神经元重要性真值指标”。现有方法测量的是不同对象：激活频率测量一个神经元是否经常响应，条件选择性测量它能否区分模态或样本类别，输出贡献与梯度归因测量它对指定预测目标的局部影响，消融才直接测量它对模型行为是否必要。相关性指标适合发现候选，不能独立证明功能；因果指标更可信，但成本更高且会受冗余、非线性交互和干预方式影响。[^1][^2][^3]

对当前三模型研究，推荐采用一条统一的三级证据链：

1. **全量低成本初筛**：记录 post-gating FFN 中间激活，计算激活概率、RMS、波动性和模态条件选择性。
2. **候选精排**：对候选神经元计算输出贡献和 `gradient × activation`；对少量高价值候选补充 Integrated Gradients。
3. **类别确认**：在独立测试集上做逐层数量匹配的零化/均值替换消融，并检查它是否“优先伤害”所声称的模态或任务，而不只是让总体性能下降。

原先的 visual / text / multimodal / unknown 四类可以保留为便于展示的一级标签，但不应把 `unknown` 解释成“不重要”，也不应把“视觉 token 和文本 token 都高激活”直接解释成真正的跨模态神经元。建议把内部分析升级为连续、多标签功能画像，并把 multimodal 拆成 **shared-additive（两种模态均响应）** 和 **interaction-dependent（存在跨模态交互效应）**；只有后者经过反事实输入和因果消融确认后，才称为强意义上的跨模态神经元。

## 1. 研究对象与统一记号

本文把第 $l$ 层 FFN 的第 $j$ 个中间维度定义为神经元。对 gated MLP，优先在送入 `down_proj` 前记录

$$
z_{l,t,j}=\phi(g_{l,t,j})\,u_{l,t,j},
$$

其中 $t$ 是 token 位置；实际激活函数和模块名从 checkpoint 配置与模型实现中发现并写入运行元数据，不以某篇论文中的普通两层 FFN 公式硬套三个模型。输出到残差流的单神经元向量为 $z_{l,t,j}W^{(l)}_{down,:,j}$。Pan 等人的输出贡献方法以及 FFN key-value memory 视角都依赖这一“激活标量 × 输出方向”的分解。[^2][^4]

统一研究 decoder FFN 是合理的，因为三个模型都把多模态 token 送入语言解码器，但它并不意味着模型内部其他部件可忽略：视觉编码器、投影器、注意力类型和局部/全局信息流会改变下游 FFN 激活。因此结论应表述为“decoder FFN 神经元的功能画像”，不能外推为整个模型的全部多模态机制。

## 2. 方法分类

| 方法族 | 代表分数或方法 | 实际测量对象 | 成本 | 能否单独确定类别 | 当前用途 |
|---|---|---|---:|---|---|
| 活跃度 | 正激活概率、超过阈值比例、RMS、均值/最大值 | 响应强度与频率 | 低 | 否 | 全量初筛、发现近似 inactive 神经元 |
| 条件选择性 | LAPE/熵、类间均值差、互信息、1-sparse probe | 激活与模态/标签的统计关联 | 低至中 | 否 | visual/text 候选与样本分布差异 |
| 输出贡献 | activation × `down_proj` × unembedding | 对指定输出 token/logit 的直接线性分量 | 中 | 否 | 概念 token、答案 token候选精排 |
| 梯度归因 | gradient × activation、Integrated Gradients | 指定目标损失/概率对神经元的局部敏感性 | 中至高 | 否 | 任务相关重要性与稳健性对照 |
| 可恢复性/剪枝代理 | FLAP、Wanda、LLM-Pruner | 删除结构后的局部输出损伤或一阶损失变化 | 低至高 | 否 | 剪枝基线，不作为语义类别真值 |
| 因果干预 | zero/mean ablation、activation patching、成组消融 | 神经元改变是否引起行为改变 | 高 | **可确认但非充分** | 最终类别验证与必要性检验 |
| 特征分解 | Sparse Autoencoder | 激活空间中的稀疏方向，而非原生神经元 | 很高 | 不适用于原生神经元四分类 | 后续独立研究，不纳入首轮主实验 |

### 2.1 活跃度指标：适合筛选，不等于重要性

MINER 将视觉、文本、特殊 token 分组后，比较了正激活概率、均值、最大值以及两种注意力加权汇总，并强调不同模态 token 数差异必须在组内归一化；其 top-$K$ 选择和消融结果说明这类前向指标可有效缩小搜索空间。[^1] LAPE 则先估计每种条件下的激活概率，再对条件概率向量做 L1 归一化并计算熵；低熵表示条件选择性强。[^5]

但“激活值 $>0$”在 OPT 的 ReLU 中代表严格开/关，在 SiLU/GELU 或 gated MLP 中通常不产生大量精确零，因此不能直接搬用“dead neuron”的含义。Voita 等人的 dead / n-gram / positional 分类建立在 OPT ReLU 的精确非零判据上；对本研究三个 checkpoint，应把 active 定义为超过在校准集冻结的正向阈值或绝对幅度阈值，并同时报告 RMS，避免把微小正值当作有效激活。[^6]

建议全量保存以下每神经元、每条件统计量：

$$
p_c=P(z>\tau_{l,j}\mid c),\quad
r_c=\sqrt{E[z^2\mid c]},\quad
\mu_c=E[z\mid c],\quad
v_c=\operatorname{Var}(z\mid c).
$$

$\tau_{l,j}$ 只在 calibration split 上确定，例如该神经元绝对激活的 95% 分位数或稳健尺度 `median + k·MAD`；不能使用包含测试集的全局最大值。Xu 等人在 LLaVA 上将每个神经元负值截为 0、最大值映射到 10，再以固定 token 计数阈值构造四类概率，这很适合作为**复现实验基线**，但其阈值来自单模型观察，最大值归一化又容易受极端样本影响，不应作为三模型共同定义。[^7]

### 2.2 条件选择性与稀疏探针：最适合做功能候选定位

将 LAPE 从“语言条件”改为“输入条件”后，可定义模态选择性熵：

$$
q_c=\frac{p_c+\epsilon}{\sum_{c'}(p_{c'}+\epsilon)},\qquad
H_{l,j}=-\sum_c q_c\log q_c.
$$

低熵只表示偏好某一条件；还必须设置最低响应门槛，否则一个几乎从不激活的神经元也可能因偶然响应而得到低熵。LAPE 原文同样同时使用低熵百分位和最低激活概率条件。[^5]

如果数据有清晰监督标签，可训练 1-sparse logistic probe：每次只允许一个神经元预测“视觉条件/文本条件”“任务标签”或样本属性，以 held-out F1、AUROC 和校准误差衡量预测性。Gurnee 等人指出 $k=1$ 可定位单神经元，较大的 $k$ 可识别稀疏组合；他们也明确警告 probing 主要提供相关性证据，会受数据混杂、冗余和特征定义影响，不能推出因果性。[^8] Skill Neurons 使用相对训练集均值的激活来预测二分类标签，并跨五次 prompt-tuning trial 聚合 predictivity；其“跨 trial 一致”思想值得采用，但其 soft-prompt 位置和 encoder 分类任务设定不宜原样迁移到生成式 VLM。[^9]

### 2.3 输出贡献：比激活幅度更接近当前预测

仅凭 $|z|$ 会忽略神经元输出方向。对目标答案 token $y$，可计算近似的有符号 logit contribution：

$$
C_{l,t,j}(y)=z_{l,t,j}\left\langle W^{(l)}_{down,:,j},u_y\right\rangle,
$$

其中 $u_y$ 是 unembedding 的目标方向。Pan 等人用等价分解对生成 caption 中的名词 token 排序，方法无需梯度，并通过敏感性、特异性和干预实验检查所定位的 multimodal neurons。[^2] 当前实验应同时保留 signed contribution 和 absolute contribution：前者区分促进/抑制，后者用于重要性排序。对多 token 答案，应对 teacher-forced 的正确答案 log-likelihood 汇总，而不是只取名词的第一个 sub-token。

这一分解是局部线性近似：LayerNorm、后续层与非线性交互意味着跨层 contribution 不能被视作完整的概率分解。因此它适合精排，但仍需因果检验。

### 2.4 梯度归因：目标明确，但更贵且会受饱和影响

`gradient × activation` 可定义为

$$
G_{l,t,j}=\left|z_{l,t,j}\frac{\partial \log p(y\mid x)}{\partial z_{l,t,j}}\right|.
$$

它直接面向正确答案概率或任务损失，适合作为当前 scaffold 中的主归因指标。Integrated Gradients（IG）沿 baseline 到真实激活的路径积分梯度，满足 Sensitivity 与 Implementation Invariance 等公理；Knowledge Neurons 将 IG 用于 FFN 中间神经元，并在多个同义 prompt 之间取稳定交集来过滤词法/句法假阳性。[^10][^11] 这一“多表达共享 + IG”设计非常适合少量关键候选，但 IG 需要多次反向传播；Knowledge Neurons 原实验用 20 个积分步，不适合直接在所有模型、所有神经元、所有样本上全量运行。[^11]

建议：全量运行 `gradient × activation`，对每层每类 top 候选和分歧样本运行 16–32 步 IG；至少比较零 baseline 与条件均值 baseline，因为 baseline 的选择本身会影响归因。

### 2.5 剪枝指标：适合作为对照，不适合定义语义类型

FLAP 的 channel 分数等于输入特征的样本方差乘相应权重列的平方范数，衡量删除该 channel 后输出能否用均值 bias 近似恢复；它还对不同层/模块的分数标准化。[^12] Wanda 使用权重幅度乘输入激活范数做无重训练的权重剪枝；若聚合到 FFN channel，可形成低成本损伤代理。[^13] LLM-Pruner 使用梯度信息评估耦合结构并进行结构化剪枝。[^14]

三者回答的是“删掉后局部损失/恢复难度如何”，不是“它属于视觉还是文本”。因此应放入 pruning baseline：与 modality typing 排名比较、检验新增分类是否带来超越通用剪枝分数的价值，而不是混入类别标签本身。

### 2.6 因果干预：类别确认的必要环节

发现集上的相关性必须在未参与阈值选择的测试集上验证。推荐至少做两种干预：

- **zero ablation**：将候选神经元的 post-gating 激活置零，测量任务指标与正确答案 log-likelihood 的变化；
- **mean ablation**：替换为校准集同层、同位置条件均值，减少零值离分布过远造成的干预伪影。

每个候选组都必须配三种对照：同层同数量随机神经元、同层同活动度匹配神经元、同层同通用重要性（RMS/FLAP）匹配神经元。MINER 的成组失活和 Pan 等人的编辑/扰动结果说明干预可验证所选集合的行为影响；ROME 的 causal tracing 也说明恢复或破坏内部状态可定位对预测有决定性的计算。[^1][^2][^15] 但单神经元消融无效不等于没有表征：稀疏 probing 研究发现预测性强的神经元有时仍因冗余或上游已完成计算而表现出很弱的单点因果效应。[^8]

## 3. “分类是如何确定的”：推荐判定协议

### 3.1 先做 2×2 反事实设计，而不只比较 token 位置

同一序列中的视觉 token 与文本 token 经过注意力后已经相互混合；MINER 也明确把 token 级模态分离称为近似假设。[^1] 因此应同时使用 token mask 和配对反事实输入：

| 条件 | 图像信息 | 文本信息 | 示例 |
|---|---|---|---|
| $x_{11}$ | 有效 | 有效 | 原图 + 原问题 |
| $x_{10}$ | 有效 | 无信息/不匹配 | 原图 + 长度匹配的中性问题或错配问题 |
| $x_{01}$ | 无信息/不匹配 | 有效 | 空白/打乱/错配图 + 原问题 |
| $x_{00}$ | 无信息/不匹配 | 无信息/不匹配 | 两者均为匹配控制 |

控制输入长度、图像 token 数、答案格式和位置；“删除图像”常会改变序列长度和模板，优先使用空白、patch shuffle 或同类错配图等长度匹配控制，并分别报告不同控制的结果。

对任一神经元分数 $s$（RMS、超阈概率、输出贡献或归因），计算

$$
M_V=s_{11}-s_{01},\quad
M_T=s_{11}-s_{10},\quad
I_{VT}=s_{11}-s_{10}-s_{01}+s_{00}.
$$

$M_V$ 和 $M_T$ 是视觉与文本主效应，$I_{VT}$ 是交互效应。这使“同时对两类 token 高响应”和“只有两种有效信息共同出现才响应”成为两个可检验的类别。

### 3.2 使用连续画像，最后再映射到标签

每个神经元至少保存以下画像，而不是只保存一个 class：

`activity`, `visual_main_effect`, `text_main_effect`, `interaction_effect`, `label_predictivity`, `output_contribution`, `gradient_attribution`, `causal_visual`, `causal_text`, `causal_crossmodal`, `stability`, `confidence`。

建议标签规则如下：

| 标签 | 统计发现条件 | 测试集确认条件 |
|---|---|---|
| inactive-on-support | 所有条件 activity 均低于预注册下限 | 扩大样本后上置信界仍低；不得称“可安全剪枝” |
| visual-selective | $M_V>0$，$M_T$ 与 $I_{VT}$ 弱 | 消融优先损害视觉任务，超过匹配对照 |
| text-selective | $M_T>0$，$M_V$ 与 $I_{VT}$ 弱 | 消融优先损害 text-only/语言任务 |
| shared-additive | $M_V,M_T>0$，$I_{VT}$ 弱 | 对两类任务均有影响，但无需跨模态配对 |
| interaction-dependent | $I_{VT}$ 稳定且效应量大 | 消融对跨模态配对/推理的额外损伤最大 |
| label/task-predictive | held-out 1-sparse probe 或互信息高 | 对相应标签/任务有特异因果效应 |
| contextual/polysemantic | 不同数据集或概念簇触发不同模式 | 保留多标签与样本簇，不强制归入单类 |
| unresolved | 证据弱、方向冲突或不稳定 | 继续保留，不等同低重要性 |

如果论文图表需要四类，可把 `shared-additive` 与 `interaction-dependent` 合并显示为 multimodal，把 inactive/contextual/unresolved 显示为 unknown；但底层表必须保留细分类和连续分数。

### 3.3 阈值与置信度

分类流程应固定为 discovery / calibration / test 三段：discovery 选方法和候选；calibration 确定每层阈值、候选比例和分数融合方式；test 只做一次最终估计。阈值优先使用每模型、每层百分位或稳健 z-score，跨模型只比较百分位、候选比例和标准化效应量，不比较原始激活值。

对 $M_V,M_T,I_{VT}$ 和因果效应做按样本 bootstrap 置信区间；对逐神经元显著性检验控制 Benjamini–Hochberg FDR。类别需满足：

- 至少两个数据子集/数据集方向一致；
- 多个 prompt 模板或随机种子中 rank/label 稳定；
- 95% bootstrap CI 不跨越预注册的最小实质效应（不仅是不跨 0）；
- 因果效应超过同层、同数量、同活动度匹配对照。

## 4. 三模型适用性筛选

| 模型 | 架构差异 | 可直接统一的部分 | 必须分层/控制的部分 | 总体适用性 |
|---|---|---|---|---|
| LLaVA-1.5-7B | 视觉编码器 + 投影器 + Vicuna/LLaMA 自回归 decoder；标准全注意力 | post-gating FFN、激活/RMS、贡献、梯度、消融 | 图像 patch token 与问题/答案 token 分开；可直接复现 Xu/Pan 类方法 | **最高，作为基准模型** |
| Qwen3.5-2B | 24 层；每 4 层为 3 个 Gated DeltaNet linear-attention + 1 个 full-attention；FFN 宽 6144 | 同一 FFN 定义和所有分数 | 必须按 linear/full attention strata 匹配；不要让层型差异冒充神经元类型差异 | **高，最适合检验架构泛化** |
| Gemma-4-E2B-it | 35 层；4 个 sliding + 1 个 full attention 周期；视觉/音频独立编码器；PLE；FFN 宽 6144 | decoder FFN 的统计、归因、消融 | 按 sliding/full 分层；图像与音频应分开建条件；PLE 可能形成层特有基线 | **高，但应在视觉主实验稳定后加入音频** |

LLaVA 模型卡将其描述为基于 LLaMA/Vicuna 的自回归视觉语言 chatbot。[^16] Qwen3.5-2B 官方模型卡给出 24 层、`3×(Gated DeltaNet→FFN)+1×(Gated Attention→FFN)` 的六次重复和 6144 的 FFN 中间维度。[^17] Gemma-4-E2B-it 官方模型卡给出 35 层、512-token sliding window、图像/音频支持、独立编码器与 PLE，并说明其局部滑窗和全局注意力交错。[^18]

跨模型对比应使用：每层 top-$q$% 候选比例、标准化效应量、Jaccard 仅用于同模型跨数据/seed 稳定性、层相对深度 $l/(L-1)$、以及按 attention type 的分层结果。不同模型没有天然一一对应的 neuron index，不能比较“同编号神经元”。如需跨模型功能对齐，应比较激活画像或概念标签，而不是权重坐标。

## 5. 方法取舍与优先级

### Tier A：首轮主实验必须实现

1. post-gating activation RMS 与稳健超阈激活概率；
2. 模态条件概率熵/主效应/交互效应；
3. teacher-forced 正确答案的 `gradient × activation`；
4. 每层 top-$q$ 候选的 zero + mean ablation；
5. 同层、同数量、同活动度匹配随机对照；
6. discovery/calibration/test、bootstrap CI、跨 seed/数据集稳定性。

这组方法覆盖“是否响应—响应什么—是否影响目标—是否因果必要”四个问题，且 activation collection 可以复用于多个前向指标。

### Tier B：候选精排与论文增强

1. Pan-style signed logit contribution；
2. 1-sparse probe 与 $k$-sparse probe 曲线；
3. 16–32 步 IG；
4. FLAP channel score、Wanda channel aggregation、LLM-Pruner 一阶分数作为剪枝对照；
5. interaction-dependent 候选的 activation patching。

### Tier C：暂不进入首轮

1. GPT-4o 自动解释与 activation simulator：适合生成语义解释，但引入闭源模型、提示词和评分器偏差，不能承担类别真值；[^7]
2. Sparse Autoencoder：能缓解 polysemanticity，但研究单位变成稀疏 latent feature，不再是原生 FFN neuron；[^19]
3. 对全部神经元做 IG 或逐神经元单点消融：成本与收益不成比例；
4. 把 unknown 直接作为优先剪枝集合：存在把分布校准、位置或广谱基础设施神经元误判为无用的风险。

## 6. 推荐的最小可发表实验矩阵

每个 `(model, dataset, seed)` 只做一次可复用 activation collection。建议先在 LLaVA 上完成 pilot，再固定定义后扩展到 Qwen 和 Gemma：

1. **Pilot（LLaVA，500–1000 样本）**：确定 hook 的张量语义、token masks、稳健阈值、反事实输入是否保持长度和模板一致。
2. **Discovery**：三数据集 × 三模型 × 至少三个 seed，收集 Tier A 全量分数；按模型/层取候选，而非全局 top-$K$。
3. **Calibration**：选择候选比例、最小实质效应、四类展示映射；冻结所有阈值。
4. **Test**：报告连续画像、细分类与四类映射；进行成组 zero/mean ablation 和匹配对照。
5. **Generalization**：训练/定阈于 dataset A，测试 B/C；再做 leave-one-dataset-out。跨模型不迁移 neuron index，只迁移判定规则与标准化阈值。
6. **Architecture analysis**：Qwen 按 linear/full attention、Gemma 按 sliding/full attention 分层；用层相对深度和 matched strata 比较类别比例与因果效应。

首轮核心结果应是：不同方法排名的一致性与分歧、类别在数据集/seed/模型间的稳定性、每类对匹配任务的特异消融效应、以及分类分数对通用 pruning proxy 的增量解释力。这样即使某个固定阈值后来调整，原始分数和证据链仍然有效。

## 7. 风险与解释边界

- 高激活可能是结果而非原因；高预测性也可能来自数据混杂。
- 单神经元可能 polysemantic，多神经元可能冗余或以组合方式表示功能；因此允许多标签和 unresolved。
- zero ablation 可能产生离分布输入，必须与 mean ablation 共同报告。
- 成组消融会产生非线性交互，组效应不能简单除以神经元数解释成平均单元效应。
- 图像 token、文本 token、答案 token 的数量差异会扭曲均值与最大值；必须先在条件内归一化。
- 模型生成答案会造成“目标随模型错误而变化”；重要性归因优先使用 teacher-forced 标准答案，同时把自由生成作为单独行为评估。
- 自动解释可以帮助命名候选概念，但不能替代 held-out 预测、反事实与消融证据。

## Sources

[^1]: Huang et al., “MINER: Mining the Underlying Pattern of Modality-Specific Neurons in Multimodal Large Language Models,” arXiv:2410.04819 (2024). https://arxiv.org/abs/2410.04819
[^2]: Pan et al., “Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers,” Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.60/
[^3]: Gurnee et al., “Finding Neurons in a Haystack: Case Studies with Sparse Probing,” arXiv:2305.01610 (2023). https://arxiv.org/abs/2305.01610
[^4]: Geva et al., “Transformer Feed-Forward Layers Are Key-Value Memories,” EMNLP 2021. https://aclanthology.org/2021.emnlp-main.446/
[^5]: Tang et al., “Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models,” ACL 2024. https://aclanthology.org/2024.acl-long.309/
[^6]: Voita, Ferrando, and Nalmpantis, “Neurons in Large Language Models: Dead, N-gram, Positional,” Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.75/
[^7]: Xu et al., “Deciphering Functions of Neurons in Vision-Language Models,” arXiv:2502.18485 (2025). https://arxiv.org/abs/2502.18485
[^8]: Gurnee et al., “Finding Neurons in a Haystack: Case Studies with Sparse Probing,” especially Sections 3 and 5.4. https://arxiv.org/abs/2305.01610
[^9]: Wang et al., “Finding Skill Neurons in Pre-trained Transformer-based Language Models,” EMNLP 2022. https://aclanthology.org/2022.emnlp-main.765/
[^10]: Sundararajan, Taly, and Yan, “Axiomatic Attribution for Deep Networks,” ICML 2017. https://proceedings.mlr.press/v70/sundararajan17a.html
[^11]: Dai et al., “Knowledge Neurons in Pretrained Transformers,” ACL 2022. https://aclanthology.org/2022.acl-long.581/
[^12]: An et al., “Fluctuation-based Adaptive Structured Pruning for Large Language Models,” AAAI 2024 / arXiv:2312.11983. https://arxiv.org/abs/2312.11983
[^13]: Sun et al., “A Simple and Effective Pruning Approach for Large Language Models (Wanda),” ICLR 2024. https://arxiv.org/abs/2306.11695
[^14]: Ma, Fang, and Wang, “LLM-Pruner: On the Structural Pruning of Large Language Models,” NeurIPS 2023. https://arxiv.org/abs/2305.11627
[^15]: Meng et al., “Locating and Editing Factual Associations in GPT,” NeurIPS 2022. https://arxiv.org/abs/2202.05262
[^16]: Hugging Face, `llava-hf/llava-1.5-7b-hf` model card. https://huggingface.co/llava-hf/llava-1.5-7b-hf
[^17]: Qwen Team, `Qwen/Qwen3.5-2B` model card. https://huggingface.co/Qwen/Qwen3.5-2B
[^18]: Google, `google/gemma-4-E2B-it` model card. https://huggingface.co/google/gemma-4-E2B-it
[^19]: Cunningham et al., “Sparse Autoencoders Find Highly Interpretable Features in Language Models,” arXiv:2309.08600 (2023). https://arxiv.org/abs/2309.08600
