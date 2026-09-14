# 多模态大模型神经元重要性与分类方法调研

**导师讨论稿**
**研究对象：** LLaVA-1.5-7B、Qwen3.5-2B、Gemma-4-E2B-it 的语言 Decoder FFN 神经元
**整理日期：** 2026-09-12

## 1. 结论与待确认选择

“神经元重要性”和“神经元分类”需要分开定义：重要性回答一个神经元是否影响模型响应或正确答案；分类回答它更偏视觉、文本、共享模态还是图文交互。

### 1.1 重要性计算建议

首轮保留四类互补指标，不强行合成为单一总分：

| 方法 | 测量内容 | 在实验中的角色 |
|---|---|---|
| 有效激活概率与 RMS | 响应频率和响应幅度 | 全量、低成本初筛 |
| Logit Contribution | 对目标答案 token 的局部促进或抑制 | 多模态输出贡献分析 |
| Gradient × Activation | 正确答案对神经元的局部敏感性 | 主任务重要性指标 |
| Zero / Mean Ablation | 干预后模型行为是否变化 | 最终因果验证 |

推荐顺序是“激活初筛—任务归因排序—因果验证”，但四种分数分别保留，避免把不同含义混入一个不可解释的总分。

### 1.2 分类方法需要导师选择

| 方案 | 核心思想 | 计算成本 | 跨模型性 | 创新性 | 建议定位 |
|---|---|---:|---:|---:|---|
| A. LLaVA 四类阈值投票 | 高激活视觉/文本 token 数量投票 | 低 | 较弱 | 低 | 文献复现基线 |
| B. 激活概率与选择性熵 | 比较各模态的稳定激活概率 | 低 | 强 | 中 | 跨模型泛化主方案 |
| C. 2×2 反事实析因分类 | 分离视觉主效应、文本主效应和图文交互 | 约 4 倍 | 强 | 高 | 跨模态机制主方案 |

**建议：** 若论文重点是“跨模型、跨数据集泛化”，选择方案 B；若重点是“什么才是真正的跨模态神经元”，选择方案 C。方案 A 无论如何都可以保留为低成本文献基线。

## 2. 神经元定义与统一符号

对于 gated MLP，语言 Decoder 第 $l$ 层、token 位置 $t$ 的 FFN 中间激活为：

$$
g_{l,t}=W^{(l)}_{\mathrm{gate}}h_{l,t},\qquad
u_{l,t}=W^{(l)}_{\mathrm{up}}h_{l,t},
$$

$$
z_{l,t}=\phi(g_{l,t})\odot u_{l,t},\qquad
o_{l,t}=W^{(l)}_{\mathrm{down}}z_{l,t}.
$$

其中：

- $l$：Decoder 层编号；
- $t$：token 位置；
- $h_{l,t}$：进入 FFN 的隐藏状态；
- $\phi$：SiLU、GELU 等激活函数；
- $\odot$：逐元素乘法；
- $z_{l,t,j}$：第 $l$ 层第 $j$ 个 FFN 神经元的激活；
- $W^{(l)}_{\mathrm{down},:,j}$：该神经元写回残差流的输出方向。

实验统一在 `down_proj` 之前记录 $z_{l,t,j}$。具体激活函数和模块名以各 checkpoint 的实际实现为准。

## 3. 重要性方法一：有效激活概率与 RMS

### 3.1 有效激活概率

$$
P_{l,j,c}
=
\frac{1}{N_c}
\sum_{(s,t)\in c}
\mathbf{1}\!\left(|z_{l,s,t,j}|>\tau_{l,j}\right).
$$

其中：

- $s$：样本编号；
- $t$：token 位置；
- $c$：视觉、文本、回答等 token 条件；
- $N_c$：条件 $c$ 下参与统计的 token 数；
- $z_{l,s,t,j}$：对应神经元激活；
- $\tau_{l,j}$：该神经元的有效激活阈值；
- $\mathbf{1}(\cdot)$：条件成立取 1，否则取 0。

$P_{l,j,c}$ 越大，表示该神经元在条件 $c$ 下越经常产生有效响应。

由于 SiLU/GELU 不像 ReLU 那样产生大量精确零，不能直接使用 $z>0$ 作为通用激活定义。建议只使用 calibration set 确定阈值，例如：

$$
\tau_{l,j}=Q_{0.95}\!\left(|z_{l,j}|\right),
$$

其中 $Q_{0.95}$ 表示该神经元绝对激活值的 95% 分位数。也可以在敏感性分析中比较 90%、95% 和 97% 分位阈值。

### 3.2 RMS 激活

$$
R_{l,j,c}
=
\sqrt{
\frac{1}{N_c}
\sum_{(s,t)\in c}
z_{l,s,t,j}^{2}
}.
$$

RMS 对正负激活均有效，同时反映响应幅度和强激活。它适合在同一个模型、同一层内排序；跨模型比较时应转换为层内百分位或稳健 z-score。

### 3.3 为什么保留

- 只需要前向传播，可以覆盖全部神经元；
- 同一次 activation collection 可以同时计算概率、RMS、均值和最大值；
- [MINER](https://arxiv.org/abs/2410.04819) 使用激活概率、均值、最大值及注意力加权激活定位 modality-specific neurons，并强调需要在模态 token 集合内归一化。

### 3.4 局限

高激活只能说明神经元“响应了”，不能证明模型为了得到正确答案“使用了它”。因此该方法只承担初筛，不独立形成任务重要性或因果结论。

## 4. 重要性方法二：Logit Contribution

单个神经元对残差流的局部写入为：

$$
v_{l,t,j}
=
z_{l,t,j}W^{(l)}_{\mathrm{down},:,j}.
$$

对目标答案 token $y$，定义该神经元的有符号 logit contribution：

$$
C_{l,t,j}(y)
=
z_{l,t,j}
\left\langle
W^{(l)}_{\mathrm{down},:,j},u_y
\right\rangle.
$$

其中：

- $u_y$：目标 token $y$ 的 unembedding 方向；
- $\langle a,b\rangle$：向量内积；
- $C_{l,t,j}(y)>0$：局部上促进目标 token；
- $C_{l,t,j}(y)<0$：局部上抑制目标 token；
- $|C_{l,t,j}(y)|$：局部贡献强度。

对包含 $R$ 个 token 的完整标准答案 $y_1,\ldots,y_R$，使用 teacher forcing 聚合：

$$
I^{\mathrm{logit}}_{l,j}
=
\frac{1}{R}
\sum_{r=1}^{R}
\sum_t
\left|C_{l,t,j}(y_r)\right|.
$$

### 4.1 为什么保留

- 同时考虑神经元激活和 `down_proj` 输出方向；
- 能区分促进和抑制作用；
- [Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers](https://aclanthology.org/2024.findings-acl.60/) 使用相近的 activation-output 分解定位多模态神经元，不需要模型梯度。

### 4.2 局限

这是局部输出分解。后续 LayerNorm、非线性和其他层会继续改变最终预测，因此 $C$ 不能被解释为完整的最终概率贡献，也不能单独建立因果结论。

## 5. 重要性方法三：Gradient × Activation

将正确答案的目标函数定义为：

$$
F(x,y)=\log p(y\mid x).
$$

单个 token 上的神经元重要性为：

$$
G_{l,t,j}
=
\left|
z_{l,t,j}
\frac{\partial F(x,y)}{\partial z_{l,t,j}}
\right|.
$$

跨样本和 token 聚合：

$$
I^{\mathrm{G\times A}}_{l,j}
=
\frac{1}{N}
\sum_{s,t}
\left|
z_{l,s,t,j}
\frac{
\partial\log p(y_s\mid x_s)
}{
\partial z_{l,s,t,j}
}
\right|.
$$

其中：

- $z_{l,s,t,j}$：神经元的实际激活；
- $\partial F/\partial z$：标准答案对该激活的局部敏感度；
- 两者乘积：把该神经元缩放到零时，目标函数变化的一阶近似；
- $N$：聚合时使用的样本-token 单元数量。

### 5.1 为什么保留

- 直接面向正确答案，而不只是观察激活；
- 一次前向和反向传播即可计算；
- 可以按视觉 token、文本 token、回答 token 分别聚合；
- 相比多步 Integrated Gradients，更适合三模型、三数据集的全量实验。

### 5.2 局限

- 它是局部一阶近似；
- 梯度饱和时可能低估重要神经元；
- 分数取决于目标函数，所以实验必须固定为 teacher-forced 标准答案 log-likelihood；
- 需要后续消融检验其行为影响。

## 6. 重要性方法四：因果消融

### 6.1 Zero ablation

将目标神经元的 post-gating 激活置零：

$$
z_{l,t,j}\leftarrow 0.
$$

定义损失增量：

$$
I^{\mathrm{zero}}_{l,j}
=
\mathbb{E}_{(x,y)}
\left[
\mathcal{L}(f_{-j}(x),y)
-
\mathcal{L}(f(x),y)
\right].
$$

其中：

- $f$：原始模型；
- $f_{-j}$：神经元 $j$ 被置零后的模型；
- $\mathcal{L}$：标准答案负对数似然；
- $I^{\mathrm{zero}}>0$：删除该神经元使模型损失增加。

### 6.2 Mean ablation

将神经元替换为 calibration set 上相应条件的平均激活：

$$
z_{l,t,j}\leftarrow\bar z_{l,j,c},
\qquad
\bar z_{l,j,c}=\mathbb{E}[z_{l,t,j}\mid c].
$$

对应的重要性为：

$$
I^{\mathrm{mean}}_{l,j}
=
\mathbb{E}_{(x,y)}
\left[
\mathcal{L}(f_{\mathrm{mean}(j)}(x),y)
-
\mathcal{L}(f(x),y)
\right].
$$

### 6.3 为什么保留

消融直接检验改变神经元后模型行为是否变化，是四种方法中最强的因果证据。但它仍受神经元冗余、组合效应以及干预是否离分布影响，因此更适合验证候选集合，而非穷举全部神经元。

### 6.4 对照要求

每个消融组至少配置：

- 同层、同数量随机神经元；
- 同层、同活动度匹配神经元；
- 同层、同 RMS 或其他通用重要性匹配神经元。

Zero 和 mean ablation 应同时报告；若两者方向一致，结论更稳健。

## 7. 分类方案 A：LLaVA 四类阈值投票

该方案用于复现 [Deciphering Functions of Neurons in Vision-Language Models](https://arxiv.org/abs/2502.18485) 的四类神经元识别思路。

### 7.1 每神经元激活归一化

$$
\tilde z_{s,t,j}
=
10\cdot
\frac{\max(z_{s,t,j},0)}
{\max_{s',t'}\max(z_{s',t',j},0)+\epsilon}.
$$

其中负激活被截为 0，该神经元观察到的最大正激活被映射到 10。

### 7.2 样本级视觉和文本响应

$$
V_{s,j}
=
\mathbf{1}\!\left(
\sum_{t\in T_s^V}
\mathbf{1}(\tilde z_{s,t,j}>T_V)>n_V
\right),
$$

$$
T_{s,j}
=
\mathbf{1}\!\left(
\sum_{t\in T_s^T}
\mathbf{1}(\tilde z_{s,t,j}>T_T)>n_T
\right).
$$

其中：

- $T_s^V,T_s^T$：样本 $s$ 的视觉和文本 token 集合；
- $T_V,T_T$：归一化激活阈值；
- $n_V,n_T$：至少需要多少个高激活 token；
- 原论文示例设置为 $T_V=2,n_V=4,T_T=3,n_T=2$。

样本级分类为：

| $V_{s,j}$ | $T_{s,j}$ | 类别 |
|---:|---:|---|
| 1 | 0 | visual |
| 0 | 1 | text |
| 1 | 1 | multimodal |
| 0 | 0 | unknown |

### 7.3 Top-$N$ 样本投票

在神经元 $j$ 激活最高的 top-$N$ 样本上计算：

$$
p_c(j)
=
\frac{1}{N}
\sum_{s\in\operatorname{TopN}(j)}
\mathbf{1}(C_{s,j}=c),
$$

并定义：

$$
C_j=\arg\max_{c\in\{V,T,M,U\}}p_c(j).
$$

可增加最低置信度要求 $\max_c p_c(j)\ge\gamma$；未达到时标记为 unresolved。

### 7.4 优点、局限和定位

**优点：** 已有 VLM 论文直接采用；计算成本低；容易复现和比较。
**局限：** 阈值来自单模型观察；最大值归一化容易受异常值影响；视觉和文本都激活不一定代表真正图文融合；unknown 的功能含义不明确。
**定位：** 复现已有分类方法，并检验它能否泛化到三个模型和三个数据集。创新性较弱，但风险最低。

## 8. 分类方案 B：模态激活概率与选择性熵

该方案借鉴 [MINER](https://arxiv.org/abs/2410.04819) 的模态 token 分组与重要性汇总，以及 [Language-Specific Neurons / LAPE](https://aclanthology.org/2024.acl-long.309/) 的条件激活概率熵思想，用于建立跨模型统一分类规则。

### 8.1 模态激活概率

$$
p_V(j)=P(|z_j|>\tau_j\mid V),
\qquad
p_T(j)=P(|z_j|>\tau_j\mid T).
$$

归一化为：

$$
q_V
=
\frac{p_V+\epsilon}{p_V+p_T+2\epsilon},
\qquad
q_T
=
\frac{p_T+\epsilon}{p_V+p_T+2\epsilon}.
$$

选择性熵为：

$$
H_j=-q_V\log q_V-q_T\log q_T.
$$

总体活跃度定义为：

$$
A_j=\max(p_V,p_T).
$$

其中：

- $p_V,p_T$：视觉和文本条件下的有效激活概率；
- $q_V,q_T$：归一化模态响应比例；
- $H_j$：选择性熵，越低表示越偏向某一模态；
- $A_j$：至少在一种模态下的活动程度。

### 8.2 分类规则

| 类别 | 建议规则 |
|---|---|
| visual | $A_j\ge a_0$ 且 $q_V\ge q_0$ |
| text | $A_j\ge a_0$ 且 $q_T\ge q_0$ |
| multimodal/shared | $p_V\ge a_0$ 且 $p_T\ge a_0$，两者差异不大 |
| inactive | $\max(p_V,p_T)<a_0$ |
| uncertain | 没有任何类别达到置信度要求 |

$\tau_j,a_0,q_0$ 均在 calibration set 上，按每个模型、每一层的分位数确定，并在 test set 上冻结。

### 8.3 优点、局限和定位

**优点：** 不依赖 LLaVA 固定阈值；只需正常前向传播；方便跨模型、跨层和跨数据集比较；比方案 A 更符合 generalization 研究主题。
**局限：** 仍属于激活相关性分类；不能区分“对两种模态都响应”和“必须依赖正确图文配对”；需要 calibration set。
**定位：** 建立能够跨 VLM 架构和数据集统一使用的神经元分类方法。风险和工作量适中。

## 9. 分类方案 C：2×2 反事实主效应与交互效应

该方案是本调研基于析因实验设计提出的研究性扩展，不是直接复刻某一篇神经元论文。它针对方案 A、B 无法区分 shared response 与真正跨模态交互的问题。

### 9.1 构造四种输入

| 条件 | 图像 | 文本 |
|---|---|---|
| $x_{11}$ | 正确图像 | 正确文本 |
| $x_{10}$ | 正确图像 | 无效或错配文本 |
| $x_{01}$ | 无效或错配图像 | 正确文本 |
| $x_{00}$ | 无效图像 | 无效文本 |

四种输入应尽量保持图像 token 数、文本长度、prompt 模板和答案格式一致。无效图像可以使用空白、patch shuffle 或同类别错配图像，并分别做敏感性分析。

### 9.2 主效应与交互效应

对每个条件计算神经元分数 $s_j(x)$，例如 RMS 或 Gradient × Activation。定义：

$$
M_{V,j}=s_j(x_{11})-s_j(x_{01}),
$$

$$
M_{T,j}=s_j(x_{11})-s_j(x_{10}),
$$

$$
I_{VT,j}
=
s_j(x_{11})-s_j(x_{10})-s_j(x_{01})+s_j(x_{00}).
$$

其中：

- $M_{V,j}$：有效视觉信息带来的神经元分数变化；
- $M_{T,j}$：有效文本信息带来的神经元分数变化；
- $I_{VT,j}$：正确图文组合超出两个单模态主效应的额外交互。

### 9.3 分类规则

| 类别 | 建议规则 |
|---|---|
| visual-selective | $M_V$ 大，$M_T$ 和 $I_{VT}$ 小 |
| text-selective | $M_T$ 大，$M_V$ 和 $I_{VT}$ 小 |
| shared-additive | $M_V,M_T$ 都大，但 $I_{VT}$ 小 |
| interaction-dependent | $I_{VT}$ 显著且跨样本/seed 稳定 |
| inactive | 所有条件下活动度都低 |
| unresolved | 效应不稳定或置信区间过宽 |

“大”和“小”不能凭观察决定，应在 calibration set 上预注册最小实质效应 $\delta_V,\delta_T,\delta_I$，并在 test set 上使用 bootstrap 置信区间判断。

### 9.4 优点、局限和定位

**优点：** 能区分 shared-additive 和 interaction-dependent；对 multimodal neuron 的定义最严格；创新空间最大。
**局限：** 每个样本需要四种输入；反事实控制设计困难；错配可能引入新的分布偏移；统计分析更复杂。
**定位：** 研究 VLM 中共享模态神经元与真正图文交互神经元的区别。风险最高，但机制研究价值也最高。

## 10. 分类后的共同验证：类别特异性消融

类别特异性消融不是第四种分类方案，而是可以附加在 A、B 或 C 后面的验证模块。

设消融某一类候选神经元后，在视觉、文本和跨模态任务上的损失增量为：

$$
\Delta_V,\qquad\Delta_T,\qquad\Delta_M.
$$

定义视觉、文本和跨模态因果特异性：

$$
C_V=\Delta_V-\frac{\Delta_T+\Delta_M}{2},
$$

$$
C_T=\Delta_T-\frac{\Delta_V+\Delta_M}{2},
$$

$$
C_M=\Delta_M-\frac{\Delta_V+\Delta_T}{2}.
$$

只有当统计分类和相应因果特异性方向一致，并且效应超过同层、同数量、同活动度匹配对照时，才标记为 confirmed visual、confirmed text 或 confirmed multimodal。

## 11. 建议提交导师确认的三个完整选项

### 选项 1：文献复现型

<div class="decision-box"><strong>方案 A + 类别级成组消融</strong></div>

工作量最低，结论容易与已有 LLaVA 论文对齐；适合优先保证复现，但跨模型阈值合理性和方法创新有限。

### 选项 2：跨模型泛化型（推荐）

<div class="decision-box"><strong>方案 B + 类别特异性消融</strong></div>

使用层内校准的激活概率与选择性熵，在三个模型、三个数据集上统一计算，再通过类别特异性消融确认。它在可实现性、跨模型比较和方法改进之间最平衡。

### 选项 3：跨模态机制型

<div class="decision-box"><strong>方案 C + 类别特异性消融</strong></div>

能够进一步区分共享响应和真正图文交互，研究问题最深入，但反事实构造、计算量和统计复杂度最高。

## 12. 参考文献与可点击链接

1. Huang et al. **MINER: Mining the Underlying Pattern of Modality-Specific Neurons in Multimodal Large Language Models.** [https://arxiv.org/abs/2410.04819](https://arxiv.org/abs/2410.04819)
2. Pan et al. **Finding and Editing Multi-Modal Neurons in Pre-Trained Transformers.** [https://aclanthology.org/2024.findings-acl.60/](https://aclanthology.org/2024.findings-acl.60/)
3. Xu et al. **Deciphering Functions of Neurons in Vision-Language Models.** [https://arxiv.org/abs/2502.18485](https://arxiv.org/abs/2502.18485)
4. Tang et al. **Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models.** [https://aclanthology.org/2024.acl-long.309/](https://aclanthology.org/2024.acl-long.309/)
5. Sundararajan, Taly, and Yan. **Axiomatic Attribution for Deep Networks.** [https://proceedings.mlr.press/v70/sundararajan17a.html](https://proceedings.mlr.press/v70/sundararajan17a.html)
6. Dai et al. **Knowledge Neurons in Pretrained Transformers.** [https://aclanthology.org/2022.acl-long.581/](https://aclanthology.org/2022.acl-long.581/)
