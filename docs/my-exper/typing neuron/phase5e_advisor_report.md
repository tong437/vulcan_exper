# Phase 5E：单样本任务语义极限实验报告

**汇报对象：** 导师/课题组

**实验对象：** Qwen3.5-0.8B 多模态模型，COCO val 固定单样本

**语义契约版本：** `phase5e_bicycles_in_train_v2`
**报告日期：** 2026-09-07

## 一、技术摘要

Phase 5E 研究的问题是：**对同一张图，真实物理删除多少 FFN neurons 后，模型仍能生成语义正确的图像
caption？** 本阶段不再要求复现原模型的思考过程或逐 token 输出，而是把任务正确性定义为一个可审计的
人工语义契约。

在固定图像、固定 prompt、确定性生成和当前 exact-budget learned-gate 搜索族下，本阶段得到的最好结果是：

- 真实删除 `62,950 / 86,016 = 73.18%` 的 FFN neurons；
- 删除 `193,382,400 / 852,985,920 = 22.67%` 的模型参数；
- 模型参数量由 `852,985,920` 降至 `659,603,520`；
- 保存后独立重载仍生成：`A group of bicycles on a subway train.`；
- 输出正常 EOS 结束，没有 thinking，没有触及 64-token 上限；
- 自动语义门与人工检查均判定通过；
- 当前经验边界为 `62,950 pass / 63,000 no-clean-pass`，区间宽度 50 neurons。

该结果应表述为：**在当前单样本、搜索算法、3 个随机 restart 和候选 mask 族下，已经构造并物理验证的语义
保持下界为 62,950 个 FFN neurons。** 63,000 的失败只是当前搜索没有找到干净候选，不是理论不可行上界，
更不是模型整体视觉能力的压缩结论。

## 二、研究动机与问题定义

Phase 5A–5D 的主要判据是严格行为等价，尤其关注前 512 个思考 tokens 是否与原模型完全一致。这一标准适合
研究内部轨迹的脆弱性，但对实际图像描述任务过于严格：模型即使在中间思考或措辞上发生分叉，最后仍可能给出
完全正确的 caption。

因此，Phase 5E 将问题从“能否复制原模型轨迹”改为“能否完成固定任务语义”：

> 对固定图像和 prompt，在真实删除 FFN neurons 后，模型能否继续生成一条完整、连贯、语义正确的 caption？

这一定义允许同义表达，不要求：

- 思考过程一致；
- 与原模型逐 token 一致；
- 与人工 caption 逐字一致。

这种改变使实验测量的对象从“轨迹克隆能力”转向“单样本任务功能保留”。

## 三、样本、模型与成功标准

### 3.1 固定样本

- 数据来源：COCO captions validation split；
- 原始 source offset：2500；
- 图像：`COCO_val2014_000000093736.jpg`；
- 人工 caption：`A group of bicycles on a subway train.`；
- 专用 prompt：要求仅输出一句简洁、完整的英文图像 caption；
- 评测数据被复制为独立的一行冻结数据集，避免后续 prompt 或数据顺序变化。

### 3.2 基础模型与剪枝对象

- 模型：Qwen3.5-0.8B 多模态模型；
- 原始参数量：`852,985,920`；
- Transformer 层数：24；
- 每层原始 FFN intermediate width：3,584；
- FFN neuron 总数：`24 × 3,584 = 86,016`；
- 剪枝对象：每层 gated MLP 的中间 neurons；
- 物理删除一个 neuron 对应删除 gate/up/down 三组关联权重。

### 3.3 人工语义契约

通过 caption 必须同时满足：

1. 出现 bicycle/bike 概念；
2. 表达多辆或一组自行车；
3. 场景是 train/subway/rail car/carriage；
4. 表达自行车位于车厢内或车上；
5. 输出完整、连贯并正常结束。

以下情况拒绝：

- 未提到自行车；
- 明确错误场景，如街道、室外或公交车；
- 明确只有一辆自行车；
- 否定自行车存在；
- 只有 thinking、乱码、角色词垃圾、重复退化、截断或没有正常结束。

额外描述车包、车门、窗户等细节不影响核心语义，除非引入明显矛盾。

## 四、实验设计

### 4.1 确定性短生成

所有候选使用统一生成配置：

```yaml
do_sample: false
temperature: null
enable_thinking: false
max_new_tokens: 64
```

模板使用 `qwen3_5` 并显式关闭 thinking。每次生成记录原始 token、最终抽取答案、EOS、是否达到 token 上限、
生成耗时和峰值 CUDA allocated memory。若模板异常地产生 thinking，只评价闭合 thinking 标签后的最终答案；
thinking 未闭合或在预算内没有输出 caption 时直接失败。

### 4.2 三层评价体系

**第一层：gold-caption teacher forcing 代理。** 在 11 个 gold tokens 上进行一次 forward，记录平均 NLL、相对
未剪枝模型的 delta NLL、token-level KL、gold-token logit margin 和 top-1 accuracy。这些指标用于排序和优化，
不直接决定任务是否通过。

**第二层：自由生成语义门。** 候选仅接收原图和 prompt，从头自由生成 caption。规则检查自行车、多辆、列车、
空间关系、否定、矛盾和正常结束，输出 `pass/fail/review`。完整 caption 和停止状态必须共同审查。

**第三层：人工确认。** 只对少量 frontier 物理候选进行人工检查。自动规则不会预填人工结论，人工确认以独立
sidecar 保存，避免改写已有 frontier 和 provenance hash。

### 4.3 两条搜索路线

**旧 checkpoint 回溯。** 重新用 Phase 5E prompt 和语义门测试原模型、core13 及已有物理 checkpoint，判断旧
模型是否只是思考轨迹分叉，而最终任务语义仍然正确。

**从原模型 fresh search。** 不沿 core13 继续扩展，而是从未剪枝模型对 gold caption 重新计算 saliency，再运行
静态排序和 learned gate。这样避免把为“严格思考轨迹一致”优化出的路径误当成语义任务的最优路径。

### 4.4 静态排序与 learned gate

静态阶段从 gold-caption CE 重新计算 activation、contribution 和 Taylor saliency，并按全局低重要度构造严格
预算 mask。

learned 阶段使用 exact-budget hard top-k gate 和 straight-through estimator。固定预算下的目标函数为：

```text
L = gold_caption_CE + reference_prefix_KL
```

由于预算已经精确固定，不再额外加入 sparsity penalty。正式预算默认运行 200 steps、3 个随机 restart。优化完成
后固化 hard mask，再单独运行 teacher-forcing 代理和自由生成。

### 4.5 物理验证

最大干净通过 mask 不能停留在 hook/gate 仿真阶段，必须：

1. 将 mask 转换为 24 层非均匀 FFN widths；
2. 同步删除 gate/up/down 投影关联权重；
3. 检查逐层宽度、删除数、参数量和 mask hash；
4. 对物理模型重新自由生成；
5. 保存 checkpoint；
6. 在独立进程重新加载；
7. 再次检查参数量、caption、EOS 和语义结果。

## 五、实验结果

### 5.1 改变判据后，旧物理下界从 13 提升至 1,000

原模型、core13、physical-250、physical-750 和 physical-1000 在新的短 caption 语义门下全部通过。此前的
core13 只是“前 512 思考 tokens 严格一致”的边界，并不是任务语义边界。

这一回溯结果直接证明：**思考轨迹发生分叉不能被等同于最终任务失败。** Phase 5E 使用任务语义作为主判据是
必要的。

### 5.2 fresh 静态搜索得到 48,000-neuron 物理下界

基于 gold Taylor saliency 的 fresh 静态排序在 48,000 删除处生成：

> A train with a few bikes in the back.

该候选通过自动语义检查，并完成物理保存/重载，参数量降至 `705,529,920`。静态 56,000 候选发生重复
和截断，说明单纯静态排序已经接近其当前候选族的能力边缘。

静态结果还表现出明显非单调性：40,000 候选描述成 “train or bus” 而失败，48,000 候选反而通过。因此预算
边界不能被当作严格数学二分。

### 5.3 learned gate 将物理下界提升到 62,950

learned gate 从 fresh gold-Taylor 初始化出发，逐步探测并缩窄：

| 删除预算 | 干净通过 restart | 主要观察 |
| ---: | ---: | --- |
| 50,000 | 3/3 | 全部正常生成 gold caption |
| 52,000 | 2/3 | 一个候选在正确句后重复角色词并截断 |
| 54,000 | 0/3 | 重复句或未正常结束 |
| 56,000 | 2/3 | 两个候选正常生成 gold caption |
| 58,000 | 3/3 | 全部正常生成 gold caption |
| 60,000 | 3/3 | 全部正常生成 gold caption |
| 62,000 | 1/3 | 一个干净通过，其余为无 EOS或空答案 |
| 62,500 | 1/3 | 一个干净通过 |
| 62,750 | 1/3 | 一个干净通过 |
| 62,875 | 2/3 | 两个干净通过 |
| 62,950 | 1/3 | 一个干净通过 |
| 63,000 | 0/3 | 无干净通过；一个自动 pass 含角色词垃圾 |
| 64,000 | 0/3 | 全部重复并触及 token 上限 |

成功率并不随删除预算单调。例如 54,000 没有干净通过，而 58,000 和 60,000 均为 3/3。这反映的是独立 mask、
优化轨迹和生成动力学的差异，而不是“删除越多一定越差”的简单曲线。

最终用 62,950 和 63,000 将经验区间缩小至 50 neurons。62,950 最优候选的 hook 指标为：

- gold mean NLL：`0.343505`；
- delta NLL：`-1.335173`；
- 自由 caption：`A group of bicycles on a subway train.`；
- 正常 EOS：是；
- 自动语义：pass。

NLL 显著下降不意味着模型通用能力提高。gate 就是针对该 gold caption 优化，因此它只说明目标 caption 的
条件概率提高，是任务特化的直接结果。

### 5.4 62,950-neuron 候选通过物理保存与独立重载

物理模型的 24 层 FFN 剩余宽度为 740–1,055，总计剩余 23,066 neurons。物理验证结果为：

| 指标 | 原模型 | 62,950-neuron 物理模型 |
| --- | ---: | ---: |
| FFN neurons | 86,016 | 23,066 |
| 模型参数量 | 852,985,920 | 659,603,520 |
| gold mean NLL | 1.678713 | 0.343445 |
| 单次生成峰值显存 | 约 1.800 GB | 约 1.414 GB |
| 最终语义 | pass | pass |

独立重载后的 caption 仍为：

> A group of bicycles on a subway train.

该输出已经人工确认通过。物理模型删除 `73.18%` 的 FFN neurons 和 `22.67%` 的模型参数，是 Phase 5E 当前
主结果。

峰值显存的单次观测下降约 21.45%，但生成长度不同，且没有进行多轮预热与计时统计，因此当前不能把该数字
表述为稳定的推理加速或显存收益结论。

## 六、关键分析与解释

### 6.1 语义边界远大于严格轨迹边界

严格轨迹边界为 13，而人工确认的任务语义下界为 62,950，相差约 4,842 倍。它们回答的是两个不同问题：

- 13 衡量“能否复制原模型前 512 个思考 tokens”；
- 62,950 衡量“能否完成这一个固定图像描述任务”。

结果说明内部生成轨迹包含大量对该单一最终语义并非必需的自由度。不能用思考 token 的早期分叉直接判断任务
功能已经丢失。

### 6.2 teacher forcing 是搜索代理，不是最终裁判

多个高预算候选具有很低的 gold NLL，却在自由生成时出现：

- 正确句子后重复 `assistant/user`；
- 重复同一句 caption；
- 文本看似正确但没有 EOS；
- 空最终答案；
- 达到 64-token 上限。

因此，仅报告 NLL 或 KL 会系统性高估模型的真实生成质量。Phase 5E 的最终通过必须同时满足自由生成语义、
停止状态和人工检查。

### 6.3 learned mask 显著优于静态 mask，但稳定性在边界附近下降

静态排序的物理下界为 48,000，learned gate 将其提高到 62,950，增加 14,950 neurons。与此同时，62k 以上
通常只有 1/3 或 2/3 restart 成功，说明接近边界时 mask 对初始化和优化轨迹敏感。

这支持使用多 restart，但也意味着 62,950 更适合解释为“已找到的最好构造”，而不是对某一固定预算成功概率
的稳定估计。

## 七、可信度、限制与不能过度解读的部分

### 7.1 结果可信的原因

- 输入图像、prompt、模板和生成参数已经冻结；
- generation 是确定性的，随机性只来自 gate 优化；
- 每个正式预算使用 3 个 restart；
- proxy 与最终语义判据严格分离；
- 保存了所有失败 caption 和停止状态，而不是只保留成功案例；
- 最大候选经过真实结构删除、参数计数、hash、保存和独立重载；
- 最终 caption 经过人工确认。

### 7.2 主要限制

1. **只有一张图。** 结果衡量单样本任务特化，不代表通用视觉描述能力。
2. **gold caption 参与优化。** NLL 下降是预期的目标特化现象，不能推断整体模型能力增强。
3. **63,000 不是理论上界。** 目前只运行 3 个 restart，其他初始化、损失权重或搜索算法仍可能找到解。
4. **自动语义门仍有假阳性。** 63,000 的一个候选包含角色词垃圾但被核心规则标为 pass，必须依赖完整输出
   审查和人工确认。
5. **缺少独立 prompt 验证。** 当前物理模型尚未用语义相同但措辞不同的 prompt 测试，无法区分视觉语义保持
   与 prompt-caption 模板记忆。
6. **原模型答案等价曲线尚未系统完成。** 当前报告完整覆盖严格轨迹等价和人工语义保持，但没有单独搜索“与
   原模型最终 caption 逐字一致”的边界。
7. **性能收益尚未规范 benchmark。** 当前只记录单次耗时和显存，生成长度不同，不足以报告稳定加速比。

## 八、结论

Phase 5E 完成了从“思考轨迹严格一致”到“任务语义保持”的评价转型，并建立了完整、可重复的单样本物理剪枝
闭环。

最核心的实验结论是：

> 对固定 COCO 图像，在当前 exact-budget learned-gate 搜索族下，Qwen3.5-0.8B 在真实删除 62,950 个 FFN
> neurons 后，仍能从原始图像和 prompt 确定性生成正确 caption。该模型删除了 73.18% 的 FFN neurons 和
> 22.67% 的总参数，并通过保存、独立重载、自动语义门和人工确认。

同时，63,000 neurons 的三个 restart 没有获得干净输出，因此当前经验区间为
`62,950 pass / 63,000 no-clean-pass`。这是一条搜索算法相关的构造性下界与上侧观测，不能声称全局最优。

## 九、建议的下一阶段

优先级从高到低：

1. 为最终物理模型增加 3–5 个语义等价、措辞不同的 held-out prompts，判断是否存在 prompt/caption 模板记忆；
2. 在 62,950–63,000 区间增加 restart 或替换初始化，评估边界对搜索预算的敏感性；
3. 将语义门升级为 v3，自动拒绝重复句、角色词垃圾和 EOS 前异常尾巴；
4. 在多张结构相似和不同场景图像上重复 Phase 5E，形成单样本极限的分布；
5. 系统补齐“原模型最终答案等价”曲线，形成严格轨迹、答案等价、人工语义三条完整边界；
6. 对原模型和物理模型进行统一输出长度、充分预热、多轮重复的 latency/VRAM benchmark。

## 十、汇报时建议使用的表述

建议表述：

> 我们没有声称找到了全局最小子网络，而是在固定单图任务上，构造并真实物理验证了一个可工作的极端稀疏
> FFN。结果表明，严格思考轨迹需要保留的结构与最终任务语义需要保留的结构存在几个数量级的差异。

避免表述：

- “模型最多只能删除 62,950 个 neurons”；
- “删除 73.18% FFN 后模型整体能力不变”；
- “剪枝后 NLL 更低，所以模型更强”；
- “63,000 已被证明不可能”。

## 附：主要可审计产物

- Phase 5E 计划：`docs/my-exper/typing neuron/phase5e_semantic_limit_plan.md`
- v2 旧 checkpoint 回溯：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_retrospective_caption_prompt_v2.json`
- fresh 静态 frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_static_gold_v2/static_frontier.json`
- 50k–56k learned frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_boundary_v2/learned_frontier.json`
- 58k–64k learned frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_upper_v2/learned_frontier.json`
- 62,950 learned frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_refine_62950_v2/learned_frontier.json`
- 62,950 物理筛查：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0/physical_screen_64.json`
- 62,950 独立重载：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0_reload.json`
- 62,950 人工确认：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0/human_confirmation.json`
