<!--
Copyright 2026 the LlamaFactory team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Visual-Anchored Rank-Margin Activation Alignment

本文总结 VQA-Med 实验中 Visual-Anchored Rank-Margin activation alignment 的实现、loss 优化方向、与旧 soft-IoU 方法的本质区别，以及已有实验中支持该方法更有效的证据。

## 1. 方法目标

对每个 Transformer MLP 层，代码 hook `down_proj` 的输入：

```text
A = silu(gate_proj(x)) * up_proj(x)
```

`A` 的最后一个维度对应 FFN 中间神经元。方法希望在语义层中，让 question token 更倾向于使用 visual token 已经高激活的神经元：

```text
visual token 选出的 top-k neuron
             ↓ 作为固定 anchor
提高 question activation 在这些 neuron 上的相对排名
```

这里的“对齐”不是要求 visual 和 text 的完整激活向量相等，也不是强制二者激活值重合，而是要求 text 侧优先使用 visual 侧最重要的 neuron 子集。

## 2. 实现流程

核心实现位于：

```text
src/llamafactory/train/vulcan/activation_align.py
```

### 2.1 Token 分区

代码根据 `input_ids`、`labels` 和 `attention_mask` 构造三个区域：

```text
visual_mask   = input_ids == image_token_id
answer_mask   = labels != IGNORE_INDEX 且不是 visual token
question_mask = labels == IGNORE_INDEX 且不是 visual token
```

因此当前 `question_mask` 包含用户问题、固定 instruction prompt，以及其他未参与 SFT label loss 的非视觉有效 token。固定 prompt 的影响是当前方法的一个待消融因素。

最佳实验采用 question-only：

```yaml
align_question_weight: 1.0
align_answer_weight: 0.0
```

这意味着 alignment loss 只直接优化 question activation，不直接约束答案 token。

### 2.2 激活池化

每个激活先取绝对值，再在对应 token 区域进行 mean pooling：

```text
p_v[l, i] = mean_visual_tokens |A[l, token, i]|
p_q[l, i] = mean_question_tokens |A[l, token, i]|
```

其中 `l` 表示 MLP 层，`i` 表示中间神经元。

### 2.3 Visual hard top-k anchor

`align_quantile=0.8` 时，代码严格选择 visual pooled activation 最高的 20% neuron：

```text
k = ceil((1 - quantile) * intermediate_size)
V_top = TopK(p_v, k)
```

top-k 索引通过 detached visual activation 计算。因此：

- visual top20 只负责定义当前层的 anchor 集合；
- anchor 选择本身不接受梯度；
- alignment loss 的直接梯度流向 text/question activation；
- 模型参数是共享的，所以参数更新后 visual activation 仍可能间接受到影响，但 loss 不会直接把 visual 向 text 拉动。

这正是“Visual-Anchored”的含义。

### 2.4 Text salience 归一化

question pooled activation 用 detached mean 做尺度归一化：

```text
q_norm[i] = p_q[i] / stop_grad(mean(p_q))
```

该操作降低不同层激活绝对尺度对固定 margin 的影响，同时不会通过分母引入额外的全局耦合梯度。

### 2.5 Positive 与 hard negative

正样本分数是 question 在 visual anchor neuron 上的平均 salience：

```text
top_score = mean(q_norm[i]), i in V_top
```

负样本不是全部非 anchor neuron，而是其中最强的同等数量 neuron：

```text
H_neg = TopK(q_norm[i], k), i not in V_top
other_score = mean(q_norm[i]), i in H_neg
```

这一步很关键。早期实现使用所有 non-anchor neuron 的平均值，由于大量低激活 neuron 拉低了 `other_score`，question 侧很容易天然满足 margin，导致：

```text
align_loss = 0
```

改用 hard negative 后，loss 比较的是“visual top20”与“text 最有竞争力的另一组 top20”，优化目标才真正对应 top-k 排名竞争。

### 2.6 Rank-Margin Loss

单层、单个 text 区域的定义为：

```text
gap = top_score - other_score
L_rank = ReLU(margin - gap)
```

当前 `margin=0.2`。其行为是：

- 当 `gap < 0.2`，loss 有梯度；
- 梯度提高 visual-anchor neuron 上的 question salience；
- 同时压低当前最强 hard-negative neuron 的 salience；
- 当 `gap >= 0.2`，该层目标已经满足，loss 变为 0，不再继续强推。

因此，margin 不是要求两个模态激活值相等，而是要求 visual anchor 在 question 侧至少取得一定的排名优势。

### 2.7 跨区域、跨层和总 loss

若同时启用 question 与 answer，则每层 loss 为：

```text
L_layer = w_q * L_question + w_a * L_answer
```

所有有效层取平均：

```text
L_align_raw = mean_layers(L_layer)
L_align = align_lambda * L_align_raw
```

最终训练目标为：

```text
L_total = L_SFT + L_align
```

当前最佳实验使用：

```yaml
align_loss_type: rank_margin
align_quantile: 0.8
align_margin: 0.2
align_lambda: 0.05
align_question_weight: 1.0
align_answer_weight: 0.0
align_layer_start_ratio: 0.5
align_layer_end_ratio: 0.95
```

对 24 层模型，这一范围对应 L12-L22，目的是避开浅层视觉/格式处理层以及最后 logits-adjacent 层。

注意：仓库中的 `examples/vulcan/qwen35_08b_vqa_med_cls_align_sft.yaml` 当前仍保留
`align_answer_weight: 0.2` 作为 question + answer 模板；本文最佳 question-only run 是训练时将该参数覆盖为
`0.0` 得到的结果。

## 3. Loss 优化方向与旧方法的区别

### 3.1 旧 soft-IoU

旧方法分别构造 visual/text soft mask：

```text
M_v = sigmoid((p_v - quantile(p_v)) / temperature)
M_t = sigmoid((p_t - quantile(p_t)) / temperature)

L_soft_iou = 1 - IoU(M_v, M_t)
```

它直接优化两个 soft mask 的整体相似度，是对称目标：

```text
visual ↔ text
```

visual 和 text 两侧都可以通过移动来降低 loss，因此可能发生共同漂移。它只关心 soft mask 数值是否接近，不明确规定哪一侧是参照，也不直接强调真实 top-k neuron 的排序。

### 3.2 Visual-Anchored Rank-Margin

新方法的方向是单向的：

```text
visual hard top-k --detach--> anchor
                             ↑
                  question 向 anchor 排名靠近
```

| 维度 | Soft-IoU | Visual-Anchored Rank-Margin |
| --- | --- | --- |
| 对齐方向 | visual/text 对称靠近 | text/question 单向靠近 visual anchor |
| anchor | 无 | visual hard top-k |
| top-k 定义 | sigmoid soft mask，可能较稠密 | 严格 hard top-k |
| 优化对象 | 整个 soft mask 的 IoU | anchor 与 hard negative 的相对排名 |
| 负样本 | 无显式负样本 | 最强 non-anchor top-k |
| 停止条件 | 持续提高整体 soft-IoU | `gap >= margin` 后停止强推 |
| 主要风险 | 两侧共同漂移、mask 密度影响目标 | hard top-k 离散切换、question prompt 污染 |
| hard IoU | 后期加入的诊断指标 | 仍是诊断指标，不直接参与 loss |

最核心的变化不是简单地把 IoU 换成另一个公式，而是把优化问题从：

```text
“让两个模糊集合整体相似”
```

改成：

```text
“固定 visual 的重要 neuron，让 question 在这些 neuron 上击败最强竞争者”
```

## 4. 日志指标如何解释

Rank-Margin 训练中应优先观察：

| 日志 | 含义 |
| --- | --- |
| `align_rank_question_top_score` | question 在 visual top-k neuron 上的平均归一化 salience |
| `align_rank_question_other_score` | question 在最强 non-anchor top-k 上的平均 salience |
| `align_rank_question_gap` | `top_score - other_score` |
| `align_rank_question_loss` | `ReLU(margin - gap)` |
| `align_raw_loss` | 所有激活层的未加 lambda 平均 loss |
| `align_loss` | `align_lambda * align_raw_loss` |
| `align_hard_topk_iou` | 当前 batch 中 visual/text 真实 top-k 集合 IoU，仅用于诊断 |
| `align_soft_iou` | 兼容旧实验的 soft-mask IoU，仅用于诊断 |

出现 `align_loss=0` 不一定是实现失效。如果对应 `gap >= margin`，表示当前 batch/层已经满足排序目标。只有在 gap 明显低于 margin 而 loss 仍为 0 时，才应怀疑实现或 mask。

这些日志量是在每层分别计算后再各自取平均，因此通常有：

```text
mean(ReLU(margin - gap_l)) != ReLU(margin - mean(gap_l))
```

不能仅用日志中的平均 gap 直接反推平均 loss；部分层可能已经满足 margin，另一些层仍有正 loss。

## 5. 支持方法有效的实验结果

### 5.1 外部任务指标

所有结果均在 VQA-Med 1501 条验证集上评估，其中 yes/no 为 188 条。总体 EM/F1 是主指标。

| 配置 | Exact Match | Token F1 | Yes/No Acc. |
| --- | ---: | ---: | ---: |
| no-align continuation, 200 steps | 76.48% | 78.59% | 82.98% |
| soft-IoU, `lambda=0.05` | 76.22% | 78.40% | 84.04% |
| soft-IoU, `lambda=0.2` | 75.75% | 78.20% | 88.83% |
| rank-hardneg, `answer_weight=0.2` | 76.95% | 79.04% | 87.23% |
| rank-hardneg, question-only, checkpoint 100 | 77.08% | 79.21% | 86.17% |
| rank-hardneg, question-only, 200 steps | **77.48%** | **79.57%** | **89.36%** |

question-only Rank-Margin 相对同训练步数 no-align：

```text
Exact Match: 76.48% -> 77.48%  (+1.00 pp)
Token F1:    78.59% -> 79.57%  (+0.98 pp)
Yes/No Acc.: 82.98% -> 89.36%  (+6.38 pp)
```

相较 `answer_weight=0.2`，question-only 的 EM、F1 和 yes/no accuracy 均更高。这支持“优先对齐 question grounding，而不直接约束答案决策表示”的设计。

### 5.2 Soft-IoU 强约束造成共同漂移

内部对照使用同一个 100 条 val 子集，比较 no-align 与不同 alignment 模型的 pooled activation 和 top20 neuron 集合。

| 指标（后半层平均） | soft-IoU `lambda=0.05` | soft-IoU `lambda=0.2` | Rank-Margin `lambda=0.05` |
| --- | ---: | ---: | ---: |
| visual top20 set change | 11.0% | 40.9% | **10.4%** |
| text top20 set change | 24.6% | 64.4% | 29.3% |
| visual cosine | 0.996 | 0.920 | **0.996** |
| text cosine | 0.989 | 0.788 | **0.985** |
| visual L2 drift | 0.81 | 3.60 | **0.70** |
| text L2 drift | 1.15 | 7.02 | **0.85** |
| parameter delta relative norm | 0.11% | 0.23% | 0.15% |
| within-model visual/text top20 IoU | 0.251 | 0.245 | **0.268** |

soft-IoU `lambda=0.2` 让 visual 和 text 都发生大幅重排，但 visual/text overlap 没有超过 no-align 的 0.245。Rank-Margin 则在 visual cosine 保持 0.996 的情况下，将 IoU 提高到 0.268。

这与代码的梯度方向一致：

- visual anchor 不直接接受 alignment 梯度；
- text top20 变化 29.3%，明显高于 visual 的 10.4%；
- visual 基本稳定，text 发生适度重组；
- 重组结果是跨模态 top20 overlap 提升，而不是两侧无方向漂移。

### 5.3 真实 top-k overlap 提升

同一内部对照中：

```text
no-align within-model top20 IoU: 0.245
Rank-Margin top20 IoU:           0.268
absolute improvement:            +0.023
```

旧 soft-IoU `lambda=0.05` 只提高到 0.251，即 `+0.006` 左右；强 soft-IoU `lambda=0.2` 则保持在 0.245。Rank-Margin 的 overlap 提升约为弱 soft-IoU 的 4 倍，同时没有强 soft-IoU 的大幅激活漂移。

逐层结果也显示提升并非由单层偶然造成：

| 层 | no-align IoU | Rank-Margin IoU | 差值 |
| --- | ---: | ---: | ---: |
| L12 | 0.207 | 0.231 | +0.024 |
| L13 | 0.198 | 0.230 | +0.032 |
| L14 | 0.300 | 0.320 | +0.020 |
| L15 | 0.228 | 0.245 | +0.017 |
| L16 | 0.175 | 0.200 | +0.025 |
| L17 | 0.183 | 0.215 | +0.032 |
| L18 | 0.231 | 0.262 | +0.031 |
| L19 | 0.230 | 0.261 | +0.031 |
| L20 | 0.234 | 0.255 | +0.021 |
| L21 | 0.242 | 0.268 | +0.026 |
| L22 | 0.298 | 0.297 | -0.001 |

主要训练层 L12-L21 均获得稳定提升，只有 L22 基本持平。

### 5.4 训练信号从无效变为有效

使用全部 non-anchor 均值作为 negative 时，question 侧常常天然满足目标，连续出现：

```text
align_loss = 0
```

改成 hard negative 后，代表性日志为：

```text
align_rank_question_loss = 0.03332
align_rank_question_gap = 0.3078
align_rank_question_top_score = 1.670
align_rank_question_other_score = 1.362

align_rank_answer_loss = 0.5189
align_rank_answer_gap = -0.3189
```

该日志证明实现能够区分：

- question 已较接近 visual anchor 的层；
- answer 明显偏离 visual anchor 的层；
- 已满足 margin 的层与仍需优化的层。

这种可解释的局部排序信号，是 soft-IoU 单一 overlap 标量所不具备的。

## 6. 当前结论

已有结果支持 Visual-Anchored Rank-Margin 比 soft-IoU 更符合当前 activation alignment 目标：

1. 优化方向明确：固定 visual top20，主要推动 question 侧。
2. 直接针对真实 top-k 排名，而不是稠密 soft mask 的整体相似度。
3. hard negative 保证优化的是最有竞争力的 non-anchor neuron。
4. margin 达成后停止施压，减少无必要的内部结构破坏。
5. within-model visual/text top20 IoU 从 0.245 提高到 0.268。
6. visual cosine 保持 0.996，说明 visual 表征漂移被明显压住。
7. 同训练步数下，overall EM/F1 相对 no-align 分别提高 1.00 和 0.98 pp。

因此，目前最合理的描述不是“Rank-Margin 让两个模态激活变得相同”，而是：

```text
Rank-Margin 以 visual top-k neuron 为固定参照，
重新排序 question 侧的 neuron salience，
使 question 更频繁地复用 visual 侧的重要 neuron，
并在较小内部扰动下提高真实 top-k overlap 和任务指标。
```

## 7. 结论边界与后续验证

当前证据仍有以下边界：

1. question mask 包含固定 prompt，尚未证明增益来自实际问题词还是公共 instruction。
2. 现有主要结果来自单个 seed，需要重复实验确认方差。
3. Rank-Margin final 的 EM/F1 仍低于原始 baseline checkpoint-1200 的 78.61%/80.59%；它证明优于同训练步数 no-align continuation，但尚未超过最佳 baseline 选点。
4. hard top-k 的索引选择是离散的，边界 neuron 可能在 batch 间切换。
5. 当前内部分析基于 100 条 val 子集，应在完整验证集或更多随机子集上复核。

建议后续依次进行：

1. 使用不同 seed 重复 question-only Rank-Margin。
2. 从 question mask 中排除固定 instruction 和 special token。
3. 分别报告 modality、plane、organ 的 EM/F1。
4. 用固定的 no-align visual anchor 计算方向性指标：

```text
IoU(text_align, visual_noalign) - IoU(text_noalign, visual_noalign)
```

5. 基于 Rank-Margin 模型重新生成 cluster_idx，再验证 collapse 和 50% 物理剪枝是否获得额外收益。

## 8. 代码与测试位置

| 内容 | 文件 |
| --- | --- |
| visual/question/answer mask | `src/llamafactory/train/vulcan/activation_align.py` |
| hard top-k anchor | `src/llamafactory/train/vulcan/activation_align.py::_hard_topk_mask` |
| Rank-Margin 核心公式 | `src/llamafactory/train/vulcan/activation_align.py::_rank_margin_loss` |
| 跨层聚合与日志 | `src/llamafactory/train/vulcan/activation_align.py::_compute_rank_margin_alignment_loss` |
| 与 SFT loss 相加 | `src/llamafactory/train/sft/trainer.py::_add_align_loss` |
| 参数定义与校验 | `src/llamafactory/hparams/finetuning_args.py` |
| VQA-Med 配置 | `examples/vulcan/qwen35_08b_vqa_med_cls_align_sft.yaml` |
| anchor、梯度和层范围测试 | `tests/train/vulcan/test_activation_align.py` |
