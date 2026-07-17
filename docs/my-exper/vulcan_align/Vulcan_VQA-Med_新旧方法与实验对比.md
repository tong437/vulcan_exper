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

# Vulcan 在 VQA-Med 上的新旧方法与实验对比

本文对比旧 Vulcan 与新 multimodal Vulcan 在 VQA-Med modality、plane、organ system 三分类任务上的方法差异和实验结果。基础模型均为 Qwen3.5-0.8B，主要实验均从最佳 baseline checkpoint-1200 开始。

## 1. 核心结论

新 multimodal Vulcan 没有修改 Vulcan 的核心 collapse 和物理剪枝公式，正式方法的主要变化集中在 cluster 构造阶段：旧方法主要依据全局平均激活和 FFN 参数距离聚类；新方法进一步区分 image、question、prediction 三类 token，并在 modality、plane、organ 三个任务间进行宏平均，再联合参数方向与多模态激活签名进行聚类。

因此，主要受控比较为：

```text
旧 Vulcan：旧 cluster + 原始 collapse + 原始物理剪枝
新 Vulcan：multimodal cluster + 原始 collapse + 原始物理剪枝
```

在每层统一保留 50% FFN 神经元的条件下，新方法保持了近零损失物理剪枝，并改善了剪枝模型的总体准确率和 yes/no 稳定性。

## 2. 处理对象

新旧方法都只处理 Qwen3.5 language decoder 中的 SwiGLU FFN：

```text
FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

每个 FFN 中间神经元对应：

- `up_proj` 的一行；
- `gate_proj` 的一行；
- `down_proj` 的一列。

vision tower 和 multimodal projector 不参与本轮结构剪枝。当前预算仍为所有 decoder 层统一 `keep_ratio=0.5`，尚未使用逐层自适应 ratio。

## 3. Cluster 阶段对比

### 3.1 数据采样

| 项目 | 旧 Vulcan | 新 multimodal Vulcan |
| --- | --- | --- |
| 数据来源 | 合并后的 `vqa_train_cls` | `vqa_train_modality`、`vqa_train_plane`、`vqa_train_organ` |
| 采样方式 | 混合数据 shuffle 后采样 | 每个任务独立采样 200 batches |
| 总采样量 | 约 600 batches | 3 × 200，共 600 batches |
| 任务权重 | 随实际样本/token 数隐式决定 | 三个任务等权宏平均 |

旧方法即使开启 shuffle，样本更多、序列更长或图像 token 更多的部分仍可能在总体激活统计中占更高权重。新方法将三个医学 VQA 子任务分开统计，避免某一任务主导 cluster。

### 3.2 Token 激活定义

旧 Vulcan 在 `down_proj` 输入处收集绝对激活，并将所有 token 混合统计：

```text
a_old(l,i) = mean_all_tokens |h(l,i)|
```

新方法将 token 分成三个区域：

- `image`：图像 token；
- `question`：问题 prompt 中非图像、非答案、非 special token；
- `prediction`：负责预测下一个答案 token 的 hidden-state 位置。

对于 causal LM，prediction 位置需要进行一位偏移：位置 `t` 的 hidden state 用于预测位置 `t+1` 的答案 token。

### 3.3 三级平均

新方法采用：

```text
token 平均 -> sample 平均 -> category 平均
```

形式化表示为：

```text
A = 1/|C| Σ_category [
        1/|S_c| Σ_sample [
            1/|T_s| Σ_token |activation|
        ]
    ]
```

三个层级分别解决：

1. token 平均：图像 token 更多或问题更长的样本不会获得更高权重；
2. sample 平均：每条样本在所属任务中等权；
3. category 平均：modality、plane、organ 三个任务等权。

旧方法近似把所有 category、sample、token 混在一起平均，因此不具备上述平衡机制。

### 3.4 Anchor 选择

旧方法在每个 cluster 内选择全局平均激活最大的神经元作为 anchor：

```text
anchor_old = argmax activation_all_tokens(i)
```

新方法为每个神经元构造三维激活签名：

```text
s(i) = [image(i), question(i), prediction(i)]
```

默认 anchor 分数为：

```text
score(i) = 0.4 * image(i) + 0.4 * question(i) + 0.2 * prediction(i)
```

此外，激活签名乘以对应 `down_proj` 列的 L2 范数：

```text
s_contribution(i) = s(i) * ||down_proj[:, i]||_2
```

这使 anchor 不仅需要激活较强，还需要对 FFN 最终输出具有较强贡献。

### 3.5 聚类距离

旧方法的聚类向量为：

```text
v_old(i) = concat(up_proj[i], gate_proj[i])
```

新方法先分别对 `up_proj` 和 `gate_proj` 的每一行进行 L2 归一化，再加入标准化后的多模态激活签名：

```text
v_new(i) = concat(
    normalize(up_proj[i]),
    normalize(gate_proj[i]),
    0.25 * zscore(s_contribution(i))
)
```

因此，新方法要求同簇神经元同时满足：

- FFN 参数方向相似；
- image token 响应相似；
- question token 响应相似；
- prediction position 响应相似。

旧方法主要回答“哪些神经元参数接近”，新方法进一步回答“哪些神经元参数接近且在多模态 VQA 中承担相似功能”。

### 3.6 Greedy Match

新旧方法仍使用相同的 greedy match 框架：

1. 计算神经元两两距离；
2. 从未分配神经元中选择 seed；
3. 将距离 seed 最近的神经元组成 cluster；
4. 在 cluster 内选择贡献分数最高的神经元作为 anchor；
5. 重复直到得到目标 cluster 数。

新方法改变的是距离特征和 anchor 分数，而不是整体 greedy 聚类流程。

## 4. Vulcan 训练阶段对比

为保证实验可归因，新 multimodal Vulcan 的正式训练恢复了旧 Vulcan 已验证的 collapse 动力学：

| 参数 | 旧 Vulcan 主实验 | 新 multimodal Vulcan |
| --- | ---: | ---: |
| 起始模型 | checkpoint-1200 | checkpoint-1200 |
| keep ratio | 0.5 | 0.5 |
| collapse reduction | `legacy` | `legacy` |
| 主学习率 | `1e-5` | `1e-5` |
| lambda 学习率（rho） | `-0.1` | `-0.1` |
| lambda 初值 | 0 | 0 |
| lambda 更新 | plain SGD 梯度上升 | plain SGD 梯度上升 |
| weight proxy | 关闭 | 关闭 |
| gradient accumulation | 4 | 4 |
| max steps | 2188 | 2188 |
| warmup/ramp | 无 | 无 |

Collapse loss 仍约束同一 cluster 内的 `up_proj` 和 `gate_proj` 行向 anchor 收缩：

```text
L = L_SFT + L_collapse

L_collapse = Σ_layers 1/K_l Σ_clusters [
    lambda1 * Σ ||w_i - w_anchor||_1
  + lambda2 * Σ ||w_i - w_anchor||_2^2
]
```

其中：

```text
w_i = concat(up_proj[i], gate_proj[i])
```

曾尝试的 `normalized collapse + linear ramp` 会同时对权重维度和有效层数取平均，使约束相比原 Vulcan 缩小数万倍。该尝试已放弃，不属于正式的新 multimodal Vulcan 方法。

## 5. 物理剪枝阶段对比

新旧方法使用相同的物理剪枝规则。对于每个 cluster：

```text
new_up[k]       = old_up[anchor]
new_gate[k]     = old_gate[anchor]
new_down[:, k]  = Σ old_down[:, member]
```

Collapse 训练使同簇成员的 `up_proj/gate_proj` 接近 anchor，因此：

```text
Σ down_i * f_i(x) ≈ (Σ down_i) * f_anchor(x)
```

剪枝时只保留 anchor 的输入侧权重，并把所有成员的 `down_proj` 列求和到新的 anchor 列，以近似保持 FFN 输出。

代码还支持逐层不同的目标宽度：每层宽度写入 `vulcan_intermediate_sizes`，Qwen3.5 自定义 loader 在权重加载前逐层重建 MLP。但当前正式结果仍是所有层统一保留 50%。

## 6. 新旧剪枝模型结果

| 剪枝模型 | Exact | Token F1 | Yes/No |
| --- | ---: | ---: | ---: |
| 旧 Vulcan | 74.62% | 77.10% | 75.00% |
| 新 multimodal Vulcan | **75.55%** | **78.21%** | **84.57%** |
| 提升 | **+0.93 pp** | **+1.11 pp** | **+9.57 pp** |

新 multimodal Vulcan 的 checkpoint-1800 在剪枝前 exact accuracy 为 75.68%，物理剪枝后为 75.55%，仅下降 0.13 pp；token F1 和 yes/no accuracy 没有下降。该结果表明新 cluster 仍能支持近零损失物理剪枝，同时比旧 cluster 更好地保持多模态任务能力。

## 7. VQA-Med 已有实验横向对比

| 实验 | Exact Accuracy | 相对 Baseline |
| --- | ---: | ---: |
| Baseline checkpoint-1200 | **78.61%** | 基准 |
| Vulcan 未剪枝 | 74.62% | -3.99 pp |
| Vulcan 剪枝后（50%） | 74.62% | -3.99 pp |
| Align（lambda=0.05，lr=3e-6） | 77.55% | -1.07 pp |

说明：

- 表中的 `Vulcan` 指旧 Vulcan 主实验，用于展示原始 collapse 方法在剪枝前后的性能保持；
- `Align` 是从 baseline checkpoint-1200 开始的独立 activation-alignment 实验，不是新 multimodal cluster 的组成部分，也没有在此表中报告物理剪枝结果；
- 新 multimodal Vulcan 的主要结果单独列于第 6 节，其剪枝后 exact accuracy 为 75.55%。

## 8. 当前判断

1. 原 Vulcan 的 collapse 与 `down_proj` 列合并机制是有效的，能够让 50% FFN 物理剪枝接近无额外损失。
2. 旧方法的主要问题不是“剪不动”，而是 cluster 与多模态任务结构不完全匹配，collapse 训练会损伤原任务能力。
3. 新 multimodal cluster 通过 token 分区、三级平均、输出贡献加权和功能签名聚类，提高了被合并神经元的多模态功能一致性。
4. 新方法相较旧 Vulcan 剪枝模型提升 0.93 pp exact accuracy，并显著改善 yes/no accuracy。
5. 新剪枝模型仍比 baseline 低 3.06 pp，后续重点应是减少 collapse SFT 的任务性能代价，而不是继续证明 50% FFN 是否能够物理剪枝。
6. 后续仍需补充 modality、plane、organ 分类型结果、同 cluster 的 baseline direct-prune 对照，以及剪枝前后的簇内 redundancy 统计。
