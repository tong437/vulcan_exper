# Vulcan Activation Alignment 问题分析

## 1. Lambda 增长快的原因

### 优化机制

Lambda 的梯度由 collapse loss 反向传播得到：

```
L_total = L_SFT + λ1 * L1 + λ2 * L2

dL_total/dλ1 = L1 = Σ|W_i - W_anchor|_1  (≥ 0)
dL_total/dλ2 = L2 = Σ|W_i - W_anchor|_2²  (≥ 0)
```

参数更新使用 **独立的 learning rate**：

```python
# optimizer param group
dict(params=[λ1, λ2], lr=collapse_lambda_lr=-0.01)
λ_new = λ_old - lr * gradient
      = λ_old - (-0.01) * positive_gradient
      = λ_old + 0.01 * positive_gradient
      → λ 持续增大
```

### 量化估算

以 Qwen3.5-0.8B 为例：

| 参数 | 值 |
|------|-----|
| `intermediate_size` | 3584 |
| `cluster_size` | 2（聚类数 = 1792） |
| `num_layers` | 24 |

每层每次前向传播：

```
diff_w.abs().sum() ≈ 1792 neurons × 2 (up+gate) × 1.0 (粗估权重差异) ≈ 3584
diff_w.pow(2).sum() ≈ 1792 × 2 × 0.5 ≈ 1792

每层 L1 + L2 ≈ 5376
24层总计 ≈ 5376 × 24 ≈ 129,024
```

每个 step 的 lambda 增量：

```
Δλ = |lr| × (L1 + L2)
   = 0.01 × 129,024
   ≈ 1290
```

经过 10 steps（0.1 epoch 的很小一部分）：

```
λ ≈ 0 + 10 × 1290 ≈ 12,900
```

这就是为什么 lambda 很快就能到 13。

### 负 lr 的问题

`lr = -0.01` 设计意图是让 lambda 增大，从而增强 collapse 约束。但实际效果是：

1. **梯度始终为正**（L1、L2 都是正值）
2. **负 lr × 正梯度 = 负的更新量 = 加到 λ 上**
3. **λ 越大 → collapse loss 越大 → 梯度越大 → λ 增长更快**

形成正反馈，失控。

---

## 2. Collapse Loss 大的原因

### 量级估算

```
collapse_loss = λ1 × L1 + λ2 × L2
              ≈ λ × (diff_w.abs().sum() + diff_w.pow(2).sum())

当 λ = 13, diff ≈ 129,024 per step（见上）
collapse_loss ≈ 13 × 129,024 ≈ 1,677,312
```

每层贡献：

| λ | 每层 L1+L2 | 每层 collapse_loss | 24层总计 |
|---|-----------|------------------|---------|
| 0.1 | 5,376 | 537.6 | 12,902 |
| 1.0 | 5,376 | 5,376 | 129,024 |
| 3.0 | 5,376 | 16,128 | 387,072 |
| 13.0 | 5,376 | 69,888 | 1,677,312 |

### 与 SFT Loss 的对比

```
SFT loss (per token CE) ≈ 6-8
per token CE = 1,000,000 / (940 × 144) ≈ 7.4

total SFT loss ≈ 7.4 × 144 ≈ 1,066 per batch
```

| λ | collapse_loss | SFT_loss | collapse/SFT 比值 |
|---|--------------|---------|-----------------|
| 0.1 | 12,902 | ~1,066 | **12x** |
| 1.0 | 129,024 | ~1,066 | **121x** |
| 13.0 | 1,677,312 | ~1,066 | **1573x** |

**Collapse loss 在 λ=13 时是 SFT loss 的 1573 倍**，完全主导训练。

### Collapse Loss 的实际效果

Collapse loss 驱动权重趋同：

```
权重趋同 → up_proj 和 gate_proj 的神经元变得相似
         → MLP 中间激活趋于相同
         → down_proj 输出趋同
         → logit 分布趋同（接近 uniform）
         → CE loss 上升
         → total loss 爆炸性增长
```

Loss 从 step 10 的 11,545 增长到 step 90 的 890,189（77 倍），就是这个过程。

---

## 3. Align Loss 小的原因

### 实测数据

```
set_input_ids shape=[1, 144]  image_tokens=120  text_tokens=24

total_loss = 0.085449
final_loss = 0.042725 (λ=0.5)
lambda     = 0.5

n_layers   = 24
visual_tokens = 120
text_tokens   = 24

layer0 统计：
  tau          = 0.0086
  soft_v_mean  = 0.4771
  soft_t_mean  = 0.4776
  |diff|       = 0.0524
```

### 量化分析

**对齐损失计算**：

```python
tau = quantile(mean_t.detach(), q=0.8)  # ~0.0086
soft_v = sigmoid((mean_v - tau) / 0.1)  # temperature=0.1
soft_t = sigmoid((mean_t - tau) / 0.1)

layer_loss = |soft_v - soft_t|.mean()
           ≈ 0.0524

total_loss = mean(layer_loss across 24 layers)
           ≈ 0.085449

final_loss = λ × total_loss
           = 0.5 × 0.085449
           = 0.042725
```

**与主任务 loss 的对比**：

```
align_loss     = 0.0427
main_loss      = 11,545.80
比例           = 0.00037%
```

**与 collapse loss 的对比**：

```
align_loss     = 0.0427
collapse_loss  ≈ 1,677,312 (λ=13)
比例           = 0.0000025%
```

---

## 4. 为什么 Text Tokens 只有 24 个

### Token 分解

每个样本 144 个 tokens：

| 类型 | token 数量 | 占比 |
|------|-----------|------|
| image token (id=248056) | 120 | 83.3% |
| text tokens | 24 | 16.7% |

Text tokens 的 24 个组成：

| 内容 | token 数 | 说明 |
|------|---------|------|
| 问题 "are regions of the brain infarcted?" | 8 | 语义查询 |
| 答案 "yes" 或 "no" | 1 | 分类标签 |
| `<|im_start|>` / `<|im_end|>` | 4 | 模板标记 |
| user / assistant role | 2 | 角色标记 |
| 换行符 `\n` | 9 | 格式噪声 |

**问题**：text 侧内部就不是同质的。把 question tokens + answer token + template tokens 混在一起做 mean-pool，得到 `mean_t`。这个 `mean_t` 的语义是模糊的。

---

## 5. 为什么 Align Loss 几乎没有影响

### 原因一：量级太小

```
align_loss     = 0.04
collapse_loss  = 1,677,312 (λ=13)
main_loss      = 11,545

align_loss / main_loss      = 0.00037%
align_loss / collapse_loss  = 0.0000025%
```

**梯度被 main_loss 淹没**。即使 align_loss 有梯度，其方向在 total gradient 中可以忽略。

### 原因二：text 侧异质，目标模糊

```
mean_t = mean(
    question_tokens激活 +
    answer_token激活 +
    template_tokens激活
)
```

让这个混合向量和 `mean_v` 对齐，目标本身不明确：

- Question 的激活方向 ≠ Answer 的激活方向
- 模板 tokens 的激活无意义
- 强行让三者混合后向 visual 靠拢，可能干扰模型区分问题/答案的能力

### 原因三：视觉主导

```
image tokens : text tokens = 120 : 24 = 5 : 1
```

`mean_v` 由 120 个 image tokens 的激活平均得到。每个 token 的权重是均等的，5:1 的数量优势让 `mean_v` 的方向完全由视觉特征主导。

---

## 6. 有什么影响

### Collapse + Alignment 组合

| 影响 | 描述 |
|------|------|
| **Collapse loss 主导训练** | λ=13 时 collapse 是 main_loss 的 1573 倍，模型主要在学习"让权重趋同"，而不是"做 VQA 任务" |
| **Alignment loss 形同虚设** | 0.04 vs 1,677,312 的比例，让 align 约束几乎不起作用 |
| **最终效果：负面影响** | vulcan-align-collapse (66.53%) < vulcan-sft (68.13%) |

### 为什么加了 alignment 更差

1. **Alignment 提供了一个混乱的额外目标** — 让异质的 text 混合激活去对齐 visual，这个方向本身就是噪声
2. **Collapse 已经让权重趋同了** — alignment 继续强化这一点，但强化的是"让所有 text token 趋同"，而不是"让问题和答案区分清楚"
3. **两者目标冲突** — collapse 让所有神经元变得相似，alignment 试图让 visual 和 text 的激活模式对齐，但 collapse 已经破坏了原本的激活多样性

---

## 7. 建议

### 对于 Alignment

| 问题 | 建议 |
|------|------|
| Text 侧异质 | 默认只对 answer tokens 做 pool，即 `labels != IGNORE_INDEX`，不包含问题 token 和模板 token |
| Question 也有价值 | 新实现保留 `align_text_mode: question` 和 `align_text_mode: qa`，用于后续消融 |
| Align loss 太小 | `align_lambda` 先设到 10 左右，并记录 `align_raw_loss`、`align_soft_iou`、token 数和 mask 均值 |
| Top-K 目标 | 默认使用 `align_loss_type: soft_iou`，直接优化 visual/text soft top-k mask 的重叠度 |
| 模态尺度差异 | visual/text 分别计算自己的 quantile threshold，避免用单侧阈值导致 mask 偏置 |

新版实现要点：

```yaml
use_activation_align: true
align_lambda: 10.0
align_temperature: 0.05
align_quantile: 0.8
align_pool_type: mean
align_loss_type: soft_iou
align_text_mode: answer
```

`answer` 模式是第一版主实验，因为它只使用 SFT 真正监督的 assistant response token。`question` 模式会使用 prompt 侧非视觉 token，能测试“视觉证据是否应对齐问题语义”；但在当前数据管线里它仍可能包含模板、role 和换行 token，因此需要作为消融而不是默认主线。

### 对于 Collapse

| 问题 | 建议 |
|------|------|
| Lambda 失控增长 | 避免 v1 的 `weight_proxy=true` + 低主学习率组合，当前服务器实验中 lambda 到 `149` 后训练不稳定 |
| 负 lr | 当前 v3 主线保留 `collapse_lambda_lr: -1.0`，配合 plain SGD 对 lambda 做梯度上升 |
| Lambda 太大 | v3 的最终 lambda 约 `55.64`，比 v2 的 `69.65` 和 v1 的 `149` 更稳 |
| 推荐配置 | `collapse_use_weight_proxy: false`、`learning_rate: 1.0e-4`、`gradient_accumulation_steps: 4`、`num_train_epochs: 6.0` |

注：本节早期分析曾建议固定正 lambda 或极小正 lr。最新服务器 v2/v3 结果显示，在当前 VQA-RAD yes/no 0.50 剪枝主线中，裸 learnable lambda + 负 lr 梯度上升可以实现剪枝零损失；因此主实验以 v3 配置为准。

### 正确的实验对照设计

```
基座: sft-yesno-multilr/checkpoint-100-hf（统一）
数据集: vqa_rad_train_yesno（统一）

实验组:
  A. vulcan-sft (collapse only)        → 68.13%
  B. vulcan-align (alignment only)     → 待测
  C. vulcan-align-collapse (both)       → 66.53%
  D. baseline-sft (no regularization)→ 待测

注意: A 和 C 的 loss 曲线几乎一样（差值 < 10），
     说明 alignment 对这个任务没有正向贡献
```
