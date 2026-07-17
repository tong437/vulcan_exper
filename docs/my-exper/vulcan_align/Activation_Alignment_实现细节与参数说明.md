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

# Activation Alignment 实现细节与参数说明

本文档说明当前仓库中 activation alignment 的实现方式，包括如何捕获视觉/文本 token 的 FFN 激活，如何构造 soft top-k 神经元 mask，如何解决 hard top-k 不可微的问题，以及各实验配置参数的含义。

相关代码：

```text
src/llamafactory/train/vulcan/activation_align.py
src/llamafactory/train/sft/trainer.py
src/llamafactory/hparams/finetuning_args.py
examples/vulcan/qwen35_08b_vqa_rad_align_sft.yaml
examples/vulcan/qwen35_08b_vqa_rad_align_collapse_sft.yaml
```

## 1. 设计动机

Vulcan 原论文主要针对 ViT 中的 FFN 神经元冗余构造和剪枝。在多模态 VLM 中，输入同时包含视觉 token 和文本 token。直接迁移 LLM/ViT 剪枝方法时，一个关键问题是：

```text
视觉 token 激活的 FFN 神经元，和文本 token 激活的 FFN 神经元，可能并不一致。
```

如果视觉和文本依赖不同的 FFN neuron 子集，那么剪枝时只依据某一侧激活或权重冗余，可能破坏跨模态推理能力。因此 activation alignment 的目标是：

```text
让视觉 token 与任务相关文本 token 在 FFN 中激活更相似的 top-k neuron 集合。
```

当前实现不是直接对齐 hidden states，而是对齐每层 MLP 中间维度上的 neuron activation mask。

## 2. 捕获哪一类激活

Qwen/Llama 风格的 gated MLP 结构可以写成：

```text
MLP(x) = down_proj( act(gate_proj(x)) * up_proj(x) )
```

其中：

```text
act(gate_proj(x)) * up_proj(x)
```

是进入 `down_proj` 前的中间激活，维度为：

```text
[batch_size, seq_len, intermediate_size]
```

这个向量的每一维对应一个 FFN neuron 的激活强度。因此当前实现把 hook 挂在 `down_proj` 上，捕获 `down_proj` 的输入：

```python
layer_ref.mlp.down_proj.register_forward_hook(...)
```

hook 中保存的是：

```python
act_store[layer_idx] = args[0]
```

这意味着当前 activation alignment 对齐的是 gated FFN 的实际中间激活，而不是 `up_proj` 或 `gate_proj` 的单独输出。

## 3. 训练流程中的接入位置

在 SFT trainer 的 `compute_loss` 中，前向传播前会把当前 batch 的 token 信息交给 aligner：

```python
self.activation_aligner.set_batch(
    input_ids=inputs["input_ids"],
    labels=inputs.get("labels"),
    attention_mask=inputs.get("attention_mask"),
)
```

模型前向传播时，forward hook 自动记录每一层 MLP 的中间激活。原始 SFT loss 计算完成后，再调用：

```python
align_loss = self.activation_aligner.compute_alignment_loss()
```

最终训练目标为：

```text
total_loss = sft_loss + align_loss
```

如果同时开启 Vulcan collapse，则顺序为：

```text
total_loss = sft_loss + collapse_loss + align_loss
```

## 4. Token Mask 构造

alignment 的第一步是区分视觉 token 和文本 token。

### 4.1 Visual Mask

视觉 token 通过 `image_token_id` 判断：

```python
visual_mask = input_ids == image_token_id
```

然后与 `attention_mask` 相与，排除 padding：

```python
visual_mask = visual_mask & valid_mask
```

日志中的 `align_visual_tokens` 就是当前 batch 中 `visual_mask=True` 的 token 数。

### 4.2 Text Mask

文本 token 先排除视觉 token 和 padding：

```python
non_visual_mask = (~visual_mask) & valid_mask
```

然后根据 `labels` 构造 answer/question mask。LlamaFactory SFT 中，prompt token 的 label 通常是 `IGNORE_INDEX`，assistant answer token 的 label 不是 `IGNORE_INDEX`：

```python
answer_mask = (labels != IGNORE_INDEX) & non_visual_mask
question_mask = (labels == IGNORE_INDEX) & non_visual_mask
```

`align_text_mode` 决定最终使用哪些文本 token：

```text
answer   只使用 assistant answer token
question 只使用 prompt/question token
qa       使用 question + answer 的所有非视觉有效 token
```

实验中观察到：

```text
answer 模式: align_text_tokens 通常约为 3
qa 模式:     align_text_tokens 通常约为 44-51
```

因此 answer-only 信号很稀疏，而 qa 模式更稳定。

## 5. 激活池化

对每一层 MLP 激活：

```text
act: [batch_size, seq_len, intermediate_size]
```

分别选出视觉 token 和文本 token 的激活：

```python
selected = act[token_mask].abs()
```

这里取绝对值，表示关注 neuron 激活强度，而不是正负号。

然后在 token 维度做 pooling，得到每层的两个向量：

```text
pooled_v: [intermediate_size]
pooled_t: [intermediate_size]
```

当前支持两种 pooling：

```text
mean  对所有选中 token 的激活取平均
max   对所有选中 token 的激活取最大值
```

当前推荐使用：

```yaml
align_pool_type: mean
```

因为 mean pooling 更稳定，不容易被单个异常 token 激活主导。

## 6. 为什么 hard top-k 不可微

最直接的 top-k alignment 可能会写成：

```python
idx_v = torch.topk(pooled_v, k).indices
idx_t = torch.topk(pooled_t, k).indices
mask_v = hard_one_hot(idx_v)
mask_t = hard_one_hot(idx_t)
loss = mismatch(mask_v, mask_t)
```

但这种做法有两个问题：

1. `topk` 返回的是离散 index，index selection 对输入 activation 不可微。
2. hard mask 是 0/1 离散变量，大多数位置梯度为 0，难以通过 loss 反向调整 neuron activation。

也就是说，hard top-k 可以用于统计和评估，但不适合作为训练中的可微信号。

## 7. 当前如何解决不可微问题

当前实现使用 `quantile threshold + sigmoid` 构造 soft top-k mask：

```python
tau = torch.quantile(pooled_activation.detach().float(), align_quantile)
soft_mask = torch.sigmoid((pooled_activation - tau) / align_temperature)
```

写成公式：

```text
soft_mask_i = sigmoid((a_i - tau) / T)
```

其中：

```text
a_i  第 i 个 FFN neuron 的 pooled activation
tau  当前层 pooled activation 的 quantile 阈值
T    align_temperature
```

### 7.1 Soft Top-k 的含义

当 `align_quantile=0.8` 时，`tau` 是 activation 的 80% 分位数。直觉上，它近似选择 top 20% neuron：

```text
a_i >> tau  -> soft_mask_i 接近 1
a_i << tau  -> soft_mask_i 接近 0
a_i ~= tau  -> soft_mask_i 在 0 和 1 之间
```

这不是硬 0/1 mask，而是连续值 mask。

### 7.2 为什么可微

虽然 `tau` 由 quantile 得到，但代码中对 `pooled_activation` 做了 detach：

```python
pooled_activation.detach()
```

因此 `tau` 只作为当前 batch 的固定参考阈值，不参与反向传播。梯度路径是：

```text
align_loss
-> soft_iou
-> soft_mask
-> pooled_activation
-> MLP activation
-> model weights
```

也就是说，不让梯度穿过不可微/不稳定的 quantile 排序操作，但保留了从 sigmoid soft mask 到 activation 的连续梯度。

### 7.3 Temperature 的作用

`align_temperature` 控制 sigmoid 的陡峭程度：

```text
temperature 越小，soft mask 越接近 hard top-k
temperature 越大，soft mask 越平滑
```

当前实验常用：

```yaml
align_temperature: 0.05
```

它比默认的 `0.1` 更接近 hard top-k，但仍然保留可微性。

## 8. 对齐损失

当前主实验使用：

```yaml
align_loss_type: soft_iou
```

对每一层，先计算视觉 soft mask 和文本 soft mask 的 soft IoU：

```text
intersection = sum(soft_v * soft_t)
union = sum(soft_v) + sum(soft_t) - intersection
soft_iou = intersection / union
```

然后：

```text
layer_loss = 1 - soft_iou
```

所有层取平均：

```text
align_raw_loss = mean(layer_loss)
```

最后乘上权重系数：

```text
align_loss = align_lambda * align_raw_loss
```

所以当 `align_loss_type=soft_iou` 时：

```text
align_raw_loss 越低越好
align_soft_iou 越高越好
```

## 9. 日志指标说明

训练日志中的 alignment 指标含义如下：

```text
align_loss          最终加入 total loss 的 alignment loss
align_raw_loss      未乘 align_lambda 的原始 alignment loss
align_soft_iou      视觉/文本 soft top-k mask 的平均 IoU
align_visual_tokens 当前 batch 参与 visual pooling 的 token 数
align_text_tokens   当前 batch 参与 text pooling 的 token 数
align_mask_v_mean   visual soft mask 的平均值
align_mask_t_mean   text soft mask 的平均值
```

例如：

```text
align_lambda = 0.5
align_raw_loss = 0.6992
align_loss = 0.3496
```

如果 `align_loss` 远大于 SFT loss，alignment 会主导训练，容易损伤任务能力。前期 `answer + lambda=10` 失败，主要就是这个问题。

## 10. 参数说明

### 10.1 use_activation_align

```yaml
use_activation_align: true
```

是否开启 activation alignment。只支持 SFT 阶段。

### 10.2 align_lambda

```yaml
align_lambda: 0.5
```

alignment loss 的权重：

```text
align_loss = align_lambda * align_raw_loss
```

实验结论：

```text
answer + lambda=10 过强，accuracy 降到 62.15%
answer + lambda=1  仍失败，accuracy 为 63.75%
qa + lambda=0.5    成功，accuracy 为 71.71%
```

当前推荐：

```yaml
align_lambda: 0.5
```

### 10.3 align_temperature

```yaml
align_temperature: 0.05
```

soft top-k sigmoid 的温度。越小越接近 hard top-k，越大越平滑。

建议：

```text
0.05  当前实验使用值
0.1   代码默认值，更平滑
```

如果训练不稳定或 mask 过硬，可以尝试调大到 `0.1`。

### 10.4 align_quantile

```yaml
align_quantile: 0.8
```

用于计算 soft top-k 阈值的分位数。`0.8` 约等价于关注 top 20% 激活 neuron。

解释：

```text
quantile=0.8  top 20% 近似激活区域
quantile=0.9  top 10% 更稀疏
quantile=0.7  top 30% 更宽松
```

当前推荐保持：

```yaml
align_quantile: 0.8
```

### 10.5 align_pool_type

```yaml
align_pool_type: mean
```

token 维度的 pooling 方式：

```text
mean  平均所有选中 token 的激活，较稳定
max   取最大激活，强调任一 token 的强响应
```

当前推荐：

```yaml
align_pool_type: mean
```

### 10.6 align_loss_type

```yaml
align_loss_type: soft_iou
```

可选：

```text
soft_iou  使用 1 - soft IoU，当前主配置
l1        使用 soft mask 的平均 L1 距离
neg_iou   使用 -soft IoU，保留早期兼容
```

当前推荐：

```yaml
align_loss_type: soft_iou
```

### 10.7 align_text_mode

```yaml
align_text_mode: qa
```

决定文本侧 pooling 使用哪些 token：

```text
answer   只使用 assistant answer token
question 只使用 prompt/question token
qa       使用 question + answer 的所有非视觉 token
```

实验结论：

```text
answer 模式 token 太少，通常只有 3 个 token，信号稀疏。
qa 模式通常有 44-51 个 text tokens，信号更稳定，当前效果最好。
```

当前推荐：

```yaml
align_text_mode: qa
```

## 11. 当前推荐配置

当前 align-only 成功配置为：

```bash
WANDB_DISABLED=true torchrun --nproc_per_node=1 --master_port=29536 src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_align_sft.yaml \
  model_name_or_path=saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa \
  dataset_dir=datasets/vqa_rad \
  output_dir=saves/qwen35-0_8b-vqa-rad/full/align-only-from-baseline-full-vqa-qa-lam05-lr5e6 \
  align_text_mode=qa \
  align_lambda=0.5 \
  learning_rate=5.0e-6 \
  lr_scheduler_type=constant \
  warmup_ratio=0.0 \
  deepspeed=null \
  overwrite_output_dir=true
```

对应结果：

```text
baseline-full-vqa yes/no accuracy = 69.32%
align-only qa-lam05 accuracy      = 71.71%
gain                              = +2.39 percentage points
```

## 12. Align + Vulcan 当前状态

同时开启 align 和 Vulcan collapse 的当前配置为：

```text
align_text_mode=qa
align_lambda=0.5
collapse_lambda_lr=-0.1
learning_rate=1.0e-5
max_steps=2188
lr_scheduler_type=constant
```

结果：

```text
align+Vulcan unpruned accuracy    = 65.74%
align+Vulcan pruned 0.50 accuracy = 66.14%
```

解释：

- 剪枝后没有明显掉点，说明 Vulcan collapse 仍然提供了剪枝鲁棒性。
- 但联合训练后的整体 accuracy 明显低于 baseline 和 Vulcan-only，说明 alignment 与 collapse 同时训练存在目标冲突。
- 当前更推荐两阶段路线：

```text
baseline-full-vqa
-> align-only qa-lam05
-> 基于 align-only 模型重新生成 cluster_idx
-> 再做 Vulcan collapse
-> 剪枝评估
```

## 13. 小结

当前 activation alignment 的关键结论：

1. 实现上对齐的是 gated FFN 的 `down_proj` 输入，即真实中间 neuron activation。
2. 视觉/文本 token 分别 pooling 后，转换为 soft top-k mask。
3. hard top-k 不可微的问题通过 `detach quantile threshold + sigmoid soft mask` 解决。
4. `answer-only` 文本侧 token 太少，实验中显著伤害 yes/no accuracy。
5. `qa + lambda=0.5 + lr=5e-6` 是当前 align-only 最优配置，将 yes/no accuracy 提升到 `71.71%`。
6. 直接 simultaneous align+Vulcan 目前不如 Vulcan-only，下一步应尝试两阶段方案。
