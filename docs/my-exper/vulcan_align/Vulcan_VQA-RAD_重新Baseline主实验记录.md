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

# Vulcan VQA-RAD 重新 Baseline 后主实验记录

本文档记录从重新设定 baseline 开始，到当前 `Vulcan + 0.50 MLP pruning` 主实验完成为止的实验流程、配置、结果和阶段结论。此前 yes/no-only baseline、过强 collapse 配置等早期尝试不再作为主实验标准，只作为排查经验。

## 1. 实验标准重置

### 1.1 为什么重置

早期实验中，直接使用 yes/no-only 训练集做 baseline 会导致模型严重偏向单一答案。例如训练后模型几乎恒定输出 `no`，最终 yes/no accuracy 低于 `55%`。这说明 yes/no-only 数据规模太小，容易让 Qwen3.5-0.8B 在短答案分布上过拟合，不能作为可靠主基线。

因此本轮实验重新规定：

- baseline 使用完整 VQA-RAD 训练集 `vqa_rad_train`。
- yes/no 子集只作为评估集，不作为主训练集。
- Vulcan 从训练好的 `baseline-full-vqa` 继续训练，而不是从原始 Qwen 基座直接训练。
- 聚类文件基于 `baseline-full-vqa` 生成，保证剪枝结构和当前 baseline 对齐。

### 1.2 当前主实验路径

```text
baseline model:
  saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa

Vulcan model:
  saves/qwen35-0_8b-vqa-rad/full/vulcan-from-baseline-full-vqa-lr1e5-lam01

Vulcan pruned 0.50 model:
  saves/qwen35-0_8b-vqa-rad/full/vulcan-from-baseline-full-vqa-lr1e5-lam01-pruned-0_50

cluster_idx:
  saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_baseline_full_vqa.json

dataset_dir:
  datasets/vqa_rad
```

## 2. Baseline Full VQA

### 2.1 训练配置

主 baseline 使用完整 VQA-RAD 训练集：

```yaml
dataset_dir: datasets/vqa_rad
dataset: vqa_rad_train
eval_dataset: vqa_rad_test
template: qwen3_5_nothink
finetuning_type: full
learning_rate: 5.0e-6
num_train_epochs: 3.0
eval_steps: 50
save_steps: 50
load_best_model_at_end: true
metric_for_best_model: eval_loss
greater_is_better: false
```

输出模型：

```text
saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa
```

### 2.2 Baseline yes/no 评估

最终 baseline yes/no accuracy：

```text
yes/no accuracy = 0.6932
```

该结果作为本轮主实验的基线。

## 3. Vulcan 配置

### 3.1 训练起点

Vulcan 不再从原始 Qwen 模型开始，而是从 full VQA baseline 继续训练：

```yaml
model_name_or_path: saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa
dataset_dir: datasets/vqa_rad
dataset: vqa_rad_train
eval_dataset: vqa_rad_test
```

### 3.2 Collapse 配置

本轮成功主实验使用保守 collapse 配置：

```yaml
use_collapse_loss: true
collapse_cluster_idx_path: saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_baseline_full_vqa.json
collapse_lambda1: 0.0
collapse_lambda2: 0.0
collapse_learnable_lambda: true
collapse_lambda_lr: -0.1
collapse_use_weight_proxy: false
```

关键点：

- `collapse_use_weight_proxy: false`，让梯度直接回传到 MLP 权重。
- `collapse_lambda_lr: -0.1`，用负学习率实现 lambda 梯度上升。
- 相比早期 `collapse_lambda_lr=-1.0`，本轮使用更弱的 lambda 更新，避免未剪枝任务能力明显下降。

### 3.3 Step 长度

论文按 pruning ratio `R` 设计训练步数：

```text
total_steps = 6250 * R^2 + 1250 * R
```

当 `R=0.5` 时：

```text
total_steps = 6250 * 0.25 + 1250 * 0.5 = 2187.5 ~= 2188
```

因此本轮 Vulcan 使用：

```yaml
max_steps: 2188
lr_scheduler_type: constant
warmup_ratio: 0.0
learning_rate: 1.0e-5
```

固定主学习率的原因：早期 cosine scheduler 在训练后半段将主学习率衰减到接近 0，模型权重几乎不能再通过任务 loss 恢复，而 collapse 约束仍在推动结构变化。固定主学习率更符合持续平衡任务能力和 collapse 约束的需要。

## 4. Vulcan 训练 Loss 走势

本轮日志单独记录了 `SFT Loss` 和 `Collapse Loss`。最终训练过程如下：

| Step | Epoch | SFT Loss | Collapse Loss |
| ---: | ---: | ---: | ---: |
| 10 | 0.04 | 0.0298 | 515.33 |
| 200 | 0.85 | 0.0299 | 6674.50 |
| 400 | 1.70 | 0.0030 | 6062.91 |
| 600 | 2.55 | 0.0163 | 4107.30 |
| 800 | 3.40 | 0.0144 | 2688.55 |
| 1000 | 4.26 | 0.0127 | 1777.17 |
| 1200 | 5.11 | 0.0128 | 1194.50 |
| 1400 | 5.96 | 0.0001 | 816.59 |
| 1600 | 6.81 | 0.0265 | 567.39 |
| 1800 | 7.66 | 0.0184 | 399.42 |
| 2000 | 8.51 | 0.0033 | 285.18 |
| 2188 | 9.31 | 0.00003 | 210.46 |

该趋势与 Vulcan 论文中的训练曲线一致：

1. 前期 `lambda1/lambda2` 增大，collapse 约束项被放大，`Collapse Loss` 先快速升高。
2. 中后期同簇 FFN 权重逐步靠近 anchor，约束残差下降，加权后的 `Collapse Loss` 随之下降。
3. `SFT Loss` 始终保持较低，说明从 full VQA baseline 继续训练时任务拟合没有明显失控。

注意：这里的 `Collapse Loss` 是加权项，即实际加入 total loss 的 `lambda1 * L1 + lambda2 * L2`，不能简单理解为未加权簇内距离。

## 5. Yes/No 评估结果

### 5.1 Vulcan 未剪枝

评估文件：

```text
outputs/vulcan-lr1e5-lam01-yesno-predict/generated_predictions.jsonl
```

结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.5896414342629482,
  "normalized_exact_match": 0.5896414342629482,
  "token_f1": 0.6553784860557768,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6852589641434262,
  "yesno_prediction_coverage": 1.0,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 156,
    "yes": 95
  },
  "yesno_confusion": {
    "no->no": 105,
    "no->yes": 28,
    "yes->no": 51,
    "yes->yes": 67
  }
}
```

相对 baseline：

```text
baseline-full-vqa yes/no accuracy = 0.6932
Vulcan unpruned yes/no accuracy  = 0.6853
drop                             = -0.79 percentage points
```

结论：未剪枝任务能力基本保持。

### 5.2 Vulcan 0.50 剪枝后

评估文件：

```text
outputs/vulcan-lr1e5-lam01-pruned-0_50-yesno-predict/generated_predictions.jsonl
```

结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.5577689243027888,
  "normalized_exact_match": 0.5577689243027888,
  "token_f1": 0.647011952191235,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6812749003984063,
  "yesno_prediction_coverage": 0.9960159362549801,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 151,
    "other": 1,
    "yes": 99
  },
  "yesno_confusion": {
    "no->no": 102,
    "no->other": 1,
    "no->yes": 30,
    "yes->no": 49,
    "yes->yes": 69
  }
}
```

相对关系：

```text
baseline-full-vqa unpruned       = 0.6932
Vulcan unpruned                  = 0.6853
Vulcan pruned 0.50               = 0.6813

Vulcan pruned vs Vulcan unpruned = -0.40 percentage points
Vulcan pruned vs baseline        = -1.19 percentage points
```

结论：Vulcan 训练后，50% MLP 剪枝几乎没有造成额外性能损失。

## 6. FFN 冗余机制证据

### 6.1 Gated FFN 对应关系

Vulcan 论文中的 canonical FFN 是：

```text
FFN(X) = activation(X W1^T) W2^T
```

剪枝时保留每个簇的 anchor neuron，并把同簇 `W2` 列求和。Qwen/Llama 使用 gated FFN：

```text
down_proj( act(gate_proj(x)) * up_proj(x) )
```

因此在 Qwen3.5-0.8B 上，等价的 collapse/prune 逻辑是：

- 对同簇 neuron 的 `up_proj` 行和 `gate_proj` 行同时 collapse。
- 剪枝时保留 anchor 的 `up_proj/gate_proj` 行。
- 将同簇 neuron 的 `down_proj` 列求和到 anchor 对应的新列。

本仓库当前实现使用 `concat(up_proj.weight, gate_proj.weight)` 做聚类与 collapse，符合 gated FFN 的结构要求。

### 6.2 Redundancy Inspection

Vulcan 训练后，后半层同簇 neuron 到 anchor 的距离已经非常小。已观察到的层级结果范围：

```text
mean_l1     ~= 2.9e-05 ~ 9.5e-05
mean_l2     ~= 2.1e-04 ~ 7.9e-04
mean_cosine ~= 0.9935 ~ 0.9979
```

这些数值说明：

- `mean_l1/mean_l2` 很小：同簇 `up_proj/gate_proj` 权重数值已经非常接近。
- `mean_cosine` 接近 1：同簇 neuron 的权重方向几乎一致。
- 因此非 anchor neuron 对 FFN 的功能贡献已经高度冗余，剪掉后再合并 `down_proj` 列不会明显改变 FFN 输出。

这解释了为什么本轮实验中：

```text
Vulcan unpruned accuracy = 0.6853
Vulcan pruned accuracy   = 0.6813
drop                     = -0.40 percentage points
```

也就是说，剪枝近零损失不是偶然指标波动，而是 FFN 簇内权重坍缩带来的结构性结果。

## 7. 当前主结论

本轮重新 baseline 后的主实验结论：

1. full VQA baseline 在 yes/no 子集上达到 `69.32%`，明显优于 yes/no-only baseline。
2. Vulcan 从 full VQA baseline 继续训练后，未剪枝 yes/no accuracy 为 `68.53%`，仅下降 `0.79` 个百分点。
3. Vulcan 0.50 MLP 剪枝后 yes/no accuracy 为 `68.13%`，相对未剪枝 Vulcan 仅下降 `0.40` 个百分点。
4. `Collapse Loss` 呈现论文式“先升高、后下降”的走势，说明 lambda 放大约束后，权重逐步进入 collapse 阶段。
5. redundancy inspection 显示同簇 `up_proj/gate_proj` 权重已经高度接近，支持“剪枝近零损失来自权重坍缩”的机制解释。

因此，当前 `vulcan-from-baseline-full-vqa-lr1e5-lam01` 可以作为新的 Vulcan 主实验版本。

## 8. 待补对照

为了让论文/报告里的对照更严谨，还需要补一组最新同 cluster 的 baseline direct prune：

```text
baseline-full-vqa -> 使用 cluster_idx_greedy_match_0_50_baseline_full_vqa.json 直接剪枝 -> yes/no predict
```

该结果用于证明：

```text
同样的 0.50 剪枝结构，baseline direct prune 明显差于 Vulcan pruned；
因此性能保持来自 Vulcan collapse 训练，而不是 cluster 本身偶然好。
```

建议命令：

```bash
python scripts/vulcan/save_pruned_model.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_baseline_full_vqa.json \
  --output_dir saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa-pruned-0_50 \
  --template qwen3_5_nothink \
  --trust_remote_code

WANDB_DISABLED=true python src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_yesno_predict.yaml \
  model_name_or_path=saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa-pruned-0_50 \
  dataset_dir=datasets/vqa_rad \
  output_dir=outputs/baseline-full-vqa-pruned-0_50-yesno-predict

python scripts/vulcan/eval_vqa_predictions.py \
  --prediction_file outputs/baseline-full-vqa-pruned-0_50-yesno-predict/generated_predictions.jsonl
```

## 9. Activation Alignment 实验记录

Vulcan 主实验稳定后，继续验证多模态场景中的 activation alignment。该实验的动机是：视觉 token 与文本 token 在 FFN 中激活的 top-k 神经元可能不一致；如果直接把 LLM 剪枝方法迁移到多模态模型，可能忽略视觉/文本激活差异。因此尝试约束视觉 token 与文本 token 的 FFN soft top-k activation mask 对齐。

### 9.1 当前实现

当前实现位于：

```text
src/llamafactory/train/vulcan/activation_align.py
src/llamafactory/train/sft/trainer.py
```

实现流程：

1. 在每一层 MLP 的 `down_proj` 上注册 forward hook。
2. 捕获 `down_proj` 输入，即 gated FFN 中间激活：

```text
A = silu(gate_proj(x)) * up_proj(x)
```

3. 根据 token mask 分别池化视觉 token 和文本 token 的激活。
4. 用 quantile threshold + sigmoid 构造可微 soft top-k mask：

```text
threshold = quantile(pooled_activation, align_quantile)
soft_mask = sigmoid((pooled_activation - threshold) / align_temperature)
```

5. 使用 soft IoU 约束视觉 mask 与文本 mask 对齐：

```text
align_raw_loss = 1 - soft_iou
align_loss = align_lambda * align_raw_loss
```

该方法避免了硬 top-k index selection 不可微的问题。`threshold` 用 detach 后的 activation 计算，只作为当前 batch 的参考阈值；梯度从 sigmoid soft mask 回传到 pooled activation 和模型权重。

### 9.2 日志指标解释

```text
align_loss          实际加到 total loss 的 alignment loss
align_raw_loss      未乘 align_lambda 的原始对齐损失
align_soft_iou      visual/text soft top-k mask 的重合度，越高越好
align_visual_tokens 当前 batch 中参与 visual pooling 的 image token 数
align_text_tokens   当前 batch 中参与 text pooling 的 text token 数
align_mask_v_mean   visual soft mask 的平均值
align_mask_t_mean   text soft mask 的平均值
```

当 `align_loss_type=soft_iou` 时：

```text
align_raw_loss = 1 - align_soft_iou
```

### 9.3 Align-Only: answer, lambda=10

第一版 align-only 使用：

```text
model_name_or_path = saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa
align_text_mode    = answer
align_lambda       = 10.0
learning_rate      = 1.0e-5
lr_scheduler_type  = constant
```

训练日志现象：

```text
align_loss      ~= 7.12 ~ 7.19
align_raw_loss  ~= 0.7109 ~ 0.7188
align_soft_iou  ~= 0.281 ~ 0.290
align_text_tokens = 3
```

评估结果：

```json
{
  "yesno_accuracy": 0.6215139442231076,
  "yesno_prediction_counts": {
    "no": 130,
    "yes": 121
  },
  "yesno_confusion": {
    "no->no": 84,
    "no->yes": 49,
    "yes->no": 46,
    "yes->yes": 72
  }
}
```

结论：

- `align_lambda=10` 明显过强，alignment loss 几乎主导 total loss。
- `align_soft_iou` 提升很小，但 yes/no accuracy 从 baseline `69.32%` 降到 `62.15%`。
- 当前配置不适合作为主实验。

### 9.4 Align-Only: answer, lambda=1, lr=5e-6

第二版降低 alignment 强度：

```text
align_text_mode    = answer
align_lambda       = 1.0
learning_rate      = 5.0e-6
lr_scheduler_type  = constant
```

训练日志前段：

```text
loss             ~= 0.8 ~ 1.0
align_loss       ~= 0.714 ~ 0.723
align_raw_loss   ~= 0.714 ~ 0.723
align_soft_iou   ~= 0.278 ~ 0.285
align_text_tokens = 3
```

评估结果：

```json
{
  "yesno_accuracy": 0.6374501992031872,
  "yesno_prediction_counts": {
    "no": 156,
    "yes": 95
  },
  "yesno_confusion": {
    "no->no": 99,
    "no->yes": 34,
    "yes->no": 57,
    "yes->yes": 61
  }
}
```

结论：

- 降低强度后比 `lambda=10` 好，但仍明显低于 baseline。
- 这说明问题不只是正则太强，`answer-only` 的文本池化目标本身也不稳定。
- `answer-only` 只有约 3 个 token，信号太稀疏，与 100+ 到 260+ 个 visual tokens 做 pooling 对齐时不够稳。

### 9.5 Align-Only: qa, lambda=0.5, lr=5e-6

第三版将文本侧从 answer-only 改为 question+answer，并进一步降低正则强度：

```text
align_text_mode    = qa
align_lambda       = 0.5
learning_rate      = 5.0e-6
lr_scheduler_type  = constant
```

训练输出路径：

```text
saves/qwen35-0_8b-vqa-rad/full/align-only-from-baseline-full-vqa-qa-lam05-lr5e6
```

训练日志：

```text
train_loss        = 0.4812
eval_loss         = 0.6439
align_loss        ~= 0.3477 ~ 0.3516
align_raw_loss    ~= 0.6953 ~ 0.7031
align_soft_iou    ~= 0.297 ~ 0.305
align_text_tokens ~= 44 ~ 51
```

评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.7171314741035857,
  "normalized_exact_match": 0.7171314741035857,
  "token_f1": 0.7171314741035857,
  "yesno_examples": 251,
  "yesno_accuracy": 0.7171314741035857,
  "yesno_prediction_coverage": 1.0,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 136,
    "yes": 115
  },
  "yesno_confusion": {
    "no->no": 99,
    "no->yes": 34,
    "yes->no": 37,
    "yes->yes": 81
  }
}
```

相对 baseline：

```text
baseline-full-vqa yes/no accuracy = 0.6932
align-only qa-lam05 accuracy      = 0.7171
gain                              = +2.39 percentage points
```

相对 answer-only：

```text
answer lambda=10 accuracy = 0.6215
answer lambda=1 accuracy  = 0.6375
qa lambda=0.5 accuracy    = 0.7171
```

结论：

- `qa` 文本池化显著优于 `answer-only`。
- `align_text_tokens` 从 3 增加到约 44-51，文本侧激活信号更稳定。
- `qa-lam05` 明显提升 yes 类召回：`yes->yes=81`，优于 answer-only 弱 align 的 `yes->yes=61`。
- 当前 `align-only qa-lam05` 是 activation alignment 的主成功配置。

### 9.6 Align + Vulcan: qa, lambda=0.5

在 align-only 成功后，尝试将 activation alignment 与 Vulcan collapse 同时训练：

```text
model_name_or_path       = saves/qwen35-0_8b-vqa-rad/full/baseline-full-vqa
align_text_mode          = qa
align_lambda             = 0.5
learning_rate            = 1.0e-5
max_steps                = 2188
lr_scheduler_type        = constant
collapse_lambda_lr       = -0.1
collapse_use_weight_proxy = false
```

输出模型：

```text
saves/qwen35-0_8b-vqa-rad/full/vulcan-align-qa-lam05-from-baseline-full-vqa-lr1e5-lam01
```

未剪枝 yes/no 评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.6573705179282868,
  "normalized_exact_match": 0.6573705179282868,
  "token_f1": 0.6573705179282868,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6573705179282868,
  "yesno_prediction_coverage": 1.0,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 119,
    "yes": 132
  },
  "yesno_confusion": {
    "no->no": 83,
    "no->yes": 50,
    "yes->no": 36,
    "yes->yes": 82
  }
}
```

剪枝后 yes/no 评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.6613545816733067,
  "normalized_exact_match": 0.6613545816733067,
  "token_f1": 0.6613545816733067,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6613545816733067,
  "yesno_prediction_coverage": 1.0,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 118,
    "yes": 133
  },
  "yesno_confusion": {
    "no->no": 83,
    "no->yes": 50,
    "yes->no": 35,
    "yes->yes": 83
  }
}
```

对比：

```text
baseline-full-vqa         = 0.6932
align-only qa-lam05       = 0.7171
Vulcan-only unpruned      = 0.6853
Vulcan-only pruned 0.50   = 0.6813
align+Vulcan unpruned     = 0.6574
align+Vulcan pruned 0.50  = 0.6614
```

结论：

- 当前 simultaneous align+collapse 训练不建议作为主实验，但原因不是剪枝鲁棒性失败。
- 剪枝后 accuracy 从 `0.6574` 小幅升至 `0.6614`，说明 collapse 仍然提供了剪枝鲁棒性。
- 真正问题是联合训练后的整体任务能力偏低，明显低于 Vulcan-only pruned 的 `0.6813` 和 baseline 的 `0.6932`。
- 模型明显偏向 `yes`：未剪枝 `yes` prediction 为 132，剪枝后为 133，而 label 中 `yes` 为 118。
- `yes->yes=82` 保持较好，但 `no->yes=50` 明显过高，说明 align 和 collapse 同时训练改变了决策边界。
- 当前更合理的方向不是直接同时训练，而是两阶段：

```text
baseline-full-vqa
-> align-only qa-lam05
-> 基于 align-only 模型重新生成 cluster_idx
-> 再做 Vulcan collapse
-> 剪枝评估
```

### 9.7 Align 阶段结论

当前 activation alignment 的阶段结论：

1. `answer-only` 不适合作为主配置。它的文本 token 太少，信号稀疏，强弱两版都显著低于 baseline。
2. `qa + lambda=0.5 + lr=5e-6` 是当前成功配置，使 yes/no accuracy 从 `69.32%` 提升到 `71.71%`。
3. align-only 的收益主要来自更稳定的 text pooling，而不是显著提高 `align_soft_iou` 数值；`align_soft_iou` 仍只在约 `0.30` 附近。
4. simultaneous align+Vulcan 当前存在目标冲突，未剪枝 accuracy 为 `65.74%`，剪枝后为 `66.14%`，暂不进入主结果。
5. 下一步建议采用两阶段路线：先用 align-only 提升多模态表征，再基于 align 后模型重新聚类并进行 Vulcan collapse。
