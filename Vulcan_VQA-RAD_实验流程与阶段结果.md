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

# Vulcan VQA-RAD 实验流程与阶段结果

本文档记录截至当前为止，在 LlamaFactory 中基于 Qwen3.5-0.8B 与 VQA-RAD 数据集进行“通过正则项构造参数冗余及剪枝”实验的完整流程、关键代码实现、已获得结果和下一步判断。

## 1. 实验目标

本实验希望验证如下链路是否成立：

1. 在 VQA-RAD 医学视觉问答任务上完成 Qwen3.5-0.8B full SFT baseline。
2. 基于 baseline 模型收集 MLP 激活，并结合 MLP 权重相似度生成神经元簇 `cluster_idx`。
3. 用 Collapse Loss 约束同簇神经元权重靠近，构造参数冗余。
4. 按簇剪枝 MLP 中间维，并观察剪枝后模型在 VQA-RAD，尤其 yes/no 子集上的性能保持情况。

当前主线模型与数据：

- 模型：`Qwen/Qwen3.5-0.8B`
- 数据集：`flaviagiammarino/vqa-rad`
- 训练方式：full SFT
- 评估重点：VQA-RAD test split 中答案严格为 `yes` 或 `no` 的 251 条样本

### 1.1 当前最新结论

最新服务器实验已完成 `keep_ratio=0.50` 的关键对照。结论从早期的“0.50 过于激进”更新为：

> Vulcan v3 在 50% MLP 剪枝下实现剪枝零损失：未剪枝 yes/no accuracy 为 `0.6056`，剪枝后仍为 `0.6056`。相比 baseline 直接 0.50 剪枝的 `0.4980`，Vulcan 剪枝模型高出 `10.76` 个百分点；但相比 baseline SFT 未剪枝的 `0.6733`，Vulcan 正则训练本身带来约 `6.77` 个百分点任务性能代价。

因此当前最准确的阶段判断是：

- Collapse Loss 成功把 0.50 结构化 MLP 剪枝从灾难性退化变成可控退化。
- 直接剪 baseline 会严重偏向 `yes`，Vulcan v3 剪枝后显著缓解该偏置。
- 当前瓶颈不再是“剪枝后是否掉点”，而是“如何降低 Collapse 训练对未剪枝模型的原任务能力损伤”。

## 2. 仓库新增内容概览

### 2.1 配置文件

```text
examples/vulcan/qwen35_08b_vqa_rad_full_sft_debug.yaml
examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml
examples/vulcan/qwen35_08b_vqa_rad_vulcan_sft.yaml
examples/vulcan/qwen35_08b_vqa_rad_yesno_predict.yaml
```

其中：

- `full_sft_debug.yaml`：小样本 debug。
- `full_sft.yaml`：baseline full SFT。
- `vulcan_sft.yaml`：加载 `cluster_idx` 后追加 Collapse Loss 的正则训练。
- `yesno_predict.yaml`：只做 predict，不训练，用于 yes/no 子集评估。

### 2.2 脚本

```text
scripts/vulcan/convert_vqa_rad.py
scripts/vulcan/filter_vqa_rad_yesno.py
scripts/vulcan/collect_cluster_idx.py
scripts/vulcan/save_pruned_model.py
scripts/vulcan/eval_vqa_predictions.py
scripts/vulcan/inspect_model_redundancy.py
```

### 2.3 训练核心模块

```text
src/llamafactory/train/vulcan/modeling.py
src/llamafactory/train/vulcan/schema.py
src/llamafactory/train/vulcan/clustering.py
src/llamafactory/train/vulcan/collapse_loss.py
src/llamafactory/train/vulcan/pruning.py
```

并在以下 LlamaFactory 原有模块中接入：

```text
src/llamafactory/hparams/finetuning_args.py
src/llamafactory/train/sft/workflow.py
src/llamafactory/train/sft/trainer.py
```

## 3. 数据准备流程

VQA-RAD 原始数据包含 `image`、`question`、`answer` 等字段。为了让 LlamaFactory 多模态 SFT 正常读取，先转换成 sharegpt 风格 JSONL。

转换命令：

```bash
python scripts/vulcan/convert_vqa_rad.py \
  --dataset_name_or_path /path/to/vqa-rad \
  --output_dir datasets/vqa_rad \
  --overwrite
```

输出结构：

```text
datasets/vqa_rad/
  dataset_info.json
  train.jsonl
  test.jsonl
  images/
```

核心样本格式：

```json
{
  "messages": [
    {"role": "user", "content": "<image>问题文本"},
    {"role": "assistant", "content": "答案文本"}
  ],
  "images": ["images/xxx.png"]
}
```

### 3.1 yes/no 子集构造

VQA-RAD 中有一部分闭合式问题，答案严格为 `yes` 或 `no`。为了避免 BLEU/ROUGE 对短答案评估不稳定，额外构造 yes/no test split：

```bash
python scripts/vulcan/filter_vqa_rad_yesno.py \
  --dataset_dir /root/autodl-tmp/.autodl/lyt/datasets/vqa_rad
```

该脚本会从 `test.jsonl` 中筛出 assistant 答案归一化后严格等于 `yes` 或 `no` 的样本，并在 `dataset_info.json` 中写入 `vqa_rad_test_yesno`。

## 4. Baseline 训练与评估

### 4.1 Baseline full SFT

服务器训练使用本地模型和数据路径，通过命令行覆盖 YAML，不直接改服务器 YAML：

```bash
WANDB_DISABLED=true torchrun --nproc_per_node=1 src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/models/Qwen3.5-0.8B \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  output_dir=saves/qwen35-0_8b-vqa-rad/full/sft
```

Baseline 完整评估结果：

```text
epoch                   = 3.0
eval_bleu-4             = 30.3035
eval_rouge-1            = 52.1436
eval_rouge-2            = 2.5216
eval_rouge-l            = 52.0736
eval_runtime            = 0:02:22.43
eval_samples_per_second = 3.166
eval_steps_per_second   = 3.166
```

### 4.2 Baseline yes/no 评估

Predict 命令：

```bash
WANDB_DISABLED=true torchrun --nproc_per_node=1 src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_yesno_predict.yaml \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/vulcan-exper/saves/qwen35-0_8b-vqa-rad/full/sft \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  output_dir=/root/autodl-tmp/.autodl/lyt/vulcan-exper/outputs/baseline-yesno-predict
```

评估命令：

```bash
python scripts/vulcan/eval_vqa_predictions.py \
  --prediction_file /root/autodl-tmp/.autodl/lyt/vulcan-exper/outputs/baseline-yesno-predict/generated_predictions.jsonl
```

Baseline yes/no 结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.6733067729083665,
  "normalized_exact_match": 0.6733067729083665,
  "token_f1": 0.6733067729083665,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6733067729083665,
  "yesno_prediction_coverage": 0.9880478087649402
}
```

解释：

- `yesno_accuracy=0.6733` 是当前 yes/no 子集主基线。
- `yesno_prediction_coverage=0.9880` 说明模型基本能按 yes/no 格式作答，后续剪枝比较不会主要被输出格式干扰。

## 5. 聚类索引生成

### 5.1 命令

聚类基于 baseline SFT 模型，而不是原始 Qwen 基座：

```bash
PYTHONPATH=src python scripts/vulcan/collect_cluster_idx.py \
  --config examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50.json \
  --keep_ratio 0.5 \
  --max_batches 200 \
  --num_workers 0 \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/vulcan-exper/saves/qwen35-0_8b-vqa-rad/full/sft \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  output_dir=/tmp/cluster-idx \
  deepspeed=null \
  dataloader_num_workers=0
```

注意：

- `deepspeed=null` 是为了绕开训练配置里的 DeepSpeed 校验；聚类脚本只做前向收集激活，不需要 DeepSpeed。
- `PYTHONPATH=src` 是源码目录直接运行脚本时的 import 兼容方式；后续脚本已增加自动加入 `src` 的修复。
- 已实际跑通 `--max_batches=200`。

### 5.2 聚类核心代码解析

`collect_mlp_activations()` 在每层 MLP 的 `down_proj` 上注册 forward hook，统计 `down_proj` 输入激活的平均绝对值：

```python
def make_hook(layer_idx: int):
    def hook_fn(module, inputs, output):
        hidden = inputs[0].detach().float()
        hidden = hidden.reshape(-1, hidden.shape[-1]).abs()
        activations[layer_idx] += hidden.sum(dim=0).cpu().double()
        token_counts[layer_idx] += hidden.shape[0]

    return hook_fn
```

解释：

- 对 Qwen/Llama gated MLP，`down_proj` 的输入维度等于 MLP intermediate size。
- 这里统计每个 intermediate neuron 在样本上的平均激活强度。
- 后续选 anchor 时，优先选择簇内激活更强的 neuron，避免把重要 neuron 合并到弱 neuron 上。

权重相似度使用 `up_proj` 与 `gate_proj` 拼接后的 neuron 向量：

```python
def get_mlp_weight_vectors(mlp: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [mlp.up_proj.weight.detach().float(), mlp.gate_proj.weight.detach().float()],
        dim=1,
    )
```

解释：

- gated MLP 中一个 intermediate neuron 同时由 `up_proj` 和 `gate_proj` 控制。
- 拼接两个投影的同一行，作为该 neuron 的权重表征。

聚类时根据权重欧氏距离找相近 neuron，并用激活最大者作为 anchor：

```python
anchor_idx = int(members[np.argmax(activation[members])])
cluster_list.append({"anchor": anchor_idx, "neuron": members.astype(int).tolist()})
```

当前 `keep_ratio=0.5` 表示每层 MLP intermediate size 保留 50%，其余 neuron 按簇合并到 anchor。

## 6. Vulcan 正则训练

### 6.1 命令

```bash
WANDB_DISABLED=true torchrun --nproc_per_node=1 --master_port=29501 src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_vulcan_sft.yaml \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/vulcan-exper/saves/qwen35-0_8b-vqa-rad/full/sft \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  collapse_cluster_idx_path=saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50.json \
  output_dir=saves/qwen35-0_8b-vqa-rad/full/vulcan-sft
```

说明：

- `model_name_or_path` 指向 baseline SFT 输出目录。
- `collapse_cluster_idx_path` 使用前一步生成的 0.50 聚类索引。
- `master_port` 只是 torchrun 通信端口，避免默认 `29500` 被占用。

### 6.2 Collapse Loss 接入点

在 SFT trainer 的 `compute_loss()` 中先计算标准 SFT loss，再追加 Vulcan collapse loss：

```python
loss_outputs = super().compute_loss(model, inputs, *args, **kwargs)
if isinstance(loss_outputs, tuple):
    loss, outputs = loss_outputs
else:
    loss, outputs = loss_outputs, None

loss = self._add_vulcan_loss(model, loss)
```

`_add_vulcan_loss()` 中加载 lambda 并调用 `weight_collapse_loss()`：

```python
lambda1, lambda2 = get_collapse_lambdas(unwrapped_model, self.finetuning_args)
loss_collapse = weight_collapse_loss(
    unwrapped_model,
    self.vulcan_cluster_idx,
    lambda1,
    lambda2,
    use_weight_proxy=self.finetuning_args.collapse_use_weight_proxy,
)
return loss + loss_collapse.to(loss.device, dtype=loss.dtype)
```

解释：

- 标准 SFT loss 保持原任务能力。
- Collapse Loss 额外推动同簇 neuron 的 `up_proj/gate_proj` 权重靠近 anchor。
- 训练日志会记录 `collapse_loss`、`collapse_lambda1`、`collapse_lambda2`。

### 6.3 Collapse Loss 核心实现

优化后的实现会把每层所有簇展开成两个索引 tensor：

```python
neuron_idxs, anchor_idxs = layer_tensors
diff_w = weight_proxy.index_select(0, neuron_idxs) - weight_proxy.index_select(0, anchor_idxs)
loss = loss + lambda1 * diff_w.abs().sum() + lambda2 * diff_w.pow(2).sum()
```

解释：

- `neuron_idxs` 是所有需要约束的 neuron 行号。
- `anchor_idxs` 是与每个 neuron 对应的 anchor 行号。
- 每层只做一次向量化 `index_select`，避免逐簇 Python 循环。

这一点是实际训练中发现的性能瓶颈修复：原始逐簇循环版本在 A40 上约 `207s/it`，不可接受；向量化后才适合继续训练。

## 7. Vulcan SFT 未剪枝评估

对 `vulcan-sft` 做 yes/no predict 后，结果如下：

```json
{
  "num_examples": 251,
  "exact_match": 0.6733067729083665,
  "normalized_exact_match": 0.6733067729083665,
  "token_f1": 0.6733067729083665,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6733067729083665,
  "yesno_prediction_coverage": 0.9960159362549801
}
```

与 baseline 对比：

| 模型 | yes/no accuracy | prediction coverage |
| --- | ---: | ---: |
| baseline SFT | 0.6733 | 0.9880 |
| Vulcan SFT 未剪枝 | 0.6733 | 0.9960 |

结论：

- 加 Collapse Loss 后，未剪枝模型在 yes/no accuracy 上没有掉点。
- prediction coverage 略升，说明输出格式没有被正则训练破坏。

## 8. MLP 剪枝

### 8.1 剪枝命令

```bash
python scripts/vulcan/save_pruned_model.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-sft \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50.json \
  --output_dir saves/qwen35-0_8b-vqa-rad/full/vulcan-pruned \
  --template qwen3_5_nothink \
  --trust_remote_code
```

### 8.2 剪枝核心代码解析

剪枝时，`up_proj` 和 `gate_proj` 保留每个簇的 anchor 行：

```python
new_up_proj.weight.copy_(old_up_proj.weight.index_select(0, anchors))
new_gate_proj.weight.copy_(old_gate_proj.weight.index_select(0, anchors))
```

解释：

- intermediate neuron 被分簇后，每簇只保留一个 anchor neuron。
- `up_proj/gate_proj` 的输出维度从原 intermediate size 缩小为 anchor 数量。

`down_proj` 对簇内列求和：

```python
for new_idx, cluster in enumerate(layer_clusters):
    members = torch.tensor(cluster["neuron"], device=device, dtype=torch.long)
    new_down_proj.weight[:, new_idx] = old_down_proj.weight.index_select(1, members).sum(dim=1)
```

解释：

- 原模型中同簇 neuron 的输出都会进入 `down_proj`。
- 剪枝后只保留 anchor，但为了近似原输出，需要把同簇成员对应的 `down_proj` 列合并。

最后更新配置中的 intermediate size：

```python
model.config.intermediate_size = target_intermediate_size
model.config.text_config.intermediate_size = target_intermediate_size
```

解释：

- 保存为 HuggingFace 模型时，config 必须和实际线性层形状一致。
- 当前实现要求所有 MLP 层使用统一剪枝比例，否则 config 无法简单表达每层不同 intermediate size。

## 9. 早期 0.50 剪枝后 yes/no 结果

本节记录的是早期 0.50 配置结果。该结果已被第 13 节的 v2/v3 服务器实验更新覆盖；最新结论以第 13 节为准。

早期 `vulcan-pruned` 的 yes/no 评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.49800796812749004,
  "normalized_exact_match": 0.49800796812749004,
  "token_f1": 0.49800796812749004,
  "yesno_examples": 251,
  "yesno_accuracy": 0.49800796812749004,
  "yesno_prediction_coverage": 0.9960159362549801,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 18,
    "other": 1,
    "yes": 232
  },
  "yesno_confusion": {
    "no->no": 13,
    "no->yes": 120,
    "yes->no": 5,
    "yes->other": 1,
    "yes->yes": 112
  }
}
```

对比表：

| 模型 | keep ratio | yes/no accuracy | prediction coverage |
| --- | ---: | ---: | ---: |
| baseline SFT | 无剪枝 | 0.6733 | 0.9880 |
| 早期 Vulcan SFT 未剪枝 | 无剪枝 | 0.6733 | 0.9960 |
| 早期 Vulcan pruned | 0.50 | 0.4980 | 0.9960 |

关键观察：

- 0.50 剪枝后 accuracy 从 `0.6733` 降到 `0.4980`，掉点明显。
- coverage 仍为 `0.9960`，说明模型仍按 yes/no 形式回答，并不是格式崩坏。
- 混淆矩阵显示模型严重偏向 `yes`：251 条中预测 `yes` 232 条。
- `no` 类被破坏最严重：真实 `no` 的 133 条中有 120 条被预测成 `yes`。

当时结论：

> 在当前 lambda 和训练轮数下，`keep_ratio=0.50` 对 VQA-RAD yes/no 子集过于激进。Collapse Loss 没有破坏未剪枝模型，但尚不足以让 50% MLP 剪枝保持判断能力。

更新：第 13 节的 v2/v3 结果显示，改用 `collapse_use_weight_proxy=false`、`learning_rate=1e-4`、6 epoch，并在 v3 中使用 plain SGD lambda 后，0.50 剪枝可以达到剪枝零损失。

## 10. 0.75 剪枝实验结果

由于早期 `keep_ratio=0.50` 掉点明显，当时继续做了更保守的 `keep_ratio=0.75` 对照。

### 10.1 0.75 聚类、训练与剪枝命令

生成 0.75 聚类：

```bash
PYTHONPATH=src python scripts/vulcan/collect_cluster_idx.py \
  --config examples/vulcan/qwen35_08b_vqa_rad_full_sft.yaml \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_75.json \
  --keep_ratio 0.75 \
  --max_batches 200 \
  --num_workers 0 \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/vulcan-exper/saves/qwen35-0_8b-vqa-rad/full/sft \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  output_dir=/tmp/cluster-idx \
  deepspeed=null \
  dataloader_num_workers=0
```

用 0.75 cluster 训练：

```bash
WANDB_DISABLED=true torchrun --nproc_per_node=1 --master_port=29501 src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_vulcan_sft.yaml \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/vulcan-exper/saves/qwen35-0_8b-vqa-rad/full/sft \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  collapse_cluster_idx_path=saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_75.json \
  output_dir=saves/qwen35-0_8b-vqa-rad/full/vulcan-sft-0_75
```

训练过程中曾在 `585/675` 附近收到 `SIGHUP` 中断：

```text
SignalException: Process 4174 got signal: 1
```

这类中断通常不是模型代码报错，而是 SSH / AutoDL 终端断开、前台 shell 被关闭或会话被系统回收。由于当时已有 `checkpoint-400`，后续通过 checkpoint 恢复：

```bash
WANDB_DISABLED=true torchrun --nproc_per_node=1 --master_port=29502 src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_vulcan_sft.yaml \
  model_name_or_path=/root/autodl-tmp/.autodl/lyt/vulcan-exper/saves/qwen35-0_8b-vqa-rad/full/sft \
  dataset_dir=/root/autodl-tmp/.autodl/lyt/datasets/vqa_rad \
  collapse_cluster_idx_path=saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_75.json \
  output_dir=saves/qwen35-0_8b-vqa-rad/full/vulcan-sft-0_75 \
  resume_from_checkpoint=saves/qwen35-0_8b-vqa-rad/full/vulcan-sft-0_75/checkpoint-400 \
  overwrite_output_dir=false
```

后续长训练建议放在 tmux 中运行：

```bash
tmux new -s vulcan075
# 启动训练
# Ctrl+b 后按 d 可 detach，训练继续
tmux attach -t vulcan075
```

剪枝：

```bash
python scripts/vulcan/save_pruned_model.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-sft-0_75 \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_75.json \
  --output_dir saves/qwen35-0_8b-vqa-rad/full/vulcan-pruned-0_75 \
  --template qwen3_5_nothink \
  --trust_remote_code
```

### 10.2 0.75 剪枝后 yes/no 评估

`vulcan-pruned-0_75` 的 yes/no 评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.6454183266932271,
  "normalized_exact_match": 0.6454183266932271,
  "token_f1": 0.6454183266932271,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6454183266932271,
  "yesno_prediction_coverage": 0.9960159362549801,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 81,
    "other": 1,
    "yes": 169
  },
  "yesno_confusion": {
    "no->no": 63,
    "no->yes": 70,
    "yes->no": 18,
    "yes->other": 1,
    "yes->yes": 99
  }
}
```

### 10.3 Baseline 直接 0.75 剪枝对照

为了判断 Collapse Loss 是否真的提升了剪枝鲁棒性，又使用同一个 `cluster_idx_greedy_match_0_75.json` 直接剪 baseline SFT，得到 `baseline-pruned-0_75`。

评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.6573705179282868,
  "normalized_exact_match": 0.6573705179282868,
  "token_f1": 0.6573705179282868,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6573705179282868,
  "yesno_prediction_coverage": 0.9880478087649402,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 60,
    "other": 3,
    "yes": 188
  },
  "yesno_confusion": {
    "no->no": 54,
    "no->other": 2,
    "no->yes": 77,
    "yes->no": 6,
    "yes->other": 1,
    "yes->yes": 111
  }
}
```

更新后的对比表：

| 模型 | keep ratio | yes/no accuracy | prediction coverage | 备注 |
| --- | ---: | ---: | ---: | --- |
| baseline SFT | 无剪枝 | 0.6733 | 0.9880 | 主基线 |
| 早期 Vulcan SFT 未剪枝 | 无剪枝 | 0.6733 | 0.9960 | 正则训练未掉点 |
| 早期 Vulcan pruned | 0.50 | 0.4980 | 0.9960 | 掉点明显，严重偏向 `yes` |
| Vulcan pruned | 0.75 | 0.6454 | 0.9960 | 只比 baseline 低约 2.79 个百分点 |
| baseline pruned | 0.75 | 0.6574 | 0.9880 | 不经 Collapse Loss，略高于 Vulcan 0.75 |

关键观察：

- `keep_ratio=0.75` 相比 baseline 从 `0.6733` 降到 `0.6454`，下降约 `2.79` 个百分点。
- baseline 直接 0.75 剪枝为 `0.6574`，比 Vulcan 0.75 剪枝的 `0.6454` 高约 `1.20` 个百分点。
- 与 `keep_ratio=0.50` 的 `0.4980` 相比，0.75 剪枝显著更稳。
- Vulcan 0.75 剪枝后仍有一定 `yes` 偏置：预测 `yes` 169 条、预测 `no` 81 条。
- baseline 0.75 剪枝也存在 `yes` 偏置，且预测 `yes` 更多：188 条。
- 从 `no` 类看，Vulcan 0.75 的 `no->yes` 为 70 条，baseline 0.75 的 `no->yes` 为 77 条；Vulcan 对 `no` 类略好，但对 `yes` 类更差。

阶段结论：

> 当前方法在 `keep_ratio=0.75` 下可以保留大部分 yes/no 能力，但与 baseline 直接 0.75 剪枝相比，当前 Collapse Loss 配置没有带来 yes/no accuracy 收益。`keep_ratio=0.50` 对当前任务、模型和正则强度过于激进。

更新：该阶段结论是 v2/v3 之前的历史判断。第 13 节显示，在最终 v3 配置下，0.50 剪枝本身已经可以做到零损失，新的主要问题转为 Collapse SFT 的未剪枝性能代价。

### 10.4 从零初始化 learnable lambda 对照实验

为了对齐原始实验指导中的“可学习 lambda，初始为 0”设定，又补充了一组 `keep_ratio=0.75` 的 learnable lambda 对照。该实验仍使用 uniform `cluster_idx_greedy_match_0_75.json`，不做分层聚类或分层剪枝。注意：本节记录的是旧 optimizer 方案下的负 lambda 问题，不代表第 13 节 v3 的 plain SGD 梯度上升方案。

配置差异：

```yaml
collapse_lambda1: 0.0
collapse_lambda2: 0.0
collapse_learnable_lambda: true
```

训练输出目录：

```text
saves/qwen35-0_8b-vqa-rad/full/vulcan-sft-0_75-learnable
```

剪枝输出目录：

```text
saves/qwen35-0_8b-vqa-rad/full/vulcan-pruned-0_75-learnable
```

yes/no 评估结果：

```json
{
  "num_examples": 251,
  "exact_match": 0.6653386454183267,
  "normalized_exact_match": 0.6653386454183267,
  "token_f1": 0.6653386454183267,
  "yesno_examples": 251,
  "yesno_accuracy": 0.6653386454183267,
  "yesno_prediction_coverage": 0.9880478087649402,
  "yesno_label_counts": {
    "no": 133,
    "yes": 118
  },
  "yesno_prediction_counts": {
    "no": 60,
    "other": 3,
    "yes": 188
  },
  "yesno_confusion": {
    "no->no": 55,
    "no->other": 2,
    "no->yes": 76,
    "yes->no": 5,
    "yes->other": 1,
    "yes->yes": 112
  }
}
```

与已有 0.75 结果对比：

| 模型 | keep ratio | yes/no accuracy | prediction coverage | 备注 |
| --- | ---: | ---: | ---: | --- |
| baseline SFT | 无剪枝 | 0.6733 | 0.9880 | 主基线 |
| baseline pruned | 0.75 | 0.6574 | 0.9880 | 不经 Collapse Loss |
| Vulcan pruned 固定 lambda | 0.75 | 0.6454 | 0.9960 | 固定小 lambda |
| Vulcan pruned learnable lambda | 0.75 | 0.6653 | 0.9880 | lambda 从 0 学习，但最终变为负值 |

训练日志中最后阶段的 lambda 与 collapse loss：

```text
step=600 collapse_loss=-1810.9670 lambda1=-0.0033722 lambda2=-0.0034027
step=650 collapse_loss=-1812.0444 lambda1=-0.0033722 lambda2=-0.0034027
step=675 collapse_loss=-1812.0928 lambda1=-0.0033722 lambda2=-0.0034027
```

关键判断：

- 该组剪枝后 yes/no accuracy 为 `0.6653`，比 baseline 直接 0.75 剪枝的 `0.6574` 高约 `0.80` 个百分点，折算到 251 条样本约为 2 条样本差异。
- 但 learnable lambda 最终变为负值，`collapse_loss` 也变为大幅负数，说明训练目标已经从“拉近簇内权重”变成了“奖励簇内距离变大”。
- 原因是 `L_total = L_sft + lambda * D`，其中簇内距离 `D >= 0`。如果 lambda 作为普通参数参与梯度下降，则 `dL/dlambda = D > 0`，优化器会自然把 lambda 往负方向推。
- 因此这组结果不能作为 Vulcan Collapse Loss 生效的证据，只能说明“裸 learnable lambda 从 0 初始化”这个设定在当前实现下不成立。

后续处理：

- 该旧对照不再作为主线。
- 最新主线使用裸 learnable lambda + `collapse_lambda_lr=-1.0` + plain SGD 梯度上升，避免普通梯度下降把 lambda 推成负值。
- 是否还需要固定正系数或手动 schedule，可作为降低未剪枝性能代价的后续对照，而不是当前 0.50 剪枝能否成立的前置条件。

## 11. 评估脚本指标解析

`eval_vqa_predictions.py` 会对 `generated_predictions.jsonl` 逐行读取：

```python
prediction = record["predict"].strip()
label = record["label"].strip()
```

文本归一化：

```python
def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())
```

yes/no 提取：

```python
def to_yesno(text: str, strict: bool = False) -> str | None:
    normalized = normalize_answer(text)
    if normalized in {"yes", "no"}:
        return normalized

    if strict:
        return None

    tokens = normalized.split()
    if tokens and tokens[0] in {"yes", "no"}:
        return tokens[0]

    return None
```

解释：

- label 使用 `strict=True`，只有严格答案为 `yes/no` 才算 yes/no 样本。
- prediction 允许以 `yes` 或 `no` 开头，用于兼容模型输出 `yes, ...` 这类短解释。
- `yesno_prediction_coverage` 衡量模型是否按 yes/no 形式回答。
- `yesno_confusion` 用来观察模型是否偏向某一个类别。

## 12. 当前问题与已修复工程点

### 12.1 `llamafactory` import 问题

直接运行：

```bash
python scripts/vulcan/collect_cluster_idx.py
```

曾出现：

```text
ModuleNotFoundError: No module named 'llamafactory'
```

原因是源码包位于 `src/llamafactory`。已在脚本中加入：

```python
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

涉及脚本：

```text
scripts/vulcan/collect_cluster_idx.py
scripts/vulcan/save_pruned_model.py
scripts/vulcan/inspect_model_redundancy.py
```

### 12.2 DeepSpeed 校验问题

聚类脚本读取训练 YAML 时，如果 YAML 中有：

```yaml
deepspeed: examples/deepspeed/ds_z2_config.json
```

会触发：

```text
ValueError: Please use `FORCE_TORCHRUN=1` to launch DeepSpeed training.
```

聚类不是训练，不需要 DeepSpeed，因此命令行覆盖：

```bash
deepspeed=null
```

### 12.3 torchrun 端口占用

曾出现：

```text
EADDRINUSE, address already in use
```

原因是默认端口 `29500` 被占。解决方式：

```bash
torchrun --nproc_per_node=1 --master_port=29501 ...
```

`master_port` 只影响 torchrun 通信，不影响训练结果。

### 12.4 Collapse Loss 训练过慢

最初 0.50 Vulcan SFT 曾出现：

```text
6/675 [20:46<38:34:40, 207.59s/it]
```

判断为 Collapse Loss 逐簇 Python 循环导致大量小 tensor 操作。已改为按层向量化并缓存 index tensor。

### 12.5 前台训练被 SIGHUP 中断

0.75 Vulcan SFT 曾在训练后段出现：

```text
torch.distributed.elastic.multiprocessing.api.SignalException: Process got signal: 1
```

该问题通常来自终端断开或前台 shell 被关闭，而不是训练代码异常。解决方式：

- 使用 `tmux` 或 `screen` 承载长训练。
- 设置较合理的 `save_steps`，避免中断后丢失太多 step。
- 若已有 checkpoint，使用 `resume_from_checkpoint=<checkpoint-path>` 恢复。

## 13. 0.50 最新服务器实验：v2/v3

### 13.1 实验配置对比

最新 0.50 实验仍使用 yes/no 子集：

- 训练集：`vqa_rad_train_yesno`，940 条样本。
- 测试集：`vqa_rad_test_yesno`，251 条样本。
- 聚类：Greedy Match，`3584 -> 1792`，特征为 `concat(up_proj.weight, gate_proj.weight)` 行向量，anchor 为簇内激活最大 neuron。

Vulcan 训练配置对比：

| 配置项 | v1（失败） | v2 | v3（最终推荐） |
| --- | --- | --- | --- |
| `collapse_use_weight_proxy` | `true` | `false` | `false` |
| lambda 优化器 | AdamW | AdamW | plain SGD（梯度上升） |
| `collapse_lambda_lr` | `-1.0` | `-1.0` | `-1.0` |
| 主学习率 | `5e-6` | `1e-4` | `1e-4` |
| `num_train_epochs` | `3` | `6` | `6` |
| `gradient_accumulation_steps` | `8` | `4` | `4` |
| 训练趋势 | loss 持续上升 | 先升后降 | 先升后降 |
| lambda 最终值 | `149`，失控 | `69.65` | `55.64` |

当前推荐配置已经写入 `examples/vulcan/qwen35_08b_vqa_rad_vulcan_sft.yaml`：

```yaml
use_collapse_loss: true
collapse_lambda1: 0.0
collapse_lambda2: 0.0
collapse_learnable_lambda: true
collapse_lambda_lr: -1.0
collapse_use_weight_proxy: false
learning_rate: 1.0e-4
gradient_accumulation_steps: 4
num_train_epochs: 6.0
```

关键工程判断：

- `collapse_use_weight_proxy=false` 是当前 VQA-RAD 服务器实验的必要设置。`weight_proxy=true` 的 v1 没有形成稳定有效的 MLP 冗余，训练 loss 持续上升，lambda 最终到 `149`。
- `collapse_lambda_lr=-1.0` 与自定义 plain SGD lambda optimizer 配合，用梯度上升让 lambda 随 collapse loss 梯度增大。
- 主学习率 `1e-4` 与 lambda lr 保持足够大的比例差异后，v2/v3 训练从“持续上升”变为“先升后降”。
- v3 的 plain SGD 去掉 AdamW 动量和自适应项后，lambda 最终值从 v2 的 `69.65` 降到 `55.64`，训练更稳。

### 13.2 Yes/No 准确率

| 模型 | keep ratio | yes/no accuracy | vs baseline SFT | 备注 |
| --- | ---: | ---: | ---: | --- |
| baseline SFT | 无剪枝 | `0.6733` | - | 主基线 |
| baseline 0.50 剪枝 | `0.50` | `0.4980` | `-0.1753` | 直接剪枝掉 `17.5` 个百分点 |
| vulcan-sft-v2 未剪枝 | 无剪枝 | `0.6135` | `-0.0598` | AdamW lambda |
| vulcan-sft-v2 0.50 剪枝 | `0.50` | `0.6135` | `-0.0598` | 剪枝零损失 |
| vulcan-sft-v3 未剪枝 | 无剪枝 | `0.6056` | `-0.0677` | plain SGD lambda |
| vulcan-sft-v3 0.50 剪枝 | `0.50` | `0.6056` | `-0.0677` | 剪枝零损失 |

核心对比：

- baseline 直接 0.50 剪枝：`0.6733 -> 0.4980`，掉 `17.53` 个百分点。
- Vulcan v3 0.50 剪枝：`0.6056 -> 0.6056`，剪枝零损失。
- Vulcan v3 剪枝模型比 baseline 直接 0.50 剪枝高 `10.76` 个百分点。
- Vulcan v3 剪枝模型仍比 baseline SFT 未剪枝低 `6.77` 个百分点，说明当前主要代价发生在 Collapse SFT 阶段，而不是剪枝阶段。

### 13.3 混淆矩阵

baseline SFT 未剪枝：

```text
no->no: 89,  no->yes: 44
yes->no: 35, yes->yes: 83
```

baseline 0.50 剪枝：

```text
no->no: 13,  no->yes: 120
yes->no: 5,  yes->yes: 112
```

baseline 直接剪枝后严重偏向 `yes`，真实 `no` 的 133 条中有 120 条被预测为 `yes`。

vulcan-sft-v3 0.50 剪枝：

```text
no->no: 85,  no->yes: 48
yes->no: 51, yes->yes: 67
```

Vulcan v3 剪枝后显著缓解了直接剪枝造成的 `yes` 偏置：`no->yes` 从 120 降到 48。代价是 `yes` 类召回下降，`yes->no` 从 baseline 未剪枝的 35 上升到 51。

### 13.4 阶段结论

当前 0.50 实验说明：

- Collapse Loss 确实提升了 MLP 结构化剪枝鲁棒性。
- “剪枝零损失”已经在 v2/v3 两组配置上复现。
- 与 baseline 直接剪枝相比，Vulcan v3 0.50 剪枝模型收益明显。
- 与 baseline 未剪枝相比，Vulcan v3 仍有约 `6-7` 个百分点任务性能代价。

因此当前下一步不应再只问“0.50 能不能剪”，而应围绕两个问题推进：

- 用 redundancy 统计证明剪枝零损失来自簇内权重坍缩。
- 降低 Collapse SFT 对未剪枝 yes/no accuracy 的副作用。

## 14. 冗余分析命令

本地仓库没有服务器上的 `saves/` 模型权重目录，因此 redundancy 统计需要在服务器上执行。建议用同一个 `cluster_idx_greedy_match_0_50_yesno.json` 分别检查 baseline SFT 和 vulcan-sft-v3：

```bash
python scripts/vulcan/inspect_model_redundancy.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/sft \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_yesno.json \
  --template qwen3_5_nothink \
  --trust_remote_code \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/redundancy_baseline_sft_0_50_yesno.json

python scripts/vulcan/inspect_model_redundancy.py \
  --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/vulcan-sft-v3 \
  --cluster_idx_path saves/qwen35-0_8b-vqa-rad/vulcan/cluster_idx_greedy_match_0_50_yesno.json \
  --template qwen3_5_nothink \
  --trust_remote_code \
  --output_path saves/qwen35-0_8b-vqa-rad/vulcan/redundancy_vulcan_sft_v3_0_50_yesno.json
```

期望现象：

- `redundancy_vulcan_sft_v3_0_50_yesno.json` 的 `mean_l1`、`mean_l2` 应明显低于 baseline。
- `redundancy_vulcan_sft_v3_0_50_yesno.json` 的 `mean_cosine` 应高于 baseline。
- 若该趋势成立，可支撑“Vulcan v3 的剪枝零损失来自 Collapse Loss 构造出的簇内权重冗余，而不是 yes/no 指标偶然波动”。

### 14.1 建议继续补充

- 完整 VQA-RAD test split 指标：baseline SFT、baseline pruned 0.50、vulcan-sft-v3、vulcan-pruned-v3 0.50。
- 剪枝前后参数量与模型目录大小。
- v2/v3 的 `trainer_log.jsonl` 中 lambda、collapse loss、main loss 曲线。
- yes/no 分类报告，重点观察 `no->yes` 偏置和 `yes->no` 代价之间的权衡。
