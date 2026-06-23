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

# Vulcan 在 VQA-Med 三分类任务上的实验记录

> 记录时间：2026 年 6 月  
> 当前任务：VQA-Med 2019 的 modality、plane、organ system 三类问题  
> 基础模型：Qwen3.5-0.8B  
> 目标：验证 Vulcan 在保留约 50% MLP 神经元时的任务性能与实际剪枝效果

## 1. 实验范围

VQA-Med 原始任务包含多种医学视觉问答类型。本轮实验只训练官方论文中适合视为分类任务的前三类：

1. `modality`：判断成像模态。
2. `plane`：判断图像切面或视角。
3. `organ system`：判断器官或解剖系统。

本轮不包含 `abnormality`。原因是前三类的答案空间相对封闭，适合用统一的标签生成方式训练和用 exact match 评估；异常描述更接近开放式生成，和当前分类实验的目标不同。

## 2. 数据转换与标签策略

### 2.1 转换后格式

数据转换为 ShareGPT 多模态格式，每条样本包含一轮问题、一个标准答案以及一张图像：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>what type of imaging modality is used to acquire the image?\nAnswer with the exact label only."
    },
    {
      "role": "assistant",
      "content": "us - ultrasound"
    }
  ],
  "images": ["images/train_000008.png"],
  "category": "modality"
}
```

训练只依赖 `messages` 和 `images`。`category` 是后续分类型统计和切分验证集所用的辅助字段，不作为模型输入。`raw_answer` 只用于保留清洗前答案、排查标签映射问题；训练不依赖该字段，因此不是必需项。

### 2.2 Prompt 策略

统一使用：

```text
<image>{original_question}
Answer with the exact label only.
```

这样做的目的不是让模型生成解释，而是把多模态生成约束成封闭标签预测。推理时必须继续使用与训练相同的 prompt，避免因输出格式漂移降低 exact match。

### 2.3 标签策略

- 保留数据集的细粒度标准标签，例如 `us - ultrasound`，不自行改成更短的同义词。
- 只做必要的大小写、空格和标点规范化，不合并医学意义不同的标签。
- 评估以规范化后的完整标签 exact match 为主，token F1 为辅。
- `yes/no` 是 modality 子集中的一种答案形式，不是整个三分类数据集的主体。

### 2.4 生成的数据文件

转换脚本：`scripts/vulcan/convert_vqa_med.py`

主要数据集条目：

| 数据集名称 | 用途 |
| --- | --- |
| `vqa_train_cls` | modality、plane、organ 合并训练集 |
| `vqa_val_cls` | 三类合并验证集 |
| `vqa_val_modality` | modality 单类验证集 |
| `vqa_val_plane` | plane 单类验证集 |
| `vqa_val_organ` | organ system 单类验证集 |

验证集共 1501 条：

| 类别 | 样本数 | 占比 |
| --- | ---: | ---: |
| modality | 574 | 38.24% |
| plane | 500 | 33.31% |
| organ system | 427 | 28.45% |

## 3. Baseline 训练

### 3.1 主要配置

配置文件：`examples/vulcan/qwen35_08b_vqa_med_cls_full_sft.yaml`

关键参数：

| 参数 | 设置 |
| --- | --- |
| 模型 | Qwen3.5-0.8B |
| 微调方式 | Full SFT |
| 视觉编码器 | 冻结 |
| Projector | 可训练 |
| Language model | 可训练 |
| 模板 | `qwen3_5_nothink` |
| 单卡 batch size | 1 |
| 梯度累积 | 8 |
| 学习率 | `3e-6` |
| 调度器 | cosine |
| warmup ratio | 0.05 |
| 初始训练长度 | 4 epochs |
| 评估与保存间隔 | 100 steps |

训练后期提升已经很小，因此提前终止训练，并从归档 checkpoint 中选择验证集表现最好的模型。

### 3.2 Checkpoint 对比

| Checkpoint | Exact Match | Token F1 | Yes/No Acc. | 结论 |
| --- | ---: | ---: | ---: | --- |
| 900 | 76.95% | 79.28% | 88.30% | 尚未充分收敛 |
| 1200 | **78.61%** | **80.59%** | **88.83%** | 最佳 baseline |
| 1900 | 77.88% | 80.00% | 88.83% | 后续训练没有继续提升 |

最终选择：

```text
saves/qwen35-0_8b-vqa-med-cls/checkpoint-archive-1200
```

这说明该任务不应只根据最后一个 checkpoint 选模型。对 0.8B 模型而言，约 1200 steps 已达到当前训练设置下的较优点，继续训练出现了轻微回落。

### 3.3 Baseline 分类别结果

| 类别 | 样本数 | Exact Match | Token F1 | 正确数 |
| --- | ---: | ---: | ---: | ---: |
| modality | 574 | 79.27% | 82.64% | 455 |
| plane | 500 | **80.00%** | 80.18% | 400 |
| organ system | 427 | 76.11% | 78.30% | 325 |
| 总体 | 1501 | **78.61%** | **80.59%** | 1180 |

类别宏平均准确率约为 78.46%，与按样本加权的总体准确率 78.61% 接近，说明总体结果没有被某一个大类别完全主导。

观察：

- `plane` 最稳定，exact match 最高。
- `organ system` 是当前主要短板。
- `modality` 的 token F1 明显高于 exact match，表明存在部分细粒度标签接近但没有完全匹配的预测。
- modality 中共有 188 个 yes/no 样本，baseline 的 yes/no accuracy 为 88.83%。

## 4. Vulcan 聚类索引生成

聚类以最佳 baseline checkpoint 1200 为起点，并保留约 50% 神经元：

```text
saves/qwen35-0_8b-vqa-med-cls/vulcan/cluster_idx_greedy_match_0_50_ckpt1200.json
```

主要设置：

| 参数 | 设置 |
| --- | --- |
| 起始模型 | baseline checkpoint 1200 |
| keep ratio | 0.5 |
| max batches | 600 |
| batch size | 1 |
| shuffle | 开启 |
| seed | 42 |

训练集文件按 modality、plane、organ 顺序拼接。如果在 `max_batches=600` 时不 shuffle，激活采样会明显偏向文件前部的 modality 样本。因此聚类采样必须打乱，并固定随机种子以便复现。

参考命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vulcan/collect_cluster_idx.py \
  --config examples/vulcan/qwen35_08b_vqa_med_cls_full_sft.yaml \
  --output_path saves/qwen35-0_8b-vqa-med-cls/vulcan/cluster_idx_greedy_match_0_50_ckpt1200.json \
  --keep_ratio 0.5 \
  --max_batches 600 \
  --batch_size 1 \
  --num_workers 4 \
  --shuffle \
  --seed 42 \
  model_name_or_path="$(pwd)/saves/qwen35-0_8b-vqa-med-cls/checkpoint-archive-1200" \
  dataset_dir=datasets/vqa_med \
  dataset=vqa_train_cls \
  deepspeed=null
```

这里显式设置 `deepspeed=null`，是因为聚类脚本是单进程模型分析，不需要从训练 YAML 继承 DeepSpeed；否则参数解析器会要求使用 `FORCE_TORCHRUN=1`。

## 5. Vulcan 实验一：lr=1e-5，rho=0.1

### 5.1 配置

本轮从 baseline checkpoint 1200 继续训练：

| 参数 | 设置 |
| --- | --- |
| keep ratio | 0.5 |
| 主任务学习率 | `1e-5` |
| lambda 学习率 | `-0.1` |
| lambda 初值 | 0 |
| lambda 是否可学习 | 是 |
| collapse 权重形式 | 直接权重，未使用 proxy |
| 单卡 batch size | 1 |
| 梯度累积 | 4 |
| max steps | 2188 |
| 调度器 | constant |

`2188` 来自此前 VQA-RAD 实验采用的步数公式：

```text
total_steps = 6250 * R^2 + 1250 * R
```

当 `R=0.5` 时，得到约 2188 steps。

模型目录：

```text
saves/qwen35-0_8b-vqa-med-cls/full/vulcan-from-ckpt1200-keep050-lr1e5-lam01
```

### 5.2 剪枝前结果

| 指标 | Baseline ckpt 1200 | Vulcan 剪枝前 | 变化 |
| --- | ---: | ---: | ---: |
| Exact Match | 78.61% | 74.62% | -4.00 pp |
| Token F1 | 80.59% | 76.85% | -3.74 pp |
| Yes/No Acc. | 88.83% | 71.28% | -17.55 pp |

Yes/no 混淆矩阵：

| Gold -> Pred | 数量 |
| --- | ---: |
| no -> no | 81 |
| no -> yes | 32 |
| yes -> no | 22 |
| yes -> yes | 53 |

三分类总体准确率仍有 74.62%，但 yes/no 能力明显退化，说明 collapse 约束不只是带来均匀的小幅损失，也改变了模型在二元标签上的决策边界。

### 5.3 实际剪枝后结果

对本轮 Vulcan 模型执行物理剪枝后：

| 指标 | Vulcan 剪枝前 | Vulcan 剪枝后 | 变化 |
| --- | ---: | ---: | ---: |
| Exact Match | 74.62% | **74.62%** | **0.00 pp** |
| Token F1 | 76.85% | 77.10% | +0.25 pp |
| Yes/No Acc. | 71.28% | 75.00% | +3.72 pp |

剪枝后 yes/no 混淆矩阵：

| Gold -> Pred | 数量 |
| --- | ---: |
| no -> no | 79 |
| no -> yes | 34 |
| yes -> no | 13 |
| yes -> yes | 62 |

这是当前实验最重要的正向结果：**在 Vulcan 已经完成 collapse 训练后，真正移除约 50% 目标神经元没有造成额外 exact-match 损失。** Token F1 和 yes/no accuracy 还有小幅提升。

但需要区分两个问题：

1. `Vulcan 训练模型 -> Vulcan 剪枝模型`：当前观察到零 exact-match 损失，说明 collapse 与剪枝映射是有效的。
2. `Baseline -> Vulcan 剪枝模型`：仍下降约 4.00 个百分点，说明训练阶段的任务保持能力还有改进空间。

## 6. Vulcan 实验二：lr=1e-4，rho=1

为了更接近论文表格中的 AdamW、`lr=1e-4`、`rho=1` 设置，进行了更激进的一轮训练。

### 6.1 结果

| 模型点 | Exact Match | Token F1 | Yes/No Acc. | 主要现象 |
| --- | ---: | ---: | ---: | --- |
| checkpoint 1200 | 52.43% | 54.58% | 39.89% | 188 个 yes/no 全部预测为 yes |
| checkpoint 1500 | 63.62% | 66.12% | 65.43% | 几乎全部预测为 no |
| final | 66.29% | 68.31% | 65.43% | 偏向 yes，仍明显低于实验一 |

最终模型的 yes/no 预测中，140 个为 yes、48 个为 no：

| Gold -> Pred | 数量 |
| --- | ---: |
| no -> no | 48 |
| no -> yes | 65 |
| yes -> yes | 75 |

### 6.2 训练日志诊断

训练早期 collapse loss 远大于 SFT loss：

| Step | SFT loss | Collapse loss | Collapse/SFT | lambda 1 |
| --- | ---: | ---: | ---: | ---: |
| 10 | 1.0988 | 4722 | 4297.6 | 8.72 |
| 20 | 2.7230 | 8483 | 3115.0 | 18.48 |
| 50 | 0.3159 | 9466 | 29970.0 | 42.49 |
| 100 | 0.4870 | 3299 | 6774.0 | 57.71 |
| 300 | 0.3910 | 152.35 | 389.6 | 63.72 |
| 1200 | 0.4770 | 71.25 | 149.5 | 66.05 |
| 1500 | 0.9180 | 70.23 | 76.5 | 66.70 |

最终 lambda 约增长到 68.14。约束从训练开始就压倒任务损失，模型随后在“全 yes”和“几乎全 no”的标签先验之间振荡。因此这不是普通的训练过拟合，而是增强拉格朗日约束尺度与当前 0.8B VLM 任务不匹配造成的优化失衡。

结论：**`lr=1e-4 + rho=1` 不适合当前 VQA-Med/Qwen3.5-0.8B 设置，不应继续作为主配置。**

## 7. 与 Vulcan 论文和技术文档的对照

参考材料：

- `ICLR2026_Vulcan_camera_ready.pdf`
- `Model+Pruning+技术文档.pdf`

论文中的典型设置包括：

- 步数：`6250R^2 + 1250R`。
- AdamW，识别任务 batch size 256。
- 学习率 `1e-4`、constant scheduler、weight decay 0.05。
- `rho=1`、seed 42。
- 论文算法按 batch 更新 anchor，并按激活统计自适应分配各层保留比例。

当前 LLM/VLM 扩展与原论文存在几项重要差异：

- 当前只对语言模型 MLP 做 collapse 和剪枝。
- 当前聚类完成后使用固定 anchor，而不是每个 batch 动态更新。
- 当前使用统一 `keep_ratio=0.5`，不是论文中的逐层自适应预算。
- 当前实际 micro batch 很小；即使增加梯度累积，也不等同于使用更大 batch 估计当前 batch 的 collapse loss。
- 当前 collapse loss 是按 cluster/counter 归一的直接求和形式，其绝对尺度与分类论文实验不完全一致。

因此，论文中的 `lr=1e-4` 和 `rho=1` 不能脱离 batch、loss 归一化、anchor 更新方式直接照搬。实验二的失败也印证了这一点。

## 8. 当前结论

1. VQA-Med 前三类适合当前封闭标签 SFT 实验，baseline checkpoint 1200 的 78.61% 是可靠起点。
2. 0.8B 模型在该任务上的主要困难是 organ system 和 modality 的细粒度标签，不是单纯 yes/no。
3. `keep_ratio=0.5` 的 Vulcan 实验一在训练后保留 74.62% 准确率，相对 baseline 下降 4.00 pp。
4. 实验一物理剪枝前后 exact match 完全一致，说明训练得到的冗余结构可以被实际删除，这是当前最有价值的结果。
5. `lr=1e-4 + rho=1` 使约束从早期就压倒任务损失，引发标签塌缩，性能显著恶化。
6. 多模态 cluster 实验在 checkpoint 1800 达到 75.68% exact match，物理剪枝后为 75.55%，仅下降 0.13 pp。
7. 多模态剪枝模型相较旧 Vulcan 剪枝模型提升 0.93 pp exact match、1.11 pp token F1 和 9.57 pp yes/no accuracy。
8. 当前不能只用最终准确率评价 Vulcan。至少要同时报告：baseline、Vulcan 剪枝前、Vulcan 剪枝后，以及参数量或模型大小变化。

## 9. 尚未完成的关键对照

### 9.1 Baseline 直接剪枝

此前使用旧 cluster 对 baseline checkpoint 1200 直接执行 50% FFN 剪枝时，exact accuracy 约为 0.03%，模型基本失效。这说明不经过 collapse 训练直接移除一半 FFN 神经元不可行。

为了与第 11 节形成严格受控对照，仍需使用同一个 `cluster_idx_multimodal.json` 直接剪枝 baseline checkpoint 1200 并重新评估：

```text
Baseline ckpt 1200
  -> 直接按 cluster/anchor 剪枝
  -> vqa_val_cls
```

该结果可以区分性能收益来自多模态 cluster 本身，还是来自后续 collapse 训练。旧 cluster 的直接剪枝结果已经证明 collapse 是必要步骤，但不能替代同 cluster 对照。

### 9.2 分类别剪枝结果

Vulcan 剪枝模型还应分别评估：

- `vqa_val_modality`
- `vqa_val_plane`
- `vqa_val_organ`

重点判断 baseline 到 Vulcan 的 4 pp 损失主要来自哪一类，并检查 yes/no 退化是否只集中在 modality。

### 9.3 历史候选配置

当前配置文件中准备的后续候选为：

| 参数 | 候选设置 |
| --- | --- |
| 主任务学习率 | `1e-5` |
| lambda 学习率 | `-0.1` |
| 梯度累积 | 16 |
| weight decay | 0.05 |
| max steps | 2188 |
| scheduler | constant |
| seed | 42 |
| 保存与评估间隔 | 200 |

该配置是多模态实验之前记录的候选方案，其中梯度累积 16 未用于第 11 节正式实验。第 11 节最终使用梯度累积 4，并恢复旧 Vulcan 实验一的完整训练动力学，以确保主要变量仅为 cluster 构造方式。

## 10. 评估命令模板

预测沿用项目的 `torchrun + src/train.py` 入口：

```bash
WANDB_DISABLED=true torchrun \
  --nproc_per_node=1 \
  --master_port=29538 \
  src/train.py \
  examples/vulcan/qwen35_08b_vqa_med_cls_predict.yaml \
  model_name_or_path="<MODEL_PATH>" \
  eval_dataset=vqa_val_cls \
  output_dir="<OUTPUT_PATH>"
```

计算指标：

```bash
python scripts/vulcan/eval_vqa_predictions.py \
  --prediction_file <OUTPUT_PATH>/generated_predictions.jsonl
```

当前实验的结果表应持续按以下顺序维护：

| 模型 | 是否经过 Vulcan 训练 | 是否物理剪枝 | Exact Match | Token F1 | Yes/No Acc. |
| --- | --- | --- | ---: | ---: | ---: |
| Baseline ckpt 1200 | 否 | 否 | 78.61% | 80.59% | 88.83% |
| Baseline direct-pruned（旧 cluster） | 否 | 是 | 约 0.03% | 待补 | 待补 |
| Baseline direct-pruned（multimodal cluster） | 否 | 是 | 待测 | 待测 | 待测 |
| Vulcan run 1 | 是 | 否 | 74.62% | 76.85% | 71.28% |
| Vulcan run 1 pruned | 是 | 是 | 74.62% | 77.10% | 75.00% |
| Vulcan run 2 | 是 | 否 | 66.29% | 68.31% | 65.43% |
| Multimodal Vulcan checkpoint 1800 | 是 | 否 | **75.68%** | **78.09%** | **82.98%** |
| Multimodal Vulcan checkpoint 1800 pruned | 是 | 是 | **75.55%** | **78.21%** | **84.57%** |

## 11. 多模态 Cluster Vulcan 实验

### 11.1 实验动机与 Cluster 适配

旧版聚类在混合训练集上对所有有效 token 的 MLP 激活统一平均，难以区分图像理解、问题理解和答案生成阶段的神经元功能。本轮保持 Vulcan collapse 训练公式不变，只调整 cluster 构造方法，以便进行受控比较。

多模态 cluster 使用三个独立训练子集：

- `vqa_train_modality`
- `vqa_train_plane`
- `vqa_train_organ`

主要适配包括：

1. 分别统计 image、question 和 causal prediction position 的 FFN 激活。
2. 先在样本内部平均 token，再在任务内部平均样本，最后对三个任务做等权宏平均。
3. anchor 分数使用 `0.4 image + 0.4 question + 0.2 prediction`。
4. 激活签名乘以对应 `down_proj` 列范数，以考虑神经元对 FFN 输出的实际贡献。
5. 聚类距离联合使用归一化后的 `up_proj/gate_proj` 权重方向和多模态激活签名，激活距离权重为 0.25。

本轮每个任务采集 200 batches，共 600 batches。生成的索引为：

```text
saves/cluster_idx_multimodal.json
```

当前仍使用统一 `keep_ratio=0.5`，即每层均保留 50% FFN 中间神经元；本轮尚未启用逐层自适应预算。

### 11.2 Collapse 训练设置

为了把结果变化归因于 cluster，本轮恢复旧 Vulcan 实验一的训练动力学：

| 参数 | 设置 |
| --- | --- |
| 起始模型 | baseline checkpoint 1200 |
| collapse reduction | `legacy` |
| 主任务学习率 | `1e-5` |
| lambda 学习率（rho） | `-0.1` |
| lambda 初值 | 0 |
| weight proxy | 关闭 |
| gradient accumulation | 4 |
| max steps | 2188 |
| collapse warmup/ramp | 0 / 0 |
| 保存与验证间隔 | 200 steps |

曾尝试 `normalized` collapse reduction，但该形式同时对权重维度和有效层数取平均，相比论文及旧 Vulcan 的“层内按 cluster 数平均、层间求和”显著缩小约束尺度，因此未作为本轮正式设置。

### 11.3 Checkpoint 选择

checkpoint 1800 与 final 的验证结果如下：

| 模型点 | Exact Match | Token F1 | Yes/No Acc. |
| --- | ---: | ---: | ---: |
| checkpoint 1800 | **75.68%** | **78.09%** | **82.98%** |
| final | 74.82% | 76.99% | 73.94% |

训练后段继续 collapse 导致 final 的 yes/no 能力明显回落，因此选择 checkpoint 1800 作为物理剪枝输入，而不是最终模型。

checkpoint 1800 的 yes/no 混淆矩阵：

| Gold -> Pred | 数量 |
| --- | ---: |
| no -> no | 103 |
| no -> yes | 10 |
| yes -> no | 22 |
| yes -> yes | 53 |

### 11.4 物理剪枝结果

使用同一个 `cluster_idx_multimodal.json` 对 checkpoint 1800 执行 50% FFN 物理剪枝：

| 指标 | 剪枝前 | 剪枝后 | 变化 |
| --- | ---: | ---: | ---: |
| Exact Match | 75.68% | 75.55% | -0.13 pp |
| Normalized Exact Match | 75.68% | 75.62% | -0.07 pp |
| Token F1 | 78.09% | 78.21% | +0.11 pp |
| Yes/No Acc. | 82.98% | 84.57% | +1.60 pp |

剪枝后 exact match 仅减少 2/1501 条，normalized exact match 仅减少 1/1501 条，可以视为近零损失剪枝。

剪枝后的 yes/no 混淆矩阵：

| Gold -> Pred | 数量 |
| --- | ---: |
| no -> no | 113 |
| no -> yes | 0 |
| yes -> no | 29 |
| yes -> yes | 46 |

剪枝后所有 `no` 样本均预测正确，但模型对 `no` 有一定偏置，`yes` recall 为 46/75（61.33%）。因此 yes/no 总准确率提升不能替代分类型和类别召回分析。

### 11.5 与旧 Vulcan 对比

| 剪枝模型 | Exact Match | Token F1 | Yes/No Acc. |
| --- | ---: | ---: | ---: |
| 旧 Vulcan run 1 pruned | 74.62% | 77.10% | 75.00% |
| Multimodal Vulcan checkpoint 1800 pruned | **75.55%** | **78.21%** | **84.57%** |
| 提升 | **+0.93 pp** | **+1.11 pp** | **+9.57 pp** |

两轮实验使用相同 baseline checkpoint、统一 50% 层内预算以及相同的主要 collapse 设置。主要差异位于 cluster 构造阶段，因此结果支持以下判断：多模态 token 分区、任务宏平均和输出贡献加权能够选择更适合 VQA-Med 的 FFN 合并关系，在保持近零损失物理剪枝的同时，降低 collapse 训练造成的任务能力损失。

### 11.6 当前结论与待补实验

1. 多模态 cluster 保持了 Vulcan 的核心性质：collapse 后执行 50% FFN 物理剪枝几乎不产生额外性能损失。
2. 相比旧 Vulcan，多模态 cluster 明显改善了总体 exact match 和 yes/no 稳定性。
3. 相比未剪枝 baseline 78.61%，当前剪枝模型仍低 3.06 pp，主要代价仍发生在 collapse SFT 阶段。
4. checkpoint 1800 优于 final，说明应按任务指标提前停止，而不能默认使用最后一个 checkpoint。
5. 后续需要补充 `vqa_val_modality`、`vqa_val_plane`、`vqa_val_organ` 分类型结果，以及 baseline/训练后模型的簇内 redundancy 对照。

## 12. Activation Alignment 探索

### 12.1 实验动机

Vulcan collapse 与物理剪枝主要依赖 FFN 中间神经元的冗余关系。多模态 VQA 中，视觉 token 与文本 token 可能依赖不同的 FFN neuron 子集；如果两者的 top-k activation 分布差异过大，后续 cluster/collapse/剪枝可能优先保留某一侧的表征而削弱跨模态推理。

本轮 activation alignment 的目标不是直接提升 yes/no 子集，而是验证能否在较小内部扰动下提高：

```text
IoU(visual top20 neurons, text top20 neurons)
```

并观察这种内部对齐是否能改善总体 exact match 和 token F1。VQA-Med 验证集共 1501 条，其中 yes/no 样本为 188 条，占比约 12.5%，因此主指标仍应是 overall EM/F1，yes/no 只作为副指标。

### 12.2 旧 soft-IoU 方案的问题

第一版 neuron-level alignment 使用 soft top-k mask 的 IoU loss：

```text
loss = 1 - soft_iou(visual_soft_mask, text_soft_mask)
```

其中 `align_quantile=0.8` 名义上关注 top 20% neuron，但训练日志显示 `align_mask_v_mean` 和 `align_mask_t_mean` 常在 0.44 左右，说明 soft mask 实际较稠密，不是严格 top20。后续加入 hard top-k 诊断后发现：

```text
align_soft_iou      ~= 0.30
align_hard_topk_iou ~= 0.30
```

这说明 soft-IoU 不是完全虚高，但该目标仍存在结构性问题：它是对称 loss，没有 anchor，visual 和 text 可以一起漂移，未必能真正提高两者 top20 集合的重叠。

代表性结果如下：

| 配置 | Exact Match | Token F1 | Yes/No Acc. | 主要现象 |
| --- | ---: | ---: | ---: | --- |
| no-align continuation, 200 steps | 76.48% | 78.59% | 82.98% | 同步训练步数对照 |
| soft-IoU, `lambda=0.05`, `temp=0.02` | 76.22% | 78.40% | 84.04% | 整体未优于 no-align |
| soft-IoU, `lambda=0.2`, `temp=0.02` | 75.75% | 78.20% | 88.83% | yes/no 提升，但 overall 下降 |

内部对照进一步说明，`lambda=0.2` 虽然能显著改变决策边界，但不是理想的温和对齐：

| 指标（后半层平均） | soft-IoU `lambda=0.05` | soft-IoU `lambda=0.2` |
| --- | ---: | ---: |
| visual top20 set change | 11.0% | 40.9% |
| text top20 set change | 24.6% | 64.4% |
| visual cosine | 0.996 | 0.920 |
| text cosine | 0.989 | 0.788 |
| visual L2 drift | 0.81 | 3.60 |
| text L2 drift | 1.15 | 7.02 |
| param delta rel | 0.11% | 0.23% |
| within-model IoU | 0.251 | 0.245 |

`lambda=0.2` 造成 text 和 visual 激活都大幅漂移，但 within-model visual/text top20 IoU 没有提升。这说明它更像后层表征重排，而不是把 text 稳定地拉向 visual anchor。

### 12.3 Rank-hardneg 方案

为避免 soft-IoU 的共同漂移问题，改为 visual-anchored rank-margin loss。核心目标是：在 text 侧，visual top20 neuron 的平均 salience 应超过 text 自己最强的一批非 visual-anchor neuron。

定义：

```text
V_top       = top20(visual pooled activation).detach()
top_score   = mean(text salience on V_top)
other_score = mean(text salience on strongest non-anchor top20)
gap         = top_score - other_score
loss        = relu(margin - gap)
```

其中 hard negative 使用 text 侧非 anchor neurons 中最强的 top20，而不是全部 non-anchor 的平均值。最初使用 `mean(all non-anchor)` 时，question 侧几乎天然满足 `margin=0.2`，导致 `align_loss=0`；改成 hard negative 后，训练日志出现有效信号：

```text
align_rank_question_loss = 0.03332
align_rank_question_gap  = 0.3078
align_rank_answer_loss   = 0.5189
align_rank_answer_gap    = -0.3189
```

该日志说明 question 侧通常已经较接近 visual anchors，而 answer 侧与 visual anchors 差异更大。因此后续探索重点从 answer 加权转向 question-only 与轻量 answer 加权。

当前 rank-hardneg 的关键设置：

```yaml
align_loss_type: rank_margin
align_quantile: 0.8
align_margin: 0.2
align_lambda: 0.05
align_question_weight: 1.0
align_answer_weight: 0.0 或 0.2
align_layer_start_ratio: 0.5
align_layer_end_ratio: 0.95
```

`align_layer_start_ratio=0.5`、`align_layer_end_ratio=0.95` 约等价于只对齐 L12-L22，避开浅层视觉/格式层和最后 logits-adjacent 层。

### 12.4 Rank-hardneg 外部指标

| 配置 | Checkpoint | Exact Match | Token F1 | Yes/No Acc. |
| --- | --- | ---: | ---: | ---: |
| no-align continuation | final / 200 steps | 76.48% | 78.59% | 82.98% |
| rank-hardneg, `answer_weight=0.2` | final / 200 steps | 76.95% | 79.04% | 87.23% |
| rank-hardneg, question-only | checkpoint 100 | 77.08% | 79.21% | 86.17% |
| rank-hardneg, question-only | final / 200 steps | **77.48%** | **79.57%** | **89.36%** |

本轮 question-only 训练产物：

```text
checkpoint 100:
outputs/qwen35-0_8b-vqa-med-cls/full/
  align-rank-hardneg-m02-lam005-qonly-lr3e6-checkpoint100/

final / 200 steps:
outputs/qwen35-0_8b-vqa-med-cls/full/
  align-rank-hardneg-m02-lam005-qonly-lr3e6-200steps/
```

两次评估均使用完整的 1501 条验证集。原始评估结果如下：

| Checkpoint | Num. Examples | Exact Match | Normalized EM | Token F1 | Yes/No Examples | Yes/No Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint 100 | 1501 | 0.7708194537 | 0.7708194537 | 0.7920708945 | 188 | 0.8617021277 |
| final / 200 steps | 1501 | **0.7748167888** | **0.7748167888** | **0.7956875311** | 188 | **0.8936170213** |

从 checkpoint 100 继续训练到 200 steps 后，EM 提升 0.40 pp，F1 提升 0.36 pp，yes/no accuracy 提升 3.19 pp。因此本轮没有出现 checkpoint 100 优于 final 的提前过拟合信号，最终结果应采用 200-step checkpoint。

question-only final 是当前 activation alignment 探索中的最佳 overall 结果。相对 no-align 200 steps：

| 指标 | no-align | question-only final | 提升 |
| --- | ---: | ---: | ---: |
| Exact Match | 76.48% | 77.48% | +1.00 pp |
| Token F1 | 78.59% | 79.57% | +0.98 pp |
| Yes/No Acc. | 82.98% | 89.36% | +6.38 pp |

question-only final 的 yes/no 混淆矩阵：

| Gold -> Pred | 数量 |
| --- | ---: |
| no -> no | 98 |
| no -> yes | 15 |
| yes -> no | 5 |
| yes -> yes | 70 |

该结果说明 answer 侧 alignment 并非必要；去掉 answer alignment 后，overall EM/F1 和 yes/no accuracy 均优于 `answer_weight=0.2`。当前更合理的解释是：question-side grounding 到 visual anchors 能改善多类标签预测，而 answer-side rank loss 容易直接扰动标签决策边界。

### 12.5 Rank-hardneg 内部对照

对同一个 100 条 val 子集，对比 no-align、soft-IoU 和 rank-hardneg 的后半层内部变化，得到：

| 指标（后半层平均） | soft-IoU `lambda=0.05` | soft-IoU `lambda=0.2` | rank-hardneg `lambda=0.05` |
| --- | ---: | ---: | ---: |
| visual top20 set change | 11.0% | 40.9% | 10.4% |
| text top20 set change | 24.6% | 64.4% | 29.3% |
| visual cosine | 0.996 | 0.920 | 0.996 |
| text cosine | 0.989 | 0.788 | 0.985 |
| visual L2 drift | 0.81 | 3.60 | 0.70 |
| text L2 drift | 1.15 | 7.02 | 0.85 |
| param delta rel | 0.11% | 0.23% | 0.15% |
| within-model IoU | 0.251 | 0.245 | **0.268** |
| no-align within-model IoU | 0.245 | 0.245 | 0.245 |

rank-hardneg 是唯一明显提升 within-model visual/text top20 IoU 的配置：

```text
0.245 -> 0.268 (+0.023)
```

同时它保持了结构稳定性：visual cosine 0.996、text cosine 0.985，参数相对变化仅 0.15%。这表明 rank-hardneg 不是通过暴力重排后层激活获得收益，而是在较小扰动下让 text top20 更接近 visual top20。

逐层看，rank-hardneg 在 L12-L21 几乎都提升 visual/text top20 IoU：

| 层 | no-align IoU | rank-hardneg IoU | 差值 |
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
| L23 | 0.418 | 0.430 | +0.012 |

由于当前实际训练范围为 L12-L22，L23 主要作为相邻层观测点。主要提升集中在 L12-L21，符合“对齐语义层、避开最后 logits-adjacent 层”的设计动机。

### 12.6 当前阶段结论

1. soft-IoU alignment 能提供正则信号，但对称目标容易造成 visual/text 共同漂移。强 lambda 可以改变 yes/no 决策边界，但不稳定提升 visual/text top-k overlap。
2. rank-hardneg 通过 visual anchor 和 hard negative ranking，直接推动 text 侧 top neuron 使用 visual top20，更符合 activation alignment 的目标。
3. 在 VQA-Med cls 上，question-only rank-hardneg 是当前最佳配置：`EM=77.48%`、`F1=79.57%`、`Yes/No=89.36%`。
4. 相比 no-align continuation，rank-hardneg question-only 提升 1.00 pp EM、0.98 pp F1 和 6.38 pp yes/no accuracy。
5. 内部分析显示 rank-hardneg 将 within-model visual/text top20 IoU 从 0.245 提高到 0.268，同时保持 visual/text activation cosine 分别为 0.996/0.985，说明它是温和而方向正确的对齐。
6. 当前仍低于原始 baseline checkpoint 1200 的 78.61% EM 和 80.59% F1，说明 alignment continuation 尚不能替代最佳 baseline 选点；但它相对同训练步数 no-align continuation 有稳定收益。
7. 后续应优先补充：不同 seed 验证、分类型结果、question mask 去除固定 prompt 的实验，以及基于 rank-hardneg 模型重新生成 cluster_idx 后再进行 Vulcan collapse/剪枝。
