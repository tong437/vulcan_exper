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
6. 当前不能只用最终准确率评价 Vulcan。至少要同时报告：baseline、Vulcan 剪枝前、Vulcan 剪枝后，以及参数量或模型大小变化。

## 9. 尚未完成的关键对照

### 9.1 Baseline 直接剪枝

需要用同一个 50% cluster index 直接剪枝 baseline checkpoint 1200，不做 Vulcan collapse 训练，然后评估：

```text
Baseline ckpt 1200
  -> 直接按 cluster/anchor 剪枝
  -> vqa_val_cls
```

这是证明 Vulcan 有效性的核心对照。如果 baseline 直接剪枝远低于 74.62%，而 Vulcan 剪枝后仍为 74.62%，才能明确量化 collapse 训练挽回了多少剪枝损失。

### 9.2 分类别剪枝结果

Vulcan 剪枝模型还应分别评估：

- `vqa_val_modality`
- `vqa_val_plane`
- `vqa_val_organ`

重点判断 baseline 到 Vulcan 的 4 pp 损失主要来自哪一类，并检查 yes/no 退化是否只集中在 modality。

### 9.3 更稳健的后续配置

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

该配置尚未验证，不能作为已完成结果。由于实验一已经证明“训练后可无损物理剪枝”，下一步优先级应是补齐 baseline 直接剪枝和分类型评估，而不是立刻扩大超参数搜索。

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
| Baseline direct-pruned | 否 | 是 | 待测 | 待测 | 待测 |
| Vulcan run 1 | 是 | 否 | 74.62% | 76.85% | 71.28% |
| Vulcan run 1 pruned | 是 | 是 | 74.62% | 77.10% | 75.00% |
| Vulcan run 2 | 是 | 否 | 66.29% | 68.31% | 65.43% |

