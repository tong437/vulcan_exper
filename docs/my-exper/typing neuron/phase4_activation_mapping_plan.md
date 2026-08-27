# Phase 4：视觉到文本 FFN 激活映射

## 研究问题

Phase 4 不重新定义 `q_multimodal`。它检验一个增量问题：

> 图像诱导的文本侧 FFN 激活可预测性，能否在 q/r 连续模态选择性之外，提供额外的因果与剪枝信息？

当前 Phase 3 已经证明 q-band 可以安全地结构化删除，并获得真实参数、checkpoint 和峰值显存下降。Phase 4 的目标是补上“视觉表示能否预测语言神经元激活”的机制链条，而不是为 Phase 3 补合法性。

## 数据与表征

对同一个问题构造：

\[
(I,Q),\qquad (\pi(I),Q)
\]

其中 shuffled image 的置换只发生在同一个 data split 内。

- `V`：第一层 decoder block 输入处的视觉 token 隐状态。这是 vision merger 进入语言骨干后的最后桥接表示。
- `Q`：同一位置的非视觉 prompt token 隐状态，不使用生成答案 token。
- `A_l^text`：第 `l` 层 `down_proj` 输入在非视觉 prompt token 上的 pooled FFN 激活。
- 预测目标：

\[
\Delta A_l=A_l^{text}(I,Q)-A_l^{text}(\pi(I),Q)
\]

默认 `V/Q` 使用 mean 与 signed max-absolute pooling 的拼接，`A_l^text` 使用 mean pooling。

## 数据隔离

1. 先过滤 Phase 1 calibration/typing 图像。
2. 再按 image ID 划分 70% train、15% validation、15% test。
3. 同一图像的所有 POPE question 必须处于同一 split。
4. shuffled image 在各 split 内独立做无固定点的一一置换。
5. alpha 在 validation 上选择。
6. neuron ranking 由 validation 的 out-of-sample 预测产生。
7. test 只用于 Gate A/B 和后续 group ablation。

## 模型与对照

每层都比较：

- Q-only；
- V-only；
- V+Q。

每个输入分别拟合：

- feature-PCA ridge；
- reduced-rank ridge。

默认 primary model 为 reduced-rank ridge。所有降维、标准化和 target basis 只能用训练集合拟合。正式实验使用 20 次 target-row permutation null。

## 指标

- variance-weighted cross-validated/test `R²`；
- per-neuron `R²` 与 Pearson correlation；
- top-k `|ΔA|` neuron Precision/Recall@k；
- `V+Q` 相对 Q-only 的增量 `R²`；
- permutation-null empirical p-value；
- q_multimodal 与 mapping signal 的 Spearman；
- matched-count group ablation 的 Accuracy/F1/yes-ratio。

用于 group ablation 的 mapping signal 为：

\[
\max(R_j^2,0)\cdot E[|\Delta A_j|]
\]

`combined_protection` 是 q percentile 与 mapping-signal percentile 的等权平均。它是待验证的启发式保护先验，不是已经成立的剪枝分数。

Phase 1 dead neuron 的原始 `q_multimodal=NaN` 会保留用于审计；仅在排名副本
`q_multimodal_rank_value` 中编码为 0，保证 matched-count mask 的分数有限。

## 预注册门槛

### Gate A：映射存在

- 各层平均 V+Q test `R² >= 0.01`；
- aggregate permutation p-value `<= 0.05`；
- 至少 4 层 test `R² > 0`。

### Gate B：映射具有增量信息

- 平均 `R²(V+Q) - R²(Q-only) >= 0.002`。

Gate A/B 任一失败时，禁止进行 mapping-driven group ablation 和结构化剪枝。当前 q/r Phase 3 结论不受影响。

### Gate C：具有因果与剪枝价值

在 image-disjoint test 图像上，以相同的每层 15% 数量比较：

- q 5–20% band；
- mapping signal 最高 15%；
- combined protection 最低 15%；
- 三个 matched-random mask。

mapping-high 必须比 matched random 表现出不更弱的因果损伤；combined-low 必须通过 Accuracy/F1/yes-ratio 安全门，并且不显著差于 q-band。只有 Gate C 通过后，才允许构建新的结构 checkpoint。

## 执行

```bash
python scripts/vulcan/neuron_typing/run_phase4_mapping.py \
  --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet \
  --mapping_vqa_file saves/neuron_typing/pope_data/coco_pope_random.json \
  --image_root /root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images \
  --output_dir saves/neuron_typing/phase4_mapping \
  --calibration_manifest saves/neuron_typing/phase1_clean_2k/calibration/sample_manifest.json \
  --typing_manifest saves/neuron_typing/phase1_clean_2k/activations/sample_manifest.json \
  --batch_size 1 \
  --causal_batch_size 4 \
  --causal_vqa random=saves/neuron_typing/pope_data/coco_pope_random.json \
  --causal_vqa popular=saves/neuron_typing/pope_data/coco_pope_popular.json \
  --causal_vqa adversarial=saves/neuron_typing/pope_data/coco_pope_adversarial.json \
  --stage all
```

正式运行中，P4.1 支持 chunk 断点续跑。总控脚本检测到完整阶段产物时会跳过该阶段。P4.2 或 P4.3 gate 失败会终止流水线，并保留完整指标文件供分析。

## 产物

```text
phase4_mapping/
├── activations/
│   ├── chunks/
│   ├── collection_state.json
│   ├── sample_manifest.json
│   └── splits.json
├── mapping/
│   ├── mapping_metrics.json
│   ├── neuron_scores_with_mapping.parquet
│   └── phase4_ablation_plan.json
├── group_causality/
│   ├── random.json
│   ├── popular.json
│   ├── adversarial.json
│   └── phase4_group_causality.json
└── phase4_pipeline.json
```
