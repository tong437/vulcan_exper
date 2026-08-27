# P4.4a：Mapping-High 因果剂量曲线

## 目标

P4.4a 只检验视觉→文本 activation mapping 的因果定位能力，不构造删除分数，也不授权结构化剪枝。

需要回答：

1. mapping-high 消融是否呈现稳定的剂量响应；
2. 输出从何种比例开始显著退化或坍缩；
3. mapping-high 是否比同数量全局随机更具破坏性；
4. 控制 q_multimodal 构成后，mapping 的因果富集是否仍然存在。

## 比例与精确数量

正式比例：

| 比例 | 每层神经元 | 总神经元 |
|---:|---:|---:|
| 1% | 36 | 864 |
| 2.5% | 90 | 2,160 |
| 5% | 180 | 4,320 |
| 10% | 359 | 8,616 |
| 15% | 538 | 12,912 |

所有 mapping-high 和 q-high mask 在比例间严格嵌套。

## 对照

每个比例包含：

- `q_multimodal` 最高同数量；
- `mapping_signal` 最高同数量；
- 20 个全局 matched-random；
- 20 个 q-stratified matched-random。

q-stratified control 在每一层把神经元按 `q_multimodal_rank_value` 排成 10 个等数量分位箱，然后精确匹配 mapping-high 在每个 q 箱中的入选数量。每个比例和随机种子独立采样，但始终保持：

- 相同层；
- 相同总数；
- 相同 q-decile 直方图；
- 无重复神经元。

## 数据隔离

沿用 Phase 4 的 69 张 image-disjoint test 图，每个 POPE split 包含 414 个问题。validation-derived mapping ranking 在 P4.4a 中被冻结。

- 主剂量曲线：random、popular、adversarial；
- 20-seed 控制分布：默认只在 random split 上运行；
- calibration、typing、mapping train/validation 图像均不进入评估。

## 指标

每个任务和比例报告：

- Accuracy/F1 变化；
- image-cluster bootstrap 95% CI；
- yes ratio 及变化；
- 首个不安全比例；
- 首个 yes/no 输出坍缩比例；
- 比例与 Accuracy damage 的 Spearman。

每个比例的控制检验报告：

- 随机 ΔAccuracy 分布；
- mapping 相对随机均值的因果富集；
- finite-sample empirical p-value；
- 随机 mask 输出坍缩频率。

经验 p-value 使用：

\[
p=\frac{1+\#\{\Delta Acc_{random}\leq\Delta Acc_{mapping}\}}{N+1}
\]

20 个控制的最小 p-value 为 `1/21=0.047619`。

## 预注册判据

### 单调剂量响应

三个任务的 Accuracy damage Spearman 中位数至少为 `0.8`。

### 因果富集

至少 3/5 个比例同时满足：

- 相对全局 random empirical `p <= 0.05`；
- 相对 q-stratified random empirical `p <= 0.05`。

无论 P4.4a 是否通过，`structural_pruning_allowed` 始终为 false。P4.4a 不能把低 mapping signal 解释为冗余。

## 正式运行

建议先跑主曲线：

```bash
python scripts/vulcan/neuron_typing/run_phase44a_dose_response.py \
  --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --mapping_metrics saves/neuron_typing/phase4_mapping/mapping/mapping_metrics.json \
  --activation_dir saves/neuron_typing/phase4_mapping/activations \
  --main_vqa random=saves/neuron_typing/pope_data/coco_pope_random.json \
  --main_vqa popular=saves/neuron_typing/pope_data/coco_pope_popular.json \
  --main_vqa adversarial=saves/neuron_typing/pope_data/coco_pope_adversarial.json \
  --control_vqa random=saves/neuron_typing/pope_data/coco_pope_random.json \
  --image_root /root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images \
  --output_dir saves/neuron_typing/phase44a_dose_response \
  --calibration_manifest saves/neuron_typing/phase1_clean_2k/calibration/sample_manifest.json \
  --typing_manifest saves/neuron_typing/phase1_clean_2k/activations/sample_manifest.json \
  --batch_size 4 \
  --stage main
```

然后运行完整控制分布：

```bash
# 使用与上面完全相同的参数，只将最后一项改为：
  --stage controls
```

最后汇总：

```bash
# 使用相同参数：
  --stage summarize
```

也可以直接使用 `--stage all`。每个评估条件都会写入 checkpoint；重复同一命令会自动恢复未完成条件。

## 产物

```text
phase44a_dose_response/
├── controls/
│   ├── neuron_scores_phase44a.parquet
│   ├── qstratified_controls.json
│   └── random.json
├── main/
│   ├── random.json
│   ├── popular.json
│   └── adversarial.json
├── test_subsets/
└── phase44a_dose_response.json
```

