# Phase 5A：单样本最小充分 FFN 子网络

## 1. 目标与边界

Phase 5A 测量固定图像—问题样本的经验剪枝前沿。模型参数保持冻结，每层 FFN channel 使用一个对所有
视觉、prompt 和 decode token 恒定的二值 mask。目标是在复现原模型自由生成、teacher-forced top-1 token
和输出分布的条件下最小化保留神经元数。

成功找到的 mask 只构成“至少可以剪到这里”的构造性证据。更小预算下搜索失败不证明不存在更小子网络。
结果不支持跨样本或跨任务泛化结论。

## 2. 当前实现

`run_phase5_single_sample_frontier.py` 完成第一阶段静态 saliency 搜索：

1. 按 `sample_offset` 固定一个 SFT 样本并保存 manifest；
2. 先生成 full-model teacher rollout，再在同一 rollout 上运行可微 teacher pass；
3. 计算 activation、down-projection contribution 和 first-order Taylor saliency；
4. 支持 uniform per-layer budget 与 non-uniform global budget；
5. 对每个预算重新计算 label NLL、teacher KL、top-1 agreement、teacher-token margin；
6. 在相同 sample-static mask 下自由生成并记录首次 token 分歧；
7. 保存完整 frontier、mask、saliency 和 teacher logits，供后续 learned-gate refinement 使用。

主可行性标准为：自由生成完全一致、teacher-forced top-1 agreement 为 100%，且平均 KL 不超过
`--kl_tolerance`。

默认 `--trace_target teacher_generation` 保证 saliency、KL 与最终自由生成使用同一条模型行为轨迹。
`--trace_target dataset_labels` 仅作为保持数据集 reference response 的次要对照，不能与主结果混用。

## 3. 正式运行示例

建议先选定一个能产生较长回答的 held-out COCO caption 样本，不要根据剪枝结果反复更换样本。

```bash
python scripts/vulcan/neuron_typing/run_phase5_single_sample_frontier.py \
  --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --output_dir saves/neuron_typing/phase5_single_sample/sample_2500 \
  --sample_offset 2500 \
  --methods activation,contribution,taylor,random \
  --selection both \
  --keep_ratios 0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0 \
  --global_normalization layer_mean \
  --max_new_tokens 32 \
  --kl_tolerance 0.001 \
  --num_workers 0
```

额外的 LlamaFactory 配置覆盖仍使用 `key=value` 形式附在命令末尾。

## 4. 输出

```text
phase5_single_sample/sample_2500/
├── sample_manifest.json
├── teacher_logits.pt
├── saliency_scores.pt
├── frontier.json
└── masks/
    └── <method>__<selection>__keep_<ratio>.json
```

`frontier.json` 会在每个条件结束后增量写回；若长 sweep 中断，已经完成的条件仍保留。
`best_feasible` 分别给出每种 saliency/selection 当前找到的最小可行子网络。
若粗网格没有覆盖边界，可使用相同配置和 `--resume` 添加新的 `--keep_ratios`；脚本会校验模型、样本、生成
长度、KL 阈值和随机种子，并跳过已经完成的条件。未显式传入 `--resume` 时不会覆盖已有 frontier。

## 5. 后续阶段

静态 sweep 定位崩溃区间后，Phase 5B 只在边界附近运行 exact-budget straight-through top-k gate，并使用
多个随机种子和 neuron swap 局部搜索。最终只对最优 mask 执行 singleton structural conversion 与
hook/structural equivalence gate。允许权重恢复和 token-dynamic routing 的实验必须单独报告，不能与本阶段
冻结权重、sample-static 的剪枝率直接比较。
