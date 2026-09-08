# Phase 5B：单样本 Exact-Budget Learned Gate

Phase 5B 在 Phase 5A 已冻结的同一条 teacher rollout 上优化 sample-static 二值 gate。模型权重保持冻结，
每次前向使用严格的 global deletion budget；硬 top-k 决定实际 mask，sigmoid straight-through estimator
为 gate logits 提供梯度。

## 运行示例

```bash
python scripts/vulcan/neuron_typing/run_phase5_learned_gate.py \
  --static_frontier saves/neuron_typing/phase5_single_sample/sample_2500_rollout64/frontier.json \
  --output_dir saves/neuron_typing/phase5_single_sample/sample_2500_learned_refine500 \
  --deletion_budgets 500,750,1000,1125,1250,1500 \
  --steps 200 \
  --restarts 3 \
  --learning_rate 0.02 \
  --temperature_start 2.0 \
  --temperature_end 0.5 \
  --init_noise 0.02 \
  --margin_weight 0.05 \
  --margin_target 0.05
```

每个 restart 从 rollout Taylor saliency 初始化；restart 0 不加噪声，其余 restart 使用确定性随机扰动。
训练目标由 teacher KL 和 cached-generated token hinge margin 组成。margin 项用于避免平均 KL 已较低但
少数低 margin token 翻转。若 BF16 cache generation 与 full-forward teacher top 本身在边界 token 上不同，
margin 以实际 cached rollout token 为目标；top-1 agreement gate 只统计 teacher 自洽位置，真实行为仍要求
cached generation exact match。训练结束后将最佳 hard mask 固定，再独立计算：

- teacher rollout mean/max KL；
- teacher-forced top-1 agreement；
- 64-token free-running exact match；
- 实际删除参数和全模型参数下降。

只有三项主 gate 同时通过才进入 `best_feasible`。使用 `--resume` 可以在优化设置不变时追加预算或更多
restart。成功结果只说明对这个固定样本和 rollout 找到了构造性 mask；搜索失败不证明该预算不可行。
