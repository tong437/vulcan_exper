# Phase 5B 单样本 Learned-Gate 结果

## 1. 结论

在 Qwen3.5-VL-0.8B 的固定 `coco_captions_val sample_offset=2500` 上，Phase 5B 已构造出一个严格通过
预设 gate 的 1000-neuron 删除 mask：

| 指标 | 结果 |
|---|---:|
| 删除 FFN neurons | 1000 / 86,016 |
| FFN neuron pruning ratio | 1.1626% |
| 删除参数 | 3,072,000 |
| 全模型参数下降 | 0.3601% |
| Mean KL | 0.000892 |
| 自洽位置 top-1 agreement | 100% |
| 64-token cached generation | exact match |

这把 Phase 5A 静态 Taylor 排序的严格可行结果从 301 提高到 1000 个 neuron，即删除数为原来的
3.32 倍。它是当前搜索得到的**构造性下界**，不是组合最优证明。

正式结果位于
`saves/neuron_typing/phase5_single_sample/sample_2500_learned_refine500/learned_frontier.json`，最佳 mask 为
`masks/delete_1000__restart_1.json`。

## 2. Exact-budget 搜索结果

所有预算使用相同设置：200 steps、3 restarts、Taylor 初始化、learning rate 0.02、temperature 2.0 到
0.5、cached-token margin weight 0.05。可行性要求同时满足 mean KL <= 0.001、自洽位置 top-1 全保留、
64-token 自由生成完全一致。

| 删除预算 | 成功数 / 3 | 最低 KL | 结果摘要 |
|---:|---:|---:|---|
| 500 | 2 / 3 | 0.000656 | 稳定可行 |
| 750 | 2 / 3 | 0.000802 | 可行，已有生成边界敏感性 |
| 1000 | 1 / 3 | 0.000892 | 当前最大严格可行构造 |
| 1125 | 0 / 3 | 0.000904 | 最低 KL run 的生成在 token 47 分叉；其余 run KL 超阈值 |
| 1250 | 0 / 3 | 0.001156 | 未找到可行解 |
| 1500 | 0 / 3 | 0.001135 | 未找到可行解 |

因此当前搜索预算下，已找到的严格下界是 1000；1125 是最近的未找到可行解预算。后者只是经验搜索
上界，不能解释为 1125 在数学上不可行。若把“稳定边界”定义为至少 2 / 3 重启成功，则当前是 750。

## 3. 层结构与 mask 多解性

最佳 1000-mask 仍明显偏向深层：

- Layer 23：340；Layer 22：142；二者合计 48.2%；
- Layer 21、14、15、20、16：分别 70、65、62、50、47；
- Layer 0--2：0；Layer 3--5：合计 5；
- mask 涉及 21 / 24 层，但早层只承担极少删除量。

最佳 1000-mask 覆盖静态 Taylor-301 mask 中的 270 个 neuron，即覆盖率 89.7%，两者 Jaccard 为
0.262。这说明 learned gate 保留了大部分强 Taylor 候选，同时通过联合优化找到了更多可兼容删除。

相同预算的两个成功 restart 并不收敛到完全相同的集合：500-budget Jaccard 为 0.453，750-budget
Jaccard 为 0.556。这支持固定样本周围存在多个近等价稀疏子网，而不是唯一的冗余 neuron 列表。

## 4. 限制与后续

- teacher 在 64-token 上限处仍未到 EOS，本结论只保证固定前 64 个生成 token；
- BF16 cached decode 与 full-forward teacher 自身有 1 个 top-1 边界差异，因此 top-1 gate 只统计 teacher
  自洽的 63 个位置，行为 gate 仍要求真实 cached generation 64 / 64 一致；
- learned mask 是 sample-static，不是按 token 动态路由，也未证明对其他样本安全；
- Phase 5C 已表明 1000-mask 在物理 BF16 模型上越过 KL gate；当前最大 64-token 物理可行结果是
  750 restart 0。扩展到 256 token 后现有 500/750 mask 都在 token 83 分叉，因此下一轮应直接使用长
  teacher rollout 重新优化，而不是继续细搜 64-token 的 1000--1125 区间。
