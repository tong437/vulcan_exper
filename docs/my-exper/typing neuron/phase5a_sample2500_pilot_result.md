# Phase 5A 单样本静态剪枝 Pilot 结果

## 1. 实验设置

- 模型：Qwen3.5-VL-0.8B；
- 样本：`coco_captions_val` 的固定 `sample_offset=2500`；
- mask：sample-static，所有视觉、prompt、decode token 共用，模型权重冻结；
- teacher target：full model 自由生成的同一条 64-token rollout；
- 主 gate：64-token exact generation、teacher cache/full-forward 自洽位置上的 top-1 agreement = 1、
  mean KL <= 0.001；
- 搜索：activation、down-projection contribution、first-order Taylor、random；uniform per-layer 与
  non-uniform global budget。

正式 artifact 为
`saves/neuron_typing/phase5_single_sample/sample_2500_rollout64/frontier.json`。早期
`sample_2500/frontier.json` 使用 dataset reference labels 计算 saliency/KL，却用模型生成轨迹做最终验证，
目标不一致，只保留为工程诊断，不作为本阶段证据。

## 2. 当前最佳构造性结果

Taylor-global 在 keep ratio `0.9965` 时通过全部 gate：

| 指标 | 结果 |
|---|---:|
| 删除 FFN neurons | 301 / 86,016 |
| FFN neuron pruning ratio | 0.3499% |
| 删除参数 | 924,672 |
| 全模型参数下降 | 0.1084% |
| Mean / max KL | 0.000993 / 0.005636 |
| 自洽位置 teacher-forced top-1 agreement | 100% |
| 64-token generation | exact match |
| Delta rollout NLL | -0.002447 |

这表示已经构造出一个满足预设条件的 301-neuron 删除 mask，不表示 301 是组合最优。

相邻 Taylor-global 粗网格呈非单调性：删除 322 个时 exact generation 与 top-1 agreement 仍通过，但
mean KL 为 0.001254；删除 258 个时 mean KL 为 0.001105，也略高于阈值。因此不能通过普通二分搜索证明
边界。

未剪枝 teacher 在 BF16 cached decode 与整段 full-forward 之间有 1 / 64 个边界 top-1 差异。因此主 gate
排除这个 teacher 自身不自洽的位置，同时仍以 cached decode 的 64-token exact match 作为最终行为约束。

## 3. 同预算对照

在 keep ratio `0.9965` 附近：

| 方法 | 删除数 | Mean KL | Top-1 agreement | Exact generation | Gate |
|---|---:|---:|---:|---:|---:|
| Taylor-global | 301 | 0.000993 | 100% | 是 | 通过 |
| Activation-global | 301 | 0.005115 | 95.31% | 否 | 失败 |
| Contribution-global | 301 | 0.015022 | 95.31% | 否 | 失败 |
| Random-global | 301 | 0.015029 | 93.75% | 否 | 失败 |
| Activation-per-layer | 288 | 0.003726 | 98.44% | 否 | 失败 |
| Contribution-per-layer | 288 | 0.006310 | 96.88% | 否 | 失败 |
| Random-per-layer | 288 | 0.013948 | 90.63% | 否 | 失败 |

这支持 first-order Taylor 对该固定 rollout 具有样本特异定位能力，也支持 non-uniform layer allocation
优于当前 uniform allocation。

## 4. 层分配

301 个删除 channel 高度集中在后层：

- Layer 23：173；
- Layer 22：48；
- Layer 14、15、20：分别 15、12、12；
- Layer 0--5：全部为 0；
- 其余层合计：53。

Layer 22--23 占全部删除的 73.4%。这表明固定样本下的可删空间具有强烈层不均匀性；它也可能部分来自
Taylor 对长因果链深层梯度的定义，因此还需要 learned-gate 和 neuron-swap 搜索验证，不能直接解释为
后层普遍冗余。

## 5. 限制与下一步

full model 的 64-token rollout 在上限处仍未生成 EOS，当前结论严格限定为“保持前 64 个生成 token”，
而不是保持完整回答。正式极限确认需要：

1. Phase 5B 在 301--更激进预算上运行 exact-budget learned gates 与多个随机种子；
2. 对候选执行 neuron-swap 局部搜索，避免一次 Taylor ranking 的组合局限；
3. 用更长 horizon 或直到 EOS 重新验证；
4. 仅对最终候选执行 singleton structural conversion 和 hook/structural equivalence gate。

当前 0.3499% 是严格 KL 条件下的静态 ranking 起点，不应被描述为单样本剪枝上限。
