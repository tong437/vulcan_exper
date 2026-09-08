# Phase 5C 单样本物理结构剪枝结果

## 1. 目标与判定

Phase 5C 把 Phase 5B 的 sample-static deletion mask 转换为真实非均匀 FFN：未删除 neuron 的层保持
原宽度，其他层只保留 mask complement 对应的 `gate_proj`、`up_proj` 行和 `down_proj` 列。

验证同时比较三条路径：

1. 原模型 + forward hook mask；
2. 原模型在内存中执行 singleton 物理裁剪；
3. 保存后重新加载的非均匀结构 checkpoint。

结构 gate 要求每层宽度、参数数和投影权重 hash 全部正确；行为 gate 沿用 Phase 5B 的 mean KL <= 0.001、
teacher 自洽位置 top-1 agreement = 1、cached generation exact match。

## 2. 最大通过候选

`delete_750__restart_0` 通过完整 64-token Phase 5C gate：

| 指标 | Hook mask | 内存物理裁剪 | 磁盘重载 |
|---|---:|---:|---:|
| Mean KL | 0.000846 | 0.000900 | 0.000900 |
| 自洽位置 top-1 agreement | 100% | 100% | 100% |
| 64-token generation | exact | exact | exact |

- 删除 FFN neurons：750 / 86,016，即 0.8719%；
- 删除参数：2,304,000；
- 全模型参数：852,985,920 -> 850,681,920，下降 0.2701%；
- hook 与物理裁剪的 BF16 logit mean absolute difference 为 0.036835；
- 内存物理模型与保存重载模型逐 logit 差异为 0；
- 所有目标层宽度、参数下降、singleton 权重 hash 和 reload 权重 hash 均通过。

正式 checkpoint 位于
`saves/neuron_typing/phase5_single_sample/sample_2500_phase5c_screen_750_r0/model`，完整 gate 位于同目录上级的
`equivalence.json`。

## 3. 1000 与另一 750-mask 为什么失败

结构转换本身没有错误，但 BF16 GEMM 在矩阵宽度改变后会产生不同的累积舍入：

| 候选 | Hook KL | 物理 KL | 物理生成 | 结论 |
|---|---:|---:|---|---|
| delete-1000 restart 1 | 0.000892 | 0.001074 | token 13 分叉 | 失败 |
| delete-750 restart 1 | 0.000802 | 0.001096 | 64 / 64 exact | KL gate 失败 |
| delete-750 restart 0 | 0.000846 | 0.000900 | 64 / 64 exact | 通过 |

三个候选的结构尺寸、权重和 reload 等价检查均正确。这说明 hook 可行并不足以保证 BF16 物理模型可行，
也说明物理误差不只由 hook KL 单调决定；Phase 5B 后续最好在优化中加入结构化 BF16 forward 复评。

## 4. 256-token 长窗口结果

Teacher 在 256-token 上限仍未输出 EOS。750 restart 0 在扩展 teacher rollout 上：

- hook 与物理模型都在第 83 个生成 token 首次偏离 teacher；
- hook mean KL 为 0.013633，物理模型为 0.013457；
- 物理模型与重载模型仍逐 logit 完全一致。

500 restart 2 同样在第 83 token 偏离，hook mean KL 为 0.010085。因此降低现有 64-token mask 的预算没有
解决外推问题。当前严格结论是：750 个 neuron 是**前 64 token**的最大已验证物理剪枝结果；现有 mask
不保证完整回答。若目标升级为保持 256 token 或直到 EOS，必须用长 teacher rollout 重新训练 learned gate。

rollout256 Phase 5B 的 `delete_250__restart_0` 在 hook 路径满足 256-token exact generation 与
KL=0.000878，但物理裁剪后虽然 KL 仍为 0.000907，生成在 token 13 分叉，因此也不能直接作为结构结果。
这推动了 Phase 5D 的 cached-path 优化与物理复评接口。

## 5. 实现文件

- `phase5_structural_utils.py`：允许零删除层的 partial-singleton 表示与严格验证；
- `build_phase5_structural_artifact.py`：冻结可行 run、mask、cluster 和来源 hash；
- `screen_phase5_structural_candidate.py`：不写 checkpoint 的内存物理筛选；
- `verify_phase5_structural_equivalence.py`：hook、内存结构、保存重载和扩展 horizon gate；
- `modeling_vulcan_qwen3_5.py`：非均匀 Qwen3.5 loader，并显式恢复 Transformers 5 远程加载后的
  embedding/lm-head 权重共享。
