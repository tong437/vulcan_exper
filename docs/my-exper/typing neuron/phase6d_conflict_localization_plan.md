# Phase 6D：负干扰边的双向因果定位预注册

## 目标

Phase 6D 不再问 union 平均是否有益，而是解释 Phase 6C 中三个确定的非单调反例：为什么加入另一个已经学得的
子网后，原本可完成的目标会丢失。第一阶段只做冲突定位，不优化新 mask，也不把低 mapping、低 q 或小权重当作
候选先验。

冻结的三条边为：

1. `bicycle_clock–violin_kitchen`：两个 constituent 均能在两个目标上通过，union 在两个目标上都失败；
2. `bicycle_clock–toilet_spatial`：`toilet_spatial` constituent 双通过，union 只破坏 toilet 目标；
3. `violin_kitchen–people_pizza`：`people_pizza` constituent 双通过，union 只破坏 people 目标。

所有结论仍限定于这四张图、三个 pair 和冻结 prompt/contract；Phase 6D 不把它们提升为总体冲突规律。

## 冻结的六个 pass-to-fail 槽

对 bicycle–violin，同时保留两个安全 base 和两个受损 target 的 2×2 方向，共四槽。另两条边各只有一个在 Phase
6C 中实际发生 pass-to-fail 的槽：

| edge | safe base | donor | target |
|---|---|---|---|
| bicycle–violin | bicycle | violin | bicycle |
| bicycle–violin | bicycle | violin | violin |
| bicycle–violin | violin | bicycle | bicycle |
| bicycle–violin | violin | bicycle | violin |
| bicycle–toilet | toilet | bicycle | toilet |
| violin–people | people | violin | people |

令安全 base 为 K_b，donor-only 增量为 D = K_d ∖ K_b，失败 union 为 U = K_b ∪ D。正式运行前必须从 Phase
6C 原始 evaluation 逐槽验证 `K_b=pass, U=fail`；Phase 6D 运行本身也重跑这两个端点。任何槽若不能复现端点，
该槽不进入层级解释。

## 6D-L：逐层双向定位

对每个槽和每层 l 运行两种互补干预：

- **add-one-layer**：从 K_b 出发，只加入 D_l。若 pass→fail，说明该层 donor identity 在安全背景中足以触发
  离散退化；
- **leave-one-layer-out**：从 U 出发，只移除 D_l。若 fail→pass，说明去掉同一层 donor identity 足以解除
  完整 union 的退化。

只有同一槽、同一层同时满足两个翻转，才进入 identity 级拆分。单方向命中只作为上下文依赖或分布式效应记录，
不能称为定位完成。gold-prefix NLL 同时用于排序和边界诊断，但不能替代严格生成语义翻转。

每个真实干预配 3 个逐层同数量随机 identity：

- add 对照从该层 `complement(U)` 采样，保证不混入真实 donor-only identity；
- LOO 对照从该层安全 base resident identity 采样，保证不误删真实 donor-only identity；
- seed 由全局 seed、contrast、layer 和 replicate 共同哈希导出，固定且可重现。

主候选判据为双向精确翻转。若 add 随机失败率与 LOO 随机恢复率都小于 50%，则额外标为
`identity_specific_against_control_majority`。三个 control seed 只提供候选级描述性对照，不用于总体显著性声称。

正式规模为 6 个槽 × `[2 endpoints + 24 layers × (2 treatments + 6 controls)]` = **1,164 个唯一 mask 与
1,164 次目标推理**。主结果为 BF16；只对离散边界翻转或 hook/结构化不一致的最终候选做 FP32 诊断。

## 6D-I：层内 identity 拆分

6D-L 的双向交集层才允许进入本阶段。对每个候选层，以完整 D_l 为已知冲突集，目标谓词固定为：

```text
base ∪ S 在目标上失败  AND  union ∖ S 在目标上恢复
```

搜索使用确定性 partition/delta-debugging，而不是把普通二分的单调假设带进来：每个粒度同时评估所有 chunk 与
complement；只有完整双向谓词保持时才缩小集合；若当前二分两侧均失败，则提高 partition 粒度而不是宣告不存在。
同大小随机 identity 必须随每个被接受的缩减重新评估。最终集合只是预算内构造；只有完整跑完 singleton-removal
粒度后才允许称为 1-minimal，任何情况下都不声称全局最小。

为避免看到层结果后无限扩展搜索，identity 阶段需在运行前冻结：最大候选层数、每层最大 evaluation budget、
partition 顺序、tie-break（更小集合优先，其次双向 NLL margin，再按 identity hash）以及随机 seed 数。若 6D-L
没有双向交集，identity 阶段按预注册规则停止，不改用单方向层追结果。

## 路由结论边界

Phase 6D 的定位结果最多支持构造冲突感知路由假设：输入 i 只激活 `C7 + S_i`，或在 union 中抑制已定位的
冲突 identity。这样的输入路由可能降低活跃宽度和避免负干扰，但如果所有 shell 常驻内存，它不减少静态参数量。
参数减少必须另由独立 checkpoint、按需加载或真正稀疏专家实现与实测支持。

## 命令

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/run_phase6d_conflict_localization.py \
  --output_dir saves/neuron_typing/phase6d_conflict_localization_v1 \
  --random_seeds 3

PYTHONPATH=src python scripts/vulcan/neuron_typing/analyze_phase6d_conflicts.py \
  --result_file saves/neuron_typing/phase6d_conflict_localization_v1/phase6d_localization.json \
  --output_file saves/neuron_typing/phase6d_conflict_localization_v1/analysis.json

# 只有 analysis 产生双向交集后才能启动；以下预算必须在启动前冻结。
PYTHONPATH=src python scripts/vulcan/neuron_typing/run_phase6d_identity_refinement.py \
  --localization_analysis saves/neuron_typing/phase6d_conflict_localization_v1/analysis.json \
  --output_dir saves/neuron_typing/phase6d_identity_refinement_v1 \
  --max_candidate_slots 12 \
  --max_subset_evaluations 96 \
  --random_seeds 3
```

runner 对每个 evaluation 做 fsync 并支持 `--resume`；resume 时模型、输入 hash、Phase 6C evaluation hash、seed
和解码配置必须完全一致。
