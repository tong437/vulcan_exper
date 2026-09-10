# Phase 6D：负干扰边双向定位与 identity 拆分结果

## 结论先行

Phase 6D 在 Phase 6C 冻结的三条负干扰边上完成了 1,164 次逐层 BF16 定位、610 次 identity 级 BF16
干预，以及 12 次最终集合 FP32 诊断。结果把“union 有负干扰”进一步收窄为三个结论：

1. **冲突不是一条 edge 对应一个固定坏层。** 六个预注册 pass-to-fail 槽、共 144 个 contrast×layer 中，只有
   3 个槽满足同层 add-one-layer 触发失败且 leave-one-layer-out 恢复；它们只来自两个 base/target 方向。
2. **最干净的层级定位是 bicycle–toilet 的 layer 20。** 435 个 bicycle-only identity 的 add/LOO 均翻转
   toilet 目标，而两个方向的等量随机对照均 0/3 命中。bicycle–violin 的 layer 0/20 也双向翻转，但完整层
   的随机 LOO 为 3/3 恢复，因此只能先称层级候选，不能在该剂量上称完整 identity-specific 层。
3. **冲突集合可被确定性非单调搜索显著缩小。** 三个候选从 2,557 个 layer increment identities 合计缩到
   797 个，三个最终集合都保持 add 失败与 LOO 恢复；最终等量随机集合为 0/9 命中联合谓词。三者在 FP32 下
   也全部复现，因此不是 BF16 特有的决策边界。

这些结果支持“有方向、依赖 base 与 target 的稀疏冲突集合”，但还没有证明它们是最小冲突集，也没有证明从
union 中抑制这些集合能同时保住 pair 的另一个任务。

## 冻结设计与完整性

- 六个槽在运行前写死：bicycle–violin 的两个安全 constituent × 两个目标共四槽，另加 toilet 和 people 各自
  在 Phase 6C 中真实受损的一槽。
- Phase 6C 原始记录和 Phase 6D 重跑均验证六槽为 `safe base=pass, union=fail`；端点复现 6/6。
- 每槽、每层都评估真实 add、真实 LOO，并分别配 3 个同层同数量随机 identity；共 12 endpoints + 288
  treatments + 864 controls = 1,164。
- localization analyzer 验证 evaluation 数量与唯一 ID、variant manifest、mask hash、target 和语义字段全部
  一致。
- identity 阶段只读取双向层 shortlist；每个粒度完整评估 chunk 与 complement，再按集合大小、双向 NLL
  margin 和 hash 做确定性 tie-break。
- identity 结果为 254 个 subset proposals × 2 个方向 + 17 次接受缩减 × 3 seeds × 2 个方向 = 610 次推理；
  最终 provenance、谓词和所有 step control 完整性均通过。

## A. 逐层双向定位

| contrast | 双向精确翻转层 | 通过 control-majority 的层 |
|---|---:|---:|
| bicycle base → bicycle target，violin donor | 0, 20 | 无 |
| bicycle base → violin target，violin donor | 无 | 无 |
| violin base → bicycle target，bicycle donor | 无 | 无 |
| violin base → violin target，bicycle donor | 无 | 无 |
| toilet base → toilet target，bicycle donor | 20 | 20 |
| people base → people target，violin donor | 无 | 无 |

这里的“双向”要求同一个 layer 同时满足：从安全 base 只加入该层 donor-only identities 后 pass→fail；从失败
union 只移除同一批 identities 后 fail→pass。144 个槽中只有 3 个命中，且 bicycle–violin 只有一个 base/target
方向命中。这排除了把负边简化为与输入目标无关的静态坏层。

### A1. bicycle–toilet layer 20

- 安全 toilet base：pass，NLL 0.3742；失败 union：缺少 `window`，NLL 0.5826。
- 只加入 435 个 layer-20 bicycle-only identities：fail，NLL 0.3831；3 个等量随机 add 全部通过。
- 从 union 移除这 435 个 identities：恢复通过，NLL 0.5769；3 个随机 resident LOO 全部仍失败。
- 因此离散语义的两个方向都与随机 identity 区分。但连续量并非全面更强：add 的 NLL 增量 +0.0090 小于随机
  均值 +0.0117，使冻结的 joint NLL advantage 为 −0.00275。该候选的主证据是可复现的严格语义翻转，NLL
  只提供混合而非一致增强。

### A2. bicycle–violin layer 0/20

安全 bicycle base 通过，失败 union 输出空 final caption。layer 0 与 20 的真实 add 均使 bicycle 目标失败，
且随机 add 0/3 失败；真实 LOO 又都恢复正常 caption。但每层的随机 resident LOO 也是 3/3 恢复。这说明：

- donor identity 对“从安全 base 触发失败”具有特异性；
- 在完整失败 union 上，移除同层大量 resident identity 本身就足以跨过生成边界；
- 故完整层剂量不能证明 LOO 方向的 identity 特异性，必须进入更小集合并重新匹配随机数量。

### A3. violin–people 没有单层双向命中

这不等于 Phase 6C 的负干扰消失；端点仍精确复现。它说明在当前逐层 partition 下，没有任何一个 violin-only
层增量既能单独触发 people 失败，又能从完整 union 中单独移除后恢复。更可能的解释包括跨层协同、多个替代
冲突路径或高阶组合，而不是一个可由单层截获的 locus。按预注册门槛，该边不进入 identity 拆分。

## B. deterministic identity delta-debugging

| contrast/layer | 初始 identity | 最终 identity | 保留比例 | 接受缩减 | proposals | 最终随机联合命中 |
|---|---:|---:|---:|---:|---:|---:|
| bicycle–toilet L20 | 435 | 135 | 31.0% | 6 | 88 | 0/3 |
| bicycle–violin L20 | 1,082 | 337 | 31.1% | 5 | 78 | 0/3 |
| bicycle–violin L0 | 1,040 | 325 | 31.3% | 6 | 88 | 0/3 |
| **合计** | **2,557** | **797** | **31.2%** | **17** | **254** | **0/9** |

最终三个真实子集全部满足冻结联合谓词：

- bicycle–toilet L20 的 135 identities：add 输出截断的 `A toilet sits next`，LOO 恢复为包含 toilet、window、
  shower/bathroom 的正常句；
- bicycle–violin L20 的 337 identities：add 的 final caption 为空，LOO 恢复 bicycle+clock 正常句；
- bicycle–violin L0 的 325 identities：同样为 add 空输出、LOO 恢复正常句。

所有接受缩减步骤共配 51 对随机 add/LOO，只有 1/51 随机对同时满足联合冲突谓词；最终三个大小上的随机对为
0/9。尤其 bicycle–violin 在完整层剂量时随机 LOO 3/3 恢复，而缩到 337/325 后最终随机 LOO 为 0/6 恢复，
说明 identity 特异性只有在控制干预剂量后才显现，不能从整层结果直接读出。

三项搜索都以 `budget_exhausted_before_full_granularity` 停止：剩余预算不足以完整枚举下一粒度，因此 runner
没有用部分候选做选择。最终集合是预算内的构造性上界，不是 1-minimal，更不是全局最小冲突集。

## C. FP32 边界诊断

对每个 BF16 最终集合，以 FP32 hook 重跑 safe base、完整 union、final add 和 final LOO，共 12 次。三项均完整
复现 `base pass, union fail, add fail, LOO pass`：

| 候选 | BF16 双向谓词 | FP32 双向谓词 |
|---|---:|---:|
| bicycle–toilet L20 / 135 | 是 | 是 |
| bicycle–violin L20 / 337 | 是 | 是 |
| bicycle–violin L0 / 325 | 是 | 是 |

因此当前最终冲突集不像 Phase 6C 那个 7/8 结构复验例一样由 BF16 决策边界单独造成。但这只是对 BF16 选择
结果的 FP32 hook 敏感性诊断，不是独立 FP32 搜索，也不是窄矩阵真实 checkpoint 的结构等价验证。

## 解释与下一门

Phase 6D 当前支持的模型是：**冲突边可由稀疏、方向性、base/target 条件化的 identity 集合介导；同一 layer
可以在不同 edge 上复现，但不存在对所有负边统一的坏层。** layer 20 在 bicycle–toilet 和 bicycle–violin
两个槽出现，值得作为共享冲突路由候选；layer 0 只在 bicycle–violin 出现；violin–people 需要跨层搜索。

下一步不能直接宣布“删掉 797 个 identity 就修好了三个 union”。应依次验证：

1. 对每个 pair 从失败 union 中抑制其最终冲突集，同时评估 pair 的两个 target，确认恢复受损目标时不破坏另一
   个目标；
2. 配同层同量随机抑制，并测试 L0、L20 以及二者组合，显式检查修复是否也非单调；
3. 对 violin–people 使用预注册的跨层 deterministic partition，而不是放宽成单方向命中；
4. 通过 hook 修复门后再物化代表 checkpoint，并分别报告 BF16/FP32 的逐 token 等价；
5. 只有实现按输入激活 `C7 + S_i` 或冲突感知稀疏门控，才能声称降低活跃计算。若所有 shell 常驻，静态参数
   存储量仍未减少。

## 产物

- 预注册：`docs/my-exper/typing neuron/phase6d_conflict_localization_plan.md`
- layer 主结果：`saves/neuron_typing/phase6d_conflict_localization_v1/phase6d_localization.json`
- layer 全部评估：`saves/neuron_typing/phase6d_conflict_localization_v1/evaluations.jsonl`
- layer 分析：`saves/neuron_typing/phase6d_conflict_localization_v1/analysis.json`
- identity 主结果：`saves/neuron_typing/phase6d_identity_refinement_v1/phase6d_identity_refinement.json`
- identity 全部评估：`saves/neuron_typing/phase6d_identity_refinement_v1/evaluations.jsonl`
- identity 分析：`saves/neuron_typing/phase6d_identity_refinement_v1/analysis.json`
- 最终 identity 列表：`saves/neuron_typing/phase6d_identity_refinement_v1/candidates/*.json`
- FP32 诊断：`saves/neuron_typing/phase6d_identity_refinement_v1/float32_diagnostic.json`
