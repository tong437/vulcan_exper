# Phase 6E：Repaired Union 结果

## 结论先行

Phase 6E 的冻结主判据在 hook 层面完整成立：

- `bicycle–toilet` 从 failed union 的 1/2，通过删除 L20 的 135 个 Phase 6D 冲突 identity，恢复为 2/2；
- `bicycle–violin` 从 0/2，通过联合删除 L20 的 337 个和 L0 的 325 个冲突 identity，恢复为 2/2；
- 每个 pair 的 10 个同来源、同层、同量随机 repair 均为 0/10 双任务通过；
- FP32 对完全相同的冻结 mask 重跑全部 52 次评估，复现相同的主判据与 0/20 随机双通过结果。

因此，Phase 6D 找到的集合不只是能在单个受损端点产生 `add fail + LOO pass`，还足以在 pair level 定向修复
两个失败 union。结果支持 **identity-specific conflict repair**，而非单纯减少宽度。

真实结构部署存在一个必须保留的限制：`bicycle–violin` 的 BF16 紧凑模型双任务稳定；`bicycle–toilet` 的
BF16 紧凑模型在 toilet 端点翻回失败。两者在 FP32 紧凑模型中均 2/2 通过且与 hook 逐 token 一致，说明前者
是可靠 repair，后者则是因果 repair 成立但 BF16 结构部署处于数值决策边界。

## 冻结输入与完整性

- Phase 6D 三个最终集合的 candidate ID、layer、数量、identity SHA-256、base/donor/target provenance 全部
  通过硬编码冻结校验。
- bicycle–toilet 的随机池严格为 `(K_bicycle \ K_toilet)[L20]`；bicycle–violin 的随机池严格为
  `(K_violin \ K_bicycle)[L0,L20]`。
- 每个随机集合与真实冲突集合允许自然重叠：bicycle–toilet 每个 seed 重叠 38–47/135；bicycle–violin
  分别约重叠 98–120/325 和 95–113/337。这是保守对照，而非刻意避开 treatment identity。
- BF16 与 FP32 各 26 个 mask、52 次双端点评估；evaluation ID、mask hash 和 variant manifest 全部一致。
- `violin–people` 没有冻结单层双向集合，本阶段未进行事后跨层搜索。

## BF16 主实验

| pair | 条件 | 双任务 | target 通过 | 两端平均 NLL |
|---|---|:---:|---:|---:|
| bicycle–toilet | safe base `K_toilet` | 是 | 2/2 | 1.267 |
| bicycle–toilet | failed union | 否 | 1/2 | 0.886 |
| bicycle–toilet | repaired union `−L20×135` | **是** | **2/2** | 0.927 |
| bicycle–toilet | 10× random repair | 0/10 | 10/20 | — |
| bicycle–violin | safe base `K_bicycle` | 是 | 2/2 | 0.881 |
| bicycle–violin | failed union | 否 | 0/2 | 0.722 |
| bicycle–violin | repaired union `−L20×337−L0×325` | **是** | **2/2** | 0.693 |
| bicycle–violin | 10× random repair | 0/10 | 4/20 | — |

恢复后的确定性输出为：

- bicycle–toilet / bicycle：`A bicycle clock with a black metal frame and white clock face.`
- bicycle–toilet / toilet：`A toilet sits next to a window and shower in a bathroom.`
- bicycle–violin / bicycle：`A bicycle clock with a black metal frame and a large round clock face.`
- bicycle–violin / violin：`A man is playing a violin in a kitchen.`

failed union 的平均 NLL 并不总比 repaired union 高，尤其 bicycle–toilet 的 failed NLL 更低。这再次说明
teacher-forced gold NLL 不能替代正常终止与完整概念契约；本阶段的主终点必须保持为严格自由生成双任务通过。

## FP32 冻结复验

FP32 没有重新搜索或更改 mask。两个 safe base 仍双通过，两个 failed union 仍分别为 1/2 和 0/2，两个
repaired union 均为 2/2，20 个 matched-random 仍无一个双通过。

| pair | FP32 safe NLL | FP32 failed NLL | FP32 repaired NLL | repaired 双任务 |
|---|---:|---:|---:|:---:|
| bicycle–toilet | 1.286 | 0.897 | 0.928 | 2/2 |
| bicycle–violin | 0.876 | 0.729 | 0.705 | 2/2 |

这排除了 repair 只依赖 BF16 hook 决策边界的解释。

## 真实结构模型

两个 repaired union 均已物化为永久 BF16 和 FP32 非均匀宽度 checkpoint。四个 checkpoint 的 24 层宽度、
gate/up/down 投影权重 hash、删参量公式全部精确匹配。

| pair | 结构参数量 | 相对原模型减少 | BF16 结构 | FP32 结构 |
|---|---:|---:|:---:|:---:|
| bicycle–toilet | 758,478,912 | 94,507,008 (11.08%) | 1/2 | **2/2，逐 token 等价** |
| bicycle–violin | 748,356,672 | 104,629,248 (12.27%) | **2/2，逐 token 等价** | **2/2，逐 token 等价** |

bicycle–toilet 的 BF16 hook 输出满足 toilet 契约，但物化后的窄矩阵输出重新缺少 `window`。其维度和权重
完全一致，hook→结构 teacher-forced logit 平均绝对差为 0.0329；FP32 中该差降为 `2.82e-6`，输出恢复逐
token 一致。该模式与 Phase 6C 的同一 pair 边界现象一致，应解释为 BF16 GEMM 归约顺序引发的部署不稳定，
不能解释为 mask 或结构复制错误。

FP32 结构模型中四个目标的 hook→结构平均绝对 logit 差为 `2.57e-6–2.93e-6`。

## Phase 6E 证明了什么

在当前冻结样本与 mask 上，证据链现在达到：

1. Phase 6C 观察到 union 非单调失败；
2. Phase 6D 把失败定位到具体、方向相关的 identity 集合；
3. Phase 6E 删除这些集合后，两个失败 union 都恢复 pair-level 双任务能力；
4. 同来源等宽随机删除 0/20 双通过，排除一般容量缩减；
5. FP32 真实结构模型完整复现，证明修复可以落到物理子网。

因此可以主张：**至少部分子网组合冲突由可定位、可删除、可结构化实现的 neuron identity 集合因果驱动。**

仍不能主张这些集合是最小、唯一或跨样本普适的；也不能把 bicycle–toilet 称为 BF16 可稳定部署的 repaired
union。对该 pair，当前可部署结论仅在 FP32 成立。

## 下一步建议

下一阶段不宜继续优化这两个冻结端点。更有信息量的方向是：

1. 在新的同类样本或 prompt 改写上测试这 797 个 identity 是否仍预测冲突/修复，区分实例特异与关系特异；
2. 对 `violin–people` 做跨层组合定位，检验无单层命中是否源于必要的 layer interaction；
3. 构建冲突感知路由，在运行时选择 safe shell 或 repaired union，并把 BF16 margin/精度稳定性纳入路由门控；
4. 对 bicycle–toilet 增加稳定性裕量目标，而不是事后按一次 BF16 结构输出继续挑 identity。

## 产物

- BF16 主结果：`saves/neuron_typing/phase6e_repaired_union_v1/phase6e_repaired_union.json`
- BF16 分析：`saves/neuron_typing/phase6e_repaired_union_v1/analysis.json`
- BF16 结构诊断：`saves/neuron_typing/phase6e_repaired_union_v1/structural_verification_auto.json`
- BF16 永久模型：`saves/neuron_typing/phase6e_repaired_union_v1/structural_models/`
- FP32 主结果：`saves/neuron_typing/phase6e_repaired_union_float32_v1/phase6e_repaired_union.json`
- FP32 分析：`saves/neuron_typing/phase6e_repaired_union_float32_v1/analysis.json`
- FP32 结构验证：`saves/neuron_typing/phase6e_repaired_union_float32_v1/structural_verification.json`
- FP32 永久模型：`saves/neuron_typing/phase6e_repaired_union_float32_v1/structural_models/`
