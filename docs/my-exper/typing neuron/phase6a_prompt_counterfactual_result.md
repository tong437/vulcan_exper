# Phase 6A：Prompt 与反事实鲁棒性审计结果

**模型：** Qwen3.5-0.8B 原模型与 Phase 5E 62,950-neuron 物理模型

**审计版本：** `phase6a_prompt_counterfactual_v2`

**生成协议：** `do_sample=false`、`enable_thinking=false`、`max_new_tokens=64`

## 一句话结论

62,950-neuron 模型保留了部分 prompt/轻扰动语义鲁棒性，也会随反事实图像改变核心物体，因此不是完全无视图像的
固定 caption 复读器；但它没有通过 Phase 6A 总 gate：15 个案例中 6 个发生角色词尾巴或重复退化，paired
retention 仅为 8/14。该模型保留了视觉条件性，却没有保留可靠的生成稳定性。

## 预注册 Gate 结果

| Gate | 原模型 | 62,950 模型 | 阈值 |
|---|---:|---:|---:|
| Canonical target | pass | pass | 必须 pass |
| 无泄露 prompt | 5/5 | 4/5 | 至少 4/5 |
| Benign image | 3/3 | 2/3 | 至少 2/3 |
| 反事实原场景复述 | 0/6 | 0/6 | 必须 0/6 |
| 反事实完整通过 | 6/6 | 2/6 | 至少 5/6 |
| 全部输出干净 | 15/15 | 9/15 | 必须 15/15 |
| 总 gate | pass | **fail** | 全部子 gate 通过 |

原模型通过全部主 gate。显式提到 bicycle 的诊断 prompt 不计入主 gate；原模型回答
`they are parked in the train car.`，由于没有在答案中再次写出 bicycle，被只看答案的 Phase 5E 自动语义门标为
fail，但结合问题语境是合理回答。

## Prompt 与 benign 图像结果

62,950 模型在 canonical、generic、main-scene、photo-question 四个 prompt，以及水平翻转和 90% 中心裁剪上正常
输出：

> A group of bicycles on a subway train.

但 alt-text prompt 和 0.65 亮度图都在正确句后继续生成 `Question User User ...`，因此分别造成 prompt 与 benign
组的一个失败。

输出多样性揭示了明显的 target attractor：

| 指标，五个主 prompt + 三个 benign 图像 | 原模型 | 62,950 模型 |
|---|---:|---:|
| 案例数 | 8 | 8 |
| 唯一答案数 | 8 | 3 |
| 与 gold caption 完全相同 | 0 | 6 |
| 以 gold caption 开头 | 0 | 8 |

因此，正向案例通过不能单独解释为自然语言鲁棒性。learned gate 对 gold caption 的优化形成了非常强的解码吸引子；
prompt 和轻微图像变化通常仍落入同一句答案，其中两个条件在正确句后越过 EOS 稳定区并进入角色词重复。

## 反事实结果

| 反事实 | 原模型 | 62,950 模型 | 62,950 主要问题 |
|---|---:|---:|---|
| Train-only crop | pass | fail | 识别 train 且不再提 bicycle，但追加 `User` 垃圾 |
| 自行车区域遮挡 | pass | fail | 不再提 bicycle，但追加 `User` 垃圾 |
| Bike-only 错图 | pass | fail | 正确识别 bicycle clock，但追加 `User` 垃圾 |
| Train-only 错图 | pass | pass | 核心事实正确，语言有重复 |
| 无关 bathroom 图 | pass | automatic pass | 未复述原场景，但 `A bluey with a life...` 人工看质量较差 |
| Blank image | pass | fail | 未复述原场景，但发生 `a group of` 长重复并截断 |

两个现象必须分开解释：

1. **视觉条件性仍在。** 62,950 模型的 6/6 反事实都满足冻结的核心事实布尔检查，且 0/6 复述完整
   bicycle-in-train 场景。bike-only 图会输出 bicycle clock，train-only 图会输出 train，遮挡自行车后不再声称有
   bicycle。这反驳了“模型完全忽略图像、无条件背诵 gold caption”的最强假设。
2. **生成可靠性显著损失。** 加入正常终止、角色词和重复检查后只有 2/6 自动通过；人工观察 bathroom 输出后，其中
   一个自动通过仍应降为低质量/review。主要失败不是反事实事实方向错误，而是正确短句后的 EOS/角色词退化或空白输入
   下的重复崩溃。

## Paired 结果与可表述结论

原模型通过 14 个自动可判定案例，62,950 模型保留其中 8 个，paired retention 为 `8/14 = 57.14%`。删除量仍为：

- FFN neurons：62,950 / 86,016，73.18%；
- 总参数：193,382,400 / 852,985,920，22.67%。

建议表述：

> Phase 5E 的极端稀疏物理模型在固定图像的多种 prompt 和轻微视觉变换下仍能恢复目标语义，并对错图和目标区域
> 删除表现出内容敏感性；但是其输出高度坍缩到优化过的 gold caption，且在 40% 的审计条件下出现生成尾部退化。
> 因此 62,950 是固定任务语义的构造性下界，不是鲁棒单样本视觉子网的下界。

不能表述为：

- 62,950 模型具有稳定的 prompt invariance；
- 该模型只是完全无视图像地背诵 caption；
- 反事实 2/6 表明其视觉识别只剩三分之一——核心事实方向实际为 6/6，失败主要来自生成稳定性；
- automatic bathroom pass 代表高质量 caption。

## 下一步建议

下一步应搜索“鲁棒 frontier”，而不是继续提高单 prompt 删除量：

1. 用同一 Phase 6A gate 评估 48k、56k、58k、60k、62k、62.5k、62.95k masks；
2. 将多个无泄露 prompt 纳入 learned-gate 优化，但保留至少两个完全 held-out prompt；
3. 加入错图/遮挡 contrastive 目标，约束 mask 对正确图像和反事实图像产生不同答案分布；
4. 把 EOS margin、角色 token margin 和重复风险作为搜索代理，最终仍由自由生成裁决；
5. 只对最大 robust-pass mask 做物理保存和独立重载。

## 可审计产物

- 预注册计划：`docs/my-exper/typing neuron/phase6a_prompt_counterfactual_plan.md`
- 完整逐案例结果：
  `saves/neuron_typing/phase6a_prompt_counterfactual_62950_v2/phase6a_results.json`
- 最终结果 SHA-256：`479516be1458591b281bf105ab79f657a8c6c1734c2323fd0974318de81a3843`
- 冻结 case manifest：
  `saves/neuron_typing/phase6a_prompt_counterfactual_62950_v2/audit_inputs/dataset/case_manifest.json`
- 派生输入图像：
  `saves/neuron_typing/phase6a_prompt_counterfactual_62950_v2/audit_inputs/derived_images/`
- v1 无图接口失败产物：`saves/neuron_typing/phase6a_prompt_counterfactual_62950_v1/`
