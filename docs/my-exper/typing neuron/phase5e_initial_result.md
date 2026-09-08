# Phase 5E 初始执行结果

## 1. 当前结论

Phase 5E 的评测、fresh 静态搜索、exact-budget learned gate、旧 checkpoint 回溯和物理结构验证链路均已实现。
在当前单样本、Taylor 初始化的 learned-gate 候选族和 `phase5e_bicycles_in_train_v2` 语义契约下，已经得到
一个经过保存、独立重载和人工确认的构造性下界：

- 删除 `62,950 / 86,016 = 73.18%` 的 FFN neurons；
- 删除 `193,382,400 / 852,985,920 = 22.67%` 的模型参数；
- 重载后确定性输出：`A group of bicycles on a subway train.`；
- 自动语义门：`pass`；
- 正常 EOS 结束，未进入 thinking，未触及 64-token 上限；
- 当前 62,950 checkpoint 已由用户人工确认；确认记录使用独立 sidecar 保存，避免改写已有 frontier 及其
  provenance hash。

这不是全局最优证明。当前搜索在 63,000 的三个 restart 中没有得到干净 caption，但这只构成当前搜索算法、
seed 数和候选族下的经验上侧观测。

## 2. 冻结评测输入

- 图像：`COCO_val2014_000000093736.jpg`（原始 COCO val source offset 2500）；
- gold caption：`A group of bicycles on a subway train.`；
- prompt：要求只输出一句简洁英文 caption；
- template：`qwen3_5`，`enable_thinking=false`；
- 生成：`do_sample=false`、`temperature=null`、`max_new_tokens=64`；
- gold teacher forcing：11 个监督 tokens，仅用于 NLL/KL/margin、静态排序和 learned-gate 优化；
- 最终成败：由从原始图像和 prompt 的自由生成 caption 及语义门决定。

## 3. 旧物理 checkpoint 回溯

当前 v2 契约下，原模型和已找到的四个旧物理 checkpoint 均自动通过：

| checkpoint | 删除 neurons | 自动语义结果 |
| --- | ---: | --- |
| original | 0 | pass |
| core13 | 13 | pass |
| physical_250 | 250 | pass |
| physical_750 | 750 | pass |
| physical_1000 | 1,000 | pass |

旧 checkpoint 回溯因此立刻把原来的严格轨迹边界 `13` 提升到了语义下界 `1,000`。这也验证了“思考轨迹
分叉不等于最终任务语义失败”。

## 4. Fresh 静态搜索

Taylor saliency 从未剪枝模型上对 gold caption 重新计算，没有沿用 core13 的轨迹等价排序。主要边界候选为：

| 删除 neurons | gold mean NLL | delta NLL | 自由 caption | 自动结果 |
| ---: | ---: | ---: | --- | --- |
| 32,000 | 2.044945 | +0.366267 | `A train car with two bicycles...` | pass |
| 40,000 | 1.815380 | +0.136702 | `An interior view of a train or a bus...` | fail |
| 48,000 | 1.688707 | +0.010029 | `A train with a few bikes in the back.` | pass |
| 56,000 | 1.787041 | +0.108364 | 单数 bicycle 且重复、截断 | fail |
| 64,000 | 2.829175 | +1.150498 | `A bike is in the back of a train...` | review（不自动通过） |

通过性并不随删除预算单调：40,000 失败而 48,000 通过。因此后续缩窄边界必须把各预算视为独立 mask
搜索，不能把它当成严格数学二分。

## 5. 静态候选物理化与保存/重载

48,000-neuron mask 已转换为 24 层非均匀 FFN 宽度，真实执行 gated-MLP 结构删除。hook 模型、内存中的
物理模型以及保存后独立重载的模型均输出：

> A train with a few bikes in the back.

物理模型参数量为 `705,529,920`；重载后的 gold mean NLL 为 `1.671587`，相对本次重载 reference 的
delta NLL 为 `-0.007090`。mask hash、逐层宽度、参数变化、caption、自动语义结果及推理统计均保存在结果
文件中。

## 6. 正式 learned-gate 边界搜索

正式搜索使用 exact-budget hard top-k + STE，在 `50,000、52,000、54,000、56,000` 每个预算运行 3 个
restart，每个 restart 运行 200 steps。自由生成结果如下：

| 删除 neurons | 干净通过 restart | 结果摘要 |
| ---: | ---: | --- |
| 50,000 | 3/3 | 三个候选均正常生成 gold caption |
| 52,000 | 2/3 | 一个候选在正确句子后重复 `assistant` 并达到 token 上限 |
| 54,000 | 0/3 | 一个候选正常结束但重复同一句三次；其余两个达到 token 上限 |
| 56,000 | 2/3 | restart 1 和 2 均正常生成 gold caption |

54,000 的重复三次候选虽然满足 v2 自动核心语义规则，但不满足单句简洁输出要求，因此在候选审查时不计作
干净通过。这也表明最终 frontier 不能只汇总自动布尔值，必须同时审查完整 caption 和停止状态。

最大干净候选 `delete_56000__restart_1` 的 hook gold mean NLL 为 `0.355123`，delta NLL 为 `-1.323554`，
自由生成正常 EOS 结束。该 mask 已完成真实结构删除、保存和独立重载：

- 重载模型参数量：`680,953,920`；
- 重载 gold mean NLL：`0.355561`；
- 重载 delta NLL：`-1.323117`；
- 重载 caption：`A group of bicycles on a subway train.`；
- 自动语义门：`pass`；
- 原模型/剪枝模型单次生成峰值显存：约 `1.800 GB / 1.457 GB`；
- 人工确认：`pass`。

56k checkpoint 已由用户人工确认，可纳入最终报告。

## 7. 上侧搜索与 50-neuron 边界缩窄

上侧搜索继续在 `58,000、60,000、62,000、64,000` 各运行 3 个 200-step restart：

| 删除 neurons | 干净通过 restart | 结果摘要 |
| ---: | ---: | --- |
| 58,000 | 3/3 | 全部正常生成 gold caption |
| 60,000 | 3/3 | 全部正常生成 gold caption |
| 62,000 | 1/3 | 一个正常通过，一个无 EOS，一个空最终答案 |
| 64,000 | 0/3 | 全部在正确开头后发生重复并触及 token 上限 |

62,000 已完成物理保存和独立重载验证。随后依次探测 63,000、62,500、62,750、62,875 和 62,950：

| 删除 neurons | 干净通过 restart | 经验结论 |
| ---: | ---: | --- |
| 62,500 | 1/3 | pass |
| 62,750 | 1/3 | pass |
| 62,875 | 2/3 | pass |
| 62,950 | 1/3 | pass |
| 63,000 | 0/3 | no-clean-pass；一个自动 pass 含重复 `user` 垃圾 |

当前经验区间为 `62,950 pass / 63,000 no-clean-pass`，宽度 50 neurons。最大干净候选
`delete_62950__restart_0` 已完成真实物理结构删除、保存和独立重载：

- 剩余 FFN neurons：`23,066`；
- 重载模型参数量：`659,603,520`；
- 重载 gold mean NLL：`0.343445`；
- 重载 delta NLL：`-1.335233`；
- 重载 caption：`A group of bicycles on a subway train.`；
- 自动语义门：`pass`；
- 原模型/剪枝模型单次生成峰值显存：约 `1.800 GB / 1.414 GB`；
- 人工确认：`pass`。

## 8. 关键产物

- v2 旧 checkpoint 回溯：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_retrospective_caption_prompt_v2.json`
- fresh 静态 frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_static_gold_v2/static_frontier.json`
- 48,000-neuron 结构 artifact、物理筛查和模型：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_static_structural_48000/`
- 48,000-neuron 独立重载结果：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_structural_48000_reload.json`
- learned-gate 冒烟结果：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_smoke/learned_frontier.json`
- learned-gate 正式边界结果：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_boundary_v2/learned_frontier.json`
- 56,000-neuron 结构 artifact、物理筛查和模型：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_56000_r1/`
- 56,000-neuron 独立重载结果：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_56000_r1_reload.json`
- 56,000-neuron 人工确认 sidecar：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_56000_r1/human_confirmation.json`
- 58k–64k learned-gate frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_upper_v2/learned_frontier.json`
- 62,950 learned-gate frontier：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_refine_62950_v2/learned_frontier.json`
- 62,950-neuron 结构 artifact、物理筛查和模型：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0/`
- 62,950-neuron 独立重载结果：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0_reload.json`
- 62,950-neuron 人工确认 sidecar：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0/human_confirmation.json`

## 9. 验证状态

- Phase 5E 定向测试：`16 passed`；
- 所有新增 Phase 5E Python 文件：Ruff check passed；
- 仓库级 `make quality` 仍会报告既存旧脚本中的问题，与本阶段新增文件无关。
