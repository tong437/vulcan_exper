# Phase 5–6：单样本 FFN 剪枝极限、子网几何与组合机制完整报告

**汇报对象：** 导师 / 课题组

**实验对象：** Qwen3.5-0.8B 多模态模型与 COCO 固定图像描述样本

**报告范围：** Phase 5A–5E、Phase 6A–6E

**报告日期：** 2026-09-08

## 一、技术摘要

Phase 5–6 围绕一个递进问题展开：**对于固定图像描述任务，模型中究竟需要保留哪些 FFN neurons；这些
单样本子网是否鲁棒、是否共享结构、能否直接组合，以及组合冲突能否被定位和修复？**

Phase 5 的核心发现是，剪枝极限高度依赖“保持什么”：

- 若要求物理模型保持原模型前 512 个生成 token、top-1 与 `mean KL <= 0.001`，当前构造性下界仅为删除
  13 个 FFN neurons；
- 若只要求固定图像仍生成正确的目标语义，则可以真实删除 `62,950 / 86,016 = 73.18%` 的 FFN neurons，
  删除 `193,382,400 / 852,985,920 = 22.67%` 的总参数；
- 62,950-neuron 候选保存并独立重载后仍输出 `A group of bicycles on a subway train.`，自动语义门和人工
  确认均通过；当前局部经验区间为 `62,950 pass / 63,000 no-clean-pass`，但 63,000 不是理论上界。

Phase 6 表明，这种极端单样本子网不是简单的“背答案模块”，但也不是鲁棒通用子网：

- Phase 6A 中，62,950-neuron 模型对 6/6 反事实都改变了核心事实方向，说明仍使用视觉输入；但 15 个审计
  条件只有 9 个输出干净，paired retention 为 `8/14 = 57.14%`，总鲁棒性 gate 失败；
- Phase 6B 在 7 个不同语义样本上都找到并结构化验证了单样本子网，最大可行删除量为 32,000–62,950，说明
  单样本充分子网的规模明显依赖样本；
- 七个赢家的共同交集为 10,970 个 neurons，是逐层宽度匹配随机期望的 31.23 倍，形成强共享核心候选；
- Phase 6C 证明共享核心在高剂量下必要但单独不充分，学得外壳具有 identity 信息；21 对 union 中 14 对
  双任务通过、7 对真正恢复，但有 3 对出现负干扰；
- Phase 6D 将三个方向性冲突集合从 2,557 个 identities 缩到 797 个，最终等量随机对照 0/9 命中；
- Phase 6E 删除这些冲突 identities 后，两个失败 union 均恢复为双任务通过，而随机 repair 为 0/20；FP32
  真实结构模型完整复现。一个 pair 的 BF16 紧凑模型仍处于数值决策边界。

综合来看，最符合现有证据的结构不是“一个通用充分核心 + 若干互斥任务插件”，而是：

> **必要但不充分的共享骨架、相互重叠且可迁移的条件外壳，以及包含协同边和冲突边的子网组合图。**

这套结果已经达到单样本机制探索的成熟阶段性汇报标准：实验链路完整、物理模型可复验、随机对照和精度复验
充分；但它仍不能支持跨样本总体、模型全能力或真实部署收益方面的普遍结论。

## 二、研究问题与 Phase 5–6 的关系

### 2.1 Phase 5：先回答“单个任务能剪到什么程度”

Phase 5 不是一次单一剪枝实验，而是逐步分离了三种不同目标：

1. **轨迹严格等价：** 保持原模型生成 token、top-1 和输出分布；
2. **最终答案等价：** 保持原模型最终 caption；该曲线尚未被系统独立搜索；
3. **人工语义保持：** 允许同义表达，只要求固定图像任务的核心语义正确。

Phase 5A–5D 主要处理第一种目标，并逐步解决静态排序、learned gate、物理结构化、cached decode 和 BF16
数值路径问题；Phase 5E 则把主终点改为第三种目标，搜索固定任务的语义极限。

### 2.2 Phase 6：再回答“这些单样本子网是什么结构”

Phase 6 并未离开单样本设定，而是把多个独立单样本子网作为研究对象：

- 6A 检验一个极端子网是否只是在固定 prompt 上复读答案；
- 6B 在更多固定样本上复制单样本极限，并分析 neuron identity 几何；
- 6C 对共享核心、条件外壳和 union 进行功能干预；
- 6D 定位 union 的负干扰 identity；
- 6E 验证删除冲突 identity 是否能因果修复失败组合。

因此，Phase 5 给出“可行子网”，Phase 6 则研究“子网之间的关系与机制”。

## 三、统一实验设置与指标定义

### 3.1 模型与剪枝对象

- 基础模型参数量：`852,985,920`；
- Transformer FFN 层数：24；
- 每层原始 intermediate width：3,584；
- FFN neuron 总数：`24 × 3,584 = 86,016`；
- 每删除一个 gated-MLP neuron，同步删除 `gate_proj`、`up_proj` 的一行和 `down_proj` 的一列；
- 对该模型，每个 FFN neuron 对应 `3 × hidden_size = 3,072` 个参数。

所有主实验冻结原模型权重。mask 是 sample-static：同一个图像的视觉 token、prompt token 和生成 token 使用
同一组二值 FFN mask，不属于 token-dynamic routing。

### 3.2 Hook mask 与物理结构模型

实验明确区分两条执行路径：

- **Hook/gate 路径：** 保留原始矩阵宽度，在 FFN 中间激活上置零；适合快速搜索；
- **物理路径：** 真正缩窄每层 `gate/up/down` 投影并保存非均匀 checkpoint；用于最终结论。

两者数学上都表示删除同一 identity，但 BF16 GEMM 的矩阵形状、kernel 和归约顺序不同，边界 token 上可能产生
可改变解码结果的数值误差。因此，最终可部署结论一律以保存、独立重载后的物理模型为准。

### 3.3 三类评价终点

| 终点 | 主指标 | 它回答的问题 | 主要限制 |
|---|---|---|---|
| 严格轨迹保持 | exact generation、top-1=100%、KL 阈值 | 是否复制原模型行为轨迹 | 对任务语义过严，且依赖 horizon |
| Teacher-forced proxy | gold NLL、KL、margin | 哪个 mask 值得优先搜索 | 不能保证自由生成、EOS 或语义 |
| 自由生成语义 | 冻结概念契约、正常终止、人工确认 | 固定任务是否仍被正确完成 | 只覆盖预定义语义与固定样本 |

Phase 5E–6E 的确定性生成统一使用 `do_sample=false`、`temperature=null`、`enable_thinking=false` 和
`max_new_tokens=64`。随机性来自 mask 优化、对照采样或搜索 restart，而不是生成采样。

## 四、Phase 5：从严格轨迹边界到任务语义极限

### 4.1 Phase 5A：静态排序建立严格轨迹搜索基线

Phase 5A 使用 activation、down-projection contribution、first-order Taylor 和 random 四类静态分数，并比较
逐层均匀预算与全局非均匀预算。固定 teacher rollout 后，对每个候选重新计算 KL、top-1、margin 和自由生成。

在 64-token 严格 gate 下，Taylor-global 删除 301 个 neurons 时通过：

- mean KL：0.000993；
- 自洽位置 top-1 agreement：100%；
- 64-token generation：完全一致；
- 全模型参数下降：0.1084%。

同预算的 activation、contribution 与 random 均失败，支持 Taylor 对固定轨迹具有更强的样本特异定位能力。
删除主要集中在后层：第 22–23 层占 73.4%，提示可删空间高度非均匀，但不能据此推断后层普遍不重要。

### 4.2 Phase 5B：Learned gate 将 64-token hook 下界提高到 1,000

Phase 5B 采用 exact-budget hard top-k gate、straight-through estimator、Taylor 初始化、200 steps 和每预算
3 个 restart。主损失保护 teacher 分布、teacher token 和生成 margin。

结果中，删除 500、750、1,000 分别有 2/3、2/3、1/3 restart 通过；1,125–1,500 未找到可行解。因此
64-token hook 严格下界从 301 提升到 1,000。相同预算成功 masks 的 Jaccard 仅约 0.45–0.56，说明固定样本
周围存在多个近等价子网，而不是唯一删除列表。

### 4.3 Phase 5C：物理结构改变了可行边界

Phase 5C 将 learned masks 真实物理化。最大完整通过候选是删除 750 个 neurons：

- hook / 内存物理 / 磁盘重载均保持 64-token exact；
- 物理与重载 mean KL 均为 0.000900；
- 删除参数 2,304,000，总参数下降 0.2701%。

Phase 5B 的 1,000-neuron hook 候选在物理模型中 KL 上升到 0.001074，并于 token 13 分叉。结构宽度、权重
hash 和保存重载均正确，差异来自 BF16 窄矩阵执行路径。这一步确立了后续原则：**hook 只作搜索代理，物理
模型才是最终 gate。**

将窗口延长到 256 tokens 后，已有 500/750 masks 都在 token 83 分叉，表明短窗口安全不能自动外推到长窗口。

### 4.4 Phase 5D：Cached-path 优化与 512-token 物理核心

Phase 5D 直接沿 teacher token 重建 cache，优化 cached KL、top-1 与生成 margin，并在候选阶段加入真实物理
筛查。

在 256-token 窗口下形成两条不同边界：

- 严格 `mean KL <= 0.001` 的物理下界：删除 100 个 neurons；
- 只要求 256-token 行为完全一致的物理、可重载下界：删除 1,000 个 neurons。

1,000-neuron 物理模型在 512/1024 外推时都于 token 276 首次分叉，因此不能称为 512-token 保持。

针对 512-token 轨迹，常规 learned masks 的 6 个 hook-exact 候选在物理路径全部失败。随后改用原子物理扫描和
physical-in-the-loop 逐 neuron 扩张：

1. 从 24 层各删一个 Taylor 候选开始，得到首个 3-neuron 物理核心；
2. 每轮固定已验证核心，再在 24 层扫描一个新增候选；
3. 在 rank-1 停滞时扩展 Taylor rank-2–8；
4. 每个正式核心都保存、独立重载并验证 512-token exact、top-1 和 KL。

最终构造出 13-neuron 物理 checkpoint，保持前 512 个生成 token 完全一致且 `mean KL <= 0.001`。搜索在 13
处因计算预算停止，尚未测试 core14，因此 13 是构造性下界，不是理论最大删除量。

### 4.5 Phase 5E：语义目标将物理删除量提高到 62,950

Phase 5E 不再要求思考轨迹或逐 token 一致，而是冻结语义契约：caption 必须表达多辆自行车位于
train/subway/carriage 内或车上，且完整、连贯、无否定、无矛盾、无角色词垃圾并正常结束。

评价采用三层结构：

1. 对 11 个 gold tokens 做 teacher forcing，NLL/KL/margin 只用于排序；
2. 从原图和 prompt 自由生成，运行保守语义门；
3. 只对 frontier 物理候选做人工确认。

旧 checkpoint 回溯中，core13、physical-250、750 和 1,000 都通过语义门，立即证明“轨迹分叉不等于任务
失败”。从原模型 fresh 计算 gold-Taylor 后，静态 mask 已在删除 48,000 时物理通过。Exact-budget learned
gate 继续把边界推进到 62,950：

| 删除预算 | 干净通过 restart | 解释 |
|---:|---:|---|
| 50,000 | 3/3 | 稳定生成 gold caption |
| 56,000 | 2/3 | 两个候选正常结束 |
| 58,000 | 3/3 | 全部通过 |
| 60,000 | 3/3 | 全部通过 |
| 62,000 | 1/3 | 其余出现无 EOS 或空答案 |
| 62,875 | 2/3 | 接近边界仍有两个通过 |
| 62,950 | 1/3 | 当前最大干净候选 |
| 63,000 | 0/3 | 一个核心语义假阳性含角色词垃圾 |
| 64,000 | 0/3 | 重复并达到 token 上限 |

62,950 候选的最终物理结果为：

| 指标 | 原模型 | 62,950-neuron 物理模型 |
|---|---:|---:|
| FFN neurons | 86,016 | 23,066 |
| 总参数 | 852,985,920 | 659,603,520 |
| 删除 FFN 比例 | 0% | 73.18% |
| 总参数缩减 | 0% | 22.67% |
| gold mean NLL | 1.678713 | 0.343445 |
| 最终语义 | pass | pass |

保存并独立重载后输出仍是：

> A group of bicycles on a subway train.

NLL 降低是因为 gold caption 直接参与优化，只表示该目标 caption 的条件概率提高，不能解释为模型整体能力
增强。

### 4.6 Phase 5 的统一结论

| 判据 | 当前物理构造性下界 | 适用范围 |
|---|---:|---|
| 前 64 token 严格轨迹 | 750 neurons | exact + top-1 + KL，物理重载 |
| 前 256 token 严格 KL | 100 neurons | `mean KL <= 0.001` |
| 前 256 token 行为等价 | 1,000 neurons | exact generation，KL 可高于 0.001 |
| 前 512 token 严格轨迹 | 13 neurons | physical-in-the-loop，exact + top-1 + KL |
| 固定 caption 人工语义 | 62,950 neurons | 64-token 短答案、自动门与人工确认 |

13 与 62,950 相差约 4,842 倍，但它们并不矛盾：前者测量长轨迹克隆，后者测量固定任务功能。Phase 5 最重要
的方法论结果正是：**剪枝极限必须与明确的功能契约绑定，不能脱离终点定义报告一个孤立百分比。**

## 五、Phase 6A：极端语义子网保留视觉条件性，但鲁棒性不足

Phase 6A 对 Phase 5E 的 62,950-neuron 物理模型运行 15 个冻结案例，包括无泄露 prompt 改写、轻微图像变换、
目标区域遮挡、错图和空白图。

| Gate | 原模型 | 62,950 模型 | 预注册阈值 |
|---|---:|---:|---:|
| Canonical target | pass | pass | 必须 pass |
| 无泄露 prompt | 5/5 | 4/5 | 至少 4/5 |
| Benign image | 3/3 | 2/3 | 至少 2/3 |
| 反事实原场景复述 | 0/6 | 0/6 | 必须 0/6 |
| 反事实完整通过 | 6/6 | 2/6 | 至少 5/6 |
| 输出干净 | 15/15 | 9/15 | 必须 15/15 |
| 总 gate | pass | **fail** | 全部子 gate 通过 |

需要区分两个现象：

1. **视觉条件性仍在。** 遮挡自行车后不再声称有自行车，bike-only 图输出 bicycle clock，train-only 图输出
   train，6/6 反事实都没有复述完整原场景；
2. **生成可靠性下降。** 多个输出在正确短句后追加 `User/Question` 角色词，或在空白图上重复崩溃，导致完整
   反事实只有 2/6 通过。

在五个主 prompt 加三个 benign 图像上，原模型产生 8 个不同答案；剪枝模型只有 3 个不同答案，6 个与 gold
caption 完全相同，8 个都以 gold caption 开头。这说明 learned gate 形成了强 target attractor。

结论是：62,950 是**固定 prompt/图像语义的构造性下界**，不是鲁棒单样本视觉子网的下界；同时，“模型完全
无视图像、只会背诵”这一最强解释也被反事实事实方向所否定。

## 六、Phase 6B：七个样本都存在可物理化子网，但规模相差 2.34 倍

Phase 6B 在七个语义层不同的 COCO 样本上重复 Phase 5E 搜索，并冻结当前搜索条件下最大已找到可行解。所有
赢家都完成了非均匀物理裁剪、保存、独立重载、层宽/权重 hash 核验和严格自由生成语义检查。

| 样本 | 语义层 | 删除 neurons | 保留 neurons | 保留率 | 总参数缩减 | 重载 |
|---|---|---:|---:|---:|---:|:---:|
| bicycle clock | 物体—属性 | 56,000 | 30,016 | 34.90% | 20.17% | pass |
| violin kitchen | 人物—动作—场景 | 40,000 | 46,016 | 53.50% | 14.41% | pass |
| toilet spatial | 多物体空间关系 | 36,000 | 50,016 | 58.15% | 12.97% | pass |
| people pizza | 数量—动作—食物 | 40,000 | 46,016 | 53.50% | 14.41% | pass |
| giraffes trees | 复数动物—自然场景 | 56,000 | 30,016 | 34.90% | 20.17% | pass |
| dogs car | 数量—动物—空间关系 | 32,000 | 54,016 | 62.80% | 11.52% | pass |
| bicycles train | 数量—物体—空间关系 | 62,950 | 23,066 | 26.82% | 22.67% | pass |

最小赢家保留 23,066 个 neurons，最大赢家保留 54,016 个，相差 2.34 倍。由于三个样本只做了单 restart 粗搜，
这些值是当前搜索的可行下界，不是七个任务的精确最小充分规模。

### 6.1 七重交集形成强共享核心候选

记第 `i` 个冻结赢家保留集合为 `K_i`，定义七重交集 `C7 = ∩K_i`，条件外壳 `S_i = K_i \ C7`。

- `|C7| = 10,970`，占全部 FFN neurons 的 12.75%；
- 至少被 6/7 个赢家保留的 near-core 为 18,584；
- 七个赢家并集为 74,067，另有 11,949 个 neuron 从未被任何赢家保留；
- 逐层宽度条件随机基线的七重交集期望为 351.3，实测为其 31.23 倍；
- 两两 Jaccard 为 0.3590–0.6132，均值 0.4698；
- 按较小子网归一化的两两覆盖率均值为 0.7875。

共享候选明显前置：0–11 层贡献 7,242 个 C7 neurons，12–23 层贡献 3,728 个。该现象既可能反映早层更通用
的视觉—语言变换，也可能来自共同 prompt、Taylor 初始化和训练目标的偏置，因此必须通过功能消融判断。

## 七、Phase 6C：共享核心必要但不充分，子网组合具有协同与负干扰

Phase 6C 固定 Phase 6B 的七个赢家，不重新优化 mask，共评估 633 个唯一 mask、924 次目标推理。

### 7.1 共享核心单独不能完成任务，但高剂量删除具有身份效应

仅保留 C7、C6+ 或 C5+ 时，七个样本均为 0/7 通过，说明高频共享集合不是独立充分的通用 caption 子网。

在每个原子网内部删除一定比例的 C7，并与同层同量随机删除比较：

| 删除剂量 | C7 删除通过率 | 随机删除通过率 | 通过率差 | C7−随机 NLL | 推断 |
|---:|---:|---:|---:|---:|---|
| 10% | 55.7% | 61.4% | −5.7 pp | +0.017 | 无显著身份效应 |
| 25% | 44.3% | 41.4% | +2.9 pp | +0.148 | NLL 变差但语义非单调 |
| 50% | 14.3% | 34.3% | −20.0 pp | +0.743 | identity 效应成立 |
| 100% | 0% | 17.1% | 描述性 | +4.342 | C7 全删破坏更强 |

50% 剂量下，语义通过 McNemar `p=0.00258`，七样本 seed 均值 NLL 的 Wilcoxon `p=0.0156`。因此证据支持
“共享核心在中高剂量下必要”，但不支持低剂量线性效应。

### 7.2 条件外壳具有专属性，也存在有向迁移

| 外壳条件 | 通过率 | 平均 NLL |
|---|---:|---:|
| Learned shell，完整 7×7 | 19/49（38.8%） | 1.856 |
| 同来源逐层等宽随机 shell，3 seeds | 5/147（3.4%） | 2.665 |
| Learned 对角线 | 7/7（100%） | 0.333 |
| Learned 非对角线 | 12/42（28.6%） | 2.109 |

对角线 7/7 说明外壳包含任务条件 identity；非对角线仍有 12/42 通过，说明外壳不是七套互斥私有插件，而是
可被不同任务以不同组合复用的中频 basis。

### 7.3 Union 能恢复多数双任务，但集合超集不保证功能单调

真实 pair union 在 target level 通过 34/42（81.0%），同规模随机扩展为 83/168（49.4%）；pair level
真实 union 为 14/21 双通过，随机扩展为 28/84。

21 对真实 union 中：

- 7 对是“真正恢复”：两个 constituent 单独都不能覆盖两个目标，而 union 双通过；
- 3 对是“负干扰”：至少一个 constituent 原本已经双通过，加入另一个子网后反而失败；
- 其余为继承覆盖或未完全组合。

这同时排除了两个过于简单的模型：

- “union 有效只是因为保留 neuron 更多”；真实 union 显著优于同规模随机扩展；
- “保留集合越大，能力必然单调不降”；三个负干扰边直接反例。

四个代表 union 完成真实结构物化。BF16 下 7/8 个目标 hook→重载逐 token 一致；剩余一个边界翻转在 FP32
下恢复一致，logit 平均绝对误差约 `3e-6`，将问题限定为低精度归约边界而非 mask 复制错误。

## 八、Phase 6D：负干扰由方向性、条件化的稀疏 identity 集合介导

Phase 6D 对 Phase 6C 的三条负干扰边运行 1,164 次逐层 BF16 定位、610 次 identity 级干预和 12 次 FP32
诊断。

六个预注册 pass-to-fail 槽、144 个 contrast×layer 中，只有三个槽满足同层 add 触发失败且 leave-one-layer-out
恢复：bicycle–toilet 的 layer 20，以及 bicycle–violin 的 layer 0 和 20。violin–people 没有单层双向命中，
提示其冲突可能需要跨层组合。

确定性 delta-debugging 结果为：

| 冲突候选 | 初始 identity | 最终 identity | 保留比例 | 最终随机命中 |
|---|---:|---:|---:|---:|
| bicycle–toilet L20 | 435 | 135 | 31.0% | 0/3 |
| bicycle–violin L20 | 1,082 | 337 | 31.1% | 0/3 |
| bicycle–violin L0 | 1,040 | 325 | 31.3% | 0/3 |
| **合计** | **2,557** | **797** | **31.2%** | **0/9** |

三个最终集合都保持冻结谓词：安全 base 通过、完整 union 失败、只加入该集合失败、从 union 删除该集合恢复。
全部集合在 FP32 hook 下复现。搜索因预算不足以完整枚举下一粒度而停止，所以这些集合是构造性上界，不是
1-minimal 或全局最小冲突集。

Phase 6D 支持的结论是：**冲突 identity 具有方向性，依赖 base、donor 和 target；同一个 layer 可以参与多个
edge，但不存在一个对所有负边统一生效的“坏层”。**

## 九、Phase 6E：删除定位到的 identity 可以定向修复失败 union

Phase 6E 从失败 union 中删除 Phase 6D 的冻结冲突集合，并配 10 个同来源、同层、同量随机 repair。主终点是
同一个 repaired mask 在 pair 的两个目标上同时通过。

| Pair | Failed union | 定向 repair | 随机双通过 |
|---|---:|---:|---:|
| bicycle–toilet | 1/2 | 删除 L20×135 后 2/2 | 0/10 |
| bicycle–violin | 0/2 | 删除 L20×337 + L0×325 后 2/2 | 0/10 |
| **合计** | — | **2/2 pairs 修复** | **0/20** |

相同冻结 masks 在 FP32 下重跑 52 次评估，两个定向 repair 仍为 2/2，20 个随机 repair 仍无双通过，排除了
repair 只是一次 BF16 hook 偶然翻转的解释。

两个 repaired union 都被物化为 BF16 和 FP32 非均匀结构 checkpoint：

| Pair | 结构参数量 | 相对原模型减少 | BF16 结构 | FP32 结构 |
|---|---:|---:|---:|---:|
| bicycle–toilet | 758,478,912 | 11.08% | 1/2 | **2/2，逐 token 一致** |
| bicycle–violin | 748,356,672 | 12.27% | **2/2，逐 token 一致** | **2/2，逐 token 一致** |

bicycle–toilet 的 BF16 紧凑模型在 toilet 端点丢失 `window` 概念，而 FP32 结构模型恢复。其层宽、参数量与
权重 hash 均精确，FP32 hook→结构 logit 平均绝对差降至 `2.82e-6`。因此因果 repair 已成立，但该 pair 的
BF16 静态部署尚不稳定。

Phase 6E 将证据链闭合为：观察负干扰 → 定位具体 identity → 删除后 pair-level 恢复 → 等量随机删除不恢复 →
FP32 真实结构模型复现。由此可以主张：**至少部分子网组合冲突由可定位、可删除、可结构化实现的 neuron
identity 集合因果驱动。**

## 十、Phase 5–6 的综合科学结论

### 10.1 单样本任务确实存在极端稀疏的物理 FFN 子网

七个不同样本都找到并重载验证了子网，最极端案例只保留 26.82% FFN neurons。该现象不是纯 hook 仿真，也
不是只看 teacher-forced NLL，而是由物理模型自由生成和语义契约支持。

### 10.2 “需要多少 neuron”不是模型常数，而是契约、样本和搜索算法的函数

同一 bicycles-train 样本，512-token 严格轨迹下界为 13，固定 caption 语义下界为 62,950；跨七个样本，
赢家保留规模又相差 2.34 倍。任何剪枝率都必须同时报告任务、horizon、判据、搜索预算和物理验证状态。

### 10.3 单样本子网包含共享结构，但共享集合不是独立通用模块

七重交集远高于随机基线，50% C7 消融也显著比随机消融更坏；但 C5+/C6+/C7-only 全部 0/7。共享核心更像
必要骨架，而不是可以单独输出所有 caption 的充分网络。

### 10.4 条件外壳不是私有插件，而是重叠、可迁移、方向不对称的 basis

Learned shell 大幅优于同宽随机 shell，对角线全部通过，但非对角线也有 28.6% 通过。不同样本不是各自占用
一组互斥 neurons，而是在共享骨架上重组一批中频 identity。

### 10.5 子网组合具有协同，也具有可定位冲突

Union 能真正恢复七对双任务，说明学习到的 identity 可零训练组合；三个负干扰边又说明集合超集不保证功能
单调。Phase 6D–6E 进一步证明其中两条冲突可由稀疏 identity 集合定向修复，随机容量缩减不能替代该作用。

### 10.6 数值精度是结构部署结论的一部分

Hook 置零、BF16 窄矩阵和 FP32 窄矩阵在理论 mask 相同的情况下可能跨过生成决策边界。因此，结构化剪枝报告
必须同时包含精度、实际矩阵形状、权重 hash、自由生成和独立重载；仅验证参数数或 teacher NLL 不够。

## 十一、证据强度与可汇报边界

### 11.1 当前可以较强主张

- 固定样本上存在可真实物理化、保存和重载的极端语义子网；
- 严格轨迹保持与任务语义保持是相差数个数量级的不同问题；
- 七个冻结赢家之间存在远高于层宽条件随机基线的共享 identity；
- 共享核心在高剂量下具有必要性证据，但单独不充分；
- Learned shell 和真实 union 显著优于 matched-random identity；
- 至少两个失败 union 的负干扰可被特定 identity 删除因果修复，并在 FP32 结构模型中复现。

### 11.2 只能作为中等强度、需继续验证的解释

- 早层 C7 较多可能反映更通用的视觉—语言骨架；
- 不同语义层需要不同子网规模；目前每个语义层只有一个样本，仍可能只是实例难度；
- layer 20 可能是跨 edge 的共享冲突路由位置；目前只在两个 edge 上出现。

### 11.3 当前不能主张

- 删除 73.18% FFN 后模型整体能力不变；
- 62,950 是全局最优，或 63,000 已被证明不可行；
- 10,970 个 C7 neurons 是一个独立充分的通用视觉核心；
- union 的任务能力随集合包含关系单调增长；
- 797 个 identities 是最小、唯一或跨样本普适的冲突集合；
- 当前单次显存/耗时观测等于稳定部署加速；
- bicycle–toilet repaired union 已能稳定 BF16 部署。

## 十二、主要局限

1. **样本规模仍小。** Phase 6B 只有七个样本，每个语义层只有一例，统计推断不能外推到 COCO 总体。
2. **Gold caption 参与 mask 优化。** 低 NLL 是任务特化现象，不能与泛化能力混为一谈。
3. **边界搜索不完全。** 多个赢家 restart 数不足，搜索 landscape 非单调，经验极限可能漏解。
4. **语义契约覆盖有限。** 自动规则只证明冻结概念、关系和输出卫生；仍需人工检查细微自然度或未编码矛盾。
5. **Prompt 鲁棒性不足。** 62,950 模型存在强 target attractor 和角色词尾部退化。
6. **后验核心。** C7 来自七个已选择赢家，其大小会随样本、restart 和赢家选择改变。
7. **随机基线仍有边界。** 层宽匹配控制了规模分布，但没有完全控制共同 prompt、Taylor 初始化和优化轨迹。
8. **BF16 决策边界。** 少数 hook/结构结果会因 GEMM 归约顺序翻转，需 FP32 或 margin 稳定性验证。
9. **没有动态路由。** 当前 union 让所有保留 shell 常驻，不能直接声称降低每次推理的活跃计算。
10. **最终答案等价曲线缺失。** Phase 5 已系统覆盖严格轨迹和人工语义，但未独立完成原模型最终 caption 等价前沿。

## 十三、建议的下一阶段

### 13.1 从“单样本极限”升级到“小规模可重复统计”

- 每个语义层扩展到至少 10–20 个样本；
- 对每个样本使用统一预算网格和至少 3 个 restart；
- 分层报告样本难度、子网规模、C7 稳定性和结构重载通过率；
- 用 bootstrap 或分层混合模型给出样本级不确定性，而不是把 restart 当独立样本。

### 13.2 搜索鲁棒而非只在 canonical prompt 上最稀疏的子网

- 把多个无泄露 prompt 和 benign 图像变换纳入训练；
- 保留完全 held-out prompt；
- 加入错图/遮挡 contrastive 目标；
- 将 EOS margin、角色 token margin 与重复风险加入代理，但仍以自由生成裁决。

### 13.3 验证核心和冲突的跨实例可迁移性

- 在新样本上测试 10,970-neuron C7 的必要性曲线；
- 测试 797 个冲突 identities 是否能预测同类新 edge；
- 对 violin–people 做预注册跨层组合定位；
- 使用不同 winner restart 重建 C7，评估核心 identity 稳定性。

### 13.4 建立冲突感知动态路由

实现按输入选择 `C7 + S_i` 或 repaired union 的稀疏路由器，并比较：

- 双任务语义通过率；
- 活跃 neuron 数与实际 FLOPs；
- BF16/FP32 稳定性；
- 路由错误时的退化模式；
- 与静态 union、全模型和等规模随机子网的差异。

### 13.5 规范化部署评测

对原模型与物理模型使用相同输入、输出长度、预热次数和重复次数，报告延迟分布、峰值显存、吞吐、checkpoint
体积和不同精度下的一致性，避免从单次运行推断加速。

## 十四、尚待回答的问题

1. 当每个语义层扩展到多个样本后，单样本充分子网的规模差异是否仍能由语义复杂度解释？
2. 七重共享核心在更换样本、prompt、Taylor 初始化或 winner restart 后，有多少 identity 能稳定复现？
3. Phase 6D 的 797 个冲突 identities 是当前实例特异、关系特异，还是能预测新样本上的同类冲突边？
4. Violin–people 的冲突能否被一个跨层集合解释，还是存在多个可替代的高阶冲突路径？
5. 动态选择 `C7 + S_i` 能否同时提高组合稳定性并降低实际活跃 FLOPs，还是路由成本会抵消稀疏收益？
6. 加入 held-out prompt、反事实和 EOS margin 后，鲁棒语义 frontier 会从 62,950 回退到什么范围？

这些问题会直接决定当前结果能否从“单样本机制构造”升级为可泛化的模块化剪枝与条件计算方法。

## 十五、导师汇报建议表述

建议使用以下总结：

> Phase 5 证明了单样本剪枝极限取决于功能契约：同一图像在保持 512-token 原轨迹时只能稳定物理删除少量
> neurons，而只保持人工语义时可以删除 73.18% 的 FFN neurons。Phase 6 进一步表明，这些单样本子网包含
> 显著共享骨架和条件外壳；它们能够组合恢复新任务，但组合不是单调的，会产生由特定 neuron identity 驱动
> 的负干扰。对两条失败组合，定向删除定位到的冲突 identities 可以恢复双任务，而等量随机删除不能恢复。

同时应主动补充：

> 这些结果是七个冻结单样本上的机制性构造证据，不是模型总体压缩结论。当前最重要的后续工作是扩大样本、
> 搜索 prompt/反事实鲁棒子网，并验证共享核心与冲突 identity 是否跨实例稳定。

## 十六、可审计产物

### Phase 5

- Phase 5A 结果：`docs/my-exper/typing neuron/phase5a_sample2500_pilot_result.md`
- Phase 5B 结果：`docs/my-exper/typing neuron/phase5b_sample2500_result.md`
- Phase 5C 结果：`docs/my-exper/typing neuron/phase5c_sample2500_result.md`
- Phase 5D 计划与累计结果：`docs/my-exper/typing neuron/phase5d_cached_gate_plan.md`
- Phase 5E 完整报告：`docs/my-exper/typing neuron/phase5e_advisor_report.md`
- Phase 5E 62,950 物理模型：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0/model/`
- Phase 5E 独立重载：
  `saves/neuron_typing/phase5_single_sample/sample_2500_phase5e_learned_structural_62950_r0_reload.json`

### Phase 6

- Phase 6A 审计：
  `saves/neuron_typing/phase6a_prompt_counterfactual_62950_v2/phase6a_results.json`
- Phase 6B 冻结赢家与几何：
  `saves/neuron_typing/phase6b_frozen_winners_v1/frozen_winners.json`
  与 `saves/neuron_typing/phase6b_frozen_winners_v1/geometry_summary.json`
- Phase 6C 组合分析：
  `saves/neuron_typing/phase6c_core_shell_composability_v1/analysis.json`
- Phase 6D 冲突定位：
  `saves/neuron_typing/phase6d_conflict_localization_v1/analysis.json`
  与 `saves/neuron_typing/phase6d_identity_refinement_v1/analysis.json`
- Phase 6E BF16 / FP32 repair：
  `saves/neuron_typing/phase6e_repaired_union_v1/analysis.json`
  与 `saves/neuron_typing/phase6e_repaired_union_float32_v1/analysis.json`

上述结果文件均保留逐案例 caption、mask/variant hash、停止状态、proxy 指标和结构校验，支持从汇总结论回溯到
具体候选与生成输出。
