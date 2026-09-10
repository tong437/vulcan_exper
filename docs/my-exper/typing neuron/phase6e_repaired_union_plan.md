# Phase 6E：Repaired Union 预注册

## 目标与冻结输入

Phase 6E 不再搜索冲突 identity，只读取 Phase 6B 冻结赢家和 Phase 6D 已落盘的最终集合，检验定向删除能否把
失败 union 修复成双任务同时通过的子网。冻结两个 repair：

1. `bicycle–toilet`：从 `K_bicycle ∪ K_toilet` 删除 Phase 6D 在 `K_bicycle \ K_toilet` 中定位的 L20
   135 个 identity；safe base 为 `K_toilet`。
2. `bicycle–violin`：从 `K_bicycle ∪ K_violin` 同时删除 Phase 6D 在 `K_violin \ K_bicycle` 中定位的
   L20 337 个与 L0 325 个 identity；safe base 为 `K_bicycle`。

三个输入文件的 identity 数量、layer、SHA-256、base/donor/target provenance 必须与 Phase 6D analysis 一致；
任一漂移均终止实验。`violin–people` 没有单层双向集合，本轮不做事后跨层搜索或强行 repair。

## 主实验与判据

每个 pair 在两个目标样本上评估四类条件：

- `safe_base`：Phase 6D 的安全 constituent；
- `failed_union`：未经修复的 Phase 6C union；
- `repaired_union`：按上述冻结 identity 做联合删除；
- `random_repair`：10 个确定性 seed 的匹配随机删除。

主判据是每个 `repaired_union` 对 pair 的两个任务都满足冻结的严格自由生成语义契约。只恢复 Phase 6D 中受损
端点而破坏另一端，记为 repair 失败。连续佐证同时报告两个端点的 gold-prefix NLL、KL、top-1 与 margin。

每个 random repair 必须逐层匹配真实 repair 的删除数量，并且只能从相同 donor 相对 safe base 的新增区域抽样：

- bicycle–toilet：只从 `(K_bicycle \ K_toilet)[L20]` 抽 135 个；
- bicycle–violin：从 `(K_violin \ K_bicycle)[L20]` 抽 337 个，同时从 L0 抽 325 个。

随机抽样池保留冻结冲突 identity，因此随机 repair 可以自然部分命中真实集合；这使 identity-specific 对照更
保守，也需记录每个 seed 与真实集合的 overlap。不得从 union 全体或 safe-base resident identity 抽样，避免
改变来源与功能位置。

支持 repaired-union 的最低证据为：

1. `safe_base` 双任务通过且 `failed_union` 至少一端失败，复现 Phase 6C 因果背景；
2. `repaired_union` 双任务通过；
3. 10 个 matched-random repair 的双任务通过率低于真实 repair，并报告 target-level 结果及 NLL，不把单次
   treatment 与随机重复误作独立样本推断。

若真实 repair 与随机 repair 相当，只能解释为宽度缩减；若真实 repair 未双通过，则 Phase 6D 的单端点
`add fail + LOO pass` 集合不足以构成 pair-level repair。

## 精度与结构复验

BF16/auto 为主实验。仅当 BF16 repaired union 满足双任务主判据后：

1. 用 FP32 对同一冻结 mask 重跑全部四类条件，不重新选择 identity；
2. 将两个成功 repaired union 物化为永久结构 checkpoint；
3. 验证 24 层宽度、gate/up/down 投影权重 hash、删参量，并在两个目标上比较 hook 记录与结构重载输出；
4. BF16 结构重载若发生 token 翻转，单独报告数值稳定性，不修改 repair 集合。

## 解释边界

- 本轮是对 Phase 6D 同一冻结端点的因果修复验证，不是独立样本或总体泛化检验。
- 三个 identity 集合是预算内构造，不是 1-minimal 或全局最小集；repair 成功也不赋予最小性。
- bicycle–violin 同时删除两个 layer 集合，只证明联合 repair；不能据此分配两个 layer 的独立贡献。
- 成功的静态 repair 仍不等价于自动路由。跨样本、跨 prompt 与路由选择留给后续阶段。
