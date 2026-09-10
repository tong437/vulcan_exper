# Phase 6B：多单样本子网极限预注册

## 目标

在进入“共享核心、条件外壳和可组合性”分析前，先为语义不同的单个图文样本分别搜索可行子网。
本阶段回答的是：在相同模型、prompt、优化器和计算预算下，不同样本各自能找到多小的 FFN 子网，
以及它们的逐层宽度和 neuron identity 有何差异。

报告中的“极限”均指当前离散预算、步数和重启数下找到的最大可行删除量，是构造性的经验下界，
不是最小充分子网的理论证明。

## 冻结样本与契约

使用 `data/phase6b_single_samples/frozen_samples.json` 中七个、经原模型预筛通过的 COCO 样本：

1. bicycle clock：物体—属性组合；
2. violin kitchen：人物—动作—场景；
3. toilet spatial：多物体空间场景；
4. people pizza：多人—食物—场景；
5. giraffes trees：复数动物—自然场景；
6. dogs car：复数动物—车内/窗关系；
7. bicycles train：复数物体—列车内部关系（既有锚点）。

统一 prompt 为：

> Write one concise English caption for this image in one complete sentence. Output only the caption.

每个样本的自动语义契约只要求原模型实际表达并经人工看图确认的核心概念；不会要求原模型遗漏的
COCO 细节。所有契约、正则和原模型 caption 在任何剪枝搜索前冻结。

## 搜索协议

- 模型：Qwen3.5-0.8B，`enable_thinking=false`；
- 掩码：24 层 FFN neuron 的 sample-static、全局 exact-budget 二值 mask；
- 轨迹：对应样本的 gold caption teacher-forced token；
- 初始化：逐层均值归一化的 Taylor saliency；
- 目标：gold CE + 原模型 gold-prefix 分布 KL；
- 最终可行性：只由确定性自由生成是否通过该样本的冻结语义契约决定；proxy 不替代生成判据；
- 粗搜索预算：删除 48,000、56,000、60,000、64,000、68,000 / 86,016 个 FFN neurons；
- 粗搜索：每预算 120 steps、1 restart，用于定位跃迁区间；
- 精搜索：围绕每个样本最后通过/首次失败区域补点，200 steps、3 restarts；补点选择在查看 mask
  重叠之前完成；
- 同预算多个可行解以更低 gold NLL 作为确定性 tie-breaker。

## 验证与输出

每个搜索 run 保存独立 mask、mask SHA-256、逐层保留宽度、参数缩减、gold proxy、原始生成文本、
停止状态和语义契约逐项判定。每个样本选出的最终 mask 至少进行结构可实现性检查；正式几何分析只读取
冻结后的获胜 mask，不反向修改本阶段的语义契约或样本集合。

主要报告：

- 七个样本的最大已找到可行删除量及区间；
- 极限处总保留 neuron 数和逐层保留宽度；
- 搜索不稳定性（同预算不同 restart 的通过率）；
- 未通过输出的具体失败类型。

后续的共享核心、条件外壳、union/composition 实验属于 Phase 6C，不参与本阶段选 mask。

## 粗搜索先导后的自适应修订

最初的 48k--68k 网格在前两个样本上从最低预算起即全部触发自由生成终止/重复失败：
bicycle-clock 的 48k 为空答案，violin-kitchen 的 48k 虽首句精确复现 gold，却随后重新进入伪
thinking 并撞到长度上限。按上述“粗搜索用于定位跃迁区间”的规则，在完成当前 run 后停止该过高网格，
原始未完成产物保留，不覆盖或选择性删除。

正式统一粗网格下移为删除 16,000、24,000、32,000、40,000、48,000 neurons，其他设置不变。
这次修改发生在查看任何跨样本 mask 重叠之前；后续精搜索仍由最后通过/首次失败点确定。
