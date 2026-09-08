# Phase 5E：单样本任务语义极限

## 1. 研究问题

对固定的 COCO 图像源样本 `coco_captions_val sample_offset=2500`，寻找能够真实物理删除的最大 FFN neuron 数，同时让模型从
原始图像和 prompt 确定性生成一个正确表达“多辆自行车位于地铁或列车车厢中”的完整 caption。

本阶段不要求复现原模型的思考过程、逐 token 轨迹或固定措辞。成功候选只构成当前搜索算法和候选族中的
构造性下界，失败预算不构成全局不可行证明。

## 2. 冻结语义契约

人工参考 caption 为：

> A group of bicycles on a subway train.

自动判定器使用三态结果：

- `pass`：明确包含自行车、多辆、列车场景、车内/车上关系，正常结束且无否定或矛盾；
- `fail`：没有最终答案、思考未结束、生成被截断、明确否定、明确只有一辆或明确错误场景；
- `review`：规则不能保守确认全部必要语义。`review` 不自动计入通过，必须人工检查。

最终报告中的通过 checkpoint 必须由人工确认。自动规则只负责批量筛选。

## 3. 确定性生成协议

- `do_sample=False`；
- `temperature=None`；
- `max_new_tokens=64`；
- 使用 `template=qwen3_5` 和 `enable_thinking=false`，让 prompt 显式包含空 thinking block；
- 使用冻结的一行数据副本 `data/phase5e_sample/sample_2500.json`，其 prompt 明确要求只输出一句简洁 caption；
- 仍保留答案抽取器：若模型输出 `<think>...</think>`，只评价最后一个闭合标签后的文本；
- 达到 token 上限且未遇到正常停止 token，或存在未闭合 thinking block，自动失败。

模板名称本身不作为“thinking 已关闭”的证据；每次生成都记录原始文本、抽取答案、thinking 状态、token
数和停止状态。

一行副本在 Phase 5E 配置中的局部 `sample_offset` 为 0；文件名、图像 ID 和本文记录保留其原始 COCO val
source offset 2500。这样既冻结 prompt，又避免修改或复制完整 5000 行数据集。

## 4. 三层评价

### 4.1 Gold-caption 代理指标

在人工 caption 的 gold prefixes 上 teacher forcing，一次 full forward 记录：

- 平均 gold NLL 和相对未剪枝模型的 `delta_nll`；
- 相对未剪枝模型分布的 token-level mean/max KL；
- gold token 的 mean/min logit margin 和 top-1 accuracy；
- 每个 neuron 的 activation、contribution 和 gold-CE Taylor saliency。

这些指标用于排序与优化，不是最终通过条件。未剪枝模型是 reference model；不使用旧的 512-token 思考
输出作为 Phase 5E teacher trajectory。

### 4.2 自由生成语义门

候选只接收图像和 prompt，自由生成短 caption。自动语义门输出完整 feature flags、原因和
`pass/fail/review`，不比较原模型或人工 caption 的逐 token 文本。

### 4.3 人工确认

每轮只检查 frontier 附近通过或待复核的少量物理候选。人工结论以独立字段记录，不能由自动判定器预填。

## 5. 搜索流程

### 5E-R：已有物理 checkpoint 回溯

依次测试未剪枝模型、core13、100/250/500/750/1000 等已保存物理模型。对每个模型重新构造 Phase 5E
non-thinking prompt，记录 gold 代理指标、自由 caption、语义判定、参数量、层宽、生成耗时和峰值显存。

### 5E-A：Fresh 静态排序与粗预算

从未剪枝模型基于 gold-caption CE 重新计算静态 saliency。按低重要度构造严格嵌套的全局 deletion masks，
优先探测 `100,250,500,1000,2000,4000,8000`。静态排序同时是可复现 baseline、learned gate 初始化和
边界局部候选池。

### 5E-B：Exact-budget learned neuron gate

每个预算使用 sample-static、global exact-budget hard mask 和 straight-through estimator。默认目标为：

```text
gold_ce_weight * gold_caption_CE
+ kl_weight * gold_prefix_reference_KL
```

精确预算下不额外使用 sparsity penalty。每个预算至少三个 restart；优化 seed 与确定性生成协议分别记录。
代理最优 mask 固定后，必须重新自由生成并过语义门。

### 5E-C：真实结构验证

有希望的 hook masks 才转换为 non-uniform 物理 FFN：内存结构生成、保存、重载、再次生成。需要同时验证
层宽、参数数、mask/provenance hash 和内存/重载输出一致性。

### 5E-D：经验边界缩窄

在已找到的最大通过预算和附近未找到通过解的预算间增加探针。语义通过性和独立 learned masks 不保证随
预算单调，因此这不是数学二分：关键预算需要独立 restart，失败点只表示当前搜索未找到可行解。

## 6. 产物

```text
phase5e/<run>/
├── sample_manifest.json
├── semantic_contract.json
├── reference_logits.pt
├── saliency_scores.pt
├── static_frontier.json
├── learned_frontier.json
├── retrospective.json
└── masks/
```

每个候选记录删除 neuron 数、参数量与比例、逐层删除、gold NLL/delta-NLL/KL/margin、原始生成、抽取
caption、自动语义结果、人工确认占位、停止状态、推理耗时和显存。物理 checkpoint 额外记录保存/重载
一致性。

## 7. 执行顺序

1. 单测语义契约、答案抽取、停止状态与 gold 代理指标；
2. 运行未剪枝模型短生成，确认真正 non-thinking 且能正常结束；
3. 回溯已有物理 checkpoints，得到即时语义下界；
4. fresh gold-caption 静态 sweep；
5. 在粗边界附近 learned-gate 多 restart；
6. 物理化并缩窄经验边界；
7. 人工确认最终 frontier，汇总严格轨迹、原答案等价和人工语义保持三条曲线。
