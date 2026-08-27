# Phase 3：q_multimodal 5--20% Band 结构化剪枝计划

> 制定日期：2026-07-22  
> 目标模型：Qwen3.5-VL 0.8B  
> Phase 3 唯一主 mask：完整 2,000-sample typing 产生的逐层 `q_multimodal` 5--20% rank band

## 1. 是否满足 Phase 3 准入条件

可以进入 Phase 3，但准入结论需要精确定义：

- Phase 2 已通过 held-out caption NLL 和 POPE random/popular/adversarial 三个 split 的功能安全检查；
- q-band 是当前唯一同时通过这些检查的 15% functional mask；
- magnitude 和 activation-frequency 对照不再作为候选，只保留为负对照；
- 500 vs 2,000 验证表明总体 `q_multimodal` 排序具有一定稳定性，但精确 band 成员在 500 样本下不稳定；
- 因此 Phase 3 必须冻结并直接使用完整 2,000-sample mask，不允许从 500-sample pilot 重建，也不在 Phase 3 内重新搜索 mask。

500 vs 2,000 的低 Jaccard 不否定完整 2k mask 已获得的因果安全证据，但限制了结论范围：当前可以声称“这个固定 mask 安全”，不能声称“任意小样本都能稳定恢复相同 mask”。

## 2. Phase 3 核心问题

Phase 3 不再回答“剪哪些神经元”，只回答以下三个问题：

1. hook 置零与物理删除相同通道是否数值等价？
2. 物理删除后，Phase 2 的任务安全性是否完整保留？
3. 15% FFN 通道删除能否转化为实际参数、显存和延迟收益？

主结论必须来自 zero-shot structural pruning。恢复训练只作为可选补充，不能用于掩盖结构化转换错误或 zero-shot 退化。

## 3. 冻结的结构规格

| 项目 | 数值 |
|---|---:|
| Transformer 层数 | 24 |
| 原始 FFN intermediate size | 3,584 / layer |
| 每层删除 | 537 |
| 每层保留 | 3,047 |
| 总删除神经元 | 12,888 / 86,016 |
| FFN 通道删除率 | 15.0% |
| hidden size | 1,024 |
| 理论删除权重数 | 39,591,936 |
| BF16/FP16 理论权重体积减少 | 约 75.5 MiB |

每个被删 FFN 通道对应：

- `gate_proj` 的一行；
- `up_proj` 的一行；
- `down_proj` 的一列。

39,591,936 是上述三个矩阵的理论删除量；正式报告必须从加载后的模型参数计数重新测量，不能只引用理论值。

## 4. 实施路线

### P3.0：冻结 mask 与实验身份

新增一个只负责构建正式结构化产物的脚本，不复用旧的 unknown-score Phase 3 脚本。

输入：

- `phase1_clean_2k/scores/neuron_type_scores.parquet`；
- Phase 1 calibration/config/manifests；
- Phase 2 正式 q-band 结果。

输出：

- `q_multimodal_band_05_20.mask.json`：每层待删除 neuron IDs；
- `q_multimodal_band_05_20.cluster_idx.json`：每个保留通道一个 singleton cluster；
- `q_multimodal_band_05_20.metadata.json`：所有输入 SHA-256、排序键、band 边界、逐层计数和总计数；
- mask 的稳定 hash，供所有后续命令引用。

硬校验：

- 24 层齐全；
- 每层恰好删除 537、保留 3,047；
- 无重复、越界或 dead-neuron 意外进入；
- 结构化 mask 与 Phase 2 hook evaluator 构造的 mask 逐 bit 相同；
- 排序固定为 `q_multimodal desc, r_multimodal desc, neuron_idx asc`；
- 输入 score/calibration/manifest hash 不匹配时立即失败。

### P3.1：结构化转换与数值等价测试

复用 `src/llamafactory/train/vulcan/pruning.py::pruning_mlp` 和
`scripts/vulcan/save_pruned_model.py`。对于普通删除，每个保留神经元写成：

```json
{"anchor": 123, "neuron": [123]}
```

这会复制保留通道的 gate/up 行和 down 列，并完全省略待删除通道。

必须完成三段等价测试：

1. 原模型 + hook q-band vs 内存中的 structural q-band；
2. 内存中的 structural q-band vs 保存并重新加载后的模型；
3. 原模型 baseline vs structural 模型，确认差异方向与 Phase 2 一致。

固定 16--32 个包含图像的 held-out 样本，保存输入 IDs 和图像 hash。比较：

- prefill logits 的 max/mean absolute error；
- label-token NLL；
- forced-choice yes/no logits 与预测；
- greedy generation 的首个分叉位置。

验收门槛：

- FP32 小模型单元测试：hook 与 structural `allclose(atol=1e-6)`；
- 每层 structural gate/up/down 权重 hash 与原模型相应 keep-index 切片完全一致；
- 内存 structural 与 reload structural 权重 hash、输出和层宽完全一致；
- BF16 原宽度 hook 与窄宽度 structural 允许由 GEMM 累积顺序产生的小幅数值漂移，smoke mean NLL 差不超过 `0.05`；
- forced-choice 预测完全一致；
- 保存前后参数量、层宽和输出一致。

不能只凭 BF16 full-logit 逐元素相等判断结构转换。改变 GEMM 的 K/N
shape 会改变低精度累积顺序，即使保留权重逐 bit 相同，深层 logits 也
可能出现小幅漂移。结构正确性的硬证据是 projection hash 与 reload
一致性，任务安全性由 P3.2 独立验收。

若 hook 与 structural 不等价，立即停止，不能进入正式任务评估。

### P3.2：Zero-shot 任务安全复验

正式比较三个条件：

1. 原始模型；
2. 原始模型 + frozen hook q-band；
3. structural q-band 模型。

首先使用 Phase 2 完全相同的 held-out 样本和评分代码，以隔离“结构转换”变量：

- held-out caption teacher-forced label NLL；
- POPE random；
- POPE popular；
- POPE adversarial。

主验收门槛：

- structural 与 hook 的 caption NLL 差不超过 `0.05`，同时不得比原模型恶化超过 `0.05`；
- BF16 structural 与 hook 在三个 POPE split 上的预测一致率至少 `99.5%`，并完整记录逐题差异、
  margin 分布和严格 `100%` 一致率；margin 仅作为诊断，因为宽/窄 GEMM 的 BF16 累积漂移随 shape
  改变，任务硬门由一致率及 aggregate accuracy/yes-ratio 共同定义；
- structural 与 hook 的 accuracy 和 yes ratio 漂移每个 split 均不超过 `0.2` percentage point；
- structural 相对原模型的 POPE accuracy 降幅每个 split 不超过 1 percentage point；
- yes ratio 相对原模型漂移每个 split 不超过 2 percentage points；
- 不允许出现 magnitude 式稳定偏向回答 `no` 的模式。

随后可增加一个未参与任何 mask 决策的新 held-out VQA split 作为外部确认，但它不是结构转换正确性的替代品。

### P3.3：参数、显存与速度基准

只比较原模型和 structural q-band；hook 条件仅用于证明它没有工程收益。

统一环境：

- 同一张 RTX 4090；
- 同一 dtype、attention backend 和 generation config；
- 固定图像尺寸、prompt token 数和输出 token 数；
- 10 次 warm-up，至少 50 次正式测量；
- 每轮同步 CUDA，固定随机种子；
- 报告 median、p95 和 bootstrap 95% CI，而非单次最快值。

场景：

| 场景 | batch size | 指标 |
|---|---:|---|
| Multimodal prefill | 1, 4 | latency、samples/s、peak memory |
| Text-only prefill | 1, 4 | latency、tokens/s、peak memory |
| End-to-end generation | 1 | TTFT、decode tok/s、总 latency |
| Checkpoint/load | 1 | disk size、load time、parameter count |

验收分为两个层次：

- 必须通过：实际参数数和 checkpoint 体积下降，任务安全性保持；
- 工程成功：至少一个主要 multimodal 场景获得稳定、置信区间不跨零的 latency/throughput 改善，且没有主要场景显著变慢。

不能把“FFN MAC 理论下降 15%”写成“端到端加速 15%”。视觉编码、attention、kernel launch 和 memory bandwidth 都会稀释收益。

### P3.4：硬件对齐分支，仅在精确 15% 不加速时启动

精确 15% 后 intermediate size 为 3,047，不是 GPU GEMM 友好的对齐宽度，可能出现参数减少但延迟不降甚至略升。

若 P3.3 没有真实加速，测试对齐宽度 3,072：

- 每层删除 512，实际 FFN 剪枝率 14.29%；
- 使用精确规格 `rank_window:multimodal:180:512`：按
  `q_multimodal desc, r_multimodal desc, neuron_idx asc` 排序，保护前 180 个通道并删除随后 512 个；
- 在结构化转换前，先对这个新 band 重新运行 Phase 2 的 caption + 三个 POPE hook 安全检查；
- 只有重新通过安全门槛，才生成 aligned structural checkpoint。

P3.3/P3.4 正式 benchmark 必须强制生成固定数量的新 token，并记录实际生成长度；同时覆盖 multimodal
和 text-only prefill，直接对 latency delta/speedup 进行 bootstrap。不得用 `max_new_tokens` 代替实际输出
长度计算 decode throughput。

这个分支必须标为 `hardware-aligned q-band`，不能与已验证的精确 5--20% mask 混为同一个实验条件。

### P3.5：可选恢复训练

默认不做。如果 zero-shot structural 与 hook 等价但相对原模型出现轻微、可接受范围外的任务退化，可增加一次短 LoRA recovery：

- 1k--5k image-text samples；
- 1 epoch；
- 独立训练集，与 calibration、typing、Phase 2/3 evaluation 全部 image-disjoint；
- 同时报告 zero-shot 和 recovered，主结论仍以 zero-shot 为准。

如果 structural 与 hook 不等价，禁止使用恢复训练修补。

## 5. 对照组策略

Phase 3 主任务是验证实际压缩收益，不再重新竞争 mask。推荐：

- 正式保存和完整评估：只做 q-band structural checkpoint；
- magnitude、activation-frequency、matched-random：沿用 Phase 2 hook 结果作为选择质量对照；
- 对 baseline mask 各做一次小规模 structural-vs-hook smoke test，验证转换器与 mask 类型无关；
- 不需要为每个已知不安全 baseline 重复完整速度测试，因为同样的逐层宽度会产生相同理论 shape 和近似相同速度。

如论文必须提供“所有方法均为物理剪枝”的表格，再追加相同宽度的 structural baseline 全量评估；这不是第一轮阻塞项。

## 6. 需要新增或修正的代码

建议新增：

1. `build_structural_pruning_artifact.py`
   - 从正式 q/r score 生成 frozen mask、singleton cluster_idx 和 provenance；
2. `verify_structural_equivalence.py`
   - 同批输入比较 hook、内存 structural、reload structural；
3. `benchmark_structural_pruning.py`
   - 参数、磁盘、显存、prefill、generation 基准；
4. `run_phase3_structural.py`
   - 串联 gate、保存、复验和 benchmark，支持断点续跑。

需要修正：

- 为 `save_pruned_model.py` 增加 mask/score/hash 摘要输出；
- 为自定义 Qwen3.5 loader 增加正式 reload 测试；
- 增加 CPU synthetic MLP deletion-equivalence 测试；
- 增加 24 层 × 537 精确计数、排序 tie-break、mask hash 和 singleton coverage 测试。

当前 `pruning/run_type_aware_pruning.py` 和 `compute_pruning_score.py` 仍使用废弃的 `p_*`/unknown-score 逻辑，且 evaluation 为占位实现。它们不能直接用于正式 Phase 3；应保留为历史脚本或明确 deprecated，避免误运行。

## 7. 目录与产物

```text
saves/neuron_typing/phase3_structural_qband/
  config.json
  provenance.json
  masks/
    q_multimodal_band_05_20.mask.json
    q_multimodal_band_05_20.cluster_idx.json
    q_multimodal_band_05_20.metadata.json
  equivalence/
    smoke_inputs.json
    equivalence_metrics.json
  model/
    ... pruned HuggingFace checkpoint ...
  evaluation/
    caption_nll.json
    pope_random.json
    pope_popular.json
    pope_adversarial.json
  benchmark/
    raw_runs.jsonl
    summary.json
  phase3_report.md
```

所有结果必须记录：git commit、模型 checkpoint hash、score hash、calibration hash、mask hash、数据 manifest hash、CUDA/GPU、PyTorch/Transformers 版本、dtype、attention backend 和完整命令。

## 8. 执行顺序与停止条件

| 顺序 | 阶段 | 预计工作量 | 继续条件 |
|---:|---|---:|---|
| 1 | P3.0 frozen artifact | 0.5 天 | mask 与 Phase 2 逐 bit 一致 |
| 2 | P3.1 converter + equivalence | 0.5--1 天 | hook/structural/reload 等价 |
| 3 | P3.2 formal zero-shot eval | 0.5--1 天 | caption + 三个 POPE 通过门槛 |
| 4 | P3.3 efficiency benchmark | 0.5 天 | 得到可复现参数/显存/速度结论 |
| 5 | P3.4 aligned variant | 可选 0.5--1 天 | 仅当 3,047 宽度无真实加速 |
| 6 | P3.5 recovery | 可选 | 仅修复轻微任务退化 |

任何阶段出现以下情况立即停止并诊断：

- frozen mask 与 Phase 2 mask 不一致；
- structural 与 hook 输出不等价；
- reload 后输出或层宽变化；
- POPE yes ratio 出现系统性向 `no` 偏移；
- 参数未实际减少；
- benchmark 条件不一致或结果方差过大。

## 9. Phase 3 最终可形成的结论

只有 P3.0--P3.3 全部通过，才可以表述：

> 基于 q/r neuron typing 的完整 2k `q_multimodal` 5--20% band 不仅在 hook-based causal ablation 下安全，也可以被物理删除并保持 caption 与 object-recognition 能力，同时带来可测量的参数/存储收益；是否带来端到端速度收益由统一硬件基准单独报告。

若参数下降但速度没有改善，结论必须限定为“结构化压缩成功，但当前非对齐宽度/执行后端未产生端到端加速”，并启动 P3.4，而不是把理论 FLOPs 当作实测 speedup。
