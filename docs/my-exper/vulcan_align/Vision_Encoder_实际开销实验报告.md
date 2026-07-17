# Vision Encoder 实际开销实验报告

## 1. 实验设置

| 项目 | 配置 |
|---|---|
| 模型 | Qwen3.5-0.8B VLM，未压缩 SFT checkpoint |
| 硬件 | NVIDIA GeForce RTX 4090 |
| 推理精度 | BF16 |
| Batch size | 1 |
| 图像数量 | 单图 |
| 图像像素上限 | `262144` |
| 测量方式 | 10 次预热，50 次正式测量 |
| 计时范围 | Vision Encoder、Projector、Prefill 和完整 Generation |

Vision Encoder 和 Projector 通过模块级 CUDA Event 分别计时。实验同时记录预处理后的视觉网格、有效像素数和实际生成 token 数，避免将原图尺寸或提前结束的生成误认为真实计算量。

## 2. 模型静态开销

| 模块 | 参数量 | 总参数占比 | BF16 权重大小 |
|---|---:|---:|---:|
| Vision Encoder | 88.00M | 7.95% | 167.85 MiB |
| Projector | 12.59M | 1.14% | 24.01 MiB |
| Language Model | 1006.67M | 90.92% | 1920.08 MiB |
| 总计 | 1107.27M | 100% | 2111.94 MiB |

Vision Encoder 仅占模型总参数的约 `8%`；包含 Projector 后，完整视觉侧约占 `9.1%`。

## 3. 1-token 短输出结果

| 样本 | 视觉网格 | 有效像素 | Prompt tokens | Prefill | Vision | Vision 占比 |
|---|---:|---:|---:|---:|---:|---:|
| Tiny（296×336） | 20×18 | 92K | 136 | 101.1 ms | 6.82 ms | 6.7% |
| Small（337×451） | 28×22 | 158K | 203 | 106.6 ms | 6.94 ms | 6.5% |
| Medium（566×555） | 32×32 | 262K | 302 | 108.7 ms | 7.18 ms | 6.6% |
| Large（1024×1309） | 36×28 | 258K | 294 | 108.7 ms | 7.19 ms | 6.6% |

在极短输出场景中，Vision Encoder 前向稳定在 `6.8–7.2 ms`，约占完整 Prefill 延迟的 `6.5%–6.7%`。

## 4. 64-token 长输出结果

| 样本 | Prefill | Vision | 完整生成中 Vision 占比 |
|---|---:|---:|---:|
| Tiny | 100.1 ms | 6.78 ms | 0.5% |
| Small | 104.9 ms | 7.16 ms | 0.5% |
| Medium | 109.0 ms | 7.30 ms | 0.5% |
| Large | 110.5 ms | 7.31 ms | 0.6% |

Vision Encoder 每个请求只在 Prefill 阶段执行一次，而 Language Model 需要持续执行 Decode。真实生成 64 tokens 时，Vision Encoder 在完整生成延迟中的占比下降至约 `0.5%`。

## 5. 关键发现

1. **Vision Encoder 的实际开销较低。**  
   其参数占比约为 `7.95%`，短输出 Prefill 延迟占比约为 `6.6%`。

2. **输出越长，压缩 Vision Encoder 的收益越小。**  
   在真实 64-token 生成中，其端到端延迟占比仅约为 `0.5%`。

3. **Vision Encoder 延迟对受控后的图像规模较稳定。**  
   原图超过 `image_max_pixels=262144` 后会被缩放到相近的有效像素预算，因此中、大图的视觉计算量和延迟基本一致。

4. **Projector 不是主要开销。**  
   Projector 约占 `1.14%` 参数，单次延迟约 `0.09 ms`，对端到端性能影响很小。

## 6. 压缩收益上限

假设将 Vision Encoder 压缩 50%，并且其延迟能够理想地线性下降 50%：

| 场景 | Vision 原始占比 | 理论端到端加速上限 |
|---|---:|---:|
| 1-token 短输出 | 约 6.6% | 约 3.3% |
| 64-token 长输出 | 约 0.5% | 约 0.25% |

以上是理想上限。受 kernel 效率、硬件对齐和固定开销影响，真实加速通常更低。

## 7. 结论

> Vision Encoder 是一个效果敏感但实际开销较低的模块，单独压缩它呈现明显不利的精度—效率权衡。

结合此前观察到的明显性能掉点，压缩 Vision Encoder 最多只能带来约 `3.3%` 的短输出理论加速；在长输出场景中，理论收益仅约 `0.25%`。因此，后续压缩预算应优先投入 Language Model，Vision Encoder 更适合保持原始容量和精度。


