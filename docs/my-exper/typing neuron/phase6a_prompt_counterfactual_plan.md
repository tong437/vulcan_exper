# Phase 6A：Prompt 与反事实鲁棒性审计

## 目标

判断 Phase 5E 的 62,950-neuron 物理模型保留的是可迁移的图像语义，还是固定
image-prompt-caption 三元组的模板记忆。原模型作为同时运行的参照，避免把基础模型本身的失败归因于剪枝。

该阶段只审计既有物理模型，不重新搜索或修改 mask。

## 冻结实验组

1. 五个不泄露 bicycle/train 答案的 prompt：原 prompt、通用描述、alt-text、主物体与场景总结、照片问答。
2. 一个显式提到 bicycle 的诊断 prompt；不计入主 prompt gate。
3. 三个语义保持图像变换：水平翻转、0.65 亮度、中心保留 90% 后恢复原分辨率。
4. 六个反事实：train-only crop、自行车区域遮挡、bike-only 错图、train-only 错图、无关室内图、纯灰 blank image。

所有生成均使用 `do_sample=false`、`enable_thinking=false`、`max_new_tokens=64`。

## 双向判据

正向样本要求保留“多辆自行车位于列车内”的 Phase 5E v2 语义，并正常结束且不含重复或角色词垃圾。

反事实样本不能复述原场景，并按冻结事实分别要求：

- train-only：提到 train，不能提到 bicycle；
- bike-only：提到 bicycle，不能提到 train；
- unrelated：不能提到 bicycle 或 train；
- blank-image：不能断言原 bicycle-in-train 场景。

## 预注册 Gate

- canonical prompt 必须通过；
- 五个无泄露 prompt 至少 4/5 通过；
- 三个 benign image 变换至少 2/3 通过；
- 六个反事实中原场景复述必须为 0/6；
- 六个反事实至少 5/6 通过各自事实检查；
- 全部 15 个输出必须正常结束、无重复和角色词垃圾。

主结果同时报告绝对 gate 和 paired retention：只在原模型通过的样本上统计剪枝模型保留率。

本审计是固定案例审计，不是总体鲁棒率估计；若失败，只能定位失败模式，不能估计其自然分布概率。

## 执行前技术修订

v1 的真正无图行与多模态行通过同一 Qwen3.5 dataset/collator 处理时产生了 `0 image tokens / 64 image
features`，在任何模型生成前即被架构校验拒绝。该输出不构成模型失败。v2 保留其他 14 个案例与全部 gate
不变，只将无图行替换为同分辨率纯灰 blank image，使模型接口和视觉 token 数与其他案例一致。v1 失败产物
保留用于审计，不纳入结果。
