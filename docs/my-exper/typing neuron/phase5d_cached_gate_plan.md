# Phase 5D：Cached-Path Learned Gate

## 1. 动机

Phase 5B 使用整段 `use_cache=False` 前向优化 gate，但最终行为由 `use_cache=True` 逐 token 生成决定。
Qwen3.5 BF16 在两条路径上存在边界差异；rollout256 的 250-mask 可以保持 hook cached generation，物理
裁剪后却在 token 13 分叉。因此 Phase 5D 直接沿 teacher token 执行 cached teacher forcing。

每个 optimization step 使用固定 gate 重建完整 cache。每个 token 计算 cached teacher/student KL 与
teacher-token margin，立即反向，然后把 hybrid cache 中的 attention、convolution 和 recurrent state detach；
整条轨迹结束后才更新 gate。这是一阶 truncated BPTT：状态值来自当前 gate，但不跨 token 反传，避免
256-token cache 图占满显存。

## 2. 双门槛

- `behavioral_feasible`：自由 cached generation 完全一致，且 cached mean KL <= 0.001；
- `strict_feasible`：在 behavioral gate 基础上，cached teacher-forcing top-1 agreement 也必须为 100%。

full-forward fidelity 继续记录为诊断，但不再决定 cached-path 主 gate。所有 hook 候选仍必须经过 Phase 5C
物理 BF16 筛选，cached hook 成功不自动授权 checkpoint。

## 3. 正式运行

建议先运行较小预算：

```bash
PYTHONPATH=src WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error \
python scripts/vulcan/neuron_typing/run_phase5_cached_gate.py \
  --static_frontier saves/neuron_typing/phase5_single_sample/sample_2500_rollout256/frontier.json \
  --output_dir saves/neuron_typing/phase5_single_sample/sample_2500_cached_gate_rollout256 \
  --deletion_budgets 50,100,250 \
  --steps 30 \
  --restarts 3 \
  --learning_rate 0.02 \
  --temperature_start 2.0 \
  --temperature_end 0.5 \
  --init_noise 0.02 \
  --margin_weight 0.05 \
  --margin_target 0.05 \
  --gradient_clip 1.0 \
  --history_interval 5 \
  --num_workers 0 \
  --preprocessing_num_workers 1
```

## 6. Robust rollout256 结果

50/100/250 三档、每档 8 个 restart 的搜索已完整结束。50 档有 5/8 strict，100 档有 5/8 strict；
250 档 8/8 都保持 256-token cached generation 和 teacher-forcing top-1 完全一致，但 cached KL 为
0.00158--0.00195，超过冻结的 0.001 门槛。

对两个代表候选完成了真实 BF16 结构剪枝筛查：

- `delete_100__restart_2`：物理剪除 100 个 neuron 后，256-token 完全一致、top-1 agreement=1.0、
  mean KL=0.000891，原 strict gate 通过；移除 307,200 个参数。
- `delete_250__restart_1`：使用显式 `physical_probe` 模式，物理剪除 250 个 neuron 后，256-token 完全
  一致、top-1 agreement=1.0、mean KL=0.001624；行为门通过但原 KL gate 不通过；移除 768,000 个参数。

因此当前单样本结论分为两条边界：原 `KL<=0.001` 定义下的已验证安全点为 100 neurons；只要求冻结
256-token 行为完全一致时，已验证点至少达到 250 neurons。250 是当前已测试上界，不是不可继续提高的
极限。高 margin 候选成功跨过 hook 到 physical 的 BF16 执行差异，说明 margin 指标比仅按 KL 排序更适合
预测结构剪枝稳定性。

`delete_250__restart_1` 随后保存为非均匀结构 checkpoint 并完成独立重载验证。重载后的每层 FFN 宽度、
参数数和投影权重 hash 与内存结构模型一致；内存与重载 logits 逐元素完全一致，重载后的 256-token
generation 也完全一致。正式 checkpoint 位于
`saves/neuron_typing/phase5_single_sample/sample_2500_phase5d_robust_structural_250_r1/model`，验证报告为同目录
上级的 `equivalence_256.json`。

## 7. 长 horizon、全 restart 物理复评与嵌套边界

对保存后的 250/r1 checkpoint 继续做长窗口验证：512 和 1024 两次实验都在生成 token 278 首次偏离
teacher。hook、内存物理模型与重载模型的首次分叉位置相同，且内存与重载 logits 仍逐元素完全一致。
因此 250 mask 从训练窗口 256 外推了 22 token，但没有保持到 512；这是 rollout horizon 泛化边界，不是
结构转换或 checkpoint 保存问题。

robust 搜索中的其余七个 250-neuron mask 随后全部完成真实 BF16 结构筛查。加上已验证的 restart 1，
250 档物理行为门成功率为 8/8。所有候选都保持 256-token generation 和 cached top-1 完全一致，物理
mean KL 为 0.001579--0.001861。这把 250 从单一构造提升为当前设置下的稳定行为预算。

以 250/r1 为冻结核心，按 rollout256 Taylor saliency 的 `layer_mean` 全局排序追加最低分 neuron，构造严格
嵌套的 300/350/400/450/500 masks。五个候选的 hook 与物理结构模型都保持 256-token 完全一致：

| 删除 neurons | 物理 mean KL | 最小生成 margin | 删除参数 |
|---:|---:|---:|---:|
| 300 | 0.001733 | 0.250 | 921,600 |
| 350 | 0.001823 | 0.125 | 1,075,200 |
| 400 | 0.001879 | 0.125 | 1,228,800 |
| 450 | 0.002047 | 0.250 | 1,382,400 |
| 500 | 0.002009 | 0.125 | 1,536,000 |

因此 rollout256 行为保持的已验证物理下界已从 250 提升到 500；严格 `KL<=0.001` 下界仍为 100。
由于 300--500 已被无训练嵌套探针覆盖，下一轮 learned-gate 不应采用 300/400/500/600 的低区间密集网格，
也无需重复以 500 为起点。推荐预算改为 600/750/1000：600 紧邻当前构造，750 和 1000 用于测试 robust
cached-margin 优化能否恢复过去只在 64-token hook/physical 路径成立的更高预算。若 1000 成功，再追加
1250；若 600 失败，则回到 500--600 做 525/550/575 的局部搜索。

## 8. High-budget learned gate 与 1000-neuron checkpoint

600/750/1000 三档、每档 8 个 restart 的 robust cached-gate 搜索已完整结束。600 档有 7/8 保持
256-token generation，750 和 1000 两档均为 8/8；所有生成成功候选的 cached teacher-forcing top-1
agreement 也为 100%。三档均因 mean KL 超过 0.001 而没有 strict 候选，但行为搜索前沿已推进到 1000。

优先物理筛查了两个 1000-neuron 候选：

- restart 2：物理 256-token exact，mean KL=0.003224，最小 margin=0.25；
- restart 3：物理 256-token exact，mean KL=0.002956，最小 margin=0.25。

二者均通过全部结构与行为检查，说明 1000-neuron 结果不是单一 restart 偶然。restart 3 具有更低的物理
mean/max KL 和略低的 hook-to-physical logit RMSE，因此被保存为正式 checkpoint。独立重载后，每层 FFN
宽度、参数数和投影权重 hash 与内存结构模型一致，内存与重载 logits 逐元素完全一致，重载生成仍为
256/256 exact。模型参数从 852,985,920 降到 849,913,920，移除 3,072,000 参数（全模型 0.3601%）。

当前证据边界更新为：严格 `KL<=0.001` 的 256-token 物理下界为 100 neurons；行为完全一致的 256-token
物理并可重载下界为 1000 neurons。1000 仍是已测试上界而不是数学极限。

对 1000/restart-3 checkpoint 继续执行 512/1024-token 外推验证。真实物理结构模型分别在生成 token 276
首次偏离 teacher；1024-token 实验中模型在生成 991 tokens 后提前 EOS。内存结构模型和独立重载模型在
两次实验中的轨迹完全相同，teacher-forced logits 也逐元素完全一致，且全部结构检查继续通过。与
250/restart-1 在 token 278 首次分叉相比，增加到 1000 neurons 没有实质缩短已验证的长程前缀，但也没有
把 256-token 安全窗口外的边界向后推进。因此目前能正式声称的是“1000-neuron 物理剪枝保持冻结样本的
前 256 个生成 token”，不能声称完整回答或 512/1024-token 行为不变。

1000/restart-3 的 hook 模型在 token 384 才首次分叉，而物理模型在 token 276 分叉。这不是 mask/cluster
映射错误：物理模型尺寸、参数数、保留权重 hash 均正确，且保存前后完全一致。hook 路径保留原始宽度，
在 `down_proj` 输入处将删除通道乘零；物理路径直接执行缩窄后的 GEMM。BF16 下矩阵形状变化会改变 kernel
与归约次序，所以两条路径的 logits 不要求逐位相同。后续所有极限结论都应以物理筛查结果为准，hook
只作为搜索代理。

对应报告：

- `sample_2500_phase5d_robust_structural_1000_r3/equivalence_512.json`
- `sample_2500_phase5d_robust_structural_1000_r3/equivalence_1024.json`

下一步若目标是保持更长回答，应把 teacher/search horizon 提升到至少 512，并针对 token 256 后的低 margin
位置训练，而不是继续只在 rollout256 上盲目增加删除预算。若目标仍限定为前 256 tokens，则可以另开
1250/1500 的预算搜索来继续抬高单样本局部剪枝下界；这两条问题应分开报告。

## 9. Rollout512 长程保护入口与 pilot

已生成独立的 rollout512 静态基线：

`saves/neuron_typing/phase5_single_sample/sample_2500_rollout512/frontier.json`

基线包含完整 512-token teacher、整段 full-forward teacher logits 和重新计算的 Taylor saliency。未剪枝
condition 为 512/512 exact。手写 cached teacher forcing 与 `generate()` teacher 对齐，mean KL 为
`1.94e-15`、top-1 agreement 为 100%。但原始 teacher 本身在 BF16 cached logits 上的最小生成 margin 为
0，共有 30 个 token 的 margin 小于 0.25，其中包括 token 278。因此 rollout256 使用的全局
`robust_margin_threshold=0.25` 不适用于 rollout512；长程搜索使用阈值 0，并继续将真实物理结构筛查作为
最终硬门。

`run_phase5_cached_gate.py` 新增两类兼容旧行为的 token 权重：

- `focus_token_start/end/weight`：提高首次物理分叉附近区间的损失权重；
- `teacher_low_margin_threshold/weight`：提高原 teacher 全轨迹脆弱位置的权重；与区间权重取最大值而不相乘。

同时，训练期间不再只按平滑 objective 最小值保存离散 mask。每一步现在记录 generated-token agreement、
teacher-top agreement、最小生成 margin 和无权 KL，候选选择依次优先最大化前两种 agreement 与最小
margin，再最小化 KL/objective。这避免平均 loss 更小但首次生成分叉反而提前的 mask 覆盖轨迹更好的候选。

100-neuron pilot 的结果表明，单独加权 256--320 区间时 5 steps 仍在 token 31 分叉；再加入全局 teacher
低-margin加权后，同规模试验曾将首次分叉推迟到 token 400。独立重复和10-step试验分别在 token 129
分叉，说明方向有效但短训练尚未稳定达到 512 exact，且离散mask路径不能用“steps越多必然更好”解释。
新版逐步 agreement 诊断在5 steps中从 508/512 提升到 510/512，并正确保留 agreement 最高的中间 step。

墙钟实测：1-step run 为99秒，5 steps 为251--253秒，10 steps 为442秒，增量约38秒/step。正式第一轮不应
直接复刻 rollout256 的三预算八restart配置；建议先跑100/250两个预算、30 steps、4 restarts，预计约
2.5--3小时：

```bash
PYTHONPATH=src WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error \
python scripts/vulcan/neuron_typing/run_phase5_cached_gate.py \
  --static_frontier saves/neuron_typing/phase5_single_sample/sample_2500_rollout512/frontier.json \
  --output_dir saves/neuron_typing/phase5_single_sample/sample_2500_cached_gate_rollout512_dual_focus \
  --deletion_budgets 100,250 \
  --steps 30 \
  --restarts 4 \
  --learning_rate 0.02 \
  --temperature_start 2.0 \
  --temperature_end 0.5 \
  --init_noise 0.03 \
  --margin_weight 0.2 \
  --margin_target 0.5 \
  --focus_token_start 256 \
  --focus_token_end 320 \
  --focus_token_weight 4.0 \
  --teacher_low_margin_threshold 0.25 \
  --teacher_low_margin_weight 4.0 \
  --robust_margin_threshold 0.0 \
  --gradient_clip 1.0 \
  --history_interval 5 \
  --num_workers 0 \
  --preprocessing_num_workers 1
```

输出按run增量落盘。中断后使用完全相同参数追加 `--resume` 即可继续。若100预算4个restart全部失败，下一轮
先降到50；若100通过而250失败，再做150/200；若250通过，再扩展500，而不是直接跳到1000。

第一轮正式搜索完成后，100预算有2/4个hook候选达到512-token strict，250预算4/4保持512-token hook
generation与cached top-1，但KL为0.001075--0.001274。对这6个hook-exact候选全部执行了真实BF16结构筛查，
物理成功率为0/6：

| 候选 | 物理首次分叉 | 物理 mean KL | 物理 top-1 agreement |
|---|---:|---:|---:|
| 100/r0 | 388 | 0.000691 | 0.9941 |
| 100/r1 | 129 | 0.000704 | 0.9922 |
| 250/r0 | 107 | 0.001183 | 0.9961 |
| 250/r1 | 400 | 0.001088 | 0.9980 |
| 250/r2 | 400 | 0.001276 | 0.9980 |
| 250/r3 | 388 | 0.001183 | 0.9980 |

所有候选的artifact、目标宽度、删除参数量和保留权重hash检查均通过，因此失败不是结构映射错误。6个物理
模型的最小生成margin均变为负值（-0.125或-0.25）；`robust_margin_threshold=0` 仍允许hook候选停留在
零margin边界，无法抵抗缩窄GEMM带来的BF16扰动。当前仍没有可声明的512-token物理剪枝下界，也没有保存
这些候选为checkpoint。下一轮应先按预案测试50预算以建立物理下界；同时把候选门提高到严格正margin
（如0.125），而不是只增加同配置restart。

## 10. 原子物理层扫描与首个 rollout512 物理下界

由于全局learned masks同时改变12--18层，新增 `scan_phase5_atomic_physical.py`，在只加载一次模型和冻结teacher
的前提下临时物理缩窄指定层，验证后恢复原投影与共享config。失败候选在首次自由生成分叉时提前停止；
exact候选继续完成完整cached teacher-forcing、KL和top-1验证。结果逐候选增量写入，支持用 `--resume`
追加单层嵌套预算与显式跨层组合。

首先对24层分别删除该层Taylor最低的1个neuron。只有第8、19、23层达到物理512/512 exact且严格
`KL<=0.001`，其余21层首次分叉集中在token 31、82、99、129和400。这说明即使删除量固定为1，物理
安全性仍高度依赖层。

对三个安全层继续做 `2/4/8/16/32/50` 嵌套扩张，并补测第23层delete-3：

- 第8层：delete-1通过，delete-2失败；
- 第19层：delete-1通过，delete-2失败；
- 第23层：delete-1和delete-2通过，delete-3及更高预算失败。

随后测试安全原子的显式组合。`8:1+23:1`、`19:1+23:1` 与 `8:1+23:2` 通过；`8:1+19:1`、
三层各删1、`19:1+23:2` 和三层总删4均失败。因此当前Taylor嵌套构造下的最大已验证候选为
`layers_08x1-23x2`：物理删除3个neuron，移除9,216参数，512-token generation与cached top-1完全一致，
mean KL=0.000581、max KL=0.007222。50个原子/嵌套/组合候选的结构检查全部通过。

完整frontier位于
`saves/neuron_typing/phase5_single_sample/sample_2500_atomic_physical_rollout512/atomic_physical_frontier.json`。
3是当前已验证物理下界，不是全局上界：扫描只测试每层Taylor最低neuron的嵌套集合，尚未枚举替代neuron或
执行physical-in-the-loop组合搜索。下一步应先将最佳3-neuron mask转成标准artifact，执行独立完整筛查、
保存和重载等价验证；之后再围绕该安全核心搜索第四个neuron。

## 11. 最佳3-neuron候选的正式结构checkpoint

新增 `build_phase5_atomic_structural_artifact.py`，将atomic frontier中的严格物理候选冻结为标准
`mask.json + cluster_idx.json + metadata.json` artifact。metadata显式记录原frontier/mask哈希、512-token
冻结horizon、KL阈值以及逐层非均匀目标宽度。最佳候选仍为 `layers_08x1-23x2`，具体删除第8层neuron
2975以及第23层neuron 3105、461；目标宽度为第8层3583、第23层3582，其余层3584。

独立完整筛查暴露出与此前learned-gate相反的数值非对称：hook置零在token 31分叉，而真实物理缩窄保持
512/512 exact，cached mean KL=0.000581、top-1 agreement=1.0，artifact、维度、参数数和权重hash均通过。
因此atomic artifact使用 `source_gate=structural_strict`：hook仍作为诊断记录，但正式门槛以实际可部署的
物理模型为准；重载门槛要求内存物理模型与磁盘checkpoint完全一致。

正式checkpoint已保存到
`saves/neuron_typing/phase5_single_sample/sample_2500_atomic_structural_best3/model`。参数从852,985,920降到
852,976,704，精确移除9,216个BF16参数（18,432 bytes，约0.0010804%）。重载验证通过全部7项结构检查；
内存物理模型与重载模型的logits max/mean/RMSE差异均为0，生成序列完全相同，重载后仍保持512/512 exact与
mean KL=0.000581。对应结果为 `physical_screen_512.json` 和 `equivalence_512.json`。

这建立了首个可保存、可重载、可复现的单样本512-token物理剪枝下界。下一轮应以这个3-neuron checkpoint
或mask为安全核心，执行physical-in-the-loop的第四neuron替代搜索，而不是继续依赖hook候选排序。

## 12. Physical-in-the-loop第四neuron搜索

新增 `scan_phase5_core_extension_physical.py`，把已验证的3-neuron artifact固定为安全核心，每个候选只增加
一个未删除neuron并立即构造真实物理模型。自由生成在首次分叉时早停，只有512-token exact候选才继续计算
完整cached fidelity。扫描逐候选增量写入并支持 `--resume`；`candidate_ranks` 表示每层排除安全核心后的一
基Taylor顺序，因此可以先做每层rank-1，再按需扩展更深排名。

第一轮测试24层各自的rank-1候选，所有候选均通过artifact、目标宽度、删除参数数和保留权重hash检查。
其中22个候选在token 12--388之间分叉，两个候选达到严格可行：

| 新增neuron | 512-token generation | cached mean KL | cached max KL |
|---|---:|---:|---:|
| layer 11 / neuron 2118 | exact | 0.000583 | 0.004697 |
| layer 23 / neuron 2072 | exact | 0.000578 | 0.007192 |

选择KL更低的layer 23 / neuron 2072作为正式第四neuron。最终mask为layer 8删除2975，layer 23删除
3105、461、2072。值得注意的是layer 23的前三个Taylor neuron单层嵌套删除此前失败，而加入layer 8 /
2975后却严格通过，说明BF16物理边界存在跨层补偿，不能用单层安全性简单相加。

`build_phase5_core_extension_artifact.py` 已将最佳候选冻结成标准artifact，完整物理筛查通过；checkpoint保存
到 `saves/neuron_typing/phase5_single_sample/sample_2500_core4_structural_best/model`。第8层宽度3583、第23层
宽度3581，其余层3584；总参数从852,985,920降到852,973,632，精确移除12,288个BF16参数。重载后全部
结构检查通过，内存物理模型与重载模型logits差异为0，二者均保持512/512 exact与mean KL=0.000578。

当前正式物理下界因此从3提高到4。rank-2及更深候选尚未扫描，因为rank-1已找到两个解；下一轮可直接以
4-neuron artifact为安全核心搜索第五neuron，同时保留layer 11 / neuron 2118作为备选分支。

## 13. 第五neuron扩张与正式checkpoint

以正式4-neuron artifact作为安全核心，再次测试24层各自rank-1的未删除Taylor neuron。23个候选在
token 31--388之间分叉，唯一严格解是上一轮的备选 `layer 11 / neuron 2118`：512/512 exact，搜索阶段
cached mean KL=0.000586。所有24个候选的结构检查均通过。由于rank-1已有唯一解，本轮没有扩展rank-2--8。

最终5-neuron mask为：layer 8删除2975；layer 11删除2118；layer 23删除3105、461、2072。标准完整物理
筛查通过，cached mean KL=0.000586、max KL=0.004694、top-1 agreement=1.0。正式checkpoint位于
`saves/neuron_typing/phase5_single_sample/sample_2500_core5_structural_best/model`；对应层宽为layer 8=3583、
layer 11=3583、layer 23=3581，其余层3584。参数从852,985,920降到852,970,560，精确移除15,360个
BF16参数（30,720 bytes，约0.0018007%）。

重载验证再次通过全部结构门槛；内存物理模型与重载模型的logits max/mean/RMSE差异均为0，重载后仍为
512/512 exact和mean KL=0.000586。因此当前正式、可保存、可重载的单样本物理下界从4提高到5。下一轮
可以5-neuron artifact为核心搜索第六neuron；若rank-1无解，再扩展rank-2--8。

结果增量写入 `cached_frontier.json`。使用相同优化参数和 `--resume` 可追加预算或 restart。

当前 2-step smoke 中，50-neuron cached KL 从 0.000609 降到 0.000585，首次生成分叉从 token 32 推迟到
token 210。单个 256-token optimization step 实测约 10--20 秒；上述 9 个 run 预计约 1.5--2 小时。

## 4. Phase 5C 接口

找到结果后，`build_phase5_structural_artifact.py` 会优先选择 `best_strict_feasible`；只有显式传入
`--allow_behavioral` 才会选择仅 behavioral 可行的候选。metadata 会保存 `source_fidelity_path=cached`，
后续 screen/equivalence 工具据此使用 cached KL 和 cached top-1 gate。

## 5. Physical robustness gate

首轮正式结果中的 50/100/250-neuron cached strict 候选在 hook 下都保持 256-token 完全一致，但物理
BF16 裁剪后分别在 token 129、154、99/129 分叉。结构尺寸、参数数量和权重 hash 检查全部通过，差异来自
改变 GEMM 形状后的 BF16 数值路径，而不是 artifact 映射错误。

后续搜索新增 `--robust_margin_threshold`。每个候选记录 teacher 生成 token 相对最强竞争 token 的最小、
1% 分位和平均 logit margin；`robust_feasible` 要求 strict gate 通过且最小 margin 达到阈值。同一预算下，
artifact builder 优先选择 robust 候选，robust 候选之间优先选择最小 margin 最大者。

建议下一轮将训练 margin target 提高到 0.5，并使用 0.25 的冻结门槛：

```bash
PYTHONPATH=src WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error \
python scripts/vulcan/neuron_typing/run_phase5_cached_gate.py \
  --static_frontier saves/neuron_typing/phase5_single_sample/sample_2500_rollout256/frontier.json \
  --output_dir saves/neuron_typing/phase5_single_sample/sample_2500_cached_gate_robust_rollout256 \
  --deletion_budgets 50,100,250 \
  --steps 50 \
  --restarts 8 \
  --learning_rate 0.02 \
  --temperature_start 2.0 \
  --temperature_end 0.5 \
  --init_noise 0.03 \
  --margin_weight 0.2 \
  --margin_target 0.5 \
  --robust_margin_threshold 0.25 \
  --gradient_clip 1.0 \
  --history_interval 5 \
  --num_workers 0 \
  --preprocessing_num_workers 1
```

## 14. 第六neuron扩张与正式checkpoint

以正式5-neuron artifact为安全核心，扫描24层各自rank-1的最低未删除Taylor neuron。全部候选的artifact、
目标宽度、参数删除量和保留权重检查均通过；22个候选在token 31、82、129、154、357或388首次分叉，两个
候选保持512/512 exact并通过`KL<=0.001`严格门：

| 新增neuron | 512-token generation | cached mean KL | cached max KL |
|---|---:|---:|---:|
| layer 22 / neuron 2525 | exact | 0.000577 | 0.004746 |
| layer 23 / neuron 1998 | exact | 0.000578 | 0.004699 |

按mean KL选择layer 22 / neuron 2525。最终6-neuron mask为：layer 8删除2975；layer 11删除2118；layer 22
删除2525；layer 23删除3105、461、2072。层宽分别为layer 8=3583、layer 11=3583、layer 22=3583、
layer 23=3581，其余层3584。参数从852,985,920降到852,967,488，精确移除18,432个BF16参数（36,864
bytes，约0.0021609%）。

标准完整物理筛查通过。保存后的checkpoint也通过全部7项结构检查；内存物理模型与重载模型的logits
max/mean/RMSE差异均为0，重载后保持512/512 exact、top-1 agreement=1.0、mean KL=0.000577、max
KL=0.004745。正式checkpoint位于
`saves/neuron_typing/phase5_single_sample/sample_2500_core6_structural_best/model`。

导出时同时修正了core-extension metadata沿用基础核心`deletions_by_layer`的问题，并加入回归断言；该问题
只影响溯源计数字段，不影响mask或模型权重。当前正式、可保存、可重载的单样本物理剪枝下界从5提高到6。
下一轮直接以6-neuron artifact为核心搜索第七neuron；若24个rank-1候选全部失败，再扩展rank-2--8。

## 15. 第七neuron扩张与正式checkpoint

以正式6-neuron artifact为安全核心，扫描24层各自rank-1的最低未删除Taylor neuron。全部候选均通过四项
结构检查；23个失败候选的首次分叉分布为token 31（5个）、99（2个）、129（8个）、154（2个）、209
（1个）和388（5个）。唯一严格解为layer 23 / neuron 1998：512/512 exact、cached mean KL=0.000575、
max KL=0.004747、top-1 agreement=1.0。由于rank-1已有解，本轮没有扩展rank-2--8。

最终7-neuron mask为：layer 8删除2975；layer 11删除2118；layer 22删除2525；layer 23删除461、1998、
2072、3105。层宽为layer 8=3583、layer 11=3583、layer 22=3583、layer 23=3580，其余层3584。参数从
852,985,920降到852,964,416，精确移除21,504个BF16参数（43,008 bytes，约0.0025210%）。

独立完整物理筛查通过；hook诊断仍在token 357分叉，但真实结构模型保持512/512 exact。正式checkpoint
位于`saves/neuron_typing/phase5_single_sample/sample_2500_core7_structural_best/model`。重载验证通过全部7项
结构检查，内存结构模型与重载模型的logits max/mean/RMSE差异均为0；重载后mean KL=0.000575、max
KL=0.004747、top-1 agreement=1.0，生成序列仍512/512 exact。artifact metadata、checkpoint中的provenance
副本及`pruning_summary.json`记录的哈希完全一致。

当前正式、可保存、可重载的单样本物理剪枝下界从6提高到7。下一轮以7-neuron artifact为核心搜索第八
neuron；仍先扫描24个rank-1，只有全部失败时才扩展rank-2--8。

## 16. 快速连续增长、深层Taylor扩张与13-neuron checkpoint

为减少每个中间核心重复screen/save/reload的开销，从正式7-neuron artifact开始使用快速连续增长：每个中间
核心只执行24个rank-1真实物理候选扫描并导出下一artifact，在停滞点才扩展rank-2--8。增长轨迹如下：

| 新核心 | 新增neuron | rank | 候选数 | 严格解数 | cached mean KL |
|---|---|---:|---:|---:|---:|
| 8 | layer 23 / 2059 | 1 | 24 | 1 | 0.000594 |
| 9 | layer 23 / 2725 | 1 | 24 | 1 | 0.000589 |
| 10 | layer 23 / 2817 | 1 | 24 | 1 | 0.000571 |
| 11 | layer 23 / 3179 | 8 | 192 | 4 | 0.000568 |
| 12 | layer 20 / 225 | 1 | 24 | 2 | 0.000556 |
| 13 | layer 19 / 2172 | 5 | 192 | 6 | 0.000556 |

core10和core12都出现24个rank-1全部失败，但rank-2--8分别找到4个和6个严格解。因此rank-1停滞不是物理
上界，静态Taylor前8名的深层扫描确实能发现补偿路径。core10的最佳第11个neuron位于rank-8；core12的
最佳第13个neuron位于rank-5。core13扫描还在layer 19、20、21、23形成多个可行分支，说明当前边界具有
明显的跨层非单调性。

正式13-neuron mask为：layer 8删除2975；layer 11删除2118；layer 19删除2172；layer 20删除225；layer 22
删除2525；layer 23删除461、1998、2059、2072、2725、2817、3105、3179。目标宽度为layer 8/11/19/20/22
=3583、layer 23=3576，其余层3584。参数从852,985,920降到852,945,984，精确移除39,936个BF16参数
（79,872 bytes，约0.0046819%）。

独立物理筛查与checkpoint重载验证均通过：真实结构模型保持512/512 exact、top-1 agreement=1.0、mean
KL=0.000556、max KL=0.005603；内存结构模型与重载模型logits max/mean/RMSE差异均为0。hook诊断在token
129分叉，再次证明正式门槛必须使用physical structural path。正式checkpoint位于
`saves/neuron_typing/phase5_single_sample/sample_2500_core13_structural_best/model`。

本轮按耗时控制停在13，并未测试core13到core14。因此13是新的正式物理下界，不是当前rank-1或rank-8
局部上界。下一步如继续增长，应先测试core13的24个rank-1；若失败，再决定是否值得继续完整rank-2--8，
或转向动态saliency与beam/swap搜索。
