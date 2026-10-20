# State entropy and effective category counts

For each neuron, let `S = -sum_c p_c log(p_c)` and `H_norm = S / log(4)`. The normalized entropy lies in `[0, 1]`.
The effective category count is `K_eff = exp(S) = 4 ** H_norm`, so one/two/three/four equally occupied states map
exactly to `1/2/3/4`. The table reports the median across all decoder MLP neurons in each run.

| Model / Dataset | MME | MMBench | TextVQA | ChartQA |
|---|---:|---:|---:|---:|
| LLaVA-1.5-7B | 2.210 | 2.297 | 2.382 | 1.999 |
| Qwen3-VL-2B | 1.939 | 2.156 | 1.942 | 1.987 |
| Qwen3.5-2B | 2.080 | 2.429 | 2.147 | 2.166 |
| Gemma-4-E2B-it | 1.888 | 1.994 | 2.012 | 1.978 |

The complete mean, quantile, and descriptive reference-band summaries are stored in
`advisor_2026_09_entropy_effective_long.csv`. The intervals around 1/2/3/4 in that file are descriptive only and are
not clustering thresholds.

![Effective category distributions](figures/advisor_2026_09_effective_category_distributions.png)

## Individual reference-style figures

| Model | MME | MMBench | TextVQA | ChartQA |
|---|---|---|---|---|
| LLaVA-1.5-7B | [MME PNG](figures/effective_category_distributions/llava_1_5_7b__mme__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/llava_1_5_7b__mme__effective_category_distribution.pdf) | [MMBench PNG](figures/effective_category_distributions/llava_1_5_7b__mmbench__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/llava_1_5_7b__mmbench__effective_category_distribution.pdf) | [TextVQA PNG](figures/effective_category_distributions/llava_1_5_7b__textvqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/llava_1_5_7b__textvqa__effective_category_distribution.pdf) | [ChartQA PNG](figures/effective_category_distributions/llava_1_5_7b__chartqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/llava_1_5_7b__chartqa__effective_category_distribution.pdf) |
| Qwen3-VL-2B | [MME PNG](figures/effective_category_distributions/qwen3_vl_2b_instruct__mme__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_vl_2b_instruct__mme__effective_category_distribution.pdf) | [MMBench PNG](figures/effective_category_distributions/qwen3_vl_2b_instruct__mmbench__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_vl_2b_instruct__mmbench__effective_category_distribution.pdf) | [TextVQA PNG](figures/effective_category_distributions/qwen3_vl_2b_instruct__textvqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_vl_2b_instruct__textvqa__effective_category_distribution.pdf) | [ChartQA PNG](figures/effective_category_distributions/qwen3_vl_2b_instruct__chartqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_vl_2b_instruct__chartqa__effective_category_distribution.pdf) |
| Qwen3.5-2B | [MME PNG](figures/effective_category_distributions/qwen3_5_2b__mme__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_5_2b__mme__effective_category_distribution.pdf) | [MMBench PNG](figures/effective_category_distributions/qwen3_5_2b__mmbench__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_5_2b__mmbench__effective_category_distribution.pdf) | [TextVQA PNG](figures/effective_category_distributions/qwen3_5_2b__textvqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_5_2b__textvqa__effective_category_distribution.pdf) | [ChartQA PNG](figures/effective_category_distributions/qwen3_5_2b__chartqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/qwen3_5_2b__chartqa__effective_category_distribution.pdf) |
| Gemma-4-E2B-it | [MME PNG](figures/effective_category_distributions/gemma_4_e2b_it__mme__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/gemma_4_e2b_it__mme__effective_category_distribution.pdf) | [MMBench PNG](figures/effective_category_distributions/gemma_4_e2b_it__mmbench__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/gemma_4_e2b_it__mmbench__effective_category_distribution.pdf) | [TextVQA PNG](figures/effective_category_distributions/gemma_4_e2b_it__textvqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/gemma_4_e2b_it__textvqa__effective_category_distribution.pdf) | [ChartQA PNG](figures/effective_category_distributions/gemma_4_e2b_it__chartqa__effective_category_distribution.png) / [PDF](figures/effective_category_distributions/gemma_4_e2b_it__chartqa__effective_category_distribution.pdf) |
