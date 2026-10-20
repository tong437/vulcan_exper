# Layerwise V/T/M/W response-state distributions

For state `c` and decoder layer `l`, this report computes
`p_c^(l) = (1 / |N^(l)|) sum_j N_c^(l,j) / N` from the frozen 1,000-sample neuron profiles. Every stacked bar is one
real decoder layer and sums to 100%. Layer widths are not forced to match: Gemma's 6,144- and 12,288-neuron layers are
averaged over their actual neurons, and no model is padded with fake neurons or layers.

This is a descriptive baseline for the planned answer-generation intervention. It shows how often neurons occupy each
response state, but does not yet measure the effect of removing or modifying any state-defined neuron set on answers.

Maximum absolute layer-sum residual across all 16 runs: `1.110e-16`.

| Model | Dataset | Layers | Neurons | V | T | M | W |
|---|---|---:|---:|---:|---:|---:|---:|
| LLaVA-1.5-7B | MME | 32 | 352,256 | 18.8% | 18.8% | 14.9% | 47.5% |
| LLaVA-1.5-7B | MMBench | 32 | 352,256 | 18.5% | 17.1% | 15.2% | 49.3% |
| LLaVA-1.5-7B | TextVQA | 32 | 352,256 | 18.5% | 18.8% | 15.1% | 47.6% |
| LLaVA-1.5-7B | ChartQA | 32 | 352,256 | 18.1% | 18.4% | 15.2% | 48.3% |
| Qwen3-VL-2B | MME | 28 | 172,032 | 17.3% | 16.5% | 14.2% | 52.1% |
| Qwen3-VL-2B | MMBench | 28 | 172,032 | 16.1% | 14.7% | 15.1% | 54.0% |
| Qwen3-VL-2B | TextVQA | 28 | 172,032 | 17.3% | 15.3% | 14.2% | 53.3% |
| Qwen3-VL-2B | ChartQA | 28 | 172,032 | 16.0% | 14.4% | 15.4% | 54.2% |
| Qwen3.5-2B | MME | 24 | 147,456 | 17.1% | 18.9% | 16.4% | 47.6% |
| Qwen3.5-2B | MMBench | 24 | 147,456 | 16.7% | 17.0% | 17.4% | 48.9% |
| Qwen3.5-2B | TextVQA | 24 | 147,456 | 16.3% | 19.7% | 16.4% | 47.6% |
| Qwen3.5-2B | ChartQA | 24 | 147,456 | 15.2% | 17.6% | 17.9% | 49.3% |
| Gemma-4-E2B-it | MME | 35 | 337,920 | 14.7% | 16.2% | 14.4% | 54.7% |
| Gemma-4-E2B-it | MMBench | 35 | 337,920 | 14.5% | 16.0% | 14.4% | 55.1% |
| Gemma-4-E2B-it | TextVQA | 35 | 337,920 | 13.7% | 15.9% | 14.8% | 55.7% |
| Gemma-4-E2B-it | ChartQA | 35 | 337,920 | 13.6% | 15.7% | 14.9% | 55.8% |

## Descriptive observations

- `W` is the largest mean state share in all 476 observed decoder layers. Across complete runs, its neuron-weighted share ranges from 47.5% to 55.8%.
- The final layer has a higher `W` share than the first layer in 16/16 runs. This pattern is especially strong in LLaVA, Qwen3-VL, and Qwen3.5; Gemma is more oscillatory across depth.
- These are occupancy proportions, not answer-impact estimates. `W` means below the within-layer importance threshold for both modalities, not zero activation or an unimportant neuron.


![Layerwise state distributions](figures/advisor_2026_09_layerwise_state_distributions.png)

The complete `476`-row layer table is `advisor_2026_09_layerwise_states_long.csv`; run summaries are in
`advisor_2026_09_layerwise_states_summary.csv`.

## Individual figures and tables

| Model | MME | MMBench | TextVQA | ChartQA |
|---|---|---|---|---|
| LLaVA-1.5-7B | [PNG](figures/layerwise_state_distributions/llava_1_5_7b__mme__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/llava_1_5_7b__mme__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/llava_1_5_7b__mme__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/llava_1_5_7b__mmbench__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/llava_1_5_7b__mmbench__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/llava_1_5_7b__mmbench__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/llava_1_5_7b__textvqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/llava_1_5_7b__textvqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/llava_1_5_7b__textvqa__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/llava_1_5_7b__chartqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/llava_1_5_7b__chartqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/llava_1_5_7b__chartqa__layerwise_state_distribution.csv) |
| Qwen3-VL-2B | [PNG](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__mme__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__mme__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_vl_2b_instruct__mme__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__mmbench__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__mmbench__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_vl_2b_instruct__mmbench__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__textvqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__textvqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_vl_2b_instruct__textvqa__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__chartqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_vl_2b_instruct__chartqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_vl_2b_instruct__chartqa__layerwise_state_distribution.csv) |
| Qwen3.5-2B | [PNG](figures/layerwise_state_distributions/qwen3_5_2b__mme__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_5_2b__mme__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_5_2b__mme__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/qwen3_5_2b__mmbench__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_5_2b__mmbench__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_5_2b__mmbench__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/qwen3_5_2b__textvqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_5_2b__textvqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_5_2b__textvqa__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/qwen3_5_2b__chartqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/qwen3_5_2b__chartqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/qwen3_5_2b__chartqa__layerwise_state_distribution.csv) |
| Gemma-4-E2B-it | [PNG](figures/layerwise_state_distributions/gemma_4_e2b_it__mme__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/gemma_4_e2b_it__mme__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/gemma_4_e2b_it__mme__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/gemma_4_e2b_it__mmbench__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/gemma_4_e2b_it__mmbench__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/gemma_4_e2b_it__mmbench__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/gemma_4_e2b_it__textvqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/gemma_4_e2b_it__textvqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/gemma_4_e2b_it__textvqa__layerwise_state_distribution.csv) | [PNG](figures/layerwise_state_distributions/gemma_4_e2b_it__chartqa__layerwise_state_distribution.png) / [PDF](figures/layerwise_state_distributions/gemma_4_e2b_it__chartqa__layerwise_state_distribution.pdf) / [CSV](tables/layerwise_state_distributions/gemma_4_e2b_it__chartqa__layerwise_state_distribution.csv) |
