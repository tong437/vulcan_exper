# State-pair decomposition

For each neuron and unordered state pair `ab`, the absolute contribution is
`R_ab = 2 N_a N_b / (N(N-1))`. Each reported `R_ab` is first averaged over all included decoder MLP neurons. The
six averaged contributions sum to the previously reported overall change rate `R`; the displayed percentages are then
computed as `R_ab / R`. This order avoids giving a low-change neuron the same composition weight as a high-change
neuron and avoids the undefined per-neuron composition when `R=0`.

Maximum absolute neuron-level identity residual across all 16 runs: `2.220e-16`.

| Model | Dataset | R | VT | VM | VW | TM | TW | MW | Dominant pair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| LLaVA-1.5-7B | MME | 0.433521 | 6.8% | 15.0% | 25.9% | 9.1% | 33.9% | 9.3% | TW |
| LLaVA-1.5-7B | MMBench | 0.449266 | 7.0% | 15.3% | 25.0% | 7.8% | 34.5% | 10.3% | TW |
| LLaVA-1.5-7B | TextVQA | 0.477110 | 7.2% | 16.7% | 21.4% | 7.2% | 37.4% | 10.2% | TW |
| LLaVA-1.5-7B | ChartQA | 0.425338 | 5.5% | 21.1% | 15.2% | 5.8% | 44.9% | 7.4% | TW |
| Qwen3-VL-2B | MME | 0.356979 | 5.5% | 13.4% | 32.4% | 11.4% | 28.4% | 8.8% | VW |
| Qwen3-VL-2B | MMBench | 0.403855 | 6.1% | 13.3% | 29.5% | 9.5% | 29.6% | 12.0% | TW |
| Qwen3-VL-2B | TextVQA | 0.363509 | 5.5% | 16.7% | 27.4% | 8.2% | 33.7% | 8.5% | TW |
| Qwen3-VL-2B | ChartQA | 0.373076 | 6.2% | 16.6% | 25.5% | 8.7% | 33.7% | 9.3% | TW |
| Qwen3.5-2B | MME | 0.396800 | 6.7% | 12.6% | 29.6% | 14.0% | 28.3% | 8.8% | VW |
| Qwen3.5-2B | MMBench | 0.456560 | 7.6% | 12.4% | 28.0% | 11.6% | 28.5% | 11.8% | TW |
| Qwen3.5-2B | TextVQA | 0.424358 | 6.8% | 16.2% | 20.5% | 9.7% | 37.9% | 8.8% | TW |
| Qwen3.5-2B | ChartQA | 0.422823 | 7.0% | 16.8% | 19.8% | 9.7% | 36.9% | 9.8% | TW |
| Gemma-4-E2B-it | MME | 0.344638 | 5.5% | 11.9% | 30.6% | 11.9% | 31.8% | 8.4% | TW |
| Gemma-4-E2B-it | MMBench | 0.374428 | 6.0% | 12.8% | 26.6% | 9.5% | 36.2% | 9.0% | TW |
| Gemma-4-E2B-it | TextVQA | 0.377963 | 6.0% | 12.7% | 25.8% | 9.7% | 36.6% | 9.2% | TW |
| Gemma-4-E2B-it | ChartQA | 0.373742 | 6.1% | 14.5% | 22.6% | 8.4% | 39.9% | 8.5% | TW |

## Descriptive observations

- `TW` is the largest component in 14/16 runs. The two exceptions are Qwen3-VL-2B × MME (VW), Qwen3.5-2B × MME (VW), where `VW` is largest.
- Giving each run equal descriptive weight, the mean composition is TW 34.5%, VW 25.4%, VM 14.9%, TM 9.5%, MW 9.4%, VT 6.3%. This is a summary of the 16 normalized panels, not a new pooled-neuron estimand.
- Pairs involving `W` (`VW + TW + MW`) account for 69.3% on average across runs (range 66.4%–71.8%). Thus most observed differences involve a sample switching between weak-on-both and one of the other response states.
- These pairs are unordered: for example, `TW` does not distinguish `T→W` from `W→T`, and none of the components is a causal transition effect.


![State-pair composition overview](figures/advisor_2026_09_state_pair_composition.png)

Absolute contributions and normalized shares are stored in `advisor_2026_09_state_pair_long.csv`; the one-row-per-run
table is `advisor_2026_09_state_pair_summary.csv`.

## Individual figures

| Model | MME | MMBench | TextVQA | ChartQA |
|---|---|---|---|---|
| LLaVA-1.5-7B | [PNG](figures/state_pair_composition/llava_1_5_7b__mme__state_pair_composition.png) / [PDF](figures/state_pair_composition/llava_1_5_7b__mme__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/llava_1_5_7b__mmbench__state_pair_composition.png) / [PDF](figures/state_pair_composition/llava_1_5_7b__mmbench__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/llava_1_5_7b__textvqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/llava_1_5_7b__textvqa__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/llava_1_5_7b__chartqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/llava_1_5_7b__chartqa__state_pair_composition.pdf) |
| Qwen3-VL-2B | [PNG](figures/state_pair_composition/qwen3_vl_2b_instruct__mme__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_vl_2b_instruct__mme__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/qwen3_vl_2b_instruct__mmbench__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_vl_2b_instruct__mmbench__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/qwen3_vl_2b_instruct__textvqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_vl_2b_instruct__textvqa__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/qwen3_vl_2b_instruct__chartqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_vl_2b_instruct__chartqa__state_pair_composition.pdf) |
| Qwen3.5-2B | [PNG](figures/state_pair_composition/qwen3_5_2b__mme__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_5_2b__mme__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/qwen3_5_2b__mmbench__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_5_2b__mmbench__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/qwen3_5_2b__textvqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_5_2b__textvqa__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/qwen3_5_2b__chartqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/qwen3_5_2b__chartqa__state_pair_composition.pdf) |
| Gemma-4-E2B-it | [PNG](figures/state_pair_composition/gemma_4_e2b_it__mme__state_pair_composition.png) / [PDF](figures/state_pair_composition/gemma_4_e2b_it__mme__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/gemma_4_e2b_it__mmbench__state_pair_composition.png) / [PDF](figures/state_pair_composition/gemma_4_e2b_it__mmbench__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/gemma_4_e2b_it__textvqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/gemma_4_e2b_it__textvqa__state_pair_composition.pdf) | [PNG](figures/state_pair_composition/gemma_4_e2b_it__chartqa__state_pair_composition.png) / [PDF](figures/state_pair_composition/gemma_4_e2b_it__chartqa__state_pair_composition.pdf) |
