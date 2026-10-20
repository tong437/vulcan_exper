# Response-state ablation (1% per layer)

Scores use the same frozen 1,000 examples as the response-state study. Positive drops indicate worse performance.

| Model | Dataset | Figure |
|---|---|---|
| Qwen3.5-2B | MME | [PNG](figures/response_state_ablation/qwen3_5_2b__mme__response_state_ablation.png) / [PDF](figures/response_state_ablation/qwen3_5_2b__mme__response_state_ablation.pdf) |
| Qwen3.5-2B | MMBench | [PNG](figures/response_state_ablation/qwen3_5_2b__mmbench__response_state_ablation.png) / [PDF](figures/response_state_ablation/qwen3_5_2b__mmbench__response_state_ablation.pdf) |
| Qwen3.5-2B | TextVQA | [PNG](figures/response_state_ablation/qwen3_5_2b__textvqa__response_state_ablation.png) / [PDF](figures/response_state_ablation/qwen3_5_2b__textvqa__response_state_ablation.pdf) |
| Qwen3.5-2B | ChartQA | [PNG](figures/response_state_ablation/qwen3_5_2b__chartqa__response_state_ablation.png) / [PDF](figures/response_state_ablation/qwen3_5_2b__chartqa__response_state_ablation.pdf) |

## Qwen3.5-2B numerical results

Baseline scores use each dataset's own metric. All other cells are score drops from baseline in percentage points; random conditions show mean +/- 1 SD across 10 runs.

| Condition | MME | MMBench | TextVQA | ChartQA |
|---|---:|---:|---:|---:|
| baseline | 83.80% | 84.10% | 77.48% | 69.20% |
| V_random | +0.25 +/- 0.59 | +1.37 +/- 0.49 | +0.70 +/- 0.49 | +0.58 +/- 0.49 |
| T_random | +2.03 +/- 0.61 | +1.86 +/- 0.47 | +0.44 +/- 0.85 | +0.38 +/- 0.54 |
| M_random | +1.72 +/- 0.50 | +3.70 +/- 0.60 | +1.50 +/- 0.64 | +2.55 +/- 0.78 |
| W_random | +0.05 +/- 0.49 | +1.10 +/- 0.37 | +0.31 +/- 0.60 | +0.12 +/- 0.42 |
| V_max | +3.00 | +2.20 | +3.42 | +5.00 |
| T_max | +15.10 | +10.50 | +4.08 | +7.00 |
| M_max | +12.40 | +43.10 | +24.51 | +29.40 |
| W_max | +0.60 | +1.60 | -0.20 | +0.60 |
| V_min | -0.60 | +1.10 | +1.22 | +0.50 |
| T_min | +0.10 | +1.20 | +0.49 | +0.20 |
| M_min | -0.40 | +2.40 | -0.30 | +0.00 |
| W_min | -0.20 | +0.60 | +0.65 | -0.60 |
| global_random | +0.02 +/- 1.02 | +1.73 +/- 0.62 | +0.27 +/- 0.88 | +0.43 +/- 0.75 |

Machine-readable results: [CSV](advisor_2026_09_response_state_ablation.csv) / [JSON](advisor_2026_09_response_state_ablation.json)
