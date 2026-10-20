# Response-state ablation (1% per layer)

Scores use the same frozen 1,000 examples as the response-state study. Positive drops indicate worse performance.

| Model | Dataset | Figure |
|---|---|---|
| Gemma-4-E2B-it | MME | [PNG](figures/response_state_ablation/gemma_4_e2b_it__mme__response_state_ablation.png) / [PDF](figures/response_state_ablation/gemma_4_e2b_it__mme__response_state_ablation.pdf) |
| Gemma-4-E2B-it | MMBench | [PNG](figures/response_state_ablation/gemma_4_e2b_it__mmbench__response_state_ablation.png) / [PDF](figures/response_state_ablation/gemma_4_e2b_it__mmbench__response_state_ablation.pdf) |
| Gemma-4-E2B-it | TextVQA | [PNG](figures/response_state_ablation/gemma_4_e2b_it__textvqa__response_state_ablation.png) / [PDF](figures/response_state_ablation/gemma_4_e2b_it__textvqa__response_state_ablation.pdf) |
| Gemma-4-E2B-it | ChartQA | [PNG](figures/response_state_ablation/gemma_4_e2b_it__chartqa__response_state_ablation.png) / [PDF](figures/response_state_ablation/gemma_4_e2b_it__chartqa__response_state_ablation.pdf) |

## Gemma-4-E2B-it numerical results

Baseline scores use each dataset's own metric. All other cells are score drops from baseline in percentage points; random conditions show mean +/- 1 SD across 10 runs.

| Condition | MME | MMBench | TextVQA | ChartQA |
|---|---:|---:|---:|---:|
| baseline | 70.40% | 73.10% | 64.89% | 44.80% |
| V_random | +1.07 +/- 0.25 | +0.42 +/- 0.53 | +0.40 +/- 0.49 | +0.92 +/- 0.35 |
| T_random | +1.24 +/- 0.63 | +1.02 +/- 0.52 | +1.00 +/- 0.64 | +0.26 +/- 0.20 |
| M_random | +4.85 +/- 0.38 | +3.00 +/- 0.85 | +4.43 +/- 0.73 | +3.43 +/- 0.57 |
| W_random | +0.78 +/- 0.29 | +0.53 +/- 0.31 | +0.29 +/- 0.33 | -0.19 +/- 0.40 |
| V_max | +4.20 | +3.70 | +8.84 | +5.80 |
| T_max | +68.60 | +14.50 | +34.43 | +19.40 |
| M_max | +23.00 | +67.30 | +64.36 | +43.60 |
| W_max | +1.40 | +1.80 | +0.19 | -0.10 |
| V_min | +1.10 | +1.30 | +0.19 | -0.10 |
| T_min | +0.90 | +1.50 | -0.19 | +0.50 |
| M_min | +1.40 | +0.10 | -0.21 | -0.40 |
| W_min | +0.80 | +0.20 | -0.08 | -0.40 |
| global_random | +1.49 +/- 0.50 | +1.10 +/- 0.91 | +0.73 +/- 0.47 | +0.83 +/- 0.67 |

Machine-readable results: [CSV](advisor_2026_09_response_state_ablation.csv) / [JSON](advisor_2026_09_response_state_ablation.json)
