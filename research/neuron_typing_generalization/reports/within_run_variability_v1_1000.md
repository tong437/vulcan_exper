# Within-run neuron variability over 1,000 frozen samples

All 12 runs have an observed minimum of 0, a minimum nonzero rate of 0.002, and a maximum close to the finite-sample
four-state ceiling of 0.750751. Consequently, central quantiles and prevalence
statistics are more informative for model and dataset comparison than raw extrema.

## Main findings

- On VQAv2, GQA, and VizWiz, median rates span 0.494–0.499 for LLaVA,
  0.475–0.480 for Qwen, and 0.383–0.400 for Gemma.
- On those same datasets, the proportion at or above 0.5 is 47.8%–49.5% for LLaVA,
  43.2%–44.9% for Qwen, and 30.5%–32.9% for Gemma.
- For CLEVR, all three distributions lie closer to stability. Rates at or above 0.5 are
  24.6% for Qwen,
  19.6% for LLaVA, and
  14.8% for Gemma; exact-zero rates rise to
  4.3%,
  3.0%, and
  6.9%, respectively.
- These are broad within-model distributions, not a claim that every neuron follows the model-level mean.

## Distribution table

| Dataset | Model | Min | Min nonzero | P10 | Median | P90 | Max | Zero | Rate ≥ 0.5 | Max neuron |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| VQAv2 | Qwen3.5-2B | 0.0000 | 0.0020 | 0.1310 | 0.4746 | 0.6800 | 0.7507 | 0.84% | 43.21% | L12:N1545 |
| VQAv2 | LLaVA-1.5-7B | 0.0000 | 0.0020 | 0.2323 | 0.4990 | 0.6888 | 0.7507 | 0.71% | 49.52% | L2:N9818 |
| VQAv2 | Gemma-4-E2B-IT | 0.0000 | 0.0020 | 0.0622 | 0.3996 | 0.6502 | 0.7507 | 1.37% | 32.95% | L29:N8434 |
| GQA Balanced | Qwen3.5-2B | 0.0000 | 0.0020 | 0.1103 | 0.4801 | 0.6880 | 0.7507 | 1.15% | 44.88% | L4:N2296 |
| GQA Balanced | LLaVA-1.5-7B | 0.0000 | 0.0020 | 0.2150 | 0.4942 | 0.6887 | 0.7507 | 0.83% | 47.75% | L24:N3350 |
| GQA Balanced | Gemma-4-E2B-IT | 0.0000 | 0.0020 | 0.0488 | 0.3830 | 0.6428 | 0.7507 | 1.87% | 30.51% | L34:N8146 |
| CLEVR | Qwen3.5-2B | 0.0000 | 0.0020 | 0.0159 | 0.3567 | 0.6188 | 0.7506 | 4.33% | 24.61% | L0:N3331 |
| CLEVR | LLaVA-1.5-7B | 0.0000 | 0.0020 | 0.0296 | 0.3448 | 0.5863 | 0.7507 | 2.96% | 19.59% | L2:N2364 |
| CLEVR | Gemma-4-E2B-IT | 0.0000 | 0.0020 | 0.0040 | 0.2524 | 0.5383 | 0.7507 | 6.89% | 14.84% | L0:N5567 |
| VizWiz VQA | Qwen3.5-2B | 0.0000 | 0.0020 | 0.1251 | 0.4763 | 0.6797 | 0.7507 | 0.69% | 44.43% | L2:N2202 |
| VizWiz VQA | LLaVA-1.5-7B | 0.0000 | 0.0020 | 0.2172 | 0.4960 | 0.6840 | 0.7507 | 0.51% | 48.57% | L7:N4229 |
| VizWiz VQA | Gemma-4-E2B-IT | 0.0000 | 0.0020 | 0.0625 | 0.3870 | 0.6416 | 0.7507 | 0.93% | 32.36% | L34:N875 |

`Lx:Ny` denotes model-local layer `x`, neuron `y`; indices are not aligned between models.

## Figures

- [Within-run change-rate distributions](figures/within_run_change_rate_distributions.png)
- [Stable and high-change neuron proportions](figures/within_run_stable_and_high_change_heatmaps.png)

## Interpretation limits

- Minimum and maximum are extreme order statistics over architecture-dependent neuron counts.
- Neuron indices are model-local and are not aligned across architectures.
- The summaries are descriptive for the frozen 1,000 samples and are not uncertainty intervals.
- A zero rate means no state transition in this sample set, not universal invariance.

## Reproducibility

- Analysis commit: `170961cf`
- Exact percentiles, counts, thresholds, run IDs, source hashes, and figure hashes are stored in the CSV and JSON.
