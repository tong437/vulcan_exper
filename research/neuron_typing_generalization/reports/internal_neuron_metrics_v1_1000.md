# Within-run entropy, effective states, and activation RMS

Each row summarizes all real decoder MLP neurons in one frozen 1,000-sample run. Pooled activation RMS is calculated
within each sample using visual/text token counts, then its per-neuron mean and standard deviation are taken over the
1,000 samples. The table reports medians across neurons; complete P10/P50/P90 values are in the CSV and JSON.

Normalized entropy and effective class count are not independent metrics:

`effective_class_count = 4 ** normalized_entropy`.

## Main findings

- Across the 12 within-run distributions, the median effective state count ranges from 1.55 to 2.35; the typical neuron therefore occupies substantially fewer than four states effectively.
- CLEVR median effective counts are 1.55–1.79, compared with 1.91–2.35 in the other frozen datasets.
- Median pooled-RMS coefficients of variation are 0.101–0.130 for CLEVR and 0.160–0.284 otherwise.
- Spearman associations between entropy and RMS strength range from -0.104 to 0.200; associations with RMS fluctuation range from 0.015 to 0.242. State diversity is therefore not reducible to activation magnitude alone.

## Internal-distribution summary

| Dataset | Model | Entropy median | Effective states median | RMS strength median | RMS fluctuation median | RMS CV median | rho(H,strength) | rho(H,fluctuation) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| VQAv2 | Qwen3.5-2B | 0.552 | 2.15 | 7.084e-02 | 1.208e-02 | 0.160 | -0.075 | 0.039 |
| VQAv2 | LLaVA-1.5-7B | 0.616 | 2.35 | 5.592e-02 | 1.105e-02 | 0.218 | 0.158 | 0.234 |
| VQAv2 | Gemma-4-E2B-IT | 0.486 | 1.96 | 5.080e-02 | 1.256e-02 | 0.255 | 0.155 | 0.232 |
| GQA Balanced | Qwen3.5-2B | 0.566 | 2.19 | 6.854e-02 | 1.264e-02 | 0.174 | -0.104 | 0.018 |
| GQA Balanced | LLaVA-1.5-7B | 0.602 | 2.31 | 5.669e-02 | 1.050e-02 | 0.205 | 0.151 | 0.229 |
| GQA Balanced | Gemma-4-E2B-IT | 0.465 | 1.91 | 5.010e-02 | 1.124e-02 | 0.233 | 0.153 | 0.242 |
| CLEVR | Qwen3.5-2B | 0.420 | 1.79 | 6.755e-02 | 7.410e-03 | 0.106 | -0.051 | 0.023 |
| CLEVR | LLaVA-1.5-7B | 0.402 | 1.75 | 4.875e-02 | 4.568e-03 | 0.101 | 0.054 | 0.096 |
| CLEVR | Gemma-4-E2B-IT | 0.317 | 1.55 | 5.132e-02 | 6.434e-03 | 0.130 | 0.126 | 0.223 |
| VizWiz VQA | Qwen3.5-2B | 0.571 | 2.21 | 7.162e-02 | 1.360e-02 | 0.179 | -0.104 | 0.015 |
| VizWiz VQA | LLaVA-1.5-7B | 0.610 | 2.33 | 5.574e-02 | 1.176e-02 | 0.228 | 0.173 | 0.227 |
| VizWiz VQA | Gemma-4-E2B-IT | 0.485 | 1.96 | 5.273e-02 | 1.454e-02 | 0.284 | 0.200 | 0.241 |

## Figures

- [Normalized entropy and effective class count distributions](figures/internal_normalized_entropy_effective_states.png)
- [Activation RMS strength versus fluctuation](figures/internal_activation_rms_strength_fluctuation.png)

## Interpretation limits

- Raw RMS scale is architecture- and layer-dependent; use it primarily within a run or layer.
- Entropy and effective class count are two representations of the same state-occupancy information.
- Association between entropy and RMS does not identify a causal mechanism.
- All summaries describe the frozen 1,000 samples and are not population guarantees.

## Reproducibility

- Analysis commit: `6eb3568c`
- Full per-neuron RMS/state profiles and their SHA-256 manifests are stored under `/root/autodl-pvt-RTX4090-hdd-1/vulcan-exper/artifacts/neuron_typing_generalization/internal_neuron_metrics/unpaired_neuron_v1_1000`.
- Exact run and layer summaries are stored in the CSV and JSON outputs.
