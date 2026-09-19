# CLEVR has the lowest descriptive response-state variability across all three models

## Executive summary

- All 12 frozen runs contain 1,000 samples. LLaVA has the highest all-distinct-sample change rate on VQAv2, GQA,
  and VizWiz; Qwen is higher on CLEVR by 0.57 percentage points. Gemma has the lowest rate on all
  four datasets. These are descriptive differences, not significance tests.
- CLEVR produces the lowest rate for every model: Qwen 33.0%, LLaVA
  32.5%, and Gemma 26.8%.
- Across datasets, Qwen spans 33.0%–43.7%, LLaVA
  32.5%–47.8%, and Gemma
  26.8%–37.3%.
- No invalid observations were recorded in any of the 12 completed runs.

## Cross-model × dataset table

| Dataset | Model | Overall distinct rate | Median neuron rate | Mean normalized entropy | W | V | T | M | Neurons |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VQAv2 | Qwen3.5-2B | 0.4361 | 0.4746 | 0.5556 | 0.463 | 0.173 | 0.201 | 0.163 | 147,456 |
| VQAv2 | LLaVA-1.5-7B | 0.4779 | 0.4990 | 0.6121 | 0.475 | 0.185 | 0.190 | 0.150 | 352,256 |
| VQAv2 | Gemma-4-E2B-IT | 0.3730 | 0.3996 | 0.4806 | 0.545 | 0.144 | 0.165 | 0.146 | 337,920 |
| GQA Balanced | Qwen3.5-2B | 0.4368 | 0.4801 | 0.5601 | 0.469 | 0.171 | 0.194 | 0.167 | 147,456 |
| GQA Balanced | LLaVA-1.5-7B | 0.4703 | 0.4942 | 0.6020 | 0.480 | 0.182 | 0.186 | 0.152 | 352,256 |
| GQA Balanced | Gemma-4-E2B-IT | 0.3593 | 0.3830 | 0.4602 | 0.546 | 0.144 | 0.163 | 0.148 | 337,920 |
| CLEVR | Qwen3.5-2B | 0.3303 | 0.3567 | 0.4090 | 0.474 | 0.164 | 0.175 | 0.187 | 147,456 |
| CLEVR | LLaVA-1.5-7B | 0.3246 | 0.3448 | 0.3938 | 0.494 | 0.175 | 0.170 | 0.161 | 352,256 |
| CLEVR | Gemma-4-E2B-IT | 0.2681 | 0.2524 | 0.3274 | 0.542 | 0.145 | 0.168 | 0.145 | 337,920 |
| VizWiz VQA | Qwen3.5-2B | 0.4350 | 0.4763 | 0.5615 | 0.469 | 0.170 | 0.198 | 0.163 | 147,456 |
| VizWiz VQA | LLaVA-1.5-7B | 0.4701 | 0.4960 | 0.6046 | 0.472 | 0.188 | 0.192 | 0.148 | 352,256 |
| VizWiz VQA | Gemma-4-E2B-IT | 0.3654 | 0.3870 | 0.4758 | 0.551 | 0.142 | 0.163 | 0.144 | 337,920 |

The overall rate is a count-weighted ratio across real neurons. The median neuron rate is an unweighted distribution
summary and therefore answers a different question; disagreement between the two is not an error.

## Figures

- [Overall change-rate heatmap](figures/unpaired_change_rate_heatmap.png)
- [Normalized-depth change-rate curves](figures/unpaired_normalized_depth_curves.png)
- [W/V/T/M occurrence composition](figures/unpaired_class_composition.png)
- [Per-neuron change-rate distributions](figures/unpaired_neuron_change_distributions.png)

## Interpretation limits

- Every rate describes the frozen natural 1,000-sample distribution; it is not a causal effect.
- Neuron indices are model-local and are not aligned across architectures.
- Normalized decoder depth is used only to compare curve shape; layers are not structurally equivalent.
- The 12 overall values are deterministic summaries of the frozen samples, not independent replications.
- M denotes joint threshold exceedance and does not by itself establish causal multimodal interaction.

## Reproducibility

- Collection commit: `1a181adc2d5cf2e079bd0064e07a50518a9bf52d`
- Analysis commit: `cb5dd5f3`
- Frozen tier: `unpaired_v1_1000`; seed 42; 1,000 samples per dataset; no controlled pair edges.
- Exact run IDs and SHA-256 provenance are stored in `unpaired_neuron_v1_1000_results.json` and the CSV table.
