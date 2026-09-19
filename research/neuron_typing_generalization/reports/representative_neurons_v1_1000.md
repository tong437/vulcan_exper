# Representative neurons and 1,000-sample traces

The deterministic selection contains 36 criterion slots across 12 runs and resolves to
24 unique model-local neurons because high-change and high-entropy selections may overlap.
Every exported trace contains exactly 1000 frozen samples.

## Selection rules

- High change: maximum all-distinct-sample response-state change rate; normalized entropy breaks ties.
- High entropy: maximum normalized W/V/T/M entropy; change rate breaks ties.
- Stable: minimum change rate, then minimum entropy, then maximum mean distance from the visual/text thresholds.
- Only fully observed neurons are eligible. Criteria are selected independently and overlap is preserved.

## Selected neurons

| Dataset | Model | Criterion | Layer | Neuron | Change rate | Normalized entropy | Dominant state |
|---|---|---|---:|---:|---:|---:|---|
| VQAv2 | Qwen3.5-2B | High change rate | 12 | 1545 | 0.7507 | 1.0000 | T (0.254) |
| VQAv2 | Qwen3.5-2B | High entropy | 12 | 1545 | 0.7507 | 1.0000 | T (0.254) |
| VQAv2 | Qwen3.5-2B | Stable | 22 | 2210 | 0.0000 | -0.0000 | M (1.000) |
| VQAv2 | LLaVA-1.5-7B | High change rate | 2 | 9818 | 0.7507 | 1.0000 | V (0.252) |
| VQAv2 | LLaVA-1.5-7B | High entropy | 2 | 9818 | 0.7507 | 1.0000 | V (0.252) |
| VQAv2 | LLaVA-1.5-7B | Stable | 31 | 3228 | 0.0000 | -0.0000 | M (1.000) |
| VQAv2 | Gemma-4-E2B-IT | High change rate | 29 | 8434 | 0.7507 | 1.0000 | M (0.252) |
| VQAv2 | Gemma-4-E2B-IT | High entropy | 29 | 8434 | 0.7507 | 1.0000 | M (0.252) |
| VQAv2 | Gemma-4-E2B-IT | Stable | 30 | 2371 | 0.0000 | -0.0000 | M (1.000) |
| GQA Balanced | Qwen3.5-2B | High change rate | 4 | 2296 | 0.7507 | 1.0000 | M (0.254) |
| GQA Balanced | Qwen3.5-2B | High entropy | 4 | 2296 | 0.7507 | 1.0000 | M (0.254) |
| GQA Balanced | Qwen3.5-2B | Stable | 22 | 2210 | 0.0000 | -0.0000 | M (1.000) |
| GQA Balanced | LLaVA-1.5-7B | High change rate | 24 | 3350 | 0.7507 | 0.9999 | T (0.254) |
| GQA Balanced | LLaVA-1.5-7B | High entropy | 24 | 3350 | 0.7507 | 0.9999 | T (0.254) |
| GQA Balanced | LLaVA-1.5-7B | Stable | 31 | 3228 | 0.0000 | -0.0000 | M (1.000) |
| GQA Balanced | Gemma-4-E2B-IT | High change rate | 34 | 8146 | 0.7507 | 1.0000 | M (0.254) |
| GQA Balanced | Gemma-4-E2B-IT | High entropy | 34 | 8146 | 0.7507 | 1.0000 | M (0.254) |
| GQA Balanced | Gemma-4-E2B-IT | Stable | 30 | 2371 | 0.0000 | -0.0000 | M (1.000) |
| CLEVR | Qwen3.5-2B | High change rate | 7 | 2394 | 0.7506 | 0.9998 | W (0.257) |
| CLEVR | Qwen3.5-2B | High entropy | 7 | 2394 | 0.7506 | 0.9998 | W (0.257) |
| CLEVR | Qwen3.5-2B | Stable | 22 | 2210 | 0.0000 | -0.0000 | M (1.000) |
| CLEVR | LLaVA-1.5-7B | High change rate | 2 | 2364 | 0.7507 | 0.9999 | V (0.256) |
| CLEVR | LLaVA-1.5-7B | High entropy | 2 | 2364 | 0.7507 | 0.9999 | V (0.256) |
| CLEVR | LLaVA-1.5-7B | Stable | 31 | 1625 | 0.0000 | -0.0000 | M (1.000) |
| CLEVR | Gemma-4-E2B-IT | High change rate | 0 | 5567 | 0.7507 | 1.0000 | W (0.252) |
| CLEVR | Gemma-4-E2B-IT | High entropy | 0 | 5567 | 0.7507 | 1.0000 | W (0.252) |
| CLEVR | Gemma-4-E2B-IT | Stable | 32 | 8474 | 0.0000 | -0.0000 | M (1.000) |
| VizWiz VQA | Qwen3.5-2B | High change rate | 2 | 2202 | 0.7507 | 1.0000 | M (0.252) |
| VizWiz VQA | Qwen3.5-2B | High entropy | 2 | 2202 | 0.7507 | 1.0000 | M (0.252) |
| VizWiz VQA | Qwen3.5-2B | Stable | 18 | 3398 | 0.0000 | -0.0000 | M (1.000) |
| VizWiz VQA | LLaVA-1.5-7B | High change rate | 7 | 4229 | 0.7507 | 1.0000 | T (0.251) |
| VizWiz VQA | LLaVA-1.5-7B | High entropy | 7 | 4229 | 0.7507 | 1.0000 | T (0.251) |
| VizWiz VQA | LLaVA-1.5-7B | Stable | 31 | 1625 | 0.0000 | -0.0000 | M (1.000) |
| VizWiz VQA | Gemma-4-E2B-IT | High change rate | 34 | 875 | 0.7507 | 1.0000 | W (0.252) |
| VizWiz VQA | Gemma-4-E2B-IT | High entropy | 34 | 875 | 0.7507 | 1.0000 | W (0.252) |
| VizWiz VQA | Gemma-4-E2B-IT | Stable | 30 | 2371 | 0.0000 | -0.0000 | M (1.000) |

## Reproducibility

- Analysis commit: `2f3a9a10`
- Trace artifact root: `/root/autodl-pvt-RTX4090-hdd-1/vulcan-exper/artifacts/neuron_typing_generalization/representative_neuron_traces/unpaired_neuron_v1_1000`
- `representative_neurons_v1_1000.json` records source-profile and trace-file SHA-256 values.
