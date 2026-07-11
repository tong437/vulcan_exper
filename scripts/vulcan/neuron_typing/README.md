# Neuron Typing Experiment - Phase 1

## Quick Start

### Full Pipeline (Recommended)

Run the complete pipeline with calibration + typing:

```bash
python scripts/vulcan/neuron_typing/run_phase1.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1 \
    --calibration_samples 500 \
    --typing_samples 5000 \
    --threshold_mode quantile \
    --quantile_idx 1
```

### Step-by-Step

#### Step 1: Pilot (Global Max)
```bash
python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
    --config scripts/vulcan/neuron_typing/configs/pilot_coco.yaml \
    --output_dir saves/neuron_typing/phase1/activations \
    --max_samples 500 \
    --pilot
```

#### Step 2: Calibration (Quantile Thresholds)
```bash
python scripts/vulcan/neuron_typing/calibrate_thresholds.py \
    --config scripts/vulcan/neuron_typing/configs/pilot_coco.yaml \
    --output_dir saves/neuron_typing/phase1/calibration \
    --max_samples 500 \
    --quantiles 0.95,0.97,0.99
```

#### Step 3: Typing (Neuron Classification)
```bash
python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1/activations \
    --max_samples 5000 \
    --threshold_mode quantile \
    --quantile_path saves/neuron_typing/phase1/calibration/neuron_quantiles.pt \
    --quantile_idx 1 \
    --visual_ratio 0.005 \
    --visual_min_count 4 \
    --text_ratio 0.10 \
    --text_min_count 2
```

#### Step 4: Scoring
```bash
python scripts/vulcan/neuron_typing/score_neuron_types.py \
    --input_dir saves/neuron_typing/phase1/activations \
    --output_dir saves/neuron_typing/phase1/scores
```

#### Step 5: Statistical Tests
```bash
python scripts/vulcan/neuron_typing/statistical_tests.py \
    --input_dir saves/neuron_typing/phase1/scores \
    --output_dir saves/neuron_typing/phase1/stats
```

## Output Structure

```
saves/neuron_typing/phase1/
├── calibration/
│   ├── global_max.pt               # Per-neuron global max (from pilot)
│   ├── neuron_quantiles.pt         # Per-neuron quantile thresholds
│   ├── calibration_summary.json    # Summary statistics
│   └── config.json
├── activations/
│   ├── global_max.pt               # Per-neuron global max
│   ├── neuron_scores.json          # Per-neuron type probabilities
│   └── config.json
├── scores/
│   ├── neuron_type_scores.parquet  # Full neuron type scores DataFrame
│   ├── layer_statistics.json       # Per-layer statistics
│   └── fa_vs_gdn_statistics.json   # FA vs GDN comparison
├── stats/
│   └── perm_test_results.json      # Permutation test and bootstrap CI
└── plots/
    ├── fig_layer_distribution.png
    ├── fig_layer_ratio.png
    ├── fig_scatter.png
    ├── fig_fa_vs_gdn.png
    ├── fig_threshold_sensitivity.png
    └── fig_quantile_sensitivity.png
```

## Threshold Modes

### Quantile Mode (Recommended)

Per-neuron, per-modality quantile thresholds:
- T_visual[j] = q97 of neuron j's activations on visual tokens
- T_text[j] = q97 of neuron j's activations on text tokens

Count thresholds are ratio-based:
- visual_required = max(4, ceil(0.005 * num_visual_tokens))
- text_required = max(2, ceil(0.10 * num_text_tokens))

### Fixed Mode (Paper Baseline)

Fixed thresholds across all neurons:
- T_visual = 2.0 (normalized to [0, 10])
- T_text = 3.0
- n_visual = 4
- n_text = 2

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--threshold_mode` | quantile | Threshold mode (fixed/quantile) |
| `--quantile_idx` | 1 | Which quantile (0=q95, 1=q97, 2=q99) |
| `--visual_ratio` | 0.005 | Min ratio of visual tokens above threshold |
| `--visual_min_count` | 4 | Min absolute visual token count |
| `--text_ratio` | 0.10 | Min ratio of text tokens above threshold |
| `--text_min_count` | 2 | Min absolute text token count |
| `--sample_score_top_m` | 5 | Top-m tokens for sample score |

## Neuron Types

- **visual**: High activation on visual tokens only
- **text**: High activation on text tokens only
- **multimodal**: High activation on both visual and text tokens
- **unknown**: Low activation on both modalities

## Sensitivity Analysis

To verify stability across quantiles, run typing with different `--quantile_idx` values and compare:

```bash
for qi in 0 1 2; do
    python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
        --config configs/formal_coco.yaml \
        --output_dir saves/neuron_typing/phase1/activations_q${qi} \
        --max_samples 5000 \
        --threshold_mode quantile \
        --quantile_path saves/neuron_typing/phase1/calibration/neuron_quantiles.pt \
        --quantile_idx ${qi}
done
```

Then compare `neuron_scores.json` across q95/q97/q99 runs.
