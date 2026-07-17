# Neuron Typing Experiment - Phase 1

For the corrected q/r scoring definition, current 2k Phase 1 results, corrected Phase 2 ablations, and the prioritized research roadmap, see [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).

## Quick Start

### Full Pipeline (Recommended)

Run the complete pipeline with calibration + typing:

```bash
python scripts/vulcan/neuron_typing/run_phase1.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1 \
    --calibration_samples 500 \
    --calibration_offset 0 \
    --typing_samples 2000 \
    --typing_offset 500 \
    --threshold_mode quantile \
    --quantile_idx_visual 1 \
    --quantile_idx_text 0
```

### Step-by-Step

#### Step 1: Pilot (Global Max)
```bash
python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1/activations \
    --max_samples 500 \
    --sample_offset 0 \
    --dataset_role calibration \
    --pilot
```

#### Step 2: Calibration (Quantile Thresholds)
```bash
python scripts/vulcan/neuron_typing/calibrate_thresholds.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1/calibration \
    --max_samples 500 \
    --sample_offset 0 \
    --quantiles 0.95,0.97,0.99
```

#### Step 3: Typing (Neuron Classification)
```bash
python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1/activations \
    --max_samples 2000 \
    --sample_offset 500 \
    --global_max_path saves/neuron_typing/phase1/activations/global_max.pt \
    --threshold_mode quantile \
    --quantile_path saves/neuron_typing/phase1/calibration/neuron_quantiles.pt \
    --quantile_idx_visual 1 \
    --quantile_idx_text 0 \
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

### Held-out Phase 2

Use rows 2500 onward, which are disjoint from calibration `[0, 500)` and
typing `[500, 2500)` under the formal config:

```bash
python scripts/vulcan/neuron_typing/run_phase2_ablation.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --score_file saves/neuron_typing/phase1/scores/neuron_type_scores.parquet \
    --output_file saves/neuron_typing/phase2/heldout_ratio_sweep.json \
    --sample_offset 2500 \
    --max_samples 500 \
    --calibration_manifest saves/neuron_typing/phase1/calibration/sample_manifest.json \
    --typing_manifest saves/neuron_typing/phase1/activations/sample_manifest.json \
    --require_data_isolation \
    --ablation multimodal:0.05 \
    --ablation rank_band:multimodal:0.05:0.20 \
    --ablation random:0.15:seed1 \
    --ablation random:0.15:seed2 \
    --ablation random:0.15:seed3
```

The no-ablation baseline is inserted automatically. Outputs include per-example
NLL, paired bootstrap intervals, improved/damaged fractions, cutoff tie
metadata, per-type nesting checks, and relative damage when matched-ratio
random seeds are present.

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
│   ├── neuron_scores.json          # Per-neuron q/r scores and dead mask
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
- T_text[j] = q95 of neuron j's activations on text tokens

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
| `--quantile_idx_visual` | 1 | Visual quantile index (0=q95, 1=q97, 2=q99) |
| `--quantile_idx_text` | 0 | Text quantile index (0=q95, 1=q97, 2=q99) |
| `--quantile_idx` | None | Optional legacy fallback that sets both indices |
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

To verify stability across quantiles, run typing with different modality-specific quantile indices and compare:

```bash
for qi in 0 1 2; do
    python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
        --config configs/formal_coco.yaml \
        --output_dir saves/neuron_typing/phase1/activations_q${qi} \
        --max_samples 5000 \
        --threshold_mode quantile \
        --quantile_path saves/neuron_typing/phase1/calibration/neuron_quantiles.pt \
        --quantile_idx_visual ${qi} \
        --quantile_idx_text ${qi}
done
```

Then compare `neuron_scores.json` across q95/q97/q99 runs.
