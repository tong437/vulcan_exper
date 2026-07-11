# Neuron Typing Experiment - Phase 1

## Quick Start

### 1. Pilot Experiment (500-1000 samples)

Run pilot to calibrate thresholds:

```bash
python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
    --config scripts/vulcan/neuron_typing/configs/pilot_coco.yaml \
    --output_dir saves/neuron_typing/pilot \
    --max_samples 500 \
    --pilot
```

Check `saves/neuron_typing/pilot/activation_stats.json` to understand activation distributions.

### 2. Full Phase 1 Pipeline

Run the complete pipeline:

```bash
python scripts/vulcan/neuron_typing/run_phase1.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1 \
    --max_samples 5000 \
    --t_visual 2.0 \
    --t_text 3.0 \
    --n_visual 4 \
    --n_text 2
```

### 3. Individual Steps

#### Step 1: Collect Activations
```bash
python scripts/vulcan/neuron_typing/collect_ffn_activations.py \
    --config configs/formal_coco.yaml \
    --output_dir saves/neuron_typing/phase1/activations \
    --max_samples 5000
```

#### Step 2: Score Neuron Types
```bash
python scripts/vulcan/neuron_typing/score_neuron_types.py \
    --input_dir saves/neuron_typing/phase1/activations \
    --output_dir saves/neuron_typing/phase1/scores \
    --high_conf_threshold 0.7
```

#### Step 3: Statistical Tests
```bash
python scripts/vulcan/neuron_typing/statistical_tests.py \
    --input_dir saves/neuron_typing/phase1/scores \
    --output_dir saves/neuron_typing/phase1/stats
```

## Output Structure

```
saves/neuron_typing/phase1/
├── activations/
│   ├── global_max.pt           # Per-neuron global max activation
│   ├── neuron_scores.json      # Per-neuron type probabilities
│   └── config.json             # Collection config
├── scores/
│   ├── neuron_type_scores.parquet  # Full neuron type scores DataFrame
│   ├── layer_statistics.json       # Per-layer statistics
│   └── fa_vs_gdn_statistics.json   # FA vs GDN comparison
├── stats/
│   └── perm_test_results.json  # Permutation test and bootstrap CI
└── plots/
    ├── fig_layer_distribution.png
    ├── fig_layer_ratio.png
    ├── fig_scatter.png
    ├── fig_fa_vs_gdn.png
    └── fig_threshold_sensitivity.png
```

## Threshold Calibration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `t_visual` | 2.0 | Normalized activation threshold for visual tokens |
| `t_text` | 3.0 | Normalized activation threshold for text tokens |
| `n_visual` | 4 | Min visual token activations for classification |
| `n_text` | 2 | Min text token activations for classification |

## Neuron Types

- **visual**: High activation on visual tokens only
- **text**: High activation on text tokens only
- **multimodal**: High activation on both visual and text tokens
- **unknown**: Low activation on both modalities
