# Neuron Typing Experiment - Phase 1

## NeuPAT language-preserving extension

The repository now includes a separately gated NeuPAT extension:

- `collect_neupat_importance.py` collects text/vision RMS importance and emits
  exhaustive language, multimodal, shared, and reserve roles;
- `prepare_neupat_text_probe.py` builds the paper-matched, macro-balanced
  four-source text probe through the Hugging Face Dataset Server, and
  `validate_neupat_text_probe.py` audits hashes, token lengths, and evaluation
  isolation;
- `analyze_neupat_stability.py` tests tau sensitivity, while
  `compare_neupat_replications.py` measures role agreement between
  sample-disjoint probing runs;
- `analyze_neupat_overlap.py` compares those roles with q/r, the frozen q-band,
  and Phase-4 mapping protection;
- `run_neupat_causality.py` treats the exact `language U shared` protection
  mask as primary, runs five per-layer matched-random controls on 500 Caption
  and 500 isolated C4 examples, and defers component roles/POPE to secondary
  analyses;
- `use_neupat: true` enables language-slice gradient masking and shared-slice
  L2/cosine preservation during full SFT;
- `run_neupat_sft_matrix.py` gates and runs matched vanilla full-SFT, LoRA, and
  NeuPAT jobs; the two `evaluate_neupat_sft_*` scripts then compare language
  retention and held-out VQA-RAD performance;
- `build_neupat_joint_candidate.py` produces a pre-causal-gate protected
  q-band candidate and never authorizes structural pruning.

See
[`neupat_integration_plan.md`](../../../docs/my-exper/typing%20neuron/neupat_integration_plan.md)
for commands, gates, and interpretation limits.

The formal protection-set causal gate passed on 2026-09-02. Excess NLL versus
the five layerwise exact-count controls was +5.440 on Caption (paired 95% CI
[5.365, 5.520]) and +5.480 on packed C4 (paired 95% CI [5.421, 5.537]). This
authorized the SFT matrix. The completed post-SFT test did not show a NeuPAT
advantage: NeuPAT minus vanilla C4 NLL was +0.001929 (paired 95% CI
[0.001120, 0.002706]), although both improved slightly over the base model.
NeuPAT VQA-RAD accuracy was 0.6853 versus vanilla's 0.6733, but the 95% CI for
the difference [-0.0558, 0.0797] failed the -0.02 non-inferiority margin. LoRA
was best on both measured endpoints (C4 NLL 3.375873; VQA accuracy 0.7052).
These results support causal language sensitivity of the protection set, not
a language-preservation benefit under the present small-data/low-LR SFT
regime.

For the corrected q/r scoring definition, current 2k Phase 1 results, corrected Phase 2 ablations, and the prioritized research roadmap, see [EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).

Phase 4 adds an image-disjoint, correct-image versus shuffled-image activation
mapping experiment. It predicts image-induced text-position FFN activation
changes from the decoder-entry visual representation, compares Q-only/V-only/V+Q
ridge and reduced-rank models, and forbids mapping-driven pruning unless the
mapping, incremental-value, and held-out group-causality gates pass. The
pre-registered design and formal command are in
[`docs/my-exper/typing neuron/phase4_activation_mapping_plan.md`](../../../docs/my-exper/typing%20neuron/phase4_activation_mapping_plan.md).

P4.4a follows the successful Phase-4 mapping and causal-localization gates with
a five-point mapping-high dose curve, 20 global matched-random controls, and
20 controls that exactly match the mapping mask's layerwise q-decile
histogram. It never authorizes pruning. See
[`phase44a_causal_dose_plan.md`](../../../docs/my-exper/typing%20neuron/phase44a_causal_dose_plan.md).

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
    --ablation multimodal:0.20 \
    --ablation rank_band:multimodal:0.05:0.20 \
    --ablation matched_random:multimodal:0.05:seed1 \
    --ablation matched_random:multimodal:0.05:seed2 \
    --ablation matched_random:multimodal:0.05:seed3 \
    --ablation matched_random:multimodal:0.20:seed1 \
    --ablation matched_random:multimodal:0.20:seed2 \
    --ablation matched_random:multimodal:0.20:seed3 \
    --ablation matched_random:rank_band:multimodal:0.05:0.20:seed1 \
    --ablation matched_random:rank_band:multimodal:0.05:0.20:seed2 \
    --ablation matched_random:rank_band:multimodal:0.05:0.20:seed3
```

The no-ablation baseline is inserted automatically. Outputs include per-example
NLL, paired bootstrap intervals, improved/damaged fractions, cutoff tie
metadata, per-type nesting checks, exact per-layer count verification for
`matched_random` controls, same-seed random partition checks (top 5% disjoint
from 5--20%, with their union equal to top 20%), and relative damage against
those matched controls.

### Exact per-layer 15% functional-mask baselines

The reference mask is the multimodal 5--20% rank band. With an FFN width of
3584, it contains exactly `ceil(0.20 * 3584) - ceil(0.05 * 3584) = 537`
neurons in every layer. Build the two extra baseline scores once:

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/build_pruning_baseline_scores.py \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores.parquet \
    --model_path /root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b \
    --output_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet
```

The four masks are: the frozen 5--20% q-band, a matched random mask, the
lowest group-L2 weight magnitudes, and the lowest threshold-response
frequencies. The latter is `1 - r_unknown`; it measures how often a neuron
crossed the Phase-1 activation threshold, not its mean activation magnitude.
Use these repeatable ablation specifications:

```bash
--ablation rank_band:multimodal:0.05:0.20 \
--ablation matched_random:rank_band:multimodal:0.05:0.20:seed1 \
--ablation matched_score:weight_magnitude:lowest:rank_band:multimodal:0.05:0.20 \
--ablation matched_score:activation_frequency:lowest:rank_band:multimodal:0.05:0.20
```

Both evaluators verify that every matched mask has exactly the same number of
selected neurons in every layer and store the result under
`matched_score_verification` or `matched_random_verification`.

### Phase 2.6 completeness sweep

`run_phase26_completeness.py` tests whether the frozen 5--20% q-band is only
the best evaluated candidate or remains preferable after a broader fixed-budget
search. It uses the 537-neuron-per-layer formal q-band budget for every mask.
The default candidate set contains:

- the frozen `q_multimodal` 5--20% reference;
- 14 `q_multimodal` tail windows starting at 20%, 25%, ..., 85%;
- the exact-budget top `q_visual`, `q_text`, and `q_unknown` masks.

The script explicitly selects the `q_unknown` column. It does not reuse the
legacy `unknown` shorthand, which intentionally ranks by `r_unknown`. The
screen evaluates every candidate on Caption NLL, text-only NLL/PPL, and POPE
random. At most five passing candidates advance to the formal Caption,
text-only, and three-split POPE evaluation. The formal stage also evaluates 20
shared, exact-count, per-layer matched-random controls.

The bundled text config uses the local 300-document C4 sample (about 120k
words), packs it into 512-token blocks, and computes pretraining-style loss on
all non-padding tokens. It is disjoint from Phase-1 image calibration and
typing data. For a paper-scale language benchmark, register a larger held-out
corpus and override the text dataset/config without changing the frozen gates.

```bash
python scripts/vulcan/neuron_typing/run_phase26_completeness.py \
    --caption_config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --text_config scripts/vulcan/neuron_typing/configs/phase26_text_only.example.yaml \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores.parquet \
    --output_dir saves/neuron_typing/phase26_completeness \
    --image_root /root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images \
    --calibration_manifest saves/neuron_typing/phase1_clean_2k/calibration/sample_manifest.json \
    --typing_manifest saves/neuron_typing/phase1_clean_2k/activations/sample_manifest.json \
    --pope random=saves/neuron_typing/pope_data/coco_pope_random.json \
    --pope popular=saves/neuron_typing/pope_data/coco_pope_popular.json \
    --pope adversarial=saves/neuron_typing/pope_data/coco_pope_adversarial.json \
    --batch_size 4 \
    --stage all
```

For long runs, execute the stages separately as `build`, `screen`, `final`,
and `summarize`. Completed output files are detected and skipped; POPE files
also use their condition-level resume support. Thresholds are frozen in
`phase26_plan.json`, screening decisions in `screen_summary.json`, and the
formal gate in `phase26_completeness.json`.

### Held-out POPE evaluation

`evaluate_pope.py` accepts the official POPE JSON-Lines files (including files
with a `.json` suffix). Use `--max_images` for pilots so all six questions for
each selected image stay together. When comparison manifests are supplied,
`--filter_manifest_overlaps` removes complete overlapping image groups before
selection and the saved manifest is checked again for zero overlap.

The evaluator provides:

- one-forward, batched forced-choice scoring for the single-token `yes` and
  `no` candidates;
- deterministic top-prefix, rank-band, and exact-count matched-random masks;
- strict per-layer count and same-seed random partition verification;
- paired Accuracy/F1 differences, exact McNemar tests, and image-cluster
  bootstrap confidence intervals;
- a deterministic shuffled-image baseline to verify that the task actually
  uses its image;
- per-condition checkpoints and `--resume` for long multi-seed runs.

Example pilot:

```bash
python scripts/vulcan/neuron_typing/evaluate_pope.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores.parquet \
    --pope_file data/pope/coco_pope_random.json \
    --image_root /path/to/coco/val2014 \
    --output_file saves/neuron_typing/phase2_pope/random_pilot.json \
    --max_images 50 \
    --batch_size 4 \
    --calibration_manifest saves/neuron_typing/phase1_clean_2k/calibration/sample_manifest.json \
    --typing_manifest saves/neuron_typing/phase1_clean_2k/activations/sample_manifest.json \
    --filter_manifest_overlaps \
    --require_data_isolation \
    --include_shuffled_image_control \
    --ablation multimodal:0.05 \
    --ablation rank_band:multimodal:0.05:0.20 \
    --ablation multimodal:0.20 \
    --ablation matched_random:multimodal:0.05:seed1 \
    --ablation matched_random:rank_band:multimodal:0.05:0.20:seed1 \
    --ablation matched_random:multimodal:0.20:seed1
```

If interrupted, repeat the identical command with `--resume`. Do not change
the dataset slice, manifests, masks, seeds, batch size, or bootstrap settings
when resuming.

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

## Typing sample-size stability

Compare a prefix run against the full typing run with the same calibration.
The command validates the calibration SHA-256, controlled typing parameters,
source-index prefix, row-level image prefix, neuron keys, and dead-neuron mask
before reporting global/per-layer Spearman correlations and mask Jaccards:

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/compare_phase1_stability.py \
    --small saves/neuron_typing/phase1_clean_500_prefix/scores/neuron_type_scores.parquet \
    --large saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores.parquet \
    --output_file saves/neuron_typing/phase1_stability_500_vs_2k.json \
    --bands 0.05:0.20 \
    --require_prefix_validation
```

## Phase 3 structural q-band pruning

The formal Phase-3 path is frozen to the full-2k per-layer
`q_multimodal` 5--20% band. It validates the exact Phase-2 score artifact,
cutoffs, manifests, calibration hash, mask counts, and singleton structural
clusters before deleting parameters.

Build the auditable mask and cluster artifacts:

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/build_structural_pruning_artifact.py \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet \
    --phase2_result saves/neuron_typing/phase2_clean_2k/heldout_functional_masks_15pct.json \
    --output_dir saves/neuron_typing/phase3_structural_qband
```

The end-to-end controller can create the checkpoint, run hook/in-memory/reload
equivalence, optionally repeat formal caption/POPE evaluation, and benchmark
the original and structural models:

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/run_phase3_structural.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --model_name_or_path /path/to/qwen3.5-0.8b \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet \
    --phase2_caption_result saves/neuron_typing/phase2_clean_2k/heldout_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/random_formal_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/popular_formal_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/adversarial_formal_functional_masks_15pct.json \
    --smoke_pope_file saves/neuron_typing/pope_data/coco_pope_random.json \
    --image_root /path/to/coco/images \
    --output_dir saves/neuron_typing/phase3_structural_qband \
    --reuse_pruned_model \
    --run_formal_evaluation \
    --run_benchmark
```

`pruning/run_type_aware_pruning.py` and `pruning/compute_pruning_score.py` are
legacy `p_*`/unknown-score prototypes and are not formal Phase-3 entry points.

## Phase 3.4 hardware-aligned q-band

When the exact 5--20% structural width (3,047) reduces parameters but does not
improve latency, P3.4 tests an exact 3,072 width. The aligned mask is
`rank_window:multimodal:180:512`: protect the first 180 ranked neurons and
remove the following 512 in every layer. Run the hook-only safety gate first:

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/run_phase34_aligned.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --model_name_or_path /path/to/qwen3.5-0.8b \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet \
    --phase2_caption_result saves/neuron_typing/phase2_clean_2k/heldout_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/random_formal_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/popular_formal_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/adversarial_formal_functional_masks_15pct.json \
    --image_root /path/to/coco/images \
    --output_dir saves/neuron_typing/phase34_aligned_3072
```

The controller stops before writing a structural checkpoint unless caption and
all three POPE hook checks pass. After they pass, reuse them and run the
structural/equivalence/formal/benchmark stages:

```bash
PYTHONPATH=src python scripts/vulcan/neuron_typing/run_phase34_aligned.py \
    --config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
    --model_name_or_path /path/to/qwen3.5-0.8b \
    --score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet \
    --phase2_caption_result saves/neuron_typing/phase2_clean_2k/heldout_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/random_formal_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/popular_formal_functional_masks_15pct.json \
    --phase2_pope_result saves/neuron_typing/phase2_pope/adversarial_formal_functional_masks_15pct.json \
    --smoke_pope_file saves/neuron_typing/pope_data/coco_pope_random.json \
    --image_root /path/to/coco/images \
    --output_dir saves/neuron_typing/phase34_aligned_3072 \
    --reuse_hook_evaluation \
    --run_structural \
    --unaligned_model_path saves/neuron_typing/phase3_structural_qband/model \
    --run_benchmark
```

The revised benchmark forces the requested generation length, validates the
actual output-token count, measures multimodal and text-only prefill, and
reports bootstrap confidence intervals directly for latency speedups/deltas.
