# NeuPAT Integration Plan for Qwen3.5-VL Neuron Typing

## Scope

This extension tests whether NeuPAT's response-based neuron roles transfer from
LLM-to-MLLM expansion to Qwen3.5-VL medical-domain adaptation. It does not
assume that a NeuPAT reserve neuron is safe to prune.

The implementation has four independently gated stages:

1. modality-associated importance probing and role allocation;
2. overlap analysis and held-out causal validation;
3. language-preserving full SFT;
4. protected q-band candidate construction.

No structural checkpoint may be created from the protected candidate until the
Caption, text-only, and all three POPE gates pass.

## P0: Importance Probing

NeuPAT importance for neuron `u` in layer `l` is:

```text
importance(D) = ||down_proj[:, u]||_2 * RMS(valid-token FFN activation[u])
```

The text and vision probe sets are processed separately. Within every layer,
the smallest deterministic set covering `tau_text` or `tau_vision` importance
mass is selected. Their set relationship defines language, multimodal, shared,
and reserve roles.

```bash
python scripts/vulcan/neuron_typing/prepare_neupat_text_probe.py

PYTHONPATH=src python scripts/vulcan/neuron_typing/validate_neupat_text_probe.py

python scripts/vulcan/neuron_typing/collect_neupat_importance.py \
  --vision_config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --text_config scripts/vulcan/neuron_typing/configs/neupat_text_probe.formal.yaml \
  --output_dir saves/neuron_typing/neupat_probe_formal_2048 \
  --vision_samples 2048 \
  --text_samples 2048 \
  --tau_vision 0.8 \
  --tau_text 0.8
```

Formal probing must use a text set independent of the text-only evaluation
corpus. For the paper-scale experiment, macro-balance instruction following,
mathematics/reasoning, factuality, and code sources instead of relying on the
single CodeAlpaca example config. The paper uses an even mixture of
CodeAlpaca-20k, MetaMathQA, databricks-dolly-15k, and HaluEval.

For an offline two-sample GPU smoke test, replace only the text probe source
without changing the vision configuration:

```bash
python scripts/vulcan/neuron_typing/collect_neupat_importance.py \
  --vision_config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --text_config scripts/vulcan/neuron_typing/configs/neupat_text_probe.example.yaml \
  --output_dir /tmp/neupat_gpu_smoke \
  --vision_samples 2 --text_samples 2 --allow_short_dataset \
  --text_override dataset=alpaca_en_demo \
  --text_override tokenized_path=null
```

Outputs:

- `neupat_scores.parquet`: continuous importance and one-hot role columns;
- `neupat_roles.json`: strict, exhaustive role artifact used during SFT;
- separate text/vision manifests;
- `neupat_summary.json`: global and layerwise role counts.

Threshold sensitivity and independent-sample replication are separate checks:

```bash
python scripts/vulcan/neuron_typing/analyze_neupat_stability.py \
  --score_file saves/neuron_typing/neupat_probe_formal_2048/neupat_scores.parquet \
  --output_file saves/neuron_typing/neupat_probe_formal_2048/neupat_tau_stability.json \
  --tau 0.7 --tau 0.8 --tau 0.9

python scripts/vulcan/neuron_typing/compare_neupat_replications.py \
  --reference_dir saves/neuron_typing/neupat_probe_replica_a_1024 \
  --comparison_dir saves/neuron_typing/neupat_probe_replica_b_1024 \
  --output_file saves/neuron_typing/neupat_probe_replication_1024.json
```

## P0.5: Overlap Analysis

Merge NeuPAT with the frozen q/r and Phase-4 mapping scores:

```bash
python scripts/vulcan/neuron_typing/analyze_neupat_overlap.py \
  --neupat_score_file saves/neuron_typing/neupat_probe_formal_2048/neupat_scores.parquet \
  --q_score_file saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores_with_baselines.parquet \
  --mapping_score_file saves/neuron_typing/phase4_mapping/mapping/neuron_scores_with_mapping.parquet \
  --output_dir saves/neuron_typing/neupat_overlap_formal_2048
```

The report includes Spearman correlations, Jaccard overlap, recall, odds-ratio
enrichment, layerwise counts, and the number of original q-band candidates
removed by NeuPAT/mapping protection.

## P1: Matched-Count Causal Validation

All four role masks use exact role membership. Each matched-random control has
the same selected count in every layer as its corresponding role.

```bash
python scripts/vulcan/neuron_typing/run_neupat_causality.py \
  --caption_config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --text_config scripts/vulcan/neuron_typing/configs/phase26_text_only.example.yaml \
  --score_file saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet \
  --output_dir saves/neuron_typing/neupat_causality \
  --image_root /root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images \
  --calibration_manifest saves/neuron_typing/phase1_clean_2k/calibration/sample_manifest.json \
  --typing_manifest saves/neuron_typing/phase1_clean_2k/activations/sample_manifest.json \
  --probe_manifest saves/neuron_typing/neupat_probe_formal_2048/vision_manifest.json \
  --probe_manifest saves/neuron_typing/neupat_probe_formal_2048/text_manifest.json \
  --pope random=saves/neuron_typing/pope_data/coco_pope_random.json \
  --pope popular=saves/neuron_typing/pope_data/coco_pope_popular.json \
  --pope adversarial=saves/neuron_typing/pope_data/coco_pope_adversarial.json \
  --control_seed_count 5 \
  --batch_size 4
```

Role transfer is supported only if:

- language/shared ablation causes more text damage than its matched controls;
- multimodal ablation causes multimodal damage without the same text-specific
  enrichment;
- conclusions are stable across at least the three primary tasks and are not
  driven by one layer or one random seed.

Reserve is treated as an update-capacity hypothesis, not a pruning hypothesis.

## P2: NeuPAT-Guided Domain SFT

The training implementation:

- masks `gate_proj` and `up_proj` language-role rows;
- masks matching `down_proj` columns;
- applies shared-role input-side squared Frobenius/L2 preservation;
- applies shared-role output-side cosine preservation;
- requires full SFT and zero weight decay.

```bash
WANDB_DISABLED=true python src/train.py \
  examples/vulcan/qwen35_08b_vqa_rad_neupat_sft.yaml
```

Required comparison matrix:

| Method | VQA-RAD | POPE | text NLL | language benchmarks |
|---|---:|---:|---:|---:|
| Original Qwen3.5-VL | required | required | required | required |
| Vanilla full SFT | required | required | required | required |
| Freeze language model | required | required | required | recommended |
| LoRA | required | required | required | recommended |
| NeuPAT | required | required | required | required |
| NeuPAT + activation alignment | optional | optional | required | recommended |

Use `tau in {0.7, 0.8, 0.9}` and `lambda in {0.05, 0.1}` only after the
default role-transfer test passes. The paper uses a sum-form regularizer; a
`mean` implementation is exposed only as an explicitly named engineering
ablation.

## P3: Joint Protection Candidate

The joint protection set is:

```text
NeuPAT language U NeuPAT shared U mapping top-1%
```

Build the conservative, variable-budget diagnostic first:

```bash
python scripts/vulcan/neuron_typing/build_neupat_joint_candidate.py \
  --score_file saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet \
  --output_dir saves/neuron_typing/neupat_joint_candidate_formal_2048
```

An exact 537-neuron-per-layer candidate requires experimental backfill below
the original q-band:

```bash
python scripts/vulcan/neuron_typing/build_neupat_joint_candidate.py \
  --score_file saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet \
  --output_dir saves/neuron_typing/neupat_joint_candidate_537 \
  --budget_per_layer 537 \
  --allow_backfill
```

The generated metadata always records `structural_pruning_allowed: false`.
Backfill changes the frozen Phase-3 mask and therefore requires a new formal
Caption, text-only, POPE, matched-random, hook/structural equivalence, and
latency evaluation. Existing Phase-3 evidence cannot be reused.

## Interpretation Rules

- NeuPAT importance is a response-weight proxy, not a direct plasticity or
  causal measurement.
- Text and vision probing datasets must be disjoint from all evaluation data.
- A role label is not a pruning label.
- Qwen3.5-VL domain SFT must be described as language-preserving multimodal
  domain adaptation, not LLM-to-MLLM expansion.
- The current Phase-2.6 zero-pass screening result must be resolved before any
  broad language-preservation claim is made.

## Current Formal Results and Gate Decision (2026-09-02)

- The formal text probe contains 2,048 unique prompts, exactly 512 from each
  paper source. Its SHA-256 is
  `69492a1c0ba9bedc94bb7d71cdf3282d7d591e43ddf038f0cd524d844cce5ed7`.
  Evaluation overlap is zero. Only 9/2,048 examples (0.44%) exceed the
  1,024-token cutoff before truncation.
- The 2,048 text + 2,048 vision probe processed 336,066 and 606,922 valid
  tokens. At `tau=0.8`, the 86,016 FFN channels divide into 15,146 language,
  12,097 multimodal, 46,750 shared, and 12,023 reserve channels.
- Two sample-disjoint 1,024+1,024 replications have 94.87% exact role
  agreement. Their language-protection Jaccard is 0.942; per-role Jaccards are
  0.881 language, 0.814 multimodal, 0.944 shared, and 0.867 reserve.
- Mapping top-1% is strongly enriched in shared channels: 729/864 (84.4%),
  odds ratio 4.58. Joint protection removes 9,086/12,888 (70.5%) original
  q-band candidates, leaving 3,802 diagnostic candidates.
- The held-out pilot supports the shared and combined protection masks. Shared
  ablation has excess NLL versus matched random of +5.10 on Caption and +3.80
  on C4 text-only. The `language U shared` mask has excess NLL +4.30 and +4.84.
  Language-only and multimodal-only causal specificity do not pass the pilot
  gate.

Decision: do not run structural pruning or claim full NeuPAT role transfer.
The 3,802-neuron joint mask remains `structural_pruning_allowed: false`.
Before P2/full formal P1, either pre-register protection-set validation as the
primary hypothesis or revise the modality-specific role definition and repeat
the pilot. The pilot is screening evidence, not a paper-scale result.
