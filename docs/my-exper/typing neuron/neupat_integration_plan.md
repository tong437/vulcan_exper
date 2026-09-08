# NeuPAT Integration Plan for Qwen3.5-VL Neuron Typing

## Scope

This extension primarily tests whether the NeuPAT protection set
(`language U shared`) preserves language ability during Qwen3.5-VL
medical-domain adaptation. Component-role specificity is exploratory. It does
not assume that a NeuPAT reserve neuron is safe to prune.

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

## P1: Protection-Set Matched-Count Causal Validation

The pre-registered primary mask is the exact `language U shared` membership.
Each of its five matched-random controls has the same selected count in every
layer. The language-only, multimodal-only, shared-only, and reserve-only masks
are optional exploratory analyses (`--include_component_roles`).

```bash
python scripts/vulcan/neuron_typing/run_neupat_causality.py \
  --caption_config scripts/vulcan/neuron_typing/configs/formal_coco.yaml \
  --text_config scripts/vulcan/neuron_typing/configs/neupat_language_eval.formal.yaml \
  --score_file saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet \
  --output_dir saves/neuron_typing/neupat_protection_formal_500 \
  --calibration_manifest saves/neuron_typing/phase1_clean_2k/calibration/sample_manifest.json \
  --typing_manifest saves/neuron_typing/phase1_clean_2k/activations/sample_manifest.json \
  --probe_manifest saves/neuron_typing/neupat_probe_formal_2048/vision_manifest.json \
  --probe_manifest saves/neuron_typing/neupat_probe_formal_2048/text_manifest.json \
  --control_seed_count 5 \
  --caption_max_samples 500 \
  --text_max_samples 500 \
  --bootstrap_samples 2000 \
  --seed 20260902 \
  --batch_size 4 \
  --skip_pope
```

The primary causal gate passes only when the text data are isolated and the
paired-bootstrap 95% lower bound for
`NLL(language U shared)-mean NLL(matched random)` is above zero. Caption is a
supportive endpoint. POPE is resumed later as a secondary multimodal check.

This gate establishes causal language sensitivity, not post-SFT language
retention. Reserve remains an update-capacity hypothesis, not a pruning
hypothesis.

## P2: NeuPAT-Guided Domain SFT

The training implementation:

- masks `gate_proj` and `up_proj` language-role rows;
- masks matching `down_proj` columns;
- applies shared-role input-side squared Frobenius/L2 preservation;
- applies shared-role output-side cosine preservation;
- requires full SFT and zero weight decay.

After the P1 gate passes, run the hash-recorded comparison matrix:

```bash
PYTHONPATH=src WANDB_DISABLED=true python \
  scripts/vulcan/neuron_typing/run_neupat_sft_matrix.py

PYTHONPATH=src WANDB_DISABLED=true python \
  scripts/vulcan/neuron_typing/evaluate_neupat_sft_language.py

PYTHONPATH=src WANDB_DISABLED=true python \
  scripts/vulcan/neuron_typing/evaluate_neupat_sft_vqa.py
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

## Current Formal Results and Gate Decision (2026-09-04)

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
- The pre-registered primary hypothesis is now the combined protection set,
  not component-role transfer. On 500 held-out Caption samples, its ablation
  increases NLL by +9.761 versus baseline; five matched controls average
  +4.321, for +5.440 excess damage (paired 95% CI [5.365, 5.520]).
- On the independent 500-example packed C4 evaluation, protection-set
  ablation increases NLL by +10.363; matched controls average +4.883, for
  +5.480 excess damage (paired 95% CI [5.421, 5.537]). Both datasets are
  isolated and all per-layer control counts match exactly.

The matched three-arm SFT matrix completed at 354 steps/3 epochs. Best
VQA-RAD eval losses were 0.198223 for vanilla full-SFT (step 50), 0.193958 for
LoRA (step 50), and 0.241588 for NeuPAT (step 100). On the fixed 500-example
C4 test, NLL was 3.381973 base, 3.378277 vanilla, 3.375873 LoRA, and 3.380206
NeuPAT. NeuPAT minus vanilla was +0.001929 (paired 95% CI [0.001120,
0.002706]), so the primary post-SFT language-retention criterion failed.

On the historical 251-question binary VQA-RAD split, accuracy was 0.6614 base, 0.6733
vanilla, 0.7052 LoRA, and 0.6853 NeuPAT. NeuPAT minus vanilla was +0.0120 by
point estimate, but its image-cluster bootstrap 95% CI [-0.0558, 0.0797]
failed the -0.02 non-inferiority margin. The causal protection-set gate remains
valid, but the experiment does not support a comparative NeuPAT
language-preservation claim in this no-forgetting regime. A stronger
forgetting stress test and a larger/grouped multimodal evaluation should be
pre-registered before retesting; do not tune on this held-out C4 slice.

NeuPAT ran without DeepSpeed because the current activation-preservation
regularizer triggers duplicate parameter reduction under ZeRO-2. The matched
seed, data, effective batch size, epochs, learning rate (versus vanilla full
SFT), and best-checkpoint rule were retained, but the backend difference is a
limitation. Do not claim full component-role transfer or authorize structural
pruning. The 3,802-neuron joint mask remains
`structural_pruning_allowed: false`.

Post-hoc image-content auditing invalidated the "held-out" interpretation of
that VQA-RAD comparison: 121 of its 135 distinct image hashes occur in the
training split. The numerical result is retained only as a diagnostic. The
replacement development protocol groups by image SHA-256 and is frozen in
`neupat_forgetting_stress_plan.md`; a separate final multimodal test source is
still required before a new formal NeuPAT comparison.
