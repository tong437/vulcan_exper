# NeuPAT Language-Forgetting Stress Discovery

## Objective

Find the least severe vanilla full-SFT setting that creates measurable
language forgetting while still improving multimodal task performance. This
is a development-only search. It does not test NeuPAT and must not access the
language lockbox.

## Data correction and frozen splits

The historical VQA-RAD question split is not image-content-disjoint. The 940
training rows contain 280 distinct image SHA-256 hashes, the 251 test rows
contain 135, and 121 test hashes occur in train. Historical VQA-RAD results
therefore remain diagnostics, not held-out multimodal evidence.

Stress discovery uses only the historical training rows, regrouped by image
content SHA-256:

| Split | Questions | Unique image hashes | no / yes |
|---|---:|---:|---:|
| stress train | 752 | 224 | 378 / 374 |
| stress dev | 188 | 56 | 95 / 93 |

The train/dev image-hash overlap is zero. Questions sharing an image hash are
kept together and rewritten to one canonical image path, so image-cluster
bootstrap treats them as one unit. Exact hashes and the legacy leakage audit
are saved in `datasets/vqa_rad/stress_split_manifest.json`.

Language selection uses a new 700-document C4 validation corpus. It packs to
649 fixed 511-token blocks; the first 500 blocks (255,000 tokens) are frozen
as language-dev. A separately fetched 700-document lockbox remains unopened
until the stress setting and later comparison design are frozen.

| Corpus | SHA-256 | Permitted use |
|---|---|---|
| stress dev | `120ed60f5a67753c9bd0aeacce0f69410ed110a08c17a898b4a0e86d6bdf2e03` | select stress strength |
| stress lockbox | `4e9a37d4f56e85f858599c0ee7bdb7a8f1a8dfbdceda8f9e8782499128953a57` | one final confirmation only |

The frozen base-dev metrics are C4 NLL 3.364215 (PPL 28.9108) and VQA-dev
accuracy 0.6436/F1 0.5786.

## Pre-registered stress criteria

A candidate qualifies only when both conditions pass:

1. Language forgetting: paired token-weighted `delta NLL >= log(1.05) =
   0.048790164`, and the paired bootstrap 95% CI lower bound is above zero.
   This is equivalent to at least a 5% perplexity increase.
2. Multimodal adaptation: VQA-dev accuracy improves by at least 0.02 over
   base, and positive-class F1 does not decrease.

Use 10,000 paired bootstrap resamples. Select the first passing candidate in
the frozen order below; do not select the largest language drop or the best
VQA score after seeing all results.

| Order | Candidate | LR | Epochs | Relative update exposure |
|---:|---|---:|---:|---:|
| 1 | `lr5e-6_ep3` | 5e-6 | 3 | 1x |
| 2 | `lr1e-5_ep3` | 1e-5 | 3 | 2x |
| 3 | `lr2e-5_ep3` | 2e-5 | 3 | 4x |
| 4 | `lr2e-5_ep6` | 2e-5 | 6 | 8x |
| 5 | `lr5e-5_ep3` | 5e-5 | 3 | 10x |

Every candidate starts independently from the same base model, uses seed
20260904 and effective batch size 8, and evaluates the final update. It does
not reload a VQA-selected best checkpoint.

## Commands

Run one candidate at a time, starting at the top of the ladder:

```bash
PYTHONPATH=src WANDB_DISABLED=true python \
  scripts/vulcan/neuron_typing/run_neupat_forgetting_stress.py \
  --candidate lr5e-6_ep3
```

Then evaluate it on development data:

```bash
PYTHONPATH=src WANDB_DISABLED=true python \
  scripts/vulcan/neuron_typing/evaluate_neupat_forgetting_stress.py \
  --candidate lr5e-6_ep3
```

Inspect the decision:

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("saves/qwen35-0_8b-vqa-rad/neupat_forgetting_stress_20260904/dev_eval/stress_selection_summary.json")
d = json.loads(p.read_text())
print("stress_regime_found:", d["stress_regime_found"])
print("selected:", d["first_qualifying_candidate"])
print("next:", d["recommended_next_candidate"])
PY
```

If `stress_regime_found` is false, replace the candidate in both commands
with `recommended_next_candidate`. Stop immediately when the first candidate
passes. Both runners skip completed outputs and resume structurally complete
checkpoints.

Do not run the lockbox config during this search. Once a stress regime is
found, first freeze the vanilla/LoRA/NeuPAT comparison, training seeds, and
multimodal final test source. Only then may the lockbox be evaluated once.

## Development Results

| Candidate | C4 dev NLL | Delta NLL vs base (95% CI) | VQA accuracy | Delta accuracy | Forgetting | Adaptation | Qualifies |
|---|---:|---:|---:|---:|---|---|---|
| base | 3.364215 | - | 0.6436 | - | - | - | - |
| `lr5e-6_ep3` | 3.370095 | +0.005880 [0.004350, 0.007464] | 0.7500 | +0.1064 | fail | pass | no |

The first candidate produces strong VQA adaptation but only about a 0.59%
perplexity increase, well below the pre-registered 5% forgetting threshold.
The next candidate is `lr1e-5_ep3`; the lockbox remains unopened.
