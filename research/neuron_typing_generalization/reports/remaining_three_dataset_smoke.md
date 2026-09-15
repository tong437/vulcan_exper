# Remaining-dataset GPU smoke acceptance

Date: 2026-09-15  
Hardware: NVIDIA GeForce RTX 4090 (24,564 MiB)  
Study config: `configs/studies/remaining_three_dataset_smoke.yaml`

All nine real-model, real-image, one-sample combinations completed on CUDA with bfloat16. Every saved activation
statistic was finite, and every decoder layer had non-empty valid visual and text token regions.

| Model | Dataset | Layers | Elapsed (s) | Peak CUDA memory (GiB) | Result |
|---|---|---:|---:|---:|---|
| Qwen3.5-2B | GQA Balanced | 24 | 2.535 | 4.26 | pass |
| Qwen3.5-2B | CLEVR | 24 | 2.510 | 4.26 | pass |
| Qwen3.5-2B | VizWiz VQA | 24 | 2.754 | 4.72 | pass |
| LLaVA-1.5-7B | GQA Balanced | 32 | 5.127 | 13.42 | pass |
| LLaVA-1.5-7B | CLEVR | 32 | 5.083 | 13.42 | pass |
| LLaVA-1.5-7B | VizWiz VQA | 32 | 5.185 | 13.42 | pass |
| Gemma-4-E2B-IT | GQA Balanced | 35 | 5.003 | 10.02 | pass |
| Gemma-4-E2B-IT | CLEVR | 35 | 4.985 | 10.02 | pass |
| Gemma-4-E2B-IT | VizWiz VQA | 35 | 4.995 | 10.02 | pass |

The smoke run exposed one pre-pilot data-boundary defect: the accepted CLEVR `audited_v1` tier referenced images
that were not part of the older `pilot` image index. The selected-only materializer now supports incremental
dataset/tier materialization, and the readiness audit checks the union of every frozen tier. CLEVR now contains 191
selected images (35,646,460 bytes); the full archive remains unmaterialized.

The smoke artifacts were acceptance evidence from a dirty worktree and are not scientific results. Formal pilots
must start only after the implementation/configuration commit is frozen; each run's `metadata.json` must report that
commit and `dirty: false`.
