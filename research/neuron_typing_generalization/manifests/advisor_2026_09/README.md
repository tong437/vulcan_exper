# Advisor 2026-09 sample manifests

These four JSONL files are the frozen seed-42 samples for the current four-model × four-dataset study. Every manifest
contains 1,000 question rows, the pinned Hugging Face revision, the source row index, question/answer metadata, image
identity, and an image-content SHA-256 digest. `sampling_audit.json` records candidate capacities, integer quotas, pair
audits, and canonical manifest digests.

| Dataset | Sampling result |
|---|---|
| MME | 500 strict two-question groups stratified by `category`; 1,000 rows |
| MMBench | 1,000 rows stratified by the literal `source` field |
| TextVQA | 1,000 validation rows sampled without replacement |
| ChartQA | 500 strict two-question `(type, image.path)` groups stratified by `type`; 1,000 rows |

ChartQA contains 1,612 `(type, image.path)` groups, but only 809 contain exactly two questions. The manifest samples
only those strict pairs: 386 `human_test` groups and 114 `augmented_test` groups. The 803 non-pair groups are excluded
and counted in the audit rather than being silently truncated or combined.

MMBench has 566 distinct literal `source` values. Some are URLs or embedded data rather than short corpus labels. The
sampler follows the requested field exactly; manifests and the audit store a SHA-256 identifier and bounded preview so
large source strings do not bloat every row.

Regenerate from the pinned downloaded assets with:

```bash
HF_DATASETS_CACHE="$PWD/artifacts/neuron_typing_generalization/advisor_2026_09/hf_cache/datasets" \
HF_HOME="$PWD/artifacts/neuron_typing_generalization/advisor_2026_09/hf_cache/home" \
python research/neuron_typing_generalization/pipelines/sample_advisor_2026_09.py \
  --output-dir research/neuron_typing_generalization/manifests/advisor_2026_09 \
  --seed 42 --offline
```
