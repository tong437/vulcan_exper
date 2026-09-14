# Frozen sampling manifests

Each dataset/tier directory contains:

- `sample_manifest.parquet`: selected sample identity, prompt, answer, group/stratum, source image locator, and seed;
- `pair_manifest.parquet`: explicit controlled comparison edges only;
- `*.sha256`: SHA-256 of canonical logical rows, independent of Parquet binary metadata.

`build_manifests.py` is idempotent when logical rows are unchanged and refuses to replace a manifest whose digest would
change. To revise sampling, change the policy and create a new named tier; do not overwrite a manifest used by a run.

The global advisor-defined statistic still uses all ordered sample pairs including self-pairs. It is computed from
per-neuron class counts and therefore is not enumerated in `pair_manifest.parquet`. The pair manifest represents only
dataset-specific controlled or candidate comparisons.

Current tiers:

| Dataset | CPU fixture | GPU smoke | Pilot | Pair status |
|---|---:|---:|---:|---|
| VQAv2 | 16 / 8 pairs | 16 / 8 pairs | 500 / 250 pairs | passed |
| GQA Balanced | 40 / 80 pairs | — | 500 / 1,000 pairs | passed |
| CLEVR | 16 / 8 pairs | — | 128 / 64 candidate pairs | `audited_v1`: 128 / 64 passed pairs |
| VizWiz-VQA | 16 / 0 pairs | — | 500 / 0 pairs | stratified cohort only |

The CLEVR `pilot` tier is retained as immutable audit-failure evidence. `audited_v1` holds the scene fixed and admits
only disjoint pairs whose functional-program token sequence has Levenshtein distance exactly one. Only `audited_v1`
is eligible for controlled maximum-pair analysis.
