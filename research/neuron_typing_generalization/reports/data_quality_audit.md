# Frozen dataset quality assessment

Overall status: `ready_with_documented_risks`.

## clevr

Source revision match: `True`; integrity failures: `0`.

- **medium / high confidence:** Pinned upstream CLEVR README contains VQAv2/visualqa statistics and no CLEVR identity. Impact: The README cannot substantiate CLEVR dataset facts; pinned config and official metadata must govern. Remediation: Retain the byte-exact README for provenance and label it as an upstream card defect.

| Tier | Samples | Pairs | Unique images | Duplicate IDs | Pair violations | Status |
|---|---:|---:|---:|---:|---:|---|
| audited_v1 | 128 | 64 | 64 | 0 | 0 | passed |
| cpu_fixture | 16 | 8 | 16 | 0 | 0 | passed |
| pilot | 128 | 64 | 128 | 0 | 0 | passed |

## gqa_balanced

Source revision match: `True`; integrity failures: `0`.

| Tier | Samples | Pairs | Unique images | Duplicate IDs | Pair violations | Status |
|---|---:|---:|---:|---:|---:|---|
| cpu_fixture | 40 | 80 | 8 | 0 | 0 | passed |
| pilot | 500 | 1000 | 100 | 0 | 0 | passed |

## vizwiz_vqa

Source revision match: `True`; integrity failures: `0`.

| Tier | Samples | Pairs | Unique images | Duplicate IDs | Pair violations | Status |
|---|---:|---:|---:|---:|---:|---|
| cpu_fixture | 16 | 0 | 16 | 0 | 0 | passed |
| pilot | 500 | 0 | 500 | 0 | 0 | passed |

## vqav2

Source revision match: `True`; integrity failures: `0`.

| Tier | Samples | Pairs | Unique images | Duplicate IDs | Pair violations | Status |
|---|---:|---:|---:|---:|---:|---|
| cpu_fixture | 16 | 8 | 16 | 0 | 0 | passed |
| gpu_smoke | 16 | 8 | 16 | 0 | 0 | passed |
| pilot | 500 | 250 | 493 | 0 | 0 | passed |

## Interpretation

Manifest grain, required-field completeness, sample-key uniqueness, pair referential integrity, and each dataset's frozen pairing rule were checked. Pending CLEVR tiers remain candidate evidence only; `audited_v1` is the controlled-analysis tier. VizWiz remains descriptive cohort data.
