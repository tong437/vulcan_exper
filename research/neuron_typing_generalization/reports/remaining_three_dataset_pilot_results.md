# Remaining three-dataset frozen-pilot report

Scope: Three models on frozen GQA Balanced, CLEVR audited_v1, and VizWiz VQA pilot tiers; seed 42.

- Collection commits: `010824e2f7af8e2f68452a8e2e8dbbf28540e27a`, `ebe8ddd4ef334b16b411118d9f8ae2a8c5eaf97f`
- Analysis commit: `cacef01e891346343d37b4f7557557fde426fb40`

| Model | Dataset | Global distinct-pair rate | Controlled estimate and 95% CI | Mean normalized entropy |
|---|---|---:|---:|---:|
| gemma_4_e2b_it | clevr | 0.2678 | 0.1455 [0.1330, 0.1579] | 0.3207 |
| llava_1_5_7b | clevr | 0.3274 | 0.1867 [0.1742, 0.1994] | 0.3924 |
| qwen3_5_2b | clevr | 0.3320 | 0.1837 [0.1722, 0.1956] | 0.4054 |
| gemma_4_e2b_it | gqa_balanced | 0.3604 | 0.2254 [0.2220, 0.2287] | 0.4601 |
| llava_1_5_7b | gqa_balanced | 0.4707 | 0.3151 [0.3112, 0.3189] | 0.6007 |
| qwen3_5_2b | gqa_balanced | 0.4397 | 0.2863 [0.2823, 0.2901] | 0.5629 |
| gemma_4_e2b_it | vizwiz_vqa | 0.3779 | descriptive only | 0.4908 |
| llava_1_5_7b | vizwiz_vqa | 0.4805 | descriptive only | 0.6151 |
| qwen3_5_2b | vizwiz_vqa | 0.4495 | descriptive only | 0.5772 |

GQA confidence intervals use 100 image-level clusters. CLEVR uses 64 disjoint manifest pairs. VizWiz has no controlled
pair manifest, so its rows are descriptive; per-cohort results are retained in the companion JSON and run artifacts.

## Frozen-tier decisions

- **gqa_balanced:** No larger tier required for a controlled-mean 95% CI half-width of 0.005.
- **clevr:** Create a new 128-disjoint-pair audited tier for a 0.010 target; do not overwrite audited_v1. The worst-model normal approximation estimates 106 pairs.
- **vizwiz_vqa:** Retain the 500-sample descriptive cohort tier; no controlled-pair precision claim.

## Interpretation limits

- Response classes are sample-conditional descriptive states, not immutable neuron identities.
- M does not establish causal cross-modal interaction; W does not mean inactive.
- GQA bootstrap units are 100 image-level pair groups, not 1,000 independent edges.
- GQA and CLEVR controlled rates are not directly comparable because their pairing interventions differ.
- VizWiz cohorts are observational and do not define controlled counterfactual pairs.
- The experiment has one seed; cross-seed uncertainty remains unmeasured.
