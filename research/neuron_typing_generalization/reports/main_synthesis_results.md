# Four-dataset main synthesis

Scope: Three frozen VLMs across VQAv2, GQA Balanced, CLEVR audited_v2_111, and VizWiz VQA.

- Analysis commit: `25b7ad3e6631e42d22d0beeaabc2892e7e56211d`
- Paired bootstrap: 20,000 replicates, seed 42
- Multiplicity: Bonferroni 95% familywise intervals across all nine prespecified contrasts

## Main results

| Dataset | Model | Primary rate | 95% CI | Mean normalized entropy | Independent bootstrap units |
|---|---|---:|---:|---:|---:|
| VQAv2 | Gemma-4-E2B-IT | 0.2195 | [0.2156, 0.2234] | 0.4830 | 250 |
| VQAv2 | LLaVA-1.5-7B | 0.2569 | [0.2514, 0.2624] | 0.6142 | 250 |
| VQAv2 | Qwen3.5-2B | 0.2231 | [0.2182, 0.2279] | 0.5572 | 250 |
| GQA Balanced | Gemma-4-E2B-IT | 0.2254 | [0.2220, 0.2287] | 0.4601 | 100 |
| GQA Balanced | LLaVA-1.5-7B | 0.3151 | [0.3112, 0.3189] | 0.6007 | 100 |
| GQA Balanced | Qwen3.5-2B | 0.2863 | [0.2823, 0.2901] | 0.5629 | 100 |
| CLEVR | Gemma-4-E2B-IT | 0.1388 | [0.1294, 0.1484] | 0.3214 | 111 |
| CLEVR | LLaVA-1.5-7B | 0.1798 | [0.1701, 0.1897] | 0.3905 | 111 |
| CLEVR | Qwen3.5-2B | 0.1779 | [0.1692, 0.1867] | 0.4034 | 111 |
| VizWiz VQA | Gemma-4-E2B-IT | 0.3779 | descriptive only | 0.4908 | 0 |
| VizWiz VQA | LLaVA-1.5-7B | 0.4805 | descriptive only | 0.6151 | 0 |
| VizWiz VQA | Qwen3.5-2B | 0.4495 | descriptive only | 0.5772 | 0 |

VizWiz reports the global all-distinct-sample rate because it has no controlled pairs. The other datasets report their
dataset-specific controlled estimand.

## Within-dataset paired model contrasts

| Dataset | Contrast | Difference | Pairwise 95% CI | Bonferroni familywise 95% CI | Units |
|---|---|---:|---:|---:|---:|
| VQAv2 | LLaVA-1.5-7B − Gemma-4-E2B-IT | +0.0374 | [+0.0339, +0.0409] | [+0.0323, +0.0424] | 250 |
| VQAv2 | Qwen3.5-2B − Gemma-4-E2B-IT | +0.0036 | [+0.0002, +0.0070] | [-0.0011, +0.0084] | 250 |
| VQAv2 | LLaVA-1.5-7B − Qwen3.5-2B | +0.0338 | [+0.0302, +0.0374] | [+0.0286, +0.0389] | 250 |
| GQA Balanced | LLaVA-1.5-7B − Gemma-4-E2B-IT | +0.0897 | [+0.0877, +0.0916] | [+0.0869, +0.0924] | 100 |
| GQA Balanced | Qwen3.5-2B − Gemma-4-E2B-IT | +0.0608 | [+0.0593, +0.0624] | [+0.0587, +0.0630] | 100 |
| GQA Balanced | LLaVA-1.5-7B − Qwen3.5-2B | +0.0288 | [+0.0277, +0.0299] | [+0.0272, +0.0304] | 100 |
| CLEVR | LLaVA-1.5-7B − Gemma-4-E2B-IT | +0.0410 | [+0.0366, +0.0455] | [+0.0344, +0.0473] | 111 |
| CLEVR | Qwen3.5-2B − Gemma-4-E2B-IT | +0.0391 | [+0.0344, +0.0437] | [+0.0324, +0.0456] | 111 |
| CLEVR | LLaVA-1.5-7B − Qwen3.5-2B | +0.0019 | [-0.0020, +0.0057] | [-0.0036, +0.0073] | 111 |

Familywise intervals excluding zero provide multiplicity-protected evidence of a model difference on the same frozen
pair units. The intervals are effect-size intervals, not a cross-dataset model leaderboard.

## Interpretation limits

- Paired model contrasts are valid only within a dataset because pairing interventions differ by dataset.
- Both unadjusted pairwise 95% intervals and Bonferroni 95% familywise intervals are reported.
- Normalized-depth interpolation is descriptive and does not assert layer-to-layer architectural equivalence.
- VizWiz has no controlled pair estimand and is excluded from model-difference inference.
- Response classes are input-conditional states and do not establish causal multimodal interaction.
