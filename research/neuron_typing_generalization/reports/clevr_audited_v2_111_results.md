# CLEVR audited_v2_111 precision-extension report

Scope: Nested CLEVR audited_v1 (64 pairs) to audited_v2_111 (111 pairs) precision extension.

- Analysis commit: `67560d5768956111a97f51ce4ce3344926bad80f`
- Target 95% CI half-width: <= 0.010
- Target achieved for all models: `True`
- Strict-contract capacity: 111 disjoint pairs

| Model | v1 estimate and 95% CI | v2 estimate and 95% CI | v2 half-width | Estimate delta | Shared activations exact |
|---|---:|---:|---:|---:|---:|
| gemma_4_e2b_it | 0.1455 [0.1330, 0.1579] | 0.1388 [0.1294, 0.1484] | 0.00947 | -0.0067 | True |
| llava_1_5_7b | 0.1867 [0.1742, 0.1994] | 0.1798 [0.1701, 0.1897] | 0.00978 | -0.0069 | True |
| qwen3_5_2b | 0.1837 [0.1722, 0.1956] | 0.1779 [0.1692, 0.1867] | 0.00874 | -0.0058 | True |

The v2 tier preserves all 64 v1 pairs and adds 47 pairs. Bootstrap resampling uses one manifest pair as the unit and
recomputes the ratio of summed changed-neuron counts to valid-neuron counts in every replicate.

## Interpretation limits

- audited_v2_111 contains audited_v1, so old and new estimates are correlated rather than independent.
- Overlapping confidence intervals do not constitute a formal test of point-estimate change.
- The tier exhausts pairs feasible under the frozen same-scene, edit-distance-one, disjoint contract.
