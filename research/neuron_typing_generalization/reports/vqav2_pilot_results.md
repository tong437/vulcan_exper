# VQAv2 three-model frozen-pilot report

Scope: VQAv2 frozen pilot: 500 samples / 250 same-question different-image pairs / seed 42.

- Collection code commit: `54f885c6db1182d31b7f6ba10a9f6315cc5ed5b8`
- Analysis code commit: `10659b7d06791da4ababb68e4bb7e34caccfe36e`
- Primary precision target: 95% CI half-width <= 0.010
- Larger-tier decision: not required for the primary controlled mean; create a new tier only for narrower precision or subgroup claims.

## Controlled same-question, different-image result

| Model | Change rate | Pair-bootstrap 95% CI | Mean normalized entropy | Pairs |
|---|---:|---:|---:|---:|
| gemma_4_e2b_it | 0.2195 | [0.2156, 0.2234] | 0.4830 | 250 |
| llava_1_5_7b | 0.2569 | [0.2514, 0.2624] | 0.6142 | 250 |
| qwen3_5_2b | 0.2231 | [0.2182, 0.2279] | 0.5572 | 250 |

The bootstrap resampling unit is one frozen manifest pair. Each replicate recomputes the ratio of summed changed-neuron
counts to summed valid-neuron counts; neurons are not treated as independent observations.

## Reproducibility

The companion JSON contains model and dataset revisions, split/config, manifest hashes, activation partition hashes,
seed, token-mask policy, runtime versions, explicit denominators, class distributions, and precision scenarios for every
model. Large activations and per-sample results remain in the ignored local artifact directories recorded there.

## Interpretation limits

- Response classes are sample-conditional descriptive states, not immutable neuron identities.
- M does not establish causal cross-modal interaction; W does not mean inactive.
- The controlled VQAv2 comparison isolates image context only subject to the frozen question/sample audit.
- One seed and one dataset do not establish cross-dataset or cross-seed generalization.
