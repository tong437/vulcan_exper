<!--
Copyright 2025 the LlamaFactory team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Advisor-confirmed experimental guidance

This file is the normative interpretation contract for the input-dependent neuron-typing study. Implementation,
configuration, tests, tables, figures, and written conclusions must agree with it. If the advisor changes a definition,
update this file first and create new versioned manifests or outputs where the change affects reproducibility.

## 1. Research object and permitted interpretation

The object of study is a neuron's **sample-conditional response state**, not an immutable neuron identity. A neuron may
be assigned different response classes for different inputs.

The four classes are descriptive:

- `V`: above-layer-average importance at visual-token positions only;
- `T`: above-layer-average importance at user-question positions only;
- `M`: above-layer-average importance at both position sets;
- `W`: at or below the layer average at both position sets.

`M` does not by itself demonstrate causal cross-modal interaction. `W` means relatively weak under this within-layer
normalization, not zero activation or irrelevance. Class changes, correlations, and natural-cohort differences are not
causal evidence unless the comparison design independently supports that interpretation.

## 2. Importance and normalization

For sample `x`, layer `l`, neuron `j`, and token set `S`, use the post-gating MLP activation `z` at the input of
`down_proj`:

```text
I(l, j; x, S) = ||W_down[l][:, j]||_2
                * sqrt(mean(z[l, t, j]^2 for t in S))
```

Compute visual and user-question importance separately. Normalize within each sample, layer, and modality:

```text
q(l, j; x, S) = I(l, j; x, S) / sum_k I(l, k; x, S)
```

A response is strong only when `q > 1 / d_ff(l)`, using the actual FFN width of layer `l`; equality is not strong.
This layer-specific notation matters for architectures such as Gemma-4 whose later decoder layers use double-wide
MLPs. Encode the visual/text strong bits as
`W=0`, `V=1`, `T=2`, and `M=3`.

Visual, user-question, control, and padding token masks must be mutually exclusive. Empty modality token sets,
non-finite values, and zero total layer importance are `INVALID`; they must not be silently classified as `W`.

## 3. Change-rate estimands

The primary global change rate uses all ordered sample pairs, including self-pairs. It must be computed from per-neuron
class counts rather than materializing an `N x N` pair table. Also report:

- the corresponding rate over distinct ordered pairs;
- both rates for every decoder layer;
- the maximum-change distinct sample pair, using bounded-memory chunked search and deterministic tie-breaking.

Invalid observations are excluded explicitly. Reports must state the valid denominator whenever invalid observations
occur; they must not mix estimands with different denominators without disclosure.

Concretely, for one neuron with valid sample count `n` and class counts `n_c`, the changed ordered-comparison count is
`n^2 - sum_c(n_c^2)`. Its including-self denominator is `n^2`, and its distinct-pair denominator is `n(n-1)`.
Overall and per-layer rates are ratios of summed changed counts to summed valid denominators. They are not unweighted
averages of per-neuron rates when neurons have different valid sample counts. The changed numerator is identical for
the including-self and distinct-pair estimands because a self-pair never changes class.

## 4. Stability summaries

For every `(layer, neuron)`, report valid-sample counts, class counts and proportions, entropy, normalized entropy,
dominant class, and an auditable stability/profile label. Transition matrices use only explicit edges from the frozen
pair manifest. Directed and symmetrized transitions must remain distinguishable.

Uncertainty for a controlled-pair aggregate is estimated by resampling frozen manifest edges, not neurons. Every
bootstrap replicate recomputes the ratio of summed changed-neuron counts to summed valid-neuron counts. Record the
bootstrap seed, replicate count, confidence level, and empirical pair-rate variance. A precision-based sample-size
decision may create a new named manifest tier, but must never overwrite the pilot tier.

## 5. Dataset-specific comparison policy

The global all-sample statistic and dataset-specific controlled comparisons are separate analyses.

| Dataset | Frozen comparison policy | Interpretation limit |
|---|---|---|
| VQAv2 | Same normalized question, different image | Controlled image-context comparison, subject to question and sample audit |
| GQA Balanced | Same image, different question | Controlled question-context comparison |
| CLEVR | Same scene and functional-program edit distance 1 in `audited_v1`; precision extension `audited_v2_111` | Original `pilot` candidates remain excluded as failed audit evidence |
| VizWiz-VQA | Stratified natural cohorts by answerability/answer type | Descriptive cohort comparison only; not a clean/degraded causal pair |

Pair manifests enumerate dataset-specific comparison edges only. They do not enumerate the global all-ordered-pairs
estimand.

## 6. Data and reproducibility boundary

Pin dataset ID, revision, configuration, split, license, and pairing policy. Acquire README/repository records and the
lightweight question/annotation metadata needed to construct samples before acquiring images.

Freeze deterministic sample and pair manifests before image materialization. Materialize only unique images referenced
by the selected manifest tier. Do not download the complete VQAv2, CLEVR, or VizWiz image archives for this preparation
stage. A changed sampling policy requires a new named manifest tier; never overwrite a manifest already used by a run.

Generated source metadata, selected images, activations, weights, and full run outputs remain local artifacts. Small
manifests, schemas, configurations, tests, and their canonical hashes belong in version control.

## 7. Staged execution boundary

Complete schemas, pure CPU metric functions, count-based analyses, model-adapter and hook interfaces, and synthetic CPU
tests before any real model forward pass. The hook target is the input of each language-decoder `mlp.down_proj`, and
captured activations should be reduced immediately rather than retained as full token-by-neuron tensors.

The first GPU check is the minimal Qwen3.5-2B x VQAv2 smoke test. It uses the frozen smoke manifest, batch size one,
BF16 inference, no generation, all 24 decoder layers, expected `d_ff=6144`, nonempty visual/text masks, and a direct
versus streaming-statistics equality check. Passing this smoke test is a prerequisite for the corresponding 16-sample
Gemma-4 and LLaVA VQAv2 smoke runs, and all three smoke runs must pass before the 500-sample VQAv2 pilot. These checks
are not themselves scientific results.

The direct-versus-streaming check must use the same real-model forward: one path retains the one-sample post-gating
activation long enough to recompute masked RMS by direct indexing, while the production path immediately reduces it.
Record per-layer/per-modality token counts and numerical tolerances, then release the retained activation.

## 8. Reporting rules

Every result must identify the model revision, dataset revision and split, manifest digest, seed, token-mask policy,
valid denominator, and whether a comparison is global, controlled, candidate/pending, or natural-cohort only. Avoid
phrasing that turns response specialization, association, or paired differences into causal function without a causal
intervention and appropriate controls.
