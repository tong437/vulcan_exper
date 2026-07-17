# Qwen3.5-VL Neuron Typing Research Status

> Last updated: 2026-07-17
> Status: corrected Phase 1 completed on 2k samples; corrected Phase 2 causal ablation completed for the first ratio sweep; Phase 3 not started formally.

## 1. Research objective

This project studies modality-related FFN neurons in the Qwen3.5-VL 0.8B hybrid architecture. The main questions are:

1. Can FFN neurons be separated by their responses to visual and text tokens?
2. Do Full Attention (FA) and GatedDeltaNet (GDN) layers have different neuron-type distributions?
3. Do the resulting neuron scores identify causally different subspaces under ablation?
4. Can these scores guide type-aware pruning more safely than random, magnitude, or activation-based pruning?

The current research direction is no longer simply “unknown neurons are safe to prune.” Corrected experiments instead suggest that modality purity and causal importance are non-monotonic:

- extremely high multimodal-purity neurons may be important;
- a broader medium-purity multimodal subspace may be highly redundant;
- neurons that are inactive on most samples may be sparse specialists rather than noise.

## 2. Model and experiment setup

| Item | Value |
|---|---|
| Model | Qwen3.5-VL 0.8B |
| Transformer layers | 24 |
| FFN neurons per layer | 3,584 |
| Total FFN neurons | 86,016 |
| FA layers | 3, 7, 11, 15, 19, 23 |
| GDN layers | Remaining 18 layers |
| Calibration samples | 500 |
| Corrected formal typing samples | 2,000 |
| Visual activation threshold | Per-neuron q97, index 1 |
| Text activation threshold | Per-neuron q95, index 0 |
| Representative sample lists | Separate visual/text top-K, then deduplicated union |
| Current top-K | 50 |
| Phase 2 primary metric | Teacher-forced label NLL; PPL is auxiliary |

The exact model checkpoint, dataset split, count thresholds, preprocessing configuration, calibration-file hash, and sample IDs should be taken from the saved run `config.json` when reporting formal results.

## 3. Corrected neuron-typing definition

For each sample and neuron, let `v_count` and `t_count` be the number of visual/text tokens whose normalized activation exceeds the calibrated threshold. The active conditions use minimum-count semantics:

```python
visual_on = v_count >= visual_required
text_on = t_count >= text_required

is_visual = visual_on & ~text_on
is_text = ~visual_on & text_on
is_multimodal = visual_on & text_on
is_unknown = ~visual_on & ~text_on
```

The four sample-level classes are mutually exclusive and exhaustive.

### 3.1 Representative-sample purity q

For neuron \(n\), let \(S_n^v\) and \(S_n^t\) be its final visual and text top-K sample lists. The representative set is the deduplicated union:

\[
S_n=S_n^v\cup S_n^t.
\]

The four purity scores are computed on the same set:

\[
q_c(n)=\frac{\#\{s\in S_n:\operatorname{type}(n,s)=c\}}{|S_n|}.
\]

For every alive neuron:

\[
q_v+q_t+q_m+q_u=1.
\]

Dead neurons have `q_* = NaN` internally and `null` in JSON.

### 3.2 Full-dataset frequency r

The full-dataset frequency is:

\[
r_c(n)=\frac{\#\{s\in D:\operatorname{type}(n,s)=c\}}{|D|},
\]

with:

\[
r_v+r_t+r_m+r_u=1.
\]

Interpretation:

- `q_*`: type purity among representative high-response samples;
- `r_*`: response-type frequency across the full typing dataset.

These quantities must not be mixed into a single probability vector.

### 3.3 Neuron-level labels

- `dominant_type`: `argmax(q_visual, q_text, q_multimodal, q_unknown)` for alive neurons;
- `high-confidence type`: a q score at or above 0.7;
- `mixed/low-confidence`: `max(q) < 0.7`;
- `dead`: `global_max <= 1e-6`.

`dominant_type` is a forced descriptive label. It must not be interpreted as a high-confidence functional class.

## 4. Bugs found and corrected

The initial reproduction exposed several implementation and transfer issues:

1. **Layer-dependent activation scale.** Deep-layer activations were approximately 3–5 times larger than shallow-layer activations, so the paper’s fixed thresholds could not be transferred directly.
2. **Float16 underflow.** A `1e-8` clamp became zero in float16, producing division by zero and NaNs. Loaded maxima are now converted to float32 and clamped safely.
3. **Visual/text token imbalance.** Approximately 258 visual tokens versus 12.8 caption tokens made the original text criterion too strict. Visual q97 and text q95 are now calibrated separately.
4. **Minimum-count off-by-one.** `count > required` incorrectly required one additional token; it was replaced with `count >= required`.
5. **Streaming top-K history contamination.** Historical type counts were incremented when samples temporarily entered top-K but were not decremented after eviction. Top-K entries now carry score, sample ID, and type code together.
6. **Non-deduplicated visual/text union.** The old “union” used two list lengths rather than a sample-ID union. The corrected q scores use a deduplicated union.
7. **Incomparable old `p_*` scores.** Visual, text, multimodal, and unknown scores used different denominators. They have been replaced by q/r.
8. **Dropped dead mask.** `dead_mask` was computed but not serialized. It is now saved and excluded from alive-neuron statistics.
9. **Quantile parameter mismatch.** The pipeline now passes separate visual and text quantile indices.
10. **Slow q computation.** The union implementation was reduced from approximately \(O(DK^2)\) Python work to \(O(DK)\).

The earlier soft-score means above 1 were direct evidence of the streaming top-K bug. Old `p_*` distributions and typed masks are not final evidence.

## 5. Corrected Phase 1 results: 2k samples

### 5.1 Integrity checks

| Quantity | Result |
|---|---:|
| Total neurons | 86,016 |
| Alive neurons | 86,013 |
| Dead neurons | 3 |

The dominant counts sum to 86,013, and the rounded mean q/r values each sum to approximately 1.

Before the formal 2k run, the corrected end-to-end pipeline also passed a 500-sample validation. Detailed 500-versus-2k rank correlations and mask-overlap statistics have not yet been reported and remain a required stability analysis.

### 5.2 Dominant type distribution

| Dominant type | Count | Percentage of all neurons |
|---|---:|---:|
| visual | 41,214 | 47.9% |
| text | 85 | 0.1% |
| multimodal | 44,388 | 51.6% |
| unknown | 326 | 0.4% |

This table reports forced `argmax(q)` labels. It does not imply that 51.6% of neurons are high-confidence multimodal neurons.

### 5.3 q statistics among alive neurons

| Score | Mean | Std | Median | Max |
|---|---:|---:|---:|---:|
| `q_visual` | 0.4556 | 0.1986 | 0.4719 | 1.0000 |
| `q_text` | 0.0213 | 0.0330 | 0.0114 | 0.7470 |
| `q_multimodal` | 0.5054 | 0.2224 | 0.5000 | 1.0000 |
| `q_unknown` | 0.0178 | 0.0558 | 0.0000 | 1.0000 |

Rounded mean check:

\[
0.4556+0.0213+0.5054+0.0178\approx1.
\]

### 5.4 r statistics among alive neurons

| Score | Mean | Std | Median | Max |
|---|---:|---:|---:|---:|
| `r_visual` | 0.7365 | 0.1888 | 0.8050 | 1.0000 |
| `r_text` | 0.0072 | 0.0179 | 0.0050 | 0.6775 |
| `r_multimodal` | 0.1908 | 0.1754 | 0.1350 | 1.0000 |
| `r_unknown` | 0.0656 | 0.1351 | 0.0225 | 1.0000 |

Rounded mean check:

\[
0.7365+0.0072+0.1908+0.0656\approx1.
\]

The q/r difference suggests that neurons are visual on most ordinary samples, while their representative high-response samples exhibit substantially more multimodal behavior.

### 5.5 Per-layer dominant counts

| Layer | Architecture | Visual | Text | Multimodal | Unknown | Dead |
|---:|---|---:|---:|---:|---:|---:|
| 0 | GDN | 2,030 | 2 | 1,541 | 11 | 0 |
| 1 | GDN | 1,596 | 1 | 1,980 | 7 | 0 |
| 2 | GDN | 1,653 | 1 | 1,923 | 7 | 0 |
| 3 | FA | 1,552 | 2 | 2,027 | 3 | 0 |
| 4 | GDN | 1,610 | 2 | 1,971 | 1 | 0 |
| 5 | GDN | 1,555 | 2 | 2,025 | 2 | 0 |
| 6 | GDN | 1,732 | 2 | 1,844 | 5 | 1 |
| 7 | FA | 1,811 | 5 | 1,765 | 3 | 0 |
| 8 | GDN | 1,842 | 1 | 1,738 | 3 | 0 |
| 9 | GDN | 1,920 | 3 | 1,658 | 3 | 0 |
| 10 | GDN | 1,963 | 4 | 1,616 | 1 | 0 |
| 11 | FA | 1,880 | 4 | 1,693 | 7 | 0 |
| 12 | GDN | 1,892 | 4 | 1,679 | 9 | 0 |
| 13 | GDN | 2,037 | 0 | 1,544 | 3 | 0 |
| 14 | GDN | 1,995 | 2 | 1,579 | 8 | 0 |
| 15 | FA | 1,718 | 3 | 1,841 | 22 | 0 |
| 16 | GDN | 1,672 | 2 | 1,890 | 20 | 0 |
| 17 | GDN | 1,591 | 4 | 1,961 | 28 | 0 |
| 18 | GDN | 1,612 | 2 | 1,946 | 24 | 0 |
| 19 | FA | 1,466 | 3 | 2,100 | 15 | 0 |
| 20 | GDN | 1,447 | 0 | 2,115 | 22 | 0 |
| 21 | GDN | 1,463 | 4 | 2,102 | 15 | 0 |
| 22 | GDN | 1,554 | 4 | 1,990 | 36 | 0 |
| 23 | FA | 1,623 | 28 | 1,860 | 71 | 2 |

Layer 23 is atypical in text, unknown, and dead counts and should be included in leave-one-FA-layer-out sensitivity analysis.

### 5.6 High-confidence FA versus GDN comparison

Threshold: `q >= 0.7`; alive neurons only.

| Type | FA count | FA ratio | GDN count | GDN ratio | Difference |
|---|---:|---:|---:|---:|---:|
| visual | 1,964 | 0.0913 | 5,415 | 0.0839 | +0.0074 |
| text | 3 | 0.0001 | 0 | 0.0000 | +0.0001 |
| multimodal | 4,587 | 0.2133 | 12,612 | 0.1955 | +0.0178 |
| unknown | 61 | 0.0028 | 113 | 0.0018 | +0.0011 |

High-confidence totals:

| Type | Count |
|---|---:|
| visual | 7,379 |
| text | 3 |
| multimodal | 17,199 |
| unknown | 174 |
| all high-confidence | 24,755 |

Approximately 28.8% of alive neurons have a q score at or above 0.7. The remaining approximately 71.2% should be treated as mixed/low-confidence rather than strongly typed.

The +1.78 percentage-point multimodal difference is currently descriptive. Significance must be evaluated at the layer/block level, not by treating all neurons as independent observations.

## 6. Historical Phase 1 results that are no longer final

The old 2k hard classification was approximately 98% visual and approximately 1.5% multimodal. The corrected result is 47.9% visual and 51.6% multimodal. Therefore:

- old hard-type distributions are invalid;
- old soft-score means are invalid;
- old FA/GDN typed proportions are invalid;
- old multimodal, unknown, and unknown-safe masks are invalid as final evidence.

The old calibration findings, float16 NaN diagnosis, token-imbalance diagnosis, `none` evaluations, and random-ablation diagnostics remain useful because they do not depend on the corrupted type-score ranking.

## 7. Corrected Phase 2 results: first q/r ablation sweep

Baseline:

| Metric | Value |
|---|---:|
| NLL | 4.7323 |
| PPL | 113.56 |

### 7.1 Full ratio sweep

| Selection score | Ratio | NLL | PPL | Delta NLL | Delta PPL |
|---|---:|---:|---:|---:|---:|
| `q_multimodal` | 5% | 5.1263 | 168.40 | +0.3940 | +54.84 |
| `q_multimodal` | 20% | 4.4064 | 81.98 | -0.3259 | -31.58 |
| `q_multimodal` | 30% | 4.0405 | 56.85 | -0.6918 | -56.71 |
| `q_multimodal` | 50% | 4.3756 | 79.49 | -0.3567 | -34.07 |
| `r_unknown` | 5% | 4.6660 | 106.27 | -0.0663 | -7.29 |
| `r_unknown` | 20% | 5.3345 | 207.36 | +0.6021 | +93.80 |
| `r_unknown` | 30% | 5.4809 | 240.07 | +0.7486 | +126.51 |
| `r_unknown` | 50% | 8.1495 | 3,461.54 | +3.4171 | +3,347.98 |
| `unknown_safe` | 5% | 4.2075 | 67.19 | -0.5248 | -46.37 |
| `unknown_safe` | 20% | 4.2220 | 68.17 | -0.5104 | -45.39 |
| `unknown_safe` | 30% | 5.4371 | 229.78 | +0.7048 | +116.22 |
| `unknown_safe` | 50% | 8.4746 | 4,791.40 | +3.7422 | +4,677.84 |
| per-layer random, current seed | 5% | 4.2896 | 72.94 | -0.4427 | -40.62 |
| per-layer random, current seed | 20% | 4.3643 | 78.59 | -0.3680 | -34.97 |
| per-layer random, current seed | 30% | 4.5743 | 96.96 | -0.1580 | -16.60 |
| per-layer random, current seed | 50% | 7.4777 | 1,768.23 | +2.7454 | +1,654.67 |

`random` and `layer_random` produced identical masks and metrics under the current per-layer selection mode, ratio, and seed. They are duplicate controls rather than independent baselines.

### 7.2 Previous random multi-seed diagnostic

These random results do not depend on neuron typing and remain useful if the evaluation setup and masks are confirmed identical:

- random 20%, five seeds: mean Delta NLL approximately -0.714, standard deviation approximately 0.235;
- random 50%, five seeds: mean Delta NLL approximately +1.674;
- random 50% seed values: +0.957, +0.906, +1.670, +2.735, +2.101;
- random 80% seed42: Delta NLL approximately +3.124.

The random 50%/80% collapse confirms that the hook and teacher-forced NLL metric can detect severe model damage.

### 7.3 Current causal interpretation

#### Multimodal-purity ranking

The cumulative `q_multimodal` curve is strongly non-monotonic:

- the top 5% is causally important on the current caption-NLL task;
- 20%–50% cumulative ablation improves NLL;
- at 50%, random ablation collapses while `q_multimodal` ablation remains better than baseline.

A plausible hypothesis is that the extreme high-purity tail contains important multimodal specialists, while a broader medium-purity multimodal subspace is redundant or overactive. This is not yet a final pruning conclusion because mask ties, nesting, held-out evaluation, and non-NLL tasks have not been verified.

#### High-r_unknown ranking

High `r_unknown` neurons are not safe noise:

- 5% has little effect;
- 20% and 30% are damaging;
- 50% is more damaging than the current random 50% mask.

High `r_unknown` should be described as `rarely-active` or `sparse-response`, not as unimportant. These neurons may encode rare concepts or specialist features.

#### Current unknown-safe score

The current score is approximately:

\[
S_{safe}=r_u-\lambda q_m-\gamma\widetilde a.
\]

It is safe at 5%–20% on the current metric but collapses at 30%–50%, and is not a validated Phase 3 method. Protecting only multimodal purity may redirect pruning toward rare visual/text specialists.

## 8. Claims currently supported

The following statements are supported as preliminary findings:

1. Corrected q/r typing produces normalized and interpretable neuron scores.
2. Neurons are predominantly visual over the full dataset, while representative high-response samples exhibit stronger multimodal purity.
3. FA layers have a descriptively higher high-confidence multimodal ratio than GDN layers by approximately 1.78 percentage points.
4. Corrected neuron scores identify subspaces with very different causal ablation curves.
5. High `r_unknown` does not identify a generally safe pruning subspace and may capture sparse specialists.
6. `q_multimodal` has a non-monotonic relationship with causal importance on the current caption-NLL task.

The following statements are not yet supported:

- FA layers have significantly more multimodal neurons;
- 51.6% of all neurons are high-confidence multimodal neurons;
- unknown neurons are noise;
- 50% multimodal neurons can be removed without general multimodal loss;
- negative Delta NLL proves universal regularization;
- the current results imply real inference speedup.

## 9. Required improvements before Phase 3

### P0: mask validity and data isolation

- [ ] Generate one deterministic full ranking per score and derive every ratio as a prefix.
- [ ] Verify strict nesting: \(M_{5}\subset M_{20}\subset M_{30}\subset M_{50}\).
- [ ] Report cutoff score, count strictly above cutoff, tie-group size, and number selected from the tie group.
- [ ] Use deterministic secondary keys, for example `r_multimodal` and neuron index after `q_multimodal`.
- [ ] Record typing and evaluation image IDs and confirm zero overlap.
- [ ] Confirm the Phase 2 evaluation split is held out from calibration and typing.

### P1: Phase 1 statistical maturity

- [ ] Compute the top-two q margin \(q_{(1)}-q_{(2)}\).
- [ ] Report exact ties and the fractions with margin below 0.05 and 0.10.
- [ ] Add an explicit mixed/low-confidence category for `max(q) < 0.7`.
- [ ] Run blocked permutation tests using layers/blocks as the statistical units.
- [ ] Report layer-level confidence intervals.
- [ ] Run leave-one-FA-layer-out analysis, especially to test sensitivity to layer 23.
- [ ] Compare corrected 500-sample and 2k-sample scores using Spearman correlation.
- [ ] Compare top-5%, 20%, 30%, and 50% masks using Jaccard overlap.
- [ ] Report visual/text top-K intersection and union-size distributions.

### P1: Phase 2 causal maturity

- [ ] Run per-layer random baselines with at least seeds 1, 2, 3, 42, and 123 at 5%, 20%, 30%, and 50%.
- [ ] Remove the duplicate `layer_random` control or implement a genuinely distinct global/matched baseline.
- [ ] Save per-example baseline NLL, ablated NLL, token count, image ID, and Delta NLL.
- [ ] Report paired bootstrap confidence intervals and the fractions of improved/damaged samples.
- [ ] Replace signed “safety ratio” with relative damage:

\[
\operatorname{RelativeDamage}=\Delta NLL_{typed}-\operatorname{mean}(\Delta NLL_{random}).
\]

- [ ] Add multimodal rank-band ablations: 0–5%, 5–20%, 20–30%, 30–50%, and 5–50%.
- [ ] Add high-r_unknown rank-band ablations using the same bands.
- [ ] Test a protected multimodal strategy that preserves the top 5% and ablates 5–X%.
- [ ] Stop treating the current `unknown_safe` score as the main method until its failure mode is understood.

### P1: task generalization

- [ ] Add held-out yes/no VQA forced-choice accuracy.
- [ ] Add POPE or a balanced COCO object-existence evaluation.
- [ ] If resources allow, add CIDEr, BLEU-4, METEOR, or SPICE for generation quality.
- [ ] Use at least caption NLL plus one non-NLL multimodal metric for pruning decisions.

### P2: pruning baselines and engineering

- [ ] Add global random and per-layer random as distinct baselines.
- [ ] Add weight-magnitude pruning.
- [ ] Add mean-activation or activation-aware pruning.
- [ ] Ensure every method removes the same number of neurons under a clearly stated layer-allocation policy.
- [ ] Add CPU unit tests for off-by-one, top-K eviction, union deduplication, q/r normalization, JSON null, score mapping, mask nesting, and tie-breaking.
- [ ] Save git commit, checkpoint, split, sample-ID hash, calibration hash, score columns, masks, seeds, token counts, and evaluation configuration with every run.
- [ ] Clean remaining Ruff warnings and stale `p_*` help text before final release.

## 10. Recommended next experiment sequence

1. Diagnose q-score ties and enforce deterministic nested masks.
2. Confirm typing/evaluation split overlap is zero.
3. Run random multi-seed baselines for every reported ratio.
4. Run multimodal and high-r_unknown band ablations.
5. Run paired per-example bootstrap analysis.
6. Complete blocked FA/GDN permutation and leave-one-layer-out analyses.
7. Compare 500 versus 2k score and mask stability.
8. Add held-out VQA/POPE evaluation.
9. Compare q/r strategies with magnitude and activation baselines.
10. Select a Phase 3 functional-pruning strategy only after these checks.

## 11. Phase 3 entry criteria

Formal Phase 3 work should begin only when:

- ratio masks are deterministic and nested;
- cutoff ties are quantified;
- typing and evaluation data are disjoint;
- random multi-seed baselines are complete;
- paired confidence intervals are available;
- the multimodal band hypothesis has been tested;
- at least one VQA/POPE metric is available;
- q/r pruning is compared against random, magnitude, and activation baselines;
- corrected 500/2k rankings are sufficiently stable.

The leading Phase 3 candidate is currently:

> Preserve the extreme top `q_multimodal` specialists and prune a medium-purity multimodal rank band, initially testing 5–20%, 5–30%, 5–40%, and 5–50% functional masks.

High `r_unknown` should not be used as the primary pruning target under the current evidence.

## 12. Working paper narrative

A concise current narrative is:

> Corrected modality-aware neuron typing separates representative high-response purity from full-dataset response frequency. In Qwen3.5-VL 0.8B, neurons are predominantly visual across ordinary samples but exhibit stronger multimodal purity among their representative high-response samples. FA layers show a modestly higher high-confidence multimodal ratio than GDN layers. Causal ablation reveals that response frequency and functional importance are not monotonic: rarely active neurons can be important sparse specialists, while the multimodal-purity ranking contains an important extreme tail and a potentially redundant broader subspace. These findings motivate rank-band, type-aware pruning rather than direct removal of an entire neuron type.
