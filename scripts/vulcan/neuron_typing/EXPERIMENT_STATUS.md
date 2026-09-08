# Qwen3.5-VL Neuron Typing: Phase 1--4 Status

> Last updated: 2026-09-05
> Source of truth: saved formal artifacts under `saves/neuron_typing/`
> Main line: continuous neuron typing -> causal ablation -> structural deletion -> visual-to-text activation mapping

## 1. Research question

This project asks whether modality-related structure in Qwen3.5-VL-0.8B FFN
channels can support safe structural compression, and whether image-induced
text-side FFN changes reveal an additional causal protection signal.

The current evidence supports two complementary signals:

- `q_multimodal` identifies a safe medium-rank deletion band;
- `mapping_signal` identifies a compact high-rank causal protection core.

The intended joint rule is therefore:

```text
q 5--20% rank band       -> deletion candidates
mapping top-1%           -> protected causal core
```

Low activation frequency, low mapping signal, and an `unknown` label must not
be interpreted as evidence that a neuron is safe to remove.

## 2. Formal setup

| Item | Value |
|---|---:|
| Model | Qwen3.5-VL-0.8B |
| Decoder layers | 24 |
| FFN width | 3,584 per layer |
| Total FFN channels | 86,016 |
| Full Attention layers | 3, 7, 11, 15, 19, 23 |
| GatedDeltaNet layers | remaining 18 layers |
| Phase-1 calibration samples | 500 |
| Phase-1 typing samples | 2,000 |
| Visual/text thresholds | per-neuron q97 / q95 |
| Primary Phase-2 metrics | Caption label NLL; POPE Accuracy/F1/yes ratio |

Formal calibration, typing, Caption evaluation, POPE evaluation, and Phase-4
train/validation/test image sets are isolated by saved manifests. Historical
results produced before the cache and image-overlap audit are diagnostics only.

## 3. Phase 1: continuous modality selectivity

### Status: complete

The authoritative score file is:

```text
saves/neuron_typing/phase1_clean_2k/scores/neuron_type_scores.parquet
```

The formal artifact contains 86,016 rows:

| Dominant label | Count |
|---|---:|
| multimodal | 54,767 |
| visual | 31,113 |
| unknown | 67 |
| text | 66 |
| dead | 3 |

Dominant labels are descriptive `argmax(q)` labels, not discrete functional
classes. Only 8,478 alive channels have `max(q) >= 0.7`; approximately 90.14%
are mixed/low-confidence. The principal Phase-1 contribution is the separation
of:

- `q`: modality purity among representative high-response samples;
- `r`: response frequency over the complete typing set.

The FA-versus-GDN enrichment hypothesis is not statistically supported. For
high-confidence multimodal channels the observed difference is +0.587
percentage point, but exact blocked permutation gives `p=0.8125`.

The 500-versus-2,000 comparison gives global `q_multimodal` Spearman 0.8182,
but the exact 5--20% band Jaccard is only 0.2641. All downstream masks must use
the frozen full-2,000 ranking.

## 4. Phase 2: held-out causal ablation

### Status: complete for the frozen q-band

The causal effect of `q_multimodal` is non-monotonic:

- the extreme top 0--5% contains important specialists;
- the per-layer 5--20% rank band is a safe deletion candidate;
- high `r_unknown` and low activation frequency are not safe deletion rules.

The frozen q-band removes 537 channels per layer, 12,888 in total
(14.983% of decoder FFN channels).

On 500 held-out Caption examples:

| Mask | NLL | Delta NLL |
|---|---:|---:|
| Original | 4.106730 | 0 |
| q 5--20% band | 3.974039 | -0.132691 |
| Matched random | 4.238996 | +0.132266 |
| Lowest magnitude | 3.974513 | -0.132217 |
| Lowest activation frequency | 4.298968 | +0.192238 |

Across POPE random/popular/adversarial, q-band Accuracy drops are only
0.69/0.40/0.62 percentage point. Matched random is worse, while magnitude and
lowest-activation masks lose roughly 9--18 points and introduce large answer
biases. Caption NLL alone is therefore not an adequate pruning safety metric.

## 5. Phase 3: structural deletion

### Status: complete

Both the exact 3,047-width q-band checkpoint and the 3,072-width
hardware-aligned checkpoint pass structural correctness and formal Caption plus
three-split POPE quality gates.

The three authoritative quality gates all contain `passed: true`:

```text
saves/neuron_typing/phase3_structural_qband/evaluation/quality_gate.json
saves/neuron_typing/phase34_aligned_3072/hook_evaluation/quality_gate.json
saves/neuron_typing/phase34_aligned_3072/structural_evaluation/quality_gate.json
```

| Width | Removed parameters | Full-model reduction | Checkpoint reduction |
|---:|---:|---:|---:|
| 3,047 | 39,591,936 | 4.642% | 117.42 MiB |
| 3,072 | 37,748,736 | 4.425% | 113.89 MiB |

The result establishes real parameter, checkpoint, and peak-memory reduction.
It does not establish overall inference acceleration on RTX 4090 + torch SDPA:
the aligned model slightly improves TTFT and fixed-length generation, but the
primary multimodal prefill scenarios remain slightly slower.

## 6. Phase 4: visual-to-text activation mapping

### Status: primary mapping and 100-control top-1% confirmation complete

Phase 4 predicts the image-induced text-side FFN change

```text
Delta A_l = A_l_text(correct image, question)
          - A_l_text(shuffled image, same question)
```

from decoder-entry visual and question representations. The formal mapping
artifact is complete and passes both pre-registered model gates:

| Metric | Result | Gate |
|---|---:|---:|
| Mean V+Q test R2 | 0.10797 | >= 0.01 |
| Mean Q-only test R2 | -0.01401 | descriptive |
| Mean incremental R2 | 0.12198 | >= 0.002 |
| Aggregate permutation p | 0.04762 | <= 0.05 |
| Layers with positive R2 | 24/24 | >= 4 |

`mapping_signal` and `q_multimodal` have weak global Spearman correlation
(0.1505), so they provide largely complementary information.

The 15% mapping-high group is more damaging than matched random, but the
combined-low deletion proposal fails its safety gate. Mapping signal is a
protection prior, not an invertible deletion score.

P4.4a tests nested mapping-high masks at 1%, 2.5%, 5%, 10%, and 15%. The broad
ranking fails the pre-registered monotonicity and 3-of-5 enrichment gates. The
clean result is restricted to top-1% (36 per layer; 864 total):

- POPE-random Accuracy delta: -3.14 percentage points;
- image-bootstrap 95% CI: [-5.31, -1.21] points;
- stronger than all 20 global matched-random controls;
- stronger than all 20 q-stratified matched-random controls;
- empirical `p=1/21=0.0476` for both comparisons;
- effect direction agrees across all three POPE splits.

The 100-seed confirmatory test is complete. It exactly reuses the first 20
controls after dataset-order and mask-tensor equivalence checks and evaluates 80
new controls per family. On POPE-random:

| Condition | Accuracy delta |
|---|---:|
| mapping top-1% | -3.140 points |
| 100 global random, mean | -0.244 point |
| 100 q-stratified random, mean | -0.297 point |

The mapping set is more damaging than all 100 global controls
(`p=1/101=0.0099`) and than 98 of 100 q-stratified controls
(`p=3/101=0.0297`). The confirmatory causal-enrichment gate passes. The effect
also remains directionally consistent on POPE popular (-2.657 points) and
adversarial (-1.932 points). This confirms a compact causal protection core;
it still does not justify deleting low-mapping neurons.

## 7. NeuPAT extension

NeuPAT is an external validation/optional extension, not the Phase 1--4 main
line. Its `language U shared` set is stable and causally language-sensitive,
but the previous small VQA-RAD SFT produced no language-forgetting regime and
therefore could not test comparative retention.

The prepared leakage-safe COCO + VQA-Med data asset is retained for later task
generalization. A full NeuPAT/LoRA/vanilla SFT matrix is deferred until the
Phase 1--4 deletion-plus-protection experiment is complete.

## 8. Active execution order

### Completed confirmatory and joint-mask evaluation

1. The 100 global and 100 q-stratified top-1% confirmation is complete under
   `saves/neuron_typing/phase44a_top1_controls100/`; its causal-enrichment gate
   passes.
2. Six four-layer blocks, FA/GDN, FA excluding layer 23, layer 23, and all-layer
   localization is complete under
   `saves/neuron_typing/phase4_top1_localization/`.
3. The equal-budget joint mask is frozen under
   `saves/neuron_typing/phase4_joint_qband_mapping_top1/`. It starts from the
   exact Phase-3 q 5--20% mask, protects mapping top-1%, and refills only from
   the immediately lower same-layer q ranks.

The joint construction changes 147 of 12,888 deletion choices while retaining
exactly 537 deletions per layer:

| Joint-mask quantity | Count |
|---|---:|
| Original q-band deletions | 12,888 |
| Mapping top-1% protected set | 864 |
| Protected neurons overlapping q-band | 147 |
| Same-layer adjacent-q refills | 147 |
| Final joint deletions | 12,888 |

Static verification confirms exact reproduction of the frozen Phase-3 mask,
equal per-layer budgets, protection/deletion disjointness, and same-layer
refill. Hook evaluation accepts only two equal-budget conditions: original
q-band and joint mapping-protected q-band.

The joint hook safety gate passes Caption and all three full held-out POPE
splits. Relative to the unablated model:

| Task | Original q-band | Joint mask |
|---|---:|---:|
| Caption delta NLL | -0.13161 | -0.18645 |
| POPE random Accuracy drop | 0.753 point | 0.430 point |
| POPE popular Accuracy drop | 0.430 point | -0.036 point |
| POPE adversarial Accuracy drop | 0.538 point | 0.215 point |

The joint mask is non-inferior to the original q-band on every endpoint. Its
Caption NLL is lower than the original q-band by 0.05485 with paired 95% CI
[-0.06412, -0.04510]. Its POPE Accuracy is higher by 0.323, 0.466, and 0.323
point on random, popular, and adversarial respectively. These task improvements
are supportive; the primary conclusion is that protecting 147 overlapping
mapping neurons and refilling the exact budget does not reduce hook safety.

The top-1% localization result is distributed rather than attributable to one
four-layer block. Mapping-mask Accuracy deltas in percentage points are:

| Group | Neurons | Random | Popular | Adversarial |
|---|---:|---:|---:|---:|
| Block 1 | 144 | -0.242 | -0.725 | +0.242 |
| Block 2 | 144 | -0.242 | -0.483 | +0.483 |
| Block 3 | 144 | -0.483 | -0.966 | 0.000 |
| Block 4 | 144 | -0.483 | +0.483 | +0.966 |
| Block 5 | 144 | +0.483 | -0.242 | -0.242 |
| Block 6 | 144 | 0.000 | -0.242 | +0.242 |
| FA | 216 | -0.725 | +0.483 | +1.208 |
| GDN | 648 | -1.449 | -1.691 | -0.966 |
| Layer 23 | 36 | 0.000 | 0.000 | -0.242 |
| All layers | 864 | -3.140 | -2.657 | -1.932 |

GDN is the only coarse partition with a consistently harmful direction across
all three splits, whereas layer 23 is negligible and does not explain the FA
partition. Individual block confidence intervals mostly include zero. The
all-layer damage is also more negative than the sum of separately ablated
blocks, especially on random and adversarial, indicating cross-layer
non-additivity. The defensible interpretation is a distributed protection core
with a GDN-leaning aggregate effect, not a uniquely causal block or layer.

### Extended gate result and structural decision

The C4 and VQA-Med hook evaluations are complete. The aggregate result is
stored in
`saves/neuron_typing/phase4_joint_qband_mapping_top1/gates/quality_gate.json`
and reports `passed: false` and `structural_checkpoint_allowed: false`.

| Task | Baseline NLL | Original q-band delta | Joint-mask delta | Joint minus q-band |
|---|---:|---:|---:|---:|
| C4 text-only, 500 packed blocks | 3.36422 | +0.46254 | +0.44032 | -0.02222 |
| VQA-Med, 1,501 examples | 6.49539 | +0.64982 | +0.71520 | +0.06539 |

The frozen absolute-safety threshold was delta NLL <= 0.05, with a joint-mask
non-inferiority margin of 0.02 relative to the original q-band. The joint mask
fails absolute safety on both extended endpoints. It modestly improves C4 over
the original q-band, but is also worse than the q-band on VQA-Med and therefore
fails that non-inferiority check. Its paired 95% CI for VQA-Med delta NLL versus
the unablated model is [0.66057, 0.77101]; the failure is not a borderline
sampling result.

Consequently, no new structural checkpoint is generated from this candidate.
The earlier Caption/POPE pass is a domain-local safety result and must not be
reported as general language or cross-domain multimodal safety. The extended
gate instead rejects the q 5--20% deletion budget as a universal safe-pruning
candidate. Mapping top-1% protection recovers a small amount of C4 loss, but
protecting only the 147 q-band overlaps is insufficient and does not generalize
to VQA-Med.

VQA-Med is measured by teacher-forced answer NLL under the hook mask; this is a
retention gate, not a generated-answer EM claim. Low mapping score must never
be used as the refill criterion. The next Phase-4 iteration should reduce the
deletion dose and/or protect text-important NeuPAT `language U shared` neurons,
then rerun the same hook gate before any structural build.

### Reduced-dose NeuPAT protection follow-up

That follow-up is complete under
`saves/neuron_typing/phase4_protected_dose_sweep/`. Four equal-budget deletion
doses were constructed from q ranks beginning at 5%: 1%, 2.5%, 5%, and 10%.
For every dose, the primary protected condition excludes both the causally
validated NeuPAT `language U shared` set and mapping top-1%, then refills from
lower same-layer q ranks. It is compared with an equal-budget q-only condition;
mapping-only is retained as an exploratory mechanism control.

The protection set contains 61,961 neurons: 61,896 NeuPAT language/shared
neurons plus only 65 mapping-top-1% neurons not already in that union. At the
1% dose, 617 of 864 original q-band choices are protected and replaced. All
candidates have exact equal per-layer budgets and are disjoint from their
declared protection sets.

The frozen C4 screen gives:

| Deletion dose | q-only delta NLL | NeuPAT + mapping delta NLL | Protected minus q-only | C4 pass |
|---|---:|---:|---:|---:|
| 1% | +0.02154 | +0.01376 | -0.00778 | yes |
| 2.5% | +0.05456 | +0.03805 | -0.01650 | yes |
| 5% | +0.12276 | +0.08680 | -0.03596 | no |
| 10% | +0.29533 | +0.17657 | -0.11876 | no |

The largest C4-safe dose, 2.5%, failed Caption: protected delta NLL was
+0.11808 versus +0.01508 for q-only. The hierarchy therefore fell back to the
1% dose. Its formal results are:

| Endpoint | q-only | mapping-only | NeuPAT + mapping | Protected absolute gate |
|---|---:|---:|---:|---:|
| Caption delta NLL | +0.09457 | +0.10390 | +0.01081 | pass |
| C4 delta NLL | +0.02154 | not run | +0.01376 | pass |
| VQA-Med delta NLL | +0.28569 | +0.32392 | -0.13788 | pass |
| POPE random Accuracy delta | -1.219 points | -1.183 points | +0.215 points | pass |
| POPE popular Accuracy delta | -0.215 points | -0.179 points | -0.502 points | pass |
| POPE adversarial Accuracy delta | +0.430 points | +0.502 points | -0.753 points | pass |

The 1% protected candidate passes every absolute task-safety check, including
the 1-point POPE accuracy and 5-point yes-ratio limits. It nevertheless fails
the preregistered POPE adversarial non-inferiority comparison: its accuracy
drop relative to q-only is 1.183 points, above the 0.5-point margin. The
aggregate result at
`saves/neuron_typing/phase4_protected_dose_sweep/gates/formal_p010/quality_gate.json`
therefore reports `passed: false` and `structural_checkpoint_allowed: false`.

This is positive mechanistic evidence for NeuPAT protection, not authorization
for pruning. At equal 1% budget it changes failing Caption, VQA-Med, and POPE
random q-only masks into safe masks, while mapping-only does not. However, the
POPE adversarial trade-off shows that language protection alone does not
uniformly preserve every multimodal behavior. No structural checkpoint is
generated. Any further pruning iteration must protect multimodal-sensitive
roles as well and must be treated as exploratory until validated on a fresh
held-out multimodal endpoint.

## 9. Canonical reports

- `docs/my-exper/typing neuron/phase1_to_phase4_advisor_report.md`
- `docs/my-exper/typing neuron/phase3_phase34_final_report.md`
- `docs/my-exper/typing neuron/phase4_activation_mapping_plan.md`
- `docs/my-exper/typing neuron/phase44a_causal_dose_plan.md`
