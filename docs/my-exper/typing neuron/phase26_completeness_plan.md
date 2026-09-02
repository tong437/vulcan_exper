# Phase 2.6 Completeness Sweep

## Question

Phase 2 established that causal damage is not monotonic along the descending
`q_multimodal` ranking and that the 5--20% band is safe among the masks tested
formally. Phase 2.6 tests the remaining scope limitation:

1. Are any equally sized windows after rank 20% as safe as, or safer than, the
   frozen 5--20% band?
2. Does an equally sized mask selected by `q_visual`, `q_text`, or `q_unknown`
   satisfy the same cross-task safety criteria?

The experiment can support "best among a broader predefined candidate set".
It still cannot establish that the chosen mask is globally optimal among all
possible channel subsets.

## Fixed Budget

For the 3,584-wide FFN, the formal 5--20% mask contains:

```text
ceil(0.20 * 3584) - ceil(0.05 * 3584) = 537 neurons/layer
```

Every Phase-2.6 candidate removes exactly 537 neurons from each of 24 layers.
The orchestrator rejects the plan before model loading if any mask violates
this count. Exact rank windows avoid the 537/538 rounding differences that
would arise from independent floating-point band boundaries.

## Candidate Set

The preregistered default set is:

- `q_multimodal` 5--20% reference;
- fixed-count `q_multimodal` windows with nominal starts 20%, 25%, ..., 85%;
- top fixed-count `q_visual`;
- top fixed-count `q_text`;
- top fixed-count `q_unknown`.

The three non-multimodal masks use the explicit parquet columns. In
particular, `q_unknown` is not replaced by `r_unknown`; the latter already has
separate evidence through the low-activation-frequency baseline.

## Stage A: Screening

All candidates are evaluated on a smaller held-out slice of:

- COCO Caption teacher-forced label NLL/PPL;
- a held-out text-only SFT dataset, also using teacher-forced label NLL/PPL;
- POPE random Accuracy, F1, yes ratio, and paired uncertainty.

Default screening gates are:

| Metric | Gate |
|---|---:|
| Caption delta NLL | <= +0.05 |
| Text-only delta NLL | <= +0.05 |
| POPE random delta Accuracy | >= -1 percentage point |
| Absolute POPE yes-ratio shift | <= 0.05 |

The thresholds are written to the frozen plan before evaluation. Passing masks
are ordered by their worst normalized safety margin, and at most five advance.
Failure is still reported for causal interpretation; it is not evidence that
the corresponding neuron type does not exist.

## Stage B: Formal Evaluation

Each finalist is evaluated on:

- full held-out Caption NLL/PPL;
- full held-out text-only NLL/PPL;
- POPE random, popular, and adversarial;
- yes ratio on all three POPE splits;
- 20 common matched-random controls with the same 537 neurons per layer.

Using common controls is valid because every candidate has the same layerwise
budget. It also reduces the formal workload from `candidate_count * 20`
control masks to 20 reusable masks per task.

The final JSON reports absolute task safety separately from matched-control
comparisons. A mask is eligible for structural follow-up only when it passes
Caption, text-only, and all three POPE safety gates. Beating the matched-control
median is reported as comparative evidence but is not allowed to rescue a mask
that fails an absolute safety gate.

## Outputs

```text
phase26_completeness/
├── phase26_plan.json
├── screen/
│   ├── caption.json
│   ├── text_only.json
│   └── pope_random.json
├── screen_summary.json
├── final/
│   ├── caption.json
│   ├── text_only.json
│   ├── pope_random.json
│   ├── pope_popular.json
│   └── pope_adversarial.json
└── phase26_completeness.json
```

The included run uses the local 300-document C4 sample, packed into 512-token
blocks with all-token pretraining labels. It is independent of the Phase-1
image calibration and typing samples. A paper-scale claim should additionally
replicate the text result on a larger standard held-out language corpus.
