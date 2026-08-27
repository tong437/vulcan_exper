# Phase 3 / P3.4 Final Report

## Scope

This report covers zero-shot structural deletion of Qwen3.5-0.8B FFN channels
selected by the full-2k `q_multimodal` typing run. No recovery training was used.

Two structural widths were tested:

- exact frozen q-band: 3,047 channels per layer, deleting 537 per layer;
- hardware-aligned q-band: 3,072 channels per layer, deleting the exact rank
  window `rank_window:multimodal:180:512` per layer.

The aligned mask SHA-256 is
`bab85f2f277ed2e168f2510090cbf12ade3be43e639e52eb2dd21ed48ca0686b`.

## Correctness and task safety

Both structural checkpoints passed projection-slice hashes, in-memory versus
reload hashes, layer-width checks, parameter-count checks, and reload-output
checks.

For the 3,072 aligned checkpoint:

- caption NLL: hook `3.837533`, structural `3.837260`, delta `-0.000274`;
- POPE random structural/hook prediction agreement: `99.744%`;
- POPE popular structural/hook prediction agreement: `99.708%`;
- POPE adversarial structural/hook prediction agreement: `99.635%`;
- maximum structural/hook accuracy drift: `0.146` percentage point;
- maximum structural/hook yes-ratio drift: `0.183` percentage point;
- all POPE accuracy and yes-ratio safety checks against the original model passed.

Strict prediction identity and a `0.125` per-mismatch margin reference are
retained as diagnostics. They are not hard correctness gates: BF16 wide and
narrow GEMMs use shape-dependent accumulation paths, while exact kept-weight
hashes and reload outputs establish structural correctness. This gate
clarification was made after observing BF16 boundary flips and must be disclosed
with the results.

## Compression

| Width | Removed parameters | Full-model reduction | Parameter storage | Checkpoint reduction | Peak-memory reduction |
|---:|---:|---:|---:|---:|---:|
| 3,047 | 39,591,936 | 4.642% | 75.52 MiB | 117.42 MiB | about 74--75 MiB |
| 3,072 | 37,748,736 | 4.425% | 72.00 MiB | 113.89 MiB | about 70--71 MiB |

Checkpoint reduction exceeds parameter-storage reduction because the saved
checkpoint also removes a tied-weight serialization duplicate. That extra
reduction must not be attributed to pruning.

## Corrected latency benchmark

Environment: NVIDIA RTX 4090, BF16, PyTorch 2.5.1 + CUDA 12.4, torch SDPA,
10 warmups, 50 measurements, 2,000 bootstrap samples, and fixed 64-token
generation. The benchmark validates actual generated-token counts and includes
multimodal and text-only prefill.

Positive percentages below mean lower latency; negative percentages mean a
slowdown relative to the original model.

| Scenario | 3,047 | 3,072 | 3,072 speedup CI excludes 1? |
|---|---:|---:|---|
| Multimodal prefill, batch 1 | -1.102% | -0.333% | yes, significant slowdown |
| Multimodal prefill, batch 4 | -0.332% | -0.401% | yes, significant slowdown |
| Text-only prefill, batch 1 | -1.128% | -0.441% | yes, significant slowdown |
| Text-only prefill, batch 4 | -0.274% | -0.645% | yes, significant slowdown |
| TTFT | -0.655% | +0.369% | yes, stable improvement |
| Fixed-64-token total latency | -0.773% | +0.128% | yes, stable improvement |
| Decode throughput | -0.777% | +0.106% | no, CI crosses 1 |

The 3,072 alignment improves generation relative to 3,047 and changes TTFT and
total generation from regressions to small stable improvements. It does not
pass the engineering gate because both primary multimodal prefill scenarios
remain significantly slower.

## Final conclusion

Phase 3 succeeds as structural compression and task-safe parameter deletion.
P3.4 demonstrates that hardware alignment partially improves execution, but
neither width provides an overall end-to-end engineering speedup under the
tested backend. The defensible conclusion is:

> q/r neuron typing identifies a fixed visual-functional FFN mask that can be
> physically removed while preserving caption and object-recognition safety,
> reducing parameters, checkpoint size, and peak memory. On the tested RTX 4090
> torch backend, these reductions do not translate into an across-scenario
> latency improvement; 3,072 alignment helps generation but not prefill.

P3.5 recovery training is not indicated because task safety already passes.
Further latency work should target compilation/kernel/backend behavior rather
than changing the mask or using recovery training. The current run used the
torch fallback for the model's fast linear-attention path, so performance claims
are specific to this execution stack.

## Result artifacts

- Exact q-band quality gate:
  `saves/neuron_typing/phase3_structural_qband/evaluation/quality_gate.json`
- Exact q-band corrected benchmark:
  `saves/neuron_typing/phase34_aligned_3072/benchmark/unaligned_3047_summary.json`
- Aligned hook gate:
  `saves/neuron_typing/phase34_aligned_3072/hook_evaluation/quality_gate.json`
- Aligned structural gate:
  `saves/neuron_typing/phase34_aligned_3072/structural_evaluation/quality_gate.json`
- Aligned corrected benchmark:
  `saves/neuron_typing/phase34_aligned_3072/benchmark/summary.json`
