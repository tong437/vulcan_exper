# Input-Dependent Neuron Typing for VLMs

This directory implements the advisor-confirmed study of sample-conditional response states in vision-language model
decoder MLPs. The historical phase1–6 code under `scripts/vulcan/neuron_typing/` remains a frozen reproducibility
baseline and is not imported here.

The normative definitions, interpretation limits, dataset comparison policies, data boundary, and reporting rules are
frozen in [`ADVISOR_GUIDANCE.md`](ADVISOR_GUIDANCE.md). When prose here is abbreviated, that guidance file takes
precedence.

## Confirmed definition

For post-gating activation `z` and a visual or user-question token set `S`, neuron `(layer, j)` has importance

```text
I(layer, j; x, S) = ||W_down[:, j]||_2 * sqrt(mean(z[layer, token, j]^2 for token in S))
```

Within each sample, layer, and modality, `q = I / sum_j(I)`. A response is strong only when
`q > 1 / d_ff(layer)`.
The visual/text strong bits encode `W=0`, `V=1`, `T=2`, and `M=3`. Equality is not strong. Empty token sets and
zero layer importance are invalid rather than silently assigned to W.

`M` means above-layer-average response at both visual and question positions. It does not by itself establish causal
cross-modal interaction. `W` means relatively weak on both sides, not zero activation.

## Study matrix

The main study contains three models and four datasets, with one seed: 12 model/dataset runs.

| Dataset | Pinned split/config | Controlled comparison | Pilot |
|---|---|---|---:|
| VQAv2 | `validation/default` | same normalized question, different image | 500 samples / 250 pairs |
| GQA Balanced | `testdev/testdev_balanced_*` | same image, different question | 500 samples / 1,000 pairs |
| CLEVR | `validation/default` | same scene, functional-program edit distance 1 | 128 samples / 64 audited pairs |
| VizWiz-VQA | `val/default` | stratified natural cohorts; no causal pairs | 500 samples / 0 pairs |

The original CLEVR `pilot` candidates remain marked `audit_status=pending` as failure evidence. Controlled analysis
uses the separately frozen `audited_v1` tier: 64 disjoint pairs holding `image_id` fixed while changing the functional
program by exactly one Levenshtein operation. VizWiz is a cohort comparison because the source has no matched
clean/degraded version of each image.

The engineering sample sizes are pilot defaults, not advisor-specified statistical targets. A later power/precision
analysis may replace them by creating a new manifest tier without overwriting these frozen manifests.

## Data boundary and current local state

`pipelines/fetch_dataset_metadata.py` downloads only pinned README/repository records and non-image question metadata.
`pipelines/build_manifests.py` deterministically selects records and refuses to overwrite a frozen manifest when its
canonical row digest changes. `pipelines/materialize_selected_images.py` then materializes unique images referenced by
the pilot manifests only.

VQAv2 reuses the machine's existing COCO validation cache. GQA uses selected Hugging Face cached assets. CLEVR and
VizWiz use bounded HTTP Range requests for selected ZIP members; the 19 GB and 3.5 GB archives are never downloaded as
complete files. Generated source metadata and images live under `artifacts/neuron_typing_generalization/` and are
ignored by Git. Small frozen Parquet manifests and their canonical SHA-256 digests belong under `manifests/` and must
be added to version control before the preparation phase is considered permanently frozen.

Audit the revision, download boundary, manifest hashes, image membership, image sizes, and image SHA-256 values:

```bash
python research/neuron_typing_generalization/pipelines/audit_cpu_ready.py
```

## CPU implementation

| Module | Responsibility |
|---|---|
| `importance.py` | weighted RMS, q normalization, strict four-way classification, mask validation |
| `analysis.py` | denominator-explicit count-based change rates, chunked maximum pair, stability, entropy, transitions |
| `tables.py` | long-form score normalization and score → class → analysis table bridge |
| `hooks.py` | decoder FFN discovery, `down_proj` pre-hooks, token spans, immediate streaming reduction |
| `adapters.py` | config-only Qwen3.5/LLaVA/Gemma layouts and processor-expanded token-role masks |
| `collection.py` | injected-model execution and immediate per-sample RMS/down-norm persistence |
| `sources.py` | size-bounded pinned metadata acquisition |
| `manifests.py` | deterministic grouping/sampling and immutable manifests |
| `materialize.py` | selected-image materialization and integrity records |
| `audit.py` | end-to-end CPU/data readiness audit |

The model-independent hook point is the input of each decoder `mlp.down_proj`, which is the gated MLP's `z`. A model
adapter must resolve expanded decoder positions into mutually exclusive visual, user-text, control, and padding masks.
Captured token activations are immediately reduced to per-sample modality RMS statistics rather than retained as raw
token-by-neuron tensors.

## Pipeline contracts

1. `collect_activations`: requires frozen sample/pair manifests; captures and streams reduced post-gating statistics.
2. `compute_importance`: verifies the activation digest and writes normalized sample/layer/neuron/modality scores.
3. `classify_neurons`: derives V/T/M/W from visual/text q values.
4. `compare_distributions`: computes controlled pair changes and cross-sample neuron profiles.
5. `evaluate`: writes overall/per-layer rates, pair-bootstrap uncertainty, both transition orientations, provenance,
   and real figures.

Schema contracts live under `schemas/`. Initializing a run automatically copies the selected immutable manifest tier
into the run directory, and refuses a conflicting pre-existing copy.

After the GPU collector writes `activations/manifest.json` and its compact sample partitions, the implemented CPU
stages can be executed directly:

```bash
python research/neuron_typing_generalization/pipelines/classify_neurons.py --run-dir /absolute/run/directory
python research/neuron_typing_generalization/pipelines/compare_distributions.py --run-dir /absolute/run/directory
python research/neuron_typing_generalization/pipelines/evaluate.py --run-dir /absolute/run/directory
```

The complete pre-GPU path can be exercised with an injected deterministic CPU model. It uses the production hook,
reduction, schemas, digests, importance normalization, classification, and analysis code without loading model weights:

```bash
python research/neuron_typing_generalization/pipelines/collect_activations.py \
  --run-dir /absolute/initialized/run --synthetic-fixture
python research/neuron_typing_generalization/pipelines/compute_importance.py \
  --run-dir /absolute/initialized/run
python research/neuron_typing_generalization/pipelines/classify_neurons.py --run-dir /absolute/initialized/run
python research/neuron_typing_generalization/pipelines/compare_distributions.py --run-dir /absolute/initialized/run
python research/neuron_typing_generalization/pipelines/evaluate.py --run-dir /absolute/initialized/run
```

The summary records both rates and their overall/per-layer changed-count and valid-comparison denominators. With invalid
observations, aggregation is `sum(changed) / sum(valid comparisons)`, never an unweighted mean of neuron-local rates.

Inspect the 12-run matrix:

```bash
python research/neuron_typing_generalization/pipelines/run_matrix.py \
  --study research/neuron_typing_generalization/configs/studies/main.yaml
```

## Three-model VQAv2 GPU launch

The GPU collector writes one compact NPZ partition per sample and stores down-projection norms once per model. A
completed partition is validated and skipped on restart, so an interrupted 500-sample run resumes without repeating
model forwards. Logical long-form row counts remain in the manifests, but the hundreds of millions of redundant rows
are never materialized.

Transformers 5.6.0 is required for the pinned Gemma-4 checkpoint and is within the repository dependency range. Run
the three-model, no-weight processor preflight before allocating a GPU:

```bash
python research/neuron_typing_generalization/pipelines/preflight_vqav2_three_models.py
```

After a GPU is visible, run one sample from each model first:

```bash
SMOKE=research/neuron_typing_generalization/configs/studies/vqav2_three_model_smoke.yaml
python research/neuron_typing_generalization/pipelines/run_vqav2_gpu.py \
  --study "$SMOKE" --model qwen3_5_2b --max-samples 1
python research/neuron_typing_generalization/pipelines/run_vqav2_gpu.py \
  --study "$SMOKE" --model gemma_4_e2b_it --max-samples 1
python research/neuron_typing_generalization/pipelines/run_vqav2_gpu.py \
  --study "$SMOKE" --model llava_1_5_7b --max-samples 1
```

Remove `--max-samples 1` to resume those same run directories through all 16 smoke samples. Only after all three
smokes pass should `SMOKE` be replaced with
`research/neuron_typing_generalization/configs/studies/vqav2_three_model_pilot.yaml` for the frozen 500-sample pilot.
Run models sequentially on a single GPU. Each command prints its absolute run directory; pass that directory to
`compute_importance.py`, `classify_neurons.py`, `compare_distributions.py`, and `evaluate.py` in that order.

Run the retained-activation versus streaming-reduction check without modifying existing activation partitions:

```bash
python research/neuron_typing_generalization/pipelines/run_vqav2_gpu.py \
  --study research/neuron_typing_generalization/configs/studies/vqav2_three_model_smoke.yaml \
  --model qwen3_5_2b --max-samples 1 --validate-direct-streaming --validation-only
```

Repeat for `gemma_4_e2b_it` and `llava_1_5_7b`. Then regenerate the frozen-pilot statistics, publication figures,
precision decision, unified Markdown/JSON report, and compact validation report:

```bash
python research/neuron_typing_generalization/pipelines/report_vqav2_pilot.py \
  --config research/neuron_typing_generalization/configs/analysis/vqav2_pilot_closure.yaml
```

Controlled confidence intervals resample the 250 frozen pair edges and recompute the ratio of summed changed counts to
summed valid counts. Transition output is an NPZ containing explicit `directed` and `symmetrized` arrays with axis order
`W,V,T,M`; it is accompanied by a JSON interpretation manifest.

## Reproduction commands

The acquisition commands are intentionally separate so image materialization cannot precede manifest freezing:

```bash
python research/neuron_typing_generalization/pipelines/fetch_dataset_metadata.py
python research/neuron_typing_generalization/pipelines/build_manifests.py
python research/neuron_typing_generalization/pipelines/materialize_selected_images.py
python research/neuron_typing_generalization/pipelines/audit_cpu_ready.py
python research/neuron_typing_generalization/pipelines/audit_clevr_pairs.py
python research/neuron_typing_generalization/pipelines/audit_data_quality.py
python research/neuron_typing_generalization/pipelines/benchmark_cpu_scale.py
python research/neuron_typing_generalization/pipelines/preflight_qwen_smoke.py
```

Run CPU tests and scoped quality checks:

```bash
WANDB_DISABLED=true pytest -q research/neuron_typing_generalization/tests
ruff check research/neuron_typing_generalization/src \
  research/neuron_typing_generalization/tests \
  research/neuron_typing_generalization/pipelines
ruff format --check research/neuron_typing_generalization/src \
  research/neuron_typing_generalization/tests \
  research/neuron_typing_generalization/pipelines
```

Do not commit raw source metadata, images, activations, model weights, checkpoints, or full per-sample run outputs.
