#!/usr/bin/env bash
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

# The artifact tree is shared by multiple AutoDL containers whose root users
# may be mapped to different identities by the storage backend.  Keep files
# created by this launcher group/world writable so a run can be resumed from
# another container without copying or changing ownership.
umask 000

phase="${1:-}"
if [[ "$phase" != "smoke" && "$phase" != "full" && "$phase" != "report" && "$phase" != "cluster" && "$phase" != "pairs" && "$phase" != "layers" && "$phase" != "ablation-smoke" && "$phase" != "ablation-full" && "$phase" != "ablation-report" ]]; then
  echo "Usage: $0 {smoke|full|report|cluster|pairs|layers|ablation-smoke|ablation-full|ablation-report}" >&2
  exit 2
fi

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
cd "$repository_root"

study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
run_root="$repository_root/artifacts/neuron_typing_generalization/runs/advisor_2026_09_main"
log_root="$repository_root/artifacts/neuron_typing_generalization/advisor_2026_09/logs"
result_root="$repository_root/research/neuron_typing_generalization/reports/advisor_2026_09"
ablation_root="$repository_root/artifacts/neuron_typing_generalization/ablation/advisor_2026_09_1pct"
mkdir -p "$log_root"
mkdir -p "$ablation_root"

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

models=(llava_1_5_7b qwen3_vl_2b_instruct qwen3_5_2b gemma_4_e2b_it)
datasets=(mme mmbench textvqa chartqa)
ablation_datasets=(mme mmbench textvqa chartqa)

resolve_replay_dir() {
  local matches=("$repository_root"/artifacts/neuron_typing_generalization/advisor_2026_09/replay.*)
  if [[ "${#matches[@]}" -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Expected exactly one frozen replay directory; found ${#matches[@]}." >&2
    exit 1
  fi
  echo "${matches[0]}"
}

require_cuda() {
  python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable on this host.")
print("CUDA ready:", torch.cuda.get_device_name(0), "VRAM bytes:", torch.cuda.get_device_properties(0).total_memory)
PY
}

resolve_run_dir() {
  local model="$1"
  local dataset="$2"
  python - "$study" "$model" "$dataset" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "research/neuron_typing_generalization/src")
from neuron_typing_generalization.config import expand_study, load_study
from neuron_typing_generalization.runtime import build_run_id, resolve_artifact_root

study_path, model, dataset = sys.argv[1:]
matches = [
    spec
    for spec in expand_study(load_study(study_path))
    if spec.model.name == model and spec.dataset.name == dataset
]
if len(matches) != 1:
    raise SystemExit(f"Expected one run, found {len(matches)}")
spec = matches[0]
print(resolve_artifact_root(spec) / "runs" / spec.study / build_run_id(spec))
PY
}

if [[ "$phase" == "smoke" ]]; then
  require_cuda
  for model in "${models[@]}"; do
    echo "[smoke] $model x mme"
    python research/neuron_typing_generalization/pipelines/run_gpu.py \
      --study "$study" \
      --model "$model" \
      --dataset mme \
      --max-samples 1 \
      --validate-direct-streaming \
      --validation-only \
      2>&1 | tee "$log_root/smoke_${model}.log"
  done
  echo "All four model smoke validations passed."
  exit 0
fi

if [[ "$phase" == "cluster" ]]; then
  python research/neuron_typing_generalization/pipelines/cluster_advisor_2026_09.py \
    --study "$study" \
    --output-dir "$result_root/hellinger_clustering_pilot" \
    --maximum-neurons 4096 \
    --threshold 0.20 \
    --seed 42
  echo "Clustering pilot: $result_root/hellinger_clustering_pilot/advisor_2026_09_hellinger_clustering.md"
  exit 0
fi

if [[ "$phase" == "pairs" ]]; then
  python research/neuron_typing_generalization/pipelines/report_advisor_state_pairs_2026_09.py \
    --study "$study" \
    --output-dir "$result_root"
  echo "State-pair report: $result_root/advisor_2026_09_state_pairs.md"
  exit 0
fi

if [[ "$phase" == "layers" ]]; then
  python research/neuron_typing_generalization/pipelines/report_advisor_layerwise_states_2026_09.py \
    --study "$study" \
    --output-dir "$result_root"
  echo "Layerwise state report: $result_root/advisor_2026_09_layerwise_states.md"
  exit 0
fi

if [[ "$phase" == "ablation-smoke" ]]; then
  require_cuda
  replay_dir="$(resolve_replay_dir)"
  python research/neuron_typing_generalization/pipelines/run_advisor_ablation_gpu_2026_09.py \
    --study "$study" \
    --model qwen3_5_2b \
    --dataset mme \
    --output-dir "$ablation_root/smoke_qwen3_5_2b__mme" \
    --replay-dir "$replay_dir" \
    --removal-percent 1 \
    --random-repeats 2 \
    --max-samples 2 \
    2>&1 | tee "$log_root/ablation_smoke.log"
  exit 0
fi

if [[ "$phase" == "ablation-full" ]]; then
  require_cuda
  replay_dir="$(resolve_replay_dir)"
  for model in "${models[@]}"; do
    for dataset in "${ablation_datasets[@]}"; do
      label="${model}__${dataset}"
      echo "[ablation] $label"
      python research/neuron_typing_generalization/pipelines/run_advisor_ablation_gpu_2026_09.py \
        --study "$study" \
        --model "$model" \
        --dataset "$dataset" \
        --output-dir "$ablation_root/$label" \
        --replay-dir "$replay_dir" \
        --removal-percent 1 \
        --random-repeats 10 \
        2>&1 | tee "$log_root/ablation_${label}.log"
    done
  done
  exit 0
fi

if [[ "$phase" == "ablation-report" ]]; then
  python research/neuron_typing_generalization/pipelines/report_advisor_ablation_2026_09.py \
    --input-dir "$ablation_root" \
    --output-dir "$result_root"
  exit 0
fi

if [[ "$phase" == "full" ]]; then
  require_cuda
  for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
      label="${model}__${dataset}"
      echo "[collect] $label"
      python research/neuron_typing_generalization/pipelines/run_gpu.py \
        --study "$study" \
        --model "$model" \
        --dataset "$dataset" \
        2>&1 | tee "$log_root/collect_${label}.log"

      run_dir="$(resolve_run_dir "$model" "$dataset")"
      echo "[analyze] $run_dir"
      python research/neuron_typing_generalization/pipelines/compute_importance.py --run-dir "$run_dir" \
        2>&1 | tee "$log_root/importance_${label}.log"
      python research/neuron_typing_generalization/pipelines/classify_neurons.py --run-dir "$run_dir" \
        2>&1 | tee "$log_root/classify_${label}.log"
      python research/neuron_typing_generalization/pipelines/compare_distributions.py --run-dir "$run_dir" \
        2>&1 | tee "$log_root/compare_${label}.log"
      python research/neuron_typing_generalization/pipelines/evaluate.py --run-dir "$run_dir" \
        2>&1 | tee "$log_root/evaluate_${label}.log"
    done
  done
fi

python research/neuron_typing_generalization/pipelines/report_advisor_2026_09.py \
  --study "$study" \
  --output-dir "$result_root"
echo "Final table: $result_root/advisor_2026_09_rates.md"
echo "Interpretability report: $result_root/advisor_2026_09_interpretability.md"
