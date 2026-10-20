#!/usr/bin/env bash
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

gpu="${1:-}"
model="${2:-}"
case "$gpu" in
  0|1|2|3) ;;
  *) echo "Usage: $0 {0|1|2|3} {model}" >&2; exit 2 ;;
esac
case "$model" in
  llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it) ;;
  *) echo "Unknown model: $model" >&2; exit 2 ;;
esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
if ! nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
  echo "GPU $gpu is unavailable; no adaptive-pruning job was started." >&2
  exit 1
fi

study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
statistics_root="artifacts/neuron_typing_generalization/control_response_2026_10_frozen"
baseline_root="artifacts/neuron_typing_generalization/weak_pruning_a1_a2_2026_10"
output_root="artifacts/neuron_typing_generalization/adaptive_pruning_a1_a2_2026_10"
replay_dir="artifacts/neuron_typing_generalization/advisor_2026_09/replay.E84Qcw"
log_root="$output_root/logs"
mkdir -p "$output_root" "$log_root"

export CUDA_VISIBLE_DEVICES="$gpu"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python_bin="python"
if [[ "$model" == "qwen3_5_2b" ]]; then
  python_bin="/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python"
fi

smoke_dir="$output_root/smoke/${model}__mme"
if [[ ! -f "$smoke_dir/summary.json" ]]; then
  "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
    --study "$study" --model "$model" --dataset mme \
    --statistics-root "$statistics_root" --baseline-root "$baseline_root" \
    --output-dir "$smoke_dir" --replay-dir "$replay_dir" --max-samples 2 \
    2>&1 | tee -a "$log_root/${model}__smoke.log"
fi

for dataset in mme mmbench textvqa chartqa; do
  label="${model}__${dataset}"
  if [[ -f "$output_root/$label/summary.json" ]]; then
    echo "[skip complete] $label"
    continue
  fi
  "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
    --study "$study" --model "$model" --dataset "$dataset" \
    --statistics-root "$statistics_root" --baseline-root "$baseline_root" \
    --output-dir "$output_root/$label" --replay-dir "$replay_dir" \
    2>&1 | tee -a "$log_root/$label.log"
done
