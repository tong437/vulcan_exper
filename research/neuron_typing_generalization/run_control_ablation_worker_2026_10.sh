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

worker="${1:-}"
case "$worker" in
  gpu1) models=(qwen3_5_2b qwen3_vl_2b_instruct) ;;
  gpu2) models=(gemma_4_e2b_it llava_1_5_7b) ;;
  *) echo "Usage: $0 {gpu1|gpu2}" >&2; exit 2 ;;
esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
statistics_root="artifacts/neuron_typing_generalization/control_response_2026_10_frozen"
output_root="artifacts/neuron_typing_generalization/control_ablation_2026_10"
replay_dir="artifacts/neuron_typing_generalization/advisor_2026_09/replay.E84Qcw"
log_root="artifacts/neuron_typing_generalization/control_ablation_2026_10/logs"
mkdir -p "$output_root" "$log_root"

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for model in "${models[@]}"; do
  python_bin="python"
  if [[ "$model" == "qwen3_5_2b" ]]; then
    python_bin="/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python"
  fi
  for dataset in mme mmbench textvqa chartqa; do
    label="${model}__${dataset}"
    if [[ -f "$output_root/$label/summary.json" ]]; then
      echo "[skip complete] $label"
      continue
    fi
    echo "[start] $label"
    "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_control_ablation_2026_10.py \
      --study "$study" \
      --model "$model" \
      --dataset "$dataset" \
      --statistics-root "$statistics_root" \
      --output-dir "$output_root/$label" \
      --replay-dir "$replay_dir" \
      --removal-percent 1 \
      --random-repeats 10 \
      2>&1 | tee -a "$log_root/$label.log"
  done
done
