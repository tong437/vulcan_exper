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

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
cd "$repository_root"

output_root="$repository_root/artifacts/neuron_typing_generalization/ablation/advisor_2026_09_5pct"
worker="$experiment_root/run_advisor_ablation_5pct_model_2026_09.sh"
interval_seconds=1800

while true; do
  if [[ -f "$output_root/qwen3_5_2b__mme/summary.json" \
    && -f "$output_root/qwen3_5_2b__mmbench/summary.json" \
    && -f "$output_root/qwen3_5_2b__textvqa/summary.json" \
    && -f "$output_root/qwen3_5_2b__chartqa/summary.json" ]]; then
    echo "[watch] Qwen3.5 5% is already complete; exiting."
    exit 0
  fi

  if pgrep -af 'run_advisor_ablation_gpu_2026_09.py' | grep -Fq -- '--model qwen3_5_2b'; then
    echo "[watch] A Qwen3.5 ablation worker is already running; exiting to avoid duplication."
    exit 0
  fi

  if gpu_status="$(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"; then
    IFS=, read -r used_mib utilization_percent <<< "$gpu_status"
    used_mib="${used_mib//[[:space:]]/}"
    utilization_percent="${utilization_percent//[[:space:]]/}"
    if [[ "$used_mib" =~ ^[0-9]+$ && "$utilization_percent" =~ ^[0-9]+$ ]]; then
      if (( used_mib <= 2048 && utilization_percent <= 20 )); then
        echo "[watch] GPU 1 is free (${used_mib} MiB, ${utilization_percent}%); starting Qwen3.5 5%."
        exec env CUDA_VISIBLE_DEVICES=1 bash "$worker" qwen3_5_2b
      fi
      echo "[watch] GPU 1 still occupied (${used_mib} MiB, ${utilization_percent}%); checking again in 30 minutes."
    else
      echo "[watch] Could not parse GPU 1 status: $gpu_status; checking again in 30 minutes."
    fi
  else
    echo "[watch] Could not query GPU 1; checking again in 30 minutes."
  fi
  sleep "$interval_seconds"
done
