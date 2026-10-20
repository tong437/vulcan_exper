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
gpu="${1:-1}"
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
root="artifacts/neuron_typing_generalization/single_layer_scan_2026_10"
mkdir -p "$root/logs" "$root/locks"
export CUDA_VISIBLE_DEVICES="$gpu" WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null
common=(
 --study research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml
 --statistics-root artifacts/neuron_typing_generalization/control_response_2026_10_frozen
 --baseline-root artifacts/neuron_typing_generalization/weak_pruning_a1_a2_2026_10
 --replay-dir artifacts/neuron_typing_generalization/advisor_2026_09/replay.E84Qcw
 --layer-scan
)
for model in qwen3_vl_2b_instruct llava_1_5_7b; do
 exec {lock_fd}>"$root/locks/${model}.lock"
 flock -n "$lock_fd" || exit 1
 smoke="$root/smoke/${model}__mme"
 if [[ ! -f "$smoke/summary.json" ]]; then
  python -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
   "${common[@]}" --model "$model" --dataset mme --scan-smoke --max-samples 2 --verify-clean --output-dir "$smoke" \
   2>&1 | tee -a "$root/logs/${model}__smoke.log"
 fi
 for dataset in mme mmbench textvqa chartqa; do
  target="$root/${model}__${dataset}"
  if [[ -f "$target/summary.json" ]]; then continue; fi
  echo "[start] $model $dataset $(date -u +%FT%TZ)"
  python -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
   "${common[@]}" --model "$model" --dataset "$dataset" --output-dir "$target" \
   2>&1 | tee -a "$root/logs/${model}__${dataset}.log"
  echo "[complete] $model $dataset $(date -u +%FT%TZ)"
 done
 flock -u "$lock_fd"
 exec {lock_fd}>&-
done
echo "[all complete] $(date -u +%FT%TZ)"
