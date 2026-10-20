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
gpu="${1:?GPU required}"
shift
[[ "$gpu" =~ ^[0-9]+$ && "$#" -gt 0 ]] || exit 2
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
output_root="artifacts/neuron_typing_generalization/nonuniform_budget_2026_10"
mkdir -p "$output_root/logs" "$output_root/locks"
export CUDA_VISIBLE_DEVICES="$gpu" WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common=(
  --study research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml
  --statistics-root artifacts/neuron_typing_generalization/control_response_2026_10_frozen
  --baseline-root artifacts/neuron_typing_generalization/weak_pruning_a1_a2_2026_10
  --replay-dir artifacts/neuron_typing_generalization/advisor_2026_09/replay.E84Qcw
  --nonuniform
)
for model in "$@"; do
  case "$model" in
    llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it) ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
  esac
  exec {lock_fd}>"$output_root/locks/${model}.lock"
  flock -n "$lock_fd" || exit 1
  python_bin="python"
  if [[ "$model" == "qwen3_5_2b" ]]; then
    python_bin="/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python"
  fi
  for epsilon in 0.00 0.30; do
    tag="${epsilon/./p}"
    target="$output_root/smoke/${model}__mme__eps_${tag}"
    if [[ ! -f "$target/summary.json" ]]; then
      echo "[smoke] $model epsilon=$epsilon $(date -u +%FT%TZ)"
      "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
        "${common[@]}" --model "$model" --dataset mme --epsilon "$epsilon" \
        --max-samples 2 --verify-clean --output-dir "$target" \
        2>&1 | tee -a "$output_root/logs/${model}__smoke_${tag}.log"
    fi
  done
  for dataset in mme mmbench textvqa chartqa; do
    for epsilon in 0.05 0.10 0.15 0.20 0.25 0.30; do
      tag="${epsilon/./p}"
      label="${model}__${dataset}__eps_${tag}"
      target="$output_root/$label"
      if [[ -f "$target/summary.json" ]]; then
        echo "[skip complete] $label"
        continue
      fi
      echo "[start] $label GPU=$gpu $(date -u +%FT%TZ)"
      "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
        "${common[@]}" --model "$model" --dataset "$dataset" --epsilon "$epsilon" --output-dir "$target" \
        2>&1 | tee -a "$output_root/logs/${label}.log"
      echo "[complete] $label $(date -u +%FT%TZ)"
    done
  done
  flock -u "$lock_fd"
  exec {lock_fd}>&-
done
echo "[all complete] $(date -u +%FT%TZ)"
