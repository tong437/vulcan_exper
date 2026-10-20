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
shift || true
if [[ ! "$gpu" =~ ^[0-9]+$ || "$#" -lt 1 ]]; then
  echo "Usage: bash $0 GPU MODEL [MODEL ...]" >&2
  exit 2
fi
if ! nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
  echo "GPU $gpu is unavailable." >&2
  exit 1
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
statistics_root="artifacts/neuron_typing_generalization/control_response_2026_10_frozen"
baseline_root="artifacts/neuron_typing_generalization/weak_pruning_a1_a2_2026_10"
prior_root="artifacts/neuron_typing_generalization/adaptive_pruning_a1_a2_2026_10"
output_root="artifacts/neuron_typing_generalization/adaptive_pruning_epsilon_sweep_2026_10"
replay_dir="artifacts/neuron_typing_generalization/advisor_2026_09/replay.E84Qcw"
mkdir -p "$output_root/logs" "$output_root/locks"

export CUDA_VISIBLE_DEVICES="$gpu"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for model in "$@"; do
  case "$model" in
    llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it) ;;
    *) echo "Unknown model: $model" >&2; exit 2 ;;
  esac
  exec {lock_fd}>"$output_root/locks/${model}.lock"
  if ! flock -n "$lock_fd"; then
    echo "Another epsilon-sweep worker already owns $model." >&2
    exit 1
  fi
  python_bin="python"
  if [[ "$model" == "qwen3_5_2b" ]]; then
    python_bin="/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python"
  fi
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python interpreter unavailable: $python_bin" >&2
    exit 1
  fi
  for dataset in mme mmbench textvqa chartqa; do
    prior="$prior_root/${model}__${dataset}/summary.json"
    if [[ ! -f "$prior" ]]; then
      echo "Missing completed epsilon=0.05 result: $prior" >&2
      exit 1
    fi
  done

  smoke="$output_root/smoke/${model}__mme__eps_0p10"
  if [[ ! -f "$smoke/summary.json" ]]; then
    echo "[smoke] $model, epsilon=0.10, GPU $gpu"
    "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
      --study "$study" --model "$model" --dataset mme \
      --statistics-root "$statistics_root" --baseline-root "$baseline_root" \
      --output-dir "$smoke" --replay-dir "$replay_dir" \
      --epsilon 0.10 --a2-only --max-samples 2 \
      2>&1 | tee -a "$output_root/logs/${model}__smoke.log"
  fi

  for dataset in mme mmbench textvqa chartqa; do
    for epsilon in 0.10 0.15 0.20 0.25 0.30; do
      tag="${epsilon/./p}"
      label="${model}__${dataset}__eps_${tag}"
      target="$output_root/$label"
      if [[ -f "$target/summary.json" ]]; then
        echo "[skip complete] $label"
        continue
      fi
      echo "[start] $label on GPU $gpu at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_adaptive_pruning_2026_10.py \
        --study "$study" --model "$model" --dataset "$dataset" \
        --statistics-root "$statistics_root" --baseline-root "$baseline_root" \
        --output-dir "$target" --replay-dir "$replay_dir" \
        --epsilon "$epsilon" --a2-only \
        2>&1 | tee -a "$output_root/logs/${label}.log"
      echo "[complete] $label at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    done
  done
  flock -u "$lock_fd"
  exec {lock_fd}>&-
done
