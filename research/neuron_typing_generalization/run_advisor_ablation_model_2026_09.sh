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

model="${1:-}"
case "$model" in
  llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it)
    ;;
  *)
    echo "Usage: $0 {llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it}" >&2
    exit 2
    ;;
esac

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
cd "$repository_root"

study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
ablation_root="$repository_root/artifacts/neuron_typing_generalization/ablation/advisor_2026_09_1pct"
log_root="$repository_root/artifacts/neuron_typing_generalization/advisor_2026_09/logs"
replay_matches=("$repository_root"/artifacts/neuron_typing_generalization/advisor_2026_09/replay.*)
if [[ "${#replay_matches[@]}" -ne 1 || ! -d "${replay_matches[0]}" ]]; then
  echo "Expected exactly one frozen replay directory; found ${#replay_matches[@]}." >&2
  exit 1
fi
replay_dir="${replay_matches[0]}"

mkdir -p "$ablation_root" "$log_root"
umask 000
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - "$model" <<'PY'
import sys

import torch
import transformers
from packaging.version import Version

model = sys.argv[1]
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expose exactly one CUDA GPU to this worker with CUDA_VISIBLE_DEVICES.")
device_name = torch.cuda.get_device_name(0)
if model == "gemma_4_e2b_it" and Version(transformers.__version__) < Version("5.6.0"):
    raise SystemExit(f"Gemma 4 requires transformers>=5.6.0; found {transformers.__version__}.")
print(f"GPU: {device_name}; torch: {torch.__version__}; transformers: {transformers.__version__}")
PY

datasets=(mme mmbench textvqa chartqa)
for dataset in "${datasets[@]}"; do
  label="${model}__${dataset}"
  if [[ -f "$ablation_root/$label/summary.json" ]]; then
    echo "[skip completed] $label"
    continue
  fi
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
