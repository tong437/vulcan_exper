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
  llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it) ;;
  *)
    echo "Usage: $0 {llava_1_5_7b|qwen3_vl_2b_instruct|qwen3_5_2b|gemma_4_e2b_it}" >&2
    exit 2
    ;;
esac

experiment_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
cd "$repository_root"

study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
ablation_root="$repository_root/artifacts/neuron_typing_generalization/ablation/advisor_2026_09_5pct"
log_root="$repository_root/artifacts/neuron_typing_generalization/advisor_2026_09/logs"
replay_matches=("$repository_root"/artifacts/neuron_typing_generalization/advisor_2026_09/replay.*)
if [[ "${#replay_matches[@]}" -ne 1 || ! -d "${replay_matches[0]}" ]]; then
  echo "Expected exactly one frozen replay directory; found ${#replay_matches[@]}." >&2
  exit 1
fi
replay_dir="${replay_matches[0]}"

python_bin="python"
if [[ "$model" == "qwen3_5_2b" ]]; then
  python_bin="/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python"
  "$python_bin" - <<'PY'
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

if not (is_causal_conv1d_available() and is_flash_linear_attention_available()):
    raise SystemExit("Qwen3.5 requires the installed linear-attention fast path.")
PY
fi

mkdir -p "$ablation_root" "$log_root"
umask 000
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

datasets=(mme mmbench textvqa chartqa)
for dataset in "${datasets[@]}"; do
  label="${model}__${dataset}"
  output_dir="$ablation_root/$label"
  if [[ -f "$output_dir/summary.json" ]]; then
    python - "$output_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("removal_fraction") != 0.05 or summary.get("expected_samples") != 1000:
    raise SystemExit(f"Existing summary is not a completed 5%/1000-sample run: {sys.argv[1]}")
PY
    echo "[skip completed] $label"
    continue
  fi
  echo "[ablation 5%] $label"
  "$python_bin" -u research/neuron_typing_generalization/pipelines/run_advisor_ablation_gpu_2026_09.py \
    --study "$study" \
    --model "$model" \
    --dataset "$dataset" \
    --output-dir "$output_dir" \
    --replay-dir "$replay_dir" \
    --removal-percent 5 \
    --insufficient-pool all \
    --random-repeats 10 \
    2>&1 | tee "$log_root/ablation_5pct_${label}.log"
done
