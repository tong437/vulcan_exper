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

python_bin="/root/.cache/vulcan-qwen35-fastpath-torch26/bin/python"
study="research/neuron_typing_generalization/configs/studies/advisor_2026_09_main.yaml"
ablation_root="$repository_root/artifacts/neuron_typing_generalization/ablation/advisor_2026_09_1pct_fastpath_4090"
log_root="$repository_root/artifacts/neuron_typing_generalization/advisor_2026_09/logs"
replay_matches=("$repository_root"/artifacts/neuron_typing_generalization/advisor_2026_09/replay.*)
if [[ "${#replay_matches[@]}" -ne 1 || ! -d "${replay_matches[0]}" ]]; then
  echo "Expected exactly one frozen replay directory; found ${#replay_matches[@]}." >&2
  exit 1
fi
replay_dir="${replay_matches[0]}"

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$python_bin" - <<'PY'
import torch
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES.")
if not (is_causal_conv1d_available() and is_flash_linear_attention_available()):
    raise SystemExit("Qwen3.5 fast path requires working fla and causal_conv1d imports.")
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: F401
from fla.modules import FusedRMSNormGated  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401

print(f"GPU: {torch.cuda.get_device_name(0)}; fast-path imports: OK", flush=True)
PY

mkdir -p "$ablation_root" "$log_root"
umask 000
for dataset in mme mmbench textvqa chartqa; do
  label="qwen3_5_2b__${dataset}"
  if [[ -f "$ablation_root/$label/summary.json" ]]; then
    echo "[skip completed] $label"
    continue
  fi
  echo "[fastpath ablation] $label"
  "$python_bin" research/neuron_typing_generalization/pipelines/run_advisor_ablation_gpu_2026_09.py \
    --study "$study" \
    --model qwen3_5_2b \
    --dataset "$dataset" \
    --output-dir "$ablation_root/$label" \
    --replay-dir "$replay_dir" \
    --removal-percent 1 \
    --random-repeats 10 \
    2>&1 | tee "$log_root/ablation_fastpath_4090_${label}.log"
done
