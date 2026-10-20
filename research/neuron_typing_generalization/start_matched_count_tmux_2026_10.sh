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
gpu="${1:-0}"
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader >/dev/null
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
mkdir -p artifacts/neuron_typing_generalization/matched_count_2026_10/logs
tmux -L matched_count_2026_10 new-session -d -s "matched_gpu${gpu}" -c "$repository_root" \
  "bash research/neuron_typing_generalization/run_matched_count_worker_2026_10.sh $gpu qwen3_vl_2b_instruct qwen3_5_2b llava_1_5_7b gemma_4_e2b_it > artifacts/neuron_typing_generalization/matched_count_2026_10/logs/gpu${gpu}_worker.log 2>&1"
tmux -L matched_count_2026_10 ls
