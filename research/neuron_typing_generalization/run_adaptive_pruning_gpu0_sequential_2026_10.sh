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

set -uo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"

worker="research/neuron_typing_generalization/run_adaptive_pruning_worker_2026_10.sh"
failed=0
for model in llava_1_5_7b gemma_4_e2b_it qwen3_5_2b; do
  echo "[start model] $model on GPU 0 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if bash "$worker" 0 "$model"; then
    echo "[complete model] $model at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "[failed model] $model at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    failed=1
  fi
done
exit "$failed"
