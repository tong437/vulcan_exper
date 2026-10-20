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

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
input_root="artifacts/neuron_typing_generalization/weak_pruning_a1_a2_2026_10"
output_dir="research/neuron_typing_generalization/reports/advisor_2026_09"

while true; do
  completed="$(find "$input_root" -mindepth 2 -maxdepth 2 -name summary.json -type f | wc -l)"
  if [[ "$completed" -eq 16 ]]; then
    break
  fi
  echo "[A1/A2 report waiting] $completed/16 complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  active=0
  for gpu in 0 1 2 3; do
    if tmux has-session -t "weak_a1a2_gpu$gpu" 2>/dev/null; then
      active=$((active + 1))
    fi
  done
  if [[ "$active" -eq 0 ]]; then
    echo "All A1/A2 workers stopped before 16 summaries were produced." >&2
    exit 1
  fi
  sleep 60
done

python research/neuron_typing_generalization/pipelines/report_weak_pruning_2026_10.py \
  --input-root "$input_root" \
  --output-dir "$output_dir"
echo "[A1/A2 report complete] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
