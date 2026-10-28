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
umask 0002
gpu="${1:?Usage: bash start_throughput_tmux_2026_10.sh GPU [--pilot-only]}"
[[ "$gpu" =~ ^[0-9]+$ ]] || exit 2
shift
[[ "$#" -eq 0 || ( "$#" -eq 1 && "$1" == --pilot-only ) ]] || exit 2
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repository_root"
python - "$gpu" <<'PY'
import subprocess, sys
s = subprocess.check_output(['nvidia-smi','-i',sys.argv[1],'--query-gpu=memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], text=True)
m, u = map(int, s.strip().split(','))
if m > 1024 or u > 10:
    raise SystemExit(f'GPU{sys.argv[1]} occupied ({m} MiB, {u}%); not starting benchmark')
PY
output="artifacts/neuron_typing_generalization/structured_throughput_1000_2026_10"
mkdir -p "$output/logs"
tmux -f /dev/null -L throughput_20261022 new-session -d -s "throughput_gpu${gpu}" -c "$repository_root" \
  "flock -n '$output/worker_gpu${gpu}.lock' python -u research/neuron_typing_generalization/pipelines/orchestrate_throughput_2026_10.py --gpu $gpu $* >> '$output/logs/gpu${gpu}.log' 2>&1"
echo "Started pilot-gated throughput worker on GPU${gpu}; log: $output/logs/gpu${gpu}.log"
