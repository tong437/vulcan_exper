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

input_dir="artifacts/neuron_typing_generalization/ablation/advisor_2026_09_1pct_fastpath_4090"
output_dir="research/neuron_typing_generalization/reports/advisor_2026_09/qwen3_5_2b_ablation"
chartqa_summary="$input_dir/qwen3_5_2b__chartqa/summary.json"

while [[ ! -f "$chartqa_summary" ]]; do
  if ! tmux has-session -t advisor_qwen35_fast_4090 2>/dev/null; then
    echo "Qwen3.5 worker exited before ChartQA summary was written." >&2
    exit 1
  fi
  sleep 60
done

python - "$input_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for dataset in ("mme", "mmbench", "textvqa", "chartqa"):
    run_dir = root / f"qwen3_5_2b__{dataset}"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if (
        summary["expected_samples"] != 1000
        or summary["removal_fraction"] != 0.01
        or summary["random_repeats"] != 10
        or len(summary["conditions"]) != 14
    ):
        raise SystemExit(f"Unexpected experiment parameters in {run_dir}.")
    with (run_dir / "predictions.jsonl").open(encoding="utf-8") as handle:
        if sum(1 for _ in handle) != 59000:
            raise SystemExit(f"Incomplete predictions in {run_dir}.")
print("All four Qwen3.5 ablation summaries and prediction files are complete.", flush=True)
PY

python research/neuron_typing_generalization/pipelines/report_advisor_ablation_2026_09.py \
  --input-dir "$input_dir" \
  --output-dir "$output_dir" \
  --model qwen3_5_2b

echo "Qwen3.5 figures and tables are ready at $output_dir"
