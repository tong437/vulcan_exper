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

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
import yaml
from neuron_typing_generalization.within_run_variability import build_within_run_variability_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze within-run per-neuron variability over 1,000 samples.")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs/analysis/unpaired_neuron_v1_1000.yaml",
    )
    parser.add_argument("--analysis-commit")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifact_root = Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPOSITORY_ROOT / artifact_root
    run_root = artifact_root / "runs" / config["study"]
    run_dirs = {
        dataset: {model: run_root / run_id for model, run_id in models.items()}
        for dataset, models in config["runs"].items()
    }
    result = build_within_run_variability_report(
        run_dirs, EXPERIMENT_ROOT / "reports", analysis_commit=args.analysis_commit or _git_head()
    )
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
