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
from neuron_typing_generalization.synthesis import build_synthesis_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the four-dataset main synthesis report.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-commit")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifact_root = Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPOSITORY_ROOT / artifact_root
    run_dirs = {
        dataset: {model: artifact_root / "runs" / values["study"] / run_id for model, run_id in values["runs"].items()}
        for dataset, values in config["datasets"].items()
    }
    result = build_synthesis_report(
        run_dirs,
        EXPERIMENT_ROOT / "reports",
        analysis_commit=args.analysis_commit or _git_head(),
        comparisons=[tuple(value) for value in config["comparisons"]],
        bootstrap_replicates=int(config["bootstrap"]["replicates"]),
        bootstrap_seed=int(config["bootstrap"]["seed"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
