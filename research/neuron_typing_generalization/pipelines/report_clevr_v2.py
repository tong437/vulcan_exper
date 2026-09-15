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
from neuron_typing_generalization.reporting import build_clevr_precision_extension_report


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the nested CLEVR precision extension.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-commit")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifact_root = Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPOSITORY_ROOT / artifact_root
    old_root = artifact_root / "runs" / config["old_study"]
    new_root = artifact_root / "runs" / config["new_study"]
    result = build_clevr_precision_extension_report(
        [old_root / run_id for run_id in config["old_run_ids"]],
        [new_root / run_id for run_id in config["new_run_ids"]],
        EXPERIMENT_ROOT / "reports",
        analysis_commit=args.analysis_commit or _git_head(),
        target_half_width=float(config["target_ci_half_width"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
