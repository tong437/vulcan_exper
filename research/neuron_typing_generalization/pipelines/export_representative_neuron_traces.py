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
from neuron_typing_generalization.representative_neurons import build_representative_neuron_exports


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select representative neurons and export their frozen traces.")
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_ROOT / "configs/analysis/unpaired_neuron_v1_1000.yaml",
    )
    parser.add_argument("--analysis-commit")
    parser.add_argument("--artifact-output-root", type=Path)
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
    output_root = args.artifact_output_root or artifact_root / "representative_neuron_traces" / config["study"]
    result = build_representative_neuron_exports(
        run_dirs,
        output_root.resolve(),
        EXPERIMENT_ROOT / "reports",
        analysis_commit=args.analysis_commit or _git_head(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "criterion_slots": result["criterion_slots"],
                "unique_neurons": result["unique_neurons"],
                "artifact_root": result["artifact_root"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
