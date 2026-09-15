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
from neuron_typing_generalization.reporting import build_unified_vqav2_report, evaluate_pilot_run


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Close and report the frozen three-model VQAv2 pilot.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--analysis-commit", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifact_root = Path(config["artifact_root"])
    if not artifact_root.is_absolute():
        artifact_root = REPOSITORY_ROOT / artifact_root
    pilot_root = artifact_root / "runs" / config["collection_study"]
    smoke_root = artifact_root / "runs" / config["smoke_study"]
    pilot_dirs = [pilot_root / run_id for run_id in config["pilot_run_ids"]]
    bootstrap = config["bootstrap"]
    for run_dir in pilot_dirs:
        evaluate_pilot_run(
            run_dir,
            bootstrap_replicates=int(bootstrap["replicates"]),
            bootstrap_seed=int(bootstrap["seed"]),
            confidence_level=float(bootstrap["confidence_level"]),
        )

    validation_models = []
    for run_id in config["smoke_run_ids"]:
        path = smoke_root / run_id / "logs/direct_vs_streaming_validation.json"
        if not path.is_file():
            raise FileNotFoundError(f"Real-model validation is missing: {path}")
        validation_models.append(json.loads(path.read_text(encoding="utf-8")))
    validation_report = {
        "status": "passed" if all(row["status"] == "passed" for row in validation_models) else "failed",
        "method": "same-forward direct retained activation versus immediate streaming reduction",
        "models": validation_models,
    }
    reports = EXPERIMENT_ROOT / "reports"
    (reports / "vqav2_direct_streaming_validation.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if validation_report["status"] != "passed":
        raise RuntimeError("At least one direct-vs-streaming validation failed.")

    result = build_unified_vqav2_report(
        pilot_dirs,
        reports,
        collection_commit=str(config["collection_code_commit"]),
        analysis_commit=args.analysis_commit or _git_head(),
        target_half_width=float(config["precision"]["primary_ci_half_width"]),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
