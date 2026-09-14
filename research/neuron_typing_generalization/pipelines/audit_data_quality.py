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
from pathlib import Path

import _bootstrap  # noqa: F401
from neuron_typing_generalization.data_quality import build_data_quality_report, render_quality_report


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen dataset grain, pairing integrity, and source records.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/neuron_typing_generalization",
    )
    args = parser.parse_args()
    report = build_data_quality_report(EXPERIMENT_ROOT, args.artifact_root.resolve())
    json_path = EXPERIMENT_ROOT / "reports/data_quality_audit.json"
    markdown_path = EXPERIMENT_ROOT / "reports/data_quality_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_quality_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    if report["critical_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
