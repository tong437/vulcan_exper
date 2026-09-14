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
from neuron_typing_generalization.clevr_audit import build_and_freeze_clevr_audit


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and freeze same-scene minimal-program-edit CLEVR pairs.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/neuron_typing_generalization",
    )
    parser.add_argument("--pair-count", type=int, default=64)
    args = parser.parse_args()
    report = build_and_freeze_clevr_audit(EXPERIMENT_ROOT, args.artifact_root.resolve(), args.pair_count)
    report_path = EXPERIMENT_ROOT / "reports/clevr_pair_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
