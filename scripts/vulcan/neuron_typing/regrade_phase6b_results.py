# Copyright 2026 the LlamaFactory team.
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

"""Regrade saved Phase 6B generations with the current frozen contract implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase6b_semantics import evaluate_contract_semantics  # noqa: E402
from run_phase5_single_sample_frontier import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regrade Phase 6B result JSON without rerunning the model.")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def grade_generation(generation: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    stop = generation["stop"]
    return evaluate_contract_semantics(
        generation["raw_text"],
        contract=contract,
        terminated_normally=stop["terminated_normally"],
        hit_token_limit=stop["hit_token_limit"],
    )


def main() -> None:
    args = parse_args()
    result = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    changes = []
    for sample_id, sample in result["samples"].items():
        contract = sample["contract"]
        if sample.get("baseline_generation"):
            sample["baseline_generation"]["semantic"] = grade_generation(sample["baseline_generation"], contract)
        best = None
        for run_name, row in sample["runs"].items():
            previous = bool(row["automatic_semantic_pass"])
            semantic = grade_generation(row["generation"], contract)
            current = bool(semantic["automatic_pass"])
            row["generation"]["semantic"] = semantic
            row["automatic_semantic_pass"] = current
            if previous != current:
                changes.append({"sample_id": sample_id, "run": run_name, "from": previous, "to": current})
            if current and (
                best is None
                or row["deletion_budget"] > best["deletion_budget"]
                or (
                    row["deletion_budget"] == best["deletion_budget"]
                    and row["gold_proxy"]["mean_nll"] < best["mean_nll"]
                )
            ):
                best = {
                    "run": run_name,
                    "deletion_budget": row["deletion_budget"],
                    "restart": row["restart"],
                    "mean_nll": row["gold_proxy"]["mean_nll"],
                    "delta_nll": row["gold_proxy"]["delta_nll"],
                    "final_caption": row["generation"]["final_caption"],
                    "mask_hash": row["mask_hash"],
                    "mask_file": row["mask_file"],
                }
        sample["best_automatic_semantic_pass"] = best
    result["regrading"] = {
        "implementation": "single-caption-sentence hygiene gate",
        "changed_verdicts": changes,
    }
    write_json(args.output_file, result)
    print(json.dumps(result["regrading"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
