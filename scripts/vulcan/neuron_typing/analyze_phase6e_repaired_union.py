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

"""Validate and summarize Phase 6E repaired-union results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 6E repaired-union output.")
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def endpoint_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: row["target_sample_id"])
    return {
        "dual_pass": len(ordered) == 2 and all(row["automatic_semantic_pass"] for row in ordered),
        "target_passes": sum(bool(row["automatic_semantic_pass"]) for row in ordered),
        "mean_nll": sum(row["gold_proxy"]["mean_nll"] for row in ordered) / len(ordered),
        "targets": [
            {
                "sample_id": row["target_sample_id"],
                "semantic_pass": row["automatic_semantic_pass"],
                "mean_nll": row["gold_proxy"]["mean_nll"],
                "caption": row["generation"]["final_caption"],
                "terminated_normally": row["generation"]["stop"]["terminated_normally"],
            }
            for row in ordered
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_path = Path(args.result_file).resolve()
    result = read_json(result_path)
    if not result.get("complete"):
        raise ValueError("Cannot analyze an incomplete Phase 6E run.")
    rows = load_jsonl(result["evaluations_file"])
    ids = [row["evaluation_id"] for row in rows]
    variants = {row["variant_id"]: row for row in result["variants"]}
    if len(rows) != result["expected_evaluations"] or len(ids) != len(set(ids)):
        raise ValueError("Phase 6E evaluation count or IDs are invalid.")
    if any(row["variant_id"] not in variants or row["kept_sha256"] != variants[row["variant_id"]]["kept_sha256"] for row in rows):
        raise ValueError("Phase 6E evaluation provenance does not match the variant manifest.")

    edge_summaries = []
    for spec in result["repair_specs"]:
        edge_id = spec["edge_id"]
        edge_rows = [row for row in rows if row["edge_id"] == edge_id]
        by_condition = {}
        for condition in ("safe_base", "failed_union", "repaired_union"):
            selected = [row for row in edge_rows if row["condition"] == condition]
            if len(selected) != 2:
                raise ValueError(f"{edge_id} lacks two {condition} endpoints.")
            by_condition[condition] = endpoint_summary(selected)
        random_conditions = sorted({row["condition"] for row in edge_rows if row["condition"].startswith("random_repair")})
        if len(random_conditions) != result["config"]["random_seeds"]:
            raise ValueError(f"{edge_id} random repair count is incomplete.")
        random_repairs = []
        for condition in random_conditions:
            selected = [row for row in edge_rows if row["condition"] == condition]
            summary = endpoint_summary(selected)
            summary.update(
                {
                    "condition": condition,
                    "replicate": selected[0]["variant_metadata"]["replicate"],
                    "overlap_with_conflicts_by_layer": selected[0]["variant_metadata"][
                        "overlap_with_conflicts_by_layer"
                    ],
                }
            )
            random_repairs.append(summary)
        random_dual = sum(row["dual_pass"] for row in random_repairs)
        primary_success = bool(
            by_condition["safe_base"]["dual_pass"]
            and not by_condition["failed_union"]["dual_pass"]
            and by_condition["repaired_union"]["dual_pass"]
            and random_dual < len(random_repairs)
        )
        edge_summaries.append(
            {
                "edge_id": edge_id,
                "safe_base": spec["safe_base"],
                "donor": spec["donor"],
                "repair_counts_by_layer": {
                    str(candidate["layer"]): candidate["size"] for candidate in spec["candidates"]
                },
                "conditions": by_condition,
                "random_repairs": random_repairs,
                "random_dual_passes": random_dual,
                "random_repairs_total": len(random_repairs),
                "random_target_passes": sum(row["target_passes"] for row in random_repairs),
                "random_target_evaluations": 2 * len(random_repairs),
                "primary_success": primary_success,
            }
        )

    analysis = {
        "complete": True,
        "source_result": str(result_path),
        "infer_dtype": result["config"]["infer_dtype"],
        "integrity": {
            "evaluations": len(rows),
            "unique_evaluation_ids": len(set(ids)),
            "variants": len(variants),
            "all_rows_match_variant_manifest": True,
            "all_conditions_have_two_targets": True,
        },
        "edges": edge_summaries,
        "summary": {
            "edges": len(edge_summaries),
            "primary_successes": sum(edge["primary_success"] for edge in edge_summaries),
            "repaired_dual_passes": sum(edge["conditions"]["repaired_union"]["dual_pass"] for edge in edge_summaries),
            "random_dual_passes": sum(edge["random_dual_passes"] for edge in edge_summaries),
            "random_repairs": sum(edge["random_repairs_total"] for edge in edge_summaries),
        },
        "interpretation_boundary": (
            "The repaired union is a single frozen treatment per edge. Random-repair replicates estimate a matched "
            "capacity control but are not independent task samples or a population-generalization test."
        ),
    }
    output_path = Path(args.output_file).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(analysis["summary"], indent=2))
    return analysis


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
