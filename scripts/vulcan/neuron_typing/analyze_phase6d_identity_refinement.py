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

"""Validate and summarize Phase 6D deterministic identity refinement."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Phase 6D identity-refinement output.")
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def paired_controls(rows: list[dict[str, Any]], expected_replicates: int) -> dict[str, Any]:
    by_replicate: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_replicate[int(row["variant_metadata"]["replicate"])][row["variant_metadata"]["direction"]] = row
    if set(by_replicate) != set(range(expected_replicates)) or any(
        set(pair) != {"add", "loo"} for pair in by_replicate.values()
    ):
        raise ValueError("Identity-refinement random controls are missing a replicate or direction.")
    pairs = []
    for replicate, pair in sorted(by_replicate.items()):
        add = pair["add"]
        loo = pair["loo"]
        pairs.append(
            {
                "replicate": replicate,
                "add_pass": add["automatic_semantic_pass"],
                "loo_pass": loo["automatic_semantic_pass"],
                "joint_conflict_predicate": bool(
                    not add["automatic_semantic_pass"] and loo["automatic_semantic_pass"]
                ),
                "add_nll": add["gold_proxy"]["mean_nll"],
                "loo_nll": loo["gold_proxy"]["mean_nll"],
            }
        )
    return {
        "replicates": expected_replicates,
        "add_failures": sum(not pair["add_pass"] for pair in pairs),
        "loo_recoveries": sum(pair["loo_pass"] for pair in pairs),
        "joint_conflict_predicates": sum(pair["joint_conflict_predicate"] for pair in pairs),
        "pairs": pairs,
    }


def analyze(result_file: str | Path) -> dict[str, Any]:
    result_path = Path(result_file).resolve()
    result = read_json(result_path)
    if not result.get("complete"):
        raise ValueError("Phase 6D identity refinement is incomplete.")
    rows = [
        json.loads(line) for line in Path(result["evaluations_file"]).read_text(encoding="utf-8").splitlines() if line
    ]
    if len(rows) != result["total_model_evaluations"] or len({row["evaluation_id"] for row in rows}) != len(rows):
        raise ValueError("Identity-refinement evaluation count or ID uniqueness check failed.")
    expected_random = int(result["config"]["random_seeds"])
    selected_ids = {f"{row['contrast_id']}__layer_{int(row['layer']):02d}" for row in result["selected_candidates"]}
    candidate_dir = result_path.parent / "candidates"
    candidates = []
    for summary in result["candidate_results"]:
        candidate_id = summary["candidate_id"]
        candidate = read_json(candidate_dir / f"{candidate_id}.json")
        if candidate_id not in selected_ids or any(candidate[key] != value for key, value in summary.items()):
            raise ValueError(f"Candidate summary/provenance mismatch for {candidate_id}.")
        candidate_rows = [row for row in rows if row["variant_metadata"]["candidate_id"] == candidate_id]
        history_count = sum(len(step["proposals"]) for step in candidate["history"])
        if history_count != candidate["subset_evaluations"]:
            raise ValueError(f"Subset-evaluation history count mismatch for {candidate_id}.")
        chosen = [step["chosen_subset_sha256"] for step in candidate["history"] if step["chosen_subset_sha256"]]
        if len(chosen) != candidate["accepted_reductions"] or chosen[-1] != candidate["final_identity_sha256"]:
            raise ValueError(f"Accepted reduction chain mismatch for {candidate_id}.")
        final_rows = [
            row
            for row in candidate_rows
            if not row["variant_metadata"].get("control")
            and row["variant_metadata"]["subset_sha256"] == candidate["final_identity_sha256"]
        ]
        by_direction = {row["variant_metadata"]["direction"]: row for row in final_rows}
        if set(by_direction) != {"add", "loo"}:
            raise ValueError(f"Final subset directions are incomplete for {candidate_id}.")
        add = by_direction["add"]
        loo = by_direction["loo"]
        if add["automatic_semantic_pass"] or not loo["automatic_semantic_pass"]:
            raise ValueError(f"Final subset no longer satisfies the bidirectional predicate for {candidate_id}.")

        controls_by_step = []
        for step in range(1, candidate["accepted_reductions"] + 1):
            control_rows = [
                row
                for row in candidate_rows
                if row["variant_metadata"].get("control") and row["variant_metadata"]["accepted_step"] == step
            ]
            paired = paired_controls(control_rows, expected_random)
            matched_counts = {row["variant_metadata"]["matched_count"] for row in control_rows}
            if len(matched_counts) != 1:
                raise ValueError(f"Matched control counts disagree for {candidate_id} step {step}.")
            controls_by_step.append({"accepted_step": step, "matched_count": matched_counts.pop(), **paired})
        final_controls = controls_by_step[-1]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "edge_id": candidate["contrast"]["edge_id"],
                "target": candidate["contrast"]["target"],
                "layer": candidate["layer"],
                "initial_size": candidate["initial_size"],
                "final_size": candidate["final_size"],
                "retained_fraction_of_layer_increment": candidate["final_size"] / candidate["initial_size"],
                "accepted_reductions": candidate["accepted_reductions"],
                "subset_evaluations": candidate["subset_evaluations"],
                "stop_reason": candidate["stop_reason"],
                "final_identity_sha256": candidate["final_identity_sha256"],
                "final_identity_file": str(candidate_dir / f"{candidate_id}.json"),
                "final_treatment": {
                    "add_pass": add["automatic_semantic_pass"],
                    "add_nll": add["gold_proxy"]["mean_nll"],
                    "add_caption": add["generation"]["final_caption"],
                    "loo_pass": loo["automatic_semantic_pass"],
                    "loo_nll": loo["gold_proxy"]["mean_nll"],
                    "loo_caption": loo["generation"]["final_caption"],
                },
                "final_matched_controls": final_controls,
                "controls_by_accepted_step": controls_by_step,
            }
        )
    if {candidate["candidate_id"] for candidate in candidates} != selected_ids:
        raise ValueError("Selected and completed identity candidates disagree.")
    final_random_pairs = sum(candidate["final_matched_controls"]["replicates"] for candidate in candidates)
    final_random_joint = sum(
        candidate["final_matched_controls"]["joint_conflict_predicates"] for candidate in candidates
    )
    return {
        "complete": True,
        "source_result": str(result_path),
        "integrity": {
            "model_evaluations": len(rows),
            "unique_evaluation_ids": len({row["evaluation_id"] for row in rows}),
            "candidate_files": len(candidates),
            "all_final_subsets_satisfy_bidirectional_predicate": True,
            "all_accepted_steps_have_complete_random_controls": True,
        },
        "summary": {
            "candidates": len(candidates),
            "initial_identities": sum(candidate["initial_size"] for candidate in candidates),
            "final_identities": sum(candidate["final_size"] for candidate in candidates),
            "accepted_reductions": sum(candidate["accepted_reductions"] for candidate in candidates),
            "subset_evaluations": sum(candidate["subset_evaluations"] for candidate in candidates),
            "final_random_joint_conflict_predicates": final_random_joint,
            "final_random_pairs": final_random_pairs,
        },
        "candidates": candidates,
        "interpretation_boundary": (
            "Every final set is a budget-limited deterministic construction. None reached a completed singleton-removal "
            "test, so the sets are not 1-minimal or globally minimal. Matched controls are descriptive at three seeds."
        ),
    }


def main() -> None:
    args = parse_args()
    output = analyze(args.result_file)
    write_json(args.output_file, output)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
