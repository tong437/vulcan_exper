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

"""Analyze Phase 6D bidirectional layer-localization interventions."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a complete Phase 6D layer-localization run.")
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_evaluations(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def get_one(rows: list[dict[str, Any]], **conditions: Any) -> dict[str, Any]:
    selected = [
        row for row in rows if all(row["variant_metadata"].get(key) == value for key, value in conditions.items())
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {conditions}, found {len(selected)}.")
    return selected[0]


def validate(result: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    if not result.get("complete"):
        raise ValueError("Phase 6D result is incomplete.")
    expected = result["expected_evaluations"]
    if len(rows) != expected or len({row["evaluation_id"] for row in rows}) != expected:
        raise ValueError("Phase 6D evaluation count or ID uniqueness check failed.")
    manifest = {variant["variant_id"]: variant for variant in result["variants"]}
    if set(manifest) != {row["variant_id"] for row in rows}:
        raise ValueError("Phase 6D result manifest and evaluations disagree.")
    for row in rows:
        variant = manifest[row["variant_id"]]
        if row["kept_sha256"] != variant["kept_sha256"] or row["target_sample_id"] != variant["target"]:
            raise ValueError(f"Phase 6D evaluation provenance mismatch for {row['evaluation_id']}.")
        semantic = row["generation"]["semantic"]
        if bool(row["automatic_semantic_pass"]) != bool(semantic["automatic_pass"]):
            raise ValueError(f"Phase 6D semantic field mismatch for {row['evaluation_id']}.")


def analyze_contrast(contrast: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    contrast_rows = [
        row for row in rows if row["variant_metadata"]["contrast"]["contrast_id"] == contrast["contrast_id"]
    ]
    base = get_one(contrast_rows, endpoint="safe_base")
    union = get_one(contrast_rows, endpoint="failed_union")
    endpoint_valid = bool(base["automatic_semantic_pass"] and not union["automatic_semantic_pass"])
    layers = sorted(
        {row["variant_metadata"]["layer"] for row in contrast_rows if row["variant_metadata"]["layer"] is not None}
    )
    per_layer = []
    for layer in layers:
        layer_rows = [row for row in contrast_rows if row["variant_metadata"]["layer"] == layer]
        add = get_one(layer_rows, intervention="add_one_layer", control=False)
        loo = get_one(layer_rows, intervention="leave_one_layer_out", control=False)
        random_add = [
            row
            for row in layer_rows
            if row["variant_metadata"]["intervention"] == "add_one_layer" and row["variant_metadata"]["control"]
        ]
        random_loo = [
            row
            for row in layer_rows
            if row["variant_metadata"]["intervention"] == "leave_one_layer_out" and row["variant_metadata"]["control"]
        ]
        if not random_add or len(random_add) != len(random_loo):
            raise ValueError(f"Missing or unbalanced random controls for {contrast['contrast_id']} layer {layer}.")
        add_flip = bool(base["automatic_semantic_pass"] and not add["automatic_semantic_pass"])
        loo_flip = bool(not union["automatic_semantic_pass"] and loo["automatic_semantic_pass"])
        random_add_failure_rate = mean([float(not row["automatic_semantic_pass"]) for row in random_add])
        random_loo_recovery_rate = mean([float(row["automatic_semantic_pass"]) for row in random_loo])
        add_nll_delta = add["gold_proxy"]["mean_nll"] - base["gold_proxy"]["mean_nll"]
        loo_nll_improvement = union["gold_proxy"]["mean_nll"] - loo["gold_proxy"]["mean_nll"]
        random_add_nll_deltas = [row["gold_proxy"]["mean_nll"] - base["gold_proxy"]["mean_nll"] for row in random_add]
        random_loo_nll_improvements = [
            union["gold_proxy"]["mean_nll"] - row["gold_proxy"]["mean_nll"] for row in random_loo
        ]
        identity_specific = bool(
            endpoint_valid
            and add_flip
            and loo_flip
            and random_add_failure_rate < 0.5
            and random_loo_recovery_rate < 0.5
        )
        per_layer.append(
            {
                "layer": layer,
                "increment_count": add["variant_metadata"]["increment_count"],
                "add_one_layer": {
                    "semantic_pass": add["automatic_semantic_pass"],
                    "exact_failure_flip": add_flip,
                    "nll": add["gold_proxy"]["mean_nll"],
                    "nll_delta_from_safe_base": add_nll_delta,
                    "random_failure_rate": random_add_failure_rate,
                    "random_nll_delta_mean": mean(random_add_nll_deltas),
                    "random_nll_delta_median": statistics.median(random_add_nll_deltas),
                },
                "leave_one_layer_out": {
                    "semantic_pass": loo["automatic_semantic_pass"],
                    "exact_recovery_flip": loo_flip,
                    "nll": loo["gold_proxy"]["mean_nll"],
                    "nll_improvement_from_failed_union": loo_nll_improvement,
                    "random_recovery_rate": random_loo_recovery_rate,
                    "random_nll_improvement_mean": mean(random_loo_nll_improvements),
                    "random_nll_improvement_median": statistics.median(random_loo_nll_improvements),
                },
                "bidirectional_exact_flip": bool(endpoint_valid and add_flip and loo_flip),
                "identity_specific_bidirectional_flip": identity_specific,
                "joint_nll_advantage_over_random_mean": min(
                    add_nll_delta - mean(random_add_nll_deltas),
                    loo_nll_improvement - mean(random_loo_nll_improvements),
                ),
            }
        )

    exact_layers = [row["layer"] for row in per_layer if row["bidirectional_exact_flip"]]
    identity_specific_layers = [row["layer"] for row in per_layer if row["identity_specific_bidirectional_flip"]]
    return {
        "contrast": contrast,
        "endpoint_reproduction": {
            "valid_pass_to_fail": endpoint_valid,
            "safe_base_pass": base["automatic_semantic_pass"],
            "safe_base_nll": base["gold_proxy"]["mean_nll"],
            "safe_base_caption": base["generation"]["final_caption"],
            "failed_union_pass": union["automatic_semantic_pass"],
            "failed_union_nll": union["gold_proxy"]["mean_nll"],
            "failed_union_caption": union["generation"]["final_caption"],
        },
        "bidirectional_exact_layers": exact_layers,
        "identity_specific_bidirectional_layers": identity_specific_layers,
        "per_layer": per_layer,
    }


def analyze(result_file: str | Path) -> dict[str, Any]:
    result = read_json(result_file)
    rows = load_evaluations(result["evaluations_file"])
    validate(result, rows)
    contrasts = [analyze_contrast(contrast, rows) for contrast in result["contrasts"]]
    shortlist = []
    edge_layer_support: dict[tuple[str, int], set[str]] = defaultdict(set)
    for entry in contrasts:
        by_layer = {row["layer"]: row for row in entry["per_layer"]}
        for layer in entry["bidirectional_exact_layers"]:
            layer_row = by_layer[layer]
            shortlist.append(
                {
                    "contrast_id": entry["contrast"]["contrast_id"],
                    "edge_id": entry["contrast"]["edge_id"],
                    "base": entry["contrast"]["base"],
                    "donor": entry["contrast"]["donor"],
                    "target": entry["contrast"]["target"],
                    "layer": layer,
                    "increment_count": layer_row["increment_count"],
                    "identity_specific_against_control_majority": layer_row["identity_specific_bidirectional_flip"],
                    "joint_nll_advantage_over_random_mean": layer_row["joint_nll_advantage_over_random_mean"],
                }
            )
            edge_layer_support[(entry["contrast"]["edge_id"], layer)].add(entry["contrast"]["contrast_id"])
    shortlist.sort(
        key=lambda row: (
            not row["identity_specific_against_control_majority"],
            -row["joint_nll_advantage_over_random_mean"],
            row["edge_id"],
            row["layer"],
            row["contrast_id"],
        )
    )
    consensus = [
        {"edge_id": edge_id, "layer": layer, "supporting_contrasts": sorted(support), "support_count": len(support)}
        for (edge_id, layer), support in edge_layer_support.items()
    ]
    consensus.sort(key=lambda row: (-row["support_count"], row["edge_id"], row["layer"]))
    return {
        "complete": True,
        "source_result": str(Path(result_file).resolve()),
        "integrity": {
            "expected_evaluations": result["expected_evaluations"],
            "observed_evaluations": len(rows),
            "endpoint_contrasts": len(contrasts),
            "all_endpoints_reproduced": all(
                entry["endpoint_reproduction"]["valid_pass_to_fail"] for entry in contrasts
            ),
        },
        "summary": {
            "bidirectional_exact_slots": len(shortlist),
            "identity_specific_slots": sum(
                bool(row["identity_specific_against_control_majority"]) for row in shortlist
            ),
            "layers_by_support_count": dict(
                sorted(Counter(row["layer"] for row in shortlist).items(), key=lambda item: (-item[1], item[0]))
            ),
        },
        "identity_refinement_shortlist": shortlist,
        "edge_layer_consensus": consensus,
        "contrasts": contrasts,
        "interpretation_boundary": (
            "The shortlist requires exact semantic flips in both causal directions. Random controls qualify identity "
            "specificity descriptively; layers are frozen-sample candidates, not population-level conflict loci."
        ),
    }


def main() -> None:
    args = parse_args()
    output = analyze(args.result_file)
    write_json(args.output_file, output)
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
