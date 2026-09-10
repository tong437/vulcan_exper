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

"""Summarize frozen Phase 6B single-sample limits and preliminary mask geometry."""

from __future__ import annotations

import argparse
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze frozen Phase 6B winner-mask geometry.")
    parser.add_argument("--winner_dir", required=True)
    parser.add_argument("--verification_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--sqlite_file", help="Optional SQLite snapshot for report-native SQL provenance.")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_sqlite(path: str | Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with sqlite3.connect(output) as connection:
        for table_name, rows in tables.items():
            if not rows:
                continue
            columns = list(rows[0])

            def sqlite_type(column: str) -> str:
                values = [row[column] for row in rows if row[column] is not None]
                if values and all(isinstance(value, (bool, int)) for value in values):
                    return "INTEGER"
                if values and all(isinstance(value, (bool, int, float)) for value in values):
                    return "REAL"
                return "TEXT"

            quoted_columns = ", ".join(f'"{column}" {sqlite_type(column)}' for column in columns)
            connection.execute(f'CREATE TABLE "{table_name}" ({quoted_columns})')
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                [
                    [int(value) if isinstance(value, bool) else value for value in (row[column] for column in columns)]
                    for row in rows
                ],
            )


def kept_by_layer(neuron_ids: dict[str, list[int]], layer_dims: dict[str, int]) -> dict[int, set[int]]:
    return {
        int(layer): set(range(int(width))) - {int(neuron) for neuron in neuron_ids[str(layer)]}
        for layer, width in layer_dims.items()
    }


def flatten(layer_sets: dict[int, set[int]]) -> set[tuple[int, int]]:
    return {(layer, neuron) for layer, neurons in layer_sets.items() for neuron in neurons}


def main() -> None:
    args = parse_args()
    winner_dir = Path(args.winner_dir).resolve()
    manifest = load_json(winner_dir / "frozen_winners.json")
    verification = load_json(Path(args.verification_file).resolve())
    if not verification.get("complete") or not verification.get("passed"):
        raise ValueError("Geometry analysis requires a complete passing structural-reload verification.")

    winner_rows = {row["sample_id"]: row for row in manifest["winners"]}
    sample_ids = list(winner_rows)
    layer_kept = {}
    kept = {}
    for sample_id, row in winner_rows.items():
        neuron_ids = load_json(winner_dir / sample_id / "mask.json")
        layer_kept[sample_id] = kept_by_layer(neuron_ids, row["layer_dims"])
        kept[sample_id] = flatten(layer_kept[sample_id])
        if len(kept[sample_id]) != row["kept_neurons"]:
            raise ValueError(f"Kept-neuron count mismatch for {sample_id}.")

    total_neurons = sum(next(iter(winner_rows.values()))["layer_dims"].values())
    counts = Counter(neuron for sample in kept.values() for neuron in sample)
    all_core = set.intersection(*kept.values())
    union = set.union(*kept.values())
    near_core = {neuron for neuron, frequency in counts.items() if frequency >= len(sample_ids) - 1}
    membership_distribution = [
        {
            "num_samples": frequency,
            "num_neurons": sum(value == frequency for value in counts.values())
            + (total_neurons - len(counts) if frequency == 0 else 0),
        }
        for frequency in range(len(sample_ids) + 1)
    ]

    sample_summary = []
    for sample_id, row in winner_rows.items():
        verification_row = verification["samples"][sample_id]
        unique = sum(counts[neuron] == 1 for neuron in kept[sample_id])
        shell = len(kept[sample_id] - all_core)
        sample_summary.append(
            {
                "sample_id": sample_id,
                "semantic_stratum": row["semantic_stratum"],
                "deletion_budget": row["deletion_budget"],
                "kept_neurons": row["kept_neurons"],
                "keep_ratio": row["keep_ratio"],
                "pruning_ratio": 1.0 - row["keep_ratio"],
                "shared_all7": len(all_core),
                "conditional_shell_vs_all7": shell,
                "sample_unique_neurons": unique,
                "removed_parameters": row["source_parameter_summary"]["removed_parameters"],
                "total_parameter_reduction_ratio": row["source_parameter_summary"]["total_parameter_reduction_ratio"],
                "structural_reload_passed": verification_row["passed"],
                "final_caption": verification_row["reloaded"]["generation"]["final_caption"],
            }
        )

    pairwise = []
    for left, right in itertools.combinations(sample_ids, 2):
        intersection = kept[left] & kept[right]
        combined = kept[left] | kept[right]
        expected_intersection = sum(
            len(layer_kept[left][layer]) * len(layer_kept[right][layer]) / winner_rows[left]["layer_dims"][str(layer)]
            for layer in layer_kept[left]
        )
        pairwise.append(
            {
                "left": left,
                "right": right,
                "intersection": len(intersection),
                "union": len(combined),
                "jaccard": len(intersection) / len(combined),
                "overlap_over_smaller": len(intersection) / min(len(kept[left]), len(kept[right])),
                "random_expected_intersection_layer_conditioned": expected_intersection,
                "observed_over_random": len(intersection) / expected_intersection,
            }
        )

    expected_all_core = sum(
        winner_rows[sample_ids[0]]["layer_dims"][str(layer)]
        * __import__("math").prod(
            len(layer_kept[sample_id][layer]) / winner_rows[sample_id]["layer_dims"][str(layer)]
            for sample_id in sample_ids
        )
        for layer in layer_kept[sample_ids[0]]
    )
    layer_summary = []
    for layer in sorted(layer_kept[sample_ids[0]]):
        layer_intersection = set.intersection(*(layer_kept[sample_id][layer] for sample_id in sample_ids))
        layer_union = set.union(*(layer_kept[sample_id][layer] for sample_id in sample_ids))
        row = {
            "layer": layer,
            "shared_all7": len(layer_intersection),
            "union_all7": len(layer_union),
            "mean_kept": sum(len(layer_kept[sample_id][layer]) for sample_id in sample_ids) / len(sample_ids),
            "min_kept": min(len(layer_kept[sample_id][layer]) for sample_id in sample_ids),
            "max_kept": max(len(layer_kept[sample_id][layer]) for sample_id in sample_ids),
        }
        row.update({sample_id: len(layer_kept[sample_id][layer]) for sample_id in sample_ids})
        layer_summary.append(row)

    validation_checks = {
        "membership_partition_equals_total": sum(row["num_neurons"] for row in membership_distribution)
        == total_neurons,
        "membership_weighted_sum_equals_kept_sum": sum(
            row["num_samples"] * row["num_neurons"] for row in membership_distribution
        )
        == sum(len(sample) for sample in kept.values()),
        "layer_core_sum_equals_global_core": sum(row["shared_all7"] for row in layer_summary) == len(all_core),
        "pairwise_row_count_is_complete": len(pairwise) == len(sample_ids) * (len(sample_ids) - 1) // 2,
        "all_structural_reloads_passed": all(row["structural_reload_passed"] for row in sample_summary),
        "removed_parameter_counts_match_ffn_width": all(
            row["removed_parameters"] == 3 * 1024 * row["deletion_budget"] for row in sample_summary
        ),
    }
    if not all(validation_checks.values()):
        raise ValueError(f"Geometry validation failed: {validation_checks}")

    result = {
        "complete": True,
        "interpretation": (
            "Identity overlaps are descriptive geometry of separately optimized feasible masks. They do not by "
            "themselves establish functional necessity, a causal shared core, or composability."
        ),
        "sources": {
            "winner_manifest": str(winner_dir / "frozen_winners.json"),
            "structural_verification": str(Path(args.verification_file).resolve()),
        },
        "definitions": {
            "total_ffn_neurons": total_neurons,
            "shared_core_all7": "Neuron identities retained by every one of the seven frozen winner masks.",
            "near_core_6plus": "Neuron identities retained by at least six of seven winner masks.",
            "conditional_shell": "For one sample, retained identities outside the seven-way intersection.",
            "random_baseline": "Independent identity selection conditioned on each mask's observed per-layer widths.",
        },
        "global_geometry": {
            "num_samples": len(sample_ids),
            "shared_core_all7": len(all_core),
            "shared_core_ratio_total": len(all_core) / total_neurons,
            "near_core_6plus": len(near_core),
            "union_all7": len(union),
            "union_ratio_total": len(union) / total_neurons,
            "never_kept": total_neurons - len(union),
            "random_expected_shared_core_all7_layer_conditioned": expected_all_core,
            "shared_core_observed_over_random": len(all_core) / expected_all_core,
        },
        "sample_summary": sample_summary,
        "membership_distribution": membership_distribution,
        "pairwise": pairwise,
        "layer_summary": layer_summary,
        "validation_checks": validation_checks,
    }
    write_json(args.output_file, result)
    if args.sqlite_file:
        write_sqlite(
            args.sqlite_file,
            {
                "sample_summary": sample_summary,
                "membership_distribution": membership_distribution,
                "pairwise_overlap": pairwise,
                "layer_summary": layer_summary,
            },
        )
    print(json.dumps(result["global_geometry"], indent=2))


if __name__ == "__main__":
    main()
