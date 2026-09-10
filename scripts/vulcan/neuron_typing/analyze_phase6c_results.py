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

"""Analyze Phase 6C causal core, shell-swap, and pair-union interventions."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a complete Phase 6C intervention run.")
    parser.add_argument("--result_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--sqlite_file", default=None)
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


def wilcoxon(values: list[float]) -> dict[str, Any]:
    if not values or all(value == 0 for value in values):
        return {"statistic": 0.0, "pvalue_two_sided": 1.0, "n": len(values)}
    result = stats.wilcoxon(values, alternative="two-sided")
    return {"statistic": float(result.statistic), "pvalue_two_sided": float(result.pvalue), "n": len(values)}


def paired_comparison(treatment: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, Any]:
    if len(treatment) != len(control) or not treatment:
        raise ValueError("Paired comparisons require equal, non-empty row lists.")
    treatment_pass = [bool(row["automatic_semantic_pass"]) for row in treatment]
    control_pass = [bool(row["automatic_semantic_pass"]) for row in control]
    treatment_only = sum(left and not right for left, right in zip(treatment_pass, control_pass, strict=True))
    control_only = sum(right and not left for left, right in zip(treatment_pass, control_pass, strict=True))
    discordant = treatment_only + control_only
    pvalue = (
        float(stats.binomtest(treatment_only, discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    )
    nll_differences = [
        left["gold_proxy"]["mean_nll"] - right["gold_proxy"]["mean_nll"]
        for left, right in zip(treatment, control, strict=True)
    ]
    return {
        "n": len(treatment),
        "treatment_passes": sum(treatment_pass),
        "control_passes": sum(control_pass),
        "pass_rate_difference": mean([float(value) for value in treatment_pass])
        - mean([float(value) for value in control_pass]),
        "discordant_treatment_only": treatment_only,
        "discordant_control_only": control_only,
        "mcnemar_exact_pvalue_two_sided": pvalue,
        "mean_nll_difference": mean(nll_differences),
        "median_nll_difference": float(stats.mstats.mquantiles(nll_differences, prob=[0.5])[0]),
        "nll_wilcoxon": wilcoxon(nll_differences),
    }


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passes = sum(bool(row["automatic_semantic_pass"]) for row in rows)
    return {
        "n": len(rows),
        "semantic_passes": passes,
        "semantic_pass_rate": passes / len(rows),
        "mean_nll": mean([row["gold_proxy"]["mean_nll"] for row in rows]),
        "median_nll": float(stats.mstats.mquantiles([row["gold_proxy"]["mean_nll"] for row in rows], prob=[0.5])[0]),
    }


def analyze_core(rows: list[dict[str, Any]], sample_ids: list[str]) -> dict[str, Any]:
    sufficiency_rows = [row for row in rows if row["family"] == "core_sufficiency"]
    sufficiency = []
    for threshold in (7, 6, 5):
        selected = [row for row in sufficiency_rows if row["variant_metadata"]["minimum_frequency"] == threshold]
        sufficiency.append({"minimum_frequency": threshold, **group_summary(selected)})

    necessity_rows = [row for row in rows if row["family"] == "core_necessity"]
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in necessity_rows:
        metadata = row["variant_metadata"]
        grouped[(metadata["sample_id"], metadata["dose"], metadata["intervention"])].append(row)
    per_sample_dose = []
    doses = sorted({key[1] for key in grouped})
    for sample_id in sample_ids:
        for dose in doses:
            core = grouped[(sample_id, dose, "core")]
            random_rows = grouped[(sample_id, dose, "matched_random")]
            per_sample_dose.append(
                {
                    "sample_id": sample_id,
                    "dose": dose,
                    "core": group_summary(core),
                    "matched_random": group_summary(random_rows),
                    "pass_rate_difference_core_minus_random": group_summary(core)["semantic_pass_rate"]
                    - group_summary(random_rows)["semantic_pass_rate"],
                    "mean_nll_difference_core_minus_random": group_summary(core)["mean_nll"]
                    - group_summary(random_rows)["mean_nll"],
                }
            )

    pooled = []
    for dose in doses:
        core = sorted(
            [
                row
                for row in necessity_rows
                if row["variant_metadata"]["dose"] == dose and row["variant_metadata"]["intervention"] == "core"
            ],
            key=lambda row: (row["variant_metadata"]["sample_id"], row["variant_metadata"]["replicate"]),
        )
        random_rows = sorted(
            [
                row
                for row in necessity_rows
                if row["variant_metadata"]["dose"] == dose
                and row["variant_metadata"]["intervention"] == "matched_random"
            ],
            key=lambda row: (row["variant_metadata"]["sample_id"], row["variant_metadata"]["replicate"]),
        )
        entry = {
            "dose": dose,
            "core": group_summary(core),
            "matched_random": group_summary(random_rows),
        }
        if dose < 1:
            entry["run_level_paired"] = paired_comparison(core, random_rows)
            sample_pass_differences = []
            sample_nll_differences = []
            for sample_id in sample_ids:
                core_sample = grouped[(sample_id, dose, "core")]
                random_sample = grouped[(sample_id, dose, "matched_random")]
                sample_pass_differences.append(
                    group_summary(core_sample)["semantic_pass_rate"]
                    - group_summary(random_sample)["semantic_pass_rate"]
                )
                sample_nll_differences.append(
                    group_summary(core_sample)["mean_nll"] - group_summary(random_sample)["mean_nll"]
                )
            entry["sample_level"] = {
                "n_samples": len(sample_ids),
                "mean_pass_rate_difference": mean(sample_pass_differences),
                "pass_difference_wilcoxon": wilcoxon(sample_pass_differences),
                "mean_nll_difference": mean(sample_nll_differences),
                "nll_difference_wilcoxon": wilcoxon(sample_nll_differences),
            }
        else:
            entry["comparison_note"] = (
                "The full C7 removal is unique and evaluated once per sample; random controls have 10 replicates. "
                "Results are descriptive rather than paired-replicate inference."
            )
        pooled.append(entry)
    return {"sufficiency": sufficiency, "necessity_per_sample_dose": per_sample_dose, "necessity_pooled": pooled}


def analyze_shell(rows: list[dict[str, Any]], sample_ids: list[str]) -> dict[str, Any]:
    shell_rows = [row for row in rows if row["family"] == "shell_swap"]
    learned = [row for row in shell_rows if row["variant_metadata"]["shell_type"] == "learned"]
    random_rows = [row for row in shell_rows if row["variant_metadata"]["shell_type"] == "matched_random"]
    learned_map = {(row["variant_metadata"]["source_sample_id"], row["target_sample_id"]): row for row in learned}
    matrix = []
    for source in sample_ids:
        for target in sample_ids:
            row = learned_map[(source, target)]
            controls = [
                candidate
                for candidate in random_rows
                if candidate["variant_metadata"]["source_sample_id"] == source
                and candidate["target_sample_id"] == target
            ]
            matrix.append(
                {
                    "source_sample_id": source,
                    "target_sample_id": target,
                    "diagonal": source == target,
                    "learned_pass": row["automatic_semantic_pass"],
                    "learned_nll": row["gold_proxy"]["mean_nll"],
                    "random_pass_rate": group_summary(controls)["semantic_pass_rate"],
                    "random_mean_nll": group_summary(controls)["mean_nll"],
                }
            )
    by_seed = []
    for replicate in sorted({row["variant_metadata"]["replicate"] for row in random_rows}):
        control = sorted(
            [row for row in random_rows if row["variant_metadata"]["replicate"] == replicate],
            key=lambda row: (row["variant_metadata"]["source_sample_id"], row["target_sample_id"]),
        )
        treatment = sorted(
            learned,
            key=lambda row: (row["variant_metadata"]["source_sample_id"], row["target_sample_id"]),
        )
        by_seed.append({"replicate": replicate, **paired_comparison(treatment, control)})
    diagonal = [row for row in learned if row["variant_metadata"]["source_sample_id"] == row["target_sample_id"]]
    off_diagonal = [row for row in learned if row["variant_metadata"]["source_sample_id"] != row["target_sample_id"]]
    return {
        "learned": group_summary(learned),
        "matched_random": group_summary(random_rows),
        "learned_diagonal": group_summary(diagonal),
        "learned_off_diagonal": group_summary(off_diagonal),
        "paired_learned_vs_each_random_seed": by_seed,
        "matrix": matrix,
    }


def analyze_union(rows: list[dict[str, Any]], shell: dict[str, Any]) -> dict[str, Any]:
    union_rows = [row for row in rows if row["family"] == "pair_union"]
    learned = [row for row in union_rows if row["variant_metadata"]["variant_type"] == "learned_union"]
    controls = [row for row in union_rows if row["variant_metadata"]["variant_type"] == "matched_random_expansion"]
    learned_shell = {
        (row["source_sample_id"], row["target_sample_id"]): bool(row["learned_pass"]) for row in shell["matrix"]
    }
    learned_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    controls_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in learned:
        metadata = row["variant_metadata"]
        learned_by_pair[(metadata["left"], metadata["right"])].append(row)
    for row in controls:
        controls_by_variant[row["variant_id"]].append(row)

    pair_summary = []
    for pair, pair_rows in learned_by_pair.items():
        left, right = pair
        by_target = {row["target_sample_id"]: row for row in pair_rows}
        pair_controls = [
            variant_rows
            for variant_rows in controls_by_variant.values()
            if (variant_rows[0]["variant_metadata"]["left"], variant_rows[0]["variant_metadata"]["right"]) == pair
        ]
        left_subnet_dual = learned_shell[(left, left)] and learned_shell[(left, right)]
        right_subnet_dual = learned_shell[(right, left)] and learned_shell[(right, right)]
        union_dual = bool(by_target[left]["automatic_semantic_pass"] and by_target[right]["automatic_semantic_pass"])
        pair_summary.append(
            {
                "left": left,
                "right": right,
                "kept_neurons": pair_rows[0]["mask_summary"]["kept_neurons"],
                "left_pass": by_target[left]["automatic_semantic_pass"],
                "right_pass": by_target[right]["automatic_semantic_pass"],
                "union_dual_pass": union_dual,
                "left_subnet_already_dual": left_subnet_dual,
                "right_subnet_already_dual": right_subnet_dual,
                "either_constituent_already_dual": left_subnet_dual or right_subnet_dual,
                "genuine_union_recovery": union_dual and not (left_subnet_dual or right_subnet_dual),
                "negative_union_interference": not union_dual and (left_subnet_dual or right_subnet_dual),
                "control_dual_passes": sum(
                    all(row["automatic_semantic_pass"] for row in variant) for variant in pair_controls
                ),
                "control_variants": len(pair_controls),
                "control_target_passes": sum(
                    row["automatic_semantic_pass"] for variant in pair_controls for row in variant
                ),
                "control_target_evaluations": sum(len(variant) for variant in pair_controls),
            }
        )

    comparisons = []
    for base_side in ("left", "right"):
        for replicate in sorted({row["variant_metadata"]["replicate"] for row in controls}):
            treatment = []
            control = []
            treatment_pair = []
            control_pair = []
            for pair in sorted(learned_by_pair):
                base = pair[0] if base_side == "left" else pair[1]
                learned_pair = sorted(learned_by_pair[pair], key=lambda row: row["target_sample_id"])
                control_pair_rows = sorted(
                    [
                        rows_for_variant
                        for rows_for_variant in controls_by_variant.values()
                        if (
                            rows_for_variant[0]["variant_metadata"]["left"],
                            rows_for_variant[0]["variant_metadata"]["right"],
                        )
                        == pair
                        and rows_for_variant[0]["variant_metadata"]["base_sample_id"] == base
                        and rows_for_variant[0]["variant_metadata"]["replicate"] == replicate
                    ][0],
                    key=lambda row: row["target_sample_id"],
                )
                treatment.extend(learned_pair)
                control.extend(control_pair_rows)
                treatment_pair.append(
                    {
                        "automatic_semantic_pass": all(row["automatic_semantic_pass"] for row in learned_pair),
                        "gold_proxy": {"mean_nll": mean([row["gold_proxy"]["mean_nll"] for row in learned_pair])},
                    }
                )
                control_pair.append(
                    {
                        "automatic_semantic_pass": all(row["automatic_semantic_pass"] for row in control_pair_rows),
                        "gold_proxy": {"mean_nll": mean([row["gold_proxy"]["mean_nll"] for row in control_pair_rows])},
                    }
                )
            comparisons.append(
                {
                    "base_side": base_side,
                    "replicate": replicate,
                    "target_level": paired_comparison(treatment, control),
                    "pair_dual_level": paired_comparison(treatment_pair, control_pair),
                }
            )

    return {
        "learned_target_level": group_summary(learned),
        "control_target_level": group_summary(controls),
        "learned_dual_pass_pairs": sum(row["union_dual_pass"] for row in pair_summary),
        "total_pairs": len(pair_summary),
        "control_dual_pass_variants": sum(row["control_dual_passes"] for row in pair_summary),
        "total_control_variants": sum(row["control_variants"] for row in pair_summary),
        "genuine_union_recovery_pairs": sum(row["genuine_union_recovery"] for row in pair_summary),
        "negative_union_interference_pairs": sum(row["negative_union_interference"] for row in pair_summary),
        "pair_summary": pair_summary,
        "paired_learned_vs_control_slots": comparisons,
    }


def sqlite_type(values: list[Any]) -> str:
    present = [value for value in values if value is not None]
    if present and all(isinstance(value, (bool, int)) for value in present):
        return "INTEGER"
    if present and all(isinstance(value, (bool, int, float)) for value in present):
        return "REAL"
    return "TEXT"


def write_sqlite(path: str | Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with sqlite3.connect(output) as connection:
        for name, rows in tables.items():
            if not rows:
                continue
            columns = list(rows[0])
            declaration = ", ".join(f'"{column}" {sqlite_type([row[column] for row in rows])}' for column in columns)
            connection.execute(f'CREATE TABLE "{name}" ({declaration})')
            placeholders = ", ".join("?" for _ in columns)
            values = [
                [
                    int(value) if isinstance(value, bool) else json.dumps(value) if isinstance(value, dict) else value
                    for value in (row[column] for column in columns)
                ]
                for row in rows
            ]
            connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})', values)


def main() -> None:
    args = parse_args()
    result_path = Path(args.result_file).resolve()
    run = read_json(result_path)
    if not run.get("complete"):
        raise ValueError("Phase 6C analysis requires a complete run.")
    rows = load_evaluations(run["evaluations_file"])
    variants = {variant["variant_id"]: variant for variant in run["variants"]}
    ids = [row["evaluation_id"] for row in rows]
    checks = {
        "evaluation_count_matches": len(rows) == run["expected_evaluations"],
        "evaluation_ids_unique": len(ids) == len(set(ids)),
        "all_variants_referenced": {row["variant_id"] for row in rows} == set(variants),
        "variant_hashes_match": all(row["kept_sha256"] == variants[row["variant_id"]]["kept_sha256"] for row in rows),
        "semantic_pass_fields_match": all(
            row["automatic_semantic_pass"] == row["generation"]["semantic"]["automatic_pass"] for row in rows
        ),
        "all_rows_have_normalized_proxy": all(math.isfinite(row["gold_proxy"]["mean_nll"]) for row in rows),
    }
    if not all(checks.values()):
        raise ValueError(f"Phase 6C validation failed: {checks}")

    core = analyze_core(rows, run["sample_ids"])
    shell = analyze_shell(rows, run["sample_ids"])
    union = analyze_union(rows, shell)
    structural_candidates = [
        {
            "variant_id": "union__bicycle_clock__violin_kitchen",
            "selection_reason": "negative interference: both constituent masks were dual-capable but their union failed both",
        },
        {
            "variant_id": "union__giraffes_trees__bicycles_train",
            "selection_reason": "smallest genuine recovery union: neither constituent was dual-capable and the union passed both",
        },
        {
            "variant_id": "union__bicycle_clock__dogs_car",
            "selection_reason": "representative genuine recovery union at a mid-range width",
        },
        {
            "variant_id": "union__toilet_spatial__dogs_car",
            "selection_reason": "largest learned union that still failed one constituent target",
        },
    ]
    output = {
        "complete": True,
        "interpretation": (
            "Phase 6C tests fixed-mask functional interventions. Statistical p-values over run-level replicates are "
            "descriptive for these seven selected samples and do not imply population-level generalization."
        ),
        "sources": {"result_file": str(result_path), "evaluations_file": run["evaluations_file"]},
        "validation_checks": checks,
        "core": core,
        "shell": shell,
        "union": union,
        "structural_candidates": structural_candidates,
    }
    write_json(args.output_file, output)
    if args.sqlite_file:
        necessity_table = []
        for row in core["necessity_per_sample_dose"]:
            necessity_table.append(
                {
                    "sample_id": row["sample_id"],
                    "dose": row["dose"],
                    "core_pass_rate": row["core"]["semantic_pass_rate"],
                    "random_pass_rate": row["matched_random"]["semantic_pass_rate"],
                    "pass_rate_difference": row["pass_rate_difference_core_minus_random"],
                    "core_mean_nll": row["core"]["mean_nll"],
                    "random_mean_nll": row["matched_random"]["mean_nll"],
                    "nll_difference": row["mean_nll_difference_core_minus_random"],
                }
            )
        write_sqlite(
            args.sqlite_file,
            {
                "necessity": necessity_table,
                "shell_matrix": shell["matrix"],
                "union_pairs": union["pair_summary"],
            },
        )
    print(
        json.dumps(
            {
                "validation": all(checks.values()),
                "shell_learned_pass_rate": shell["learned"]["semantic_pass_rate"],
                "shell_random_pass_rate": shell["matched_random"]["semantic_pass_rate"],
                "learned_union_dual_pairs": union["learned_dual_pass_pairs"],
                "total_pairs": union["total_pairs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
