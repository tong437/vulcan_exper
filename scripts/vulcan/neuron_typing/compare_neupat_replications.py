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

"""Compare NeuPAT roles from two independently sampled probe runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROLES = ("language", "multimodal", "shared", "reserve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two NeuPAT probing replications.")
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--comparison_dir", required=True)
    parser.add_argument("--output_file", required=True)
    return parser.parse_args()


def mask_overlap(reference: pd.Series, comparison: pd.Series) -> dict[str, float | int]:
    left = reference.astype(bool)
    right = comparison.astype(bool)
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    return {
        "reference_count": int(left.sum()),
        "comparison_count": int(right.sum()),
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
        "reference_recall": intersection / int(left.sum()) if bool(left.any()) else 1.0,
        "comparison_recall": intersection / int(right.sum()) if bool(right.any()) else 1.0,
    }


def compare_tables(reference: pd.DataFrame, comparison: pd.DataFrame) -> dict[str, Any]:
    keys = ["layer", "neuron_idx"]
    required = {*keys, "neupat_role", *(f"neupat_{role}" for role in ROLES)}
    for name, table in (("reference", reference), ("comparison", comparison)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} score table is missing columns: {missing}.")
        if bool(table.duplicated(keys).any()):
            raise ValueError(f"{name} score table has duplicate neuron keys.")

    merged = reference.merge(comparison, on=keys, suffixes=("_reference", "_comparison"), validate="one_to_one")
    if len(merged) != len(reference) or len(merged) != len(comparison):
        raise ValueError("Replication score tables do not contain identical neuron keys.")

    role_overlap = {
        role: mask_overlap(merged[f"neupat_{role}_reference"], merged[f"neupat_{role}_comparison"]) for role in ROLES
    }
    protect_reference = merged["neupat_language_reference"] | merged["neupat_shared_reference"]
    protect_comparison = merged["neupat_language_comparison"] | merged["neupat_shared_comparison"]
    correlations = {}
    for column in (
        "neupat_text_importance",
        "neupat_vision_importance",
        "neupat_overall_importance",
        "neupat_preference",
    ):
        if f"{column}_reference" in merged and f"{column}_comparison" in merged:
            value = merged[[f"{column}_reference", f"{column}_comparison"]].corr(method="spearman").iloc[0, 1]
            correlations[column] = None if pd.isna(value) else float(value)
    return {
        "neurons": len(merged),
        "layers": int(merged["layer"].nunique()),
        "exact_role_agreement": float((merged["neupat_role_reference"] == merged["neupat_role_comparison"]).mean()),
        "role_overlap": role_overlap,
        "language_protection_overlap": mask_overlap(protect_reference, protect_comparison),
        "score_spearman": correlations,
    }


def manifest_disjointness(reference_dir: Path, comparison_dir: Path) -> dict[str, Any]:
    result = {}
    for modality in ("text", "vision"):
        reference = json.loads((reference_dir / f"{modality}_manifest.json").read_text(encoding="utf-8"))
        comparison = json.loads((comparison_dir / f"{modality}_manifest.json").read_text(encoding="utf-8"))
        reference_indices = set(reference["source_indices"])
        comparison_indices = set(comparison["source_indices"])
        reference_images = set(reference.get("image_ids", []))
        comparison_images = set(comparison.get("image_ids", []))
        result[modality] = {
            "reference_rows": int(reference["num_rows"]),
            "comparison_rows": int(comparison["num_rows"]),
            "source_index_overlap": len(reference_indices & comparison_indices),
            "image_id_overlap": len(reference_images & comparison_images),
            "disjoint": not (reference_indices & comparison_indices) and not (reference_images & comparison_images),
        }
    return result


def main() -> None:
    args = parse_args()
    reference_dir = Path(args.reference_dir)
    comparison_dir = Path(args.comparison_dir)
    comparison = compare_tables(
        pd.read_parquet(reference_dir / "neupat_scores.parquet"),
        pd.read_parquet(comparison_dir / "neupat_scores.parquet"),
    )
    disjointness = manifest_disjointness(reference_dir, comparison_dir)
    report = {
        "artifact_version": 1,
        "method": "neupat_independent_probe_replication",
        "reference_dir": str(reference_dir.resolve()),
        "comparison_dir": str(comparison_dir.resolve()),
        "manifest_disjointness": disjointness,
        **comparison,
        "passed_disjointness": all(item["disjoint"] for item in disjointness.values()),
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed_disjointness"]:
        raise ValueError("Probe replications are not sample-disjoint.")


if __name__ == "__main__":
    main()
