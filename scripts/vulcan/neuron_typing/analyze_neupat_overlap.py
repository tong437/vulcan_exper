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

"""Compare NeuPAT roles with q/r typing, mapping protection, and q-band masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from run_phase2_ablation import (
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    parse_ablation_spec,
    read_score_table,
)


ROLE_COLUMNS = ("neupat_language", "neupat_multimodal", "neupat_shared", "neupat_reserve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NeuPAT overlap with existing neuron scores.")
    parser.add_argument("--neupat_score_file", required=True)
    parser.add_argument("--q_score_file", required=True)
    parser.add_argument("--mapping_score_file", default=None)
    parser.add_argument("--mapping_score_column", default="mapping_signal")
    parser.add_argument("--mapping_protect_ratio", type=float, default=0.01)
    parser.add_argument("--qband_start", type=float, default=0.05)
    parser.add_argument("--qband_end", type=float, default=0.20)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def _validate_keys(table: pd.DataFrame, name: str) -> None:
    required = {"layer", "neuron_idx"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}.")
    if bool(table.duplicated(["layer", "neuron_idx"]).any()):
        raise ValueError(f"{name} contains duplicate layer/neuron keys.")


def _merge_one_to_one(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = left.merge(right, on=["layer", "neuron_idx"], how="left", validate="one_to_one")
    right_columns = [column for column in right.columns if column not in {"layer", "neuron_idx"}]
    if right_columns and bool(merged[right_columns].isna().all(axis=1).any()):
        raise ValueError(f"{name} does not cover every NeuPAT neuron key.")
    return merged


def masks_to_column(table: pd.DataFrame, masks: dict[int, torch.Tensor]) -> pd.Series:
    return pd.Series(
        [bool(masks[int(layer_idx)][int(neuron_idx)]) for layer_idx, neuron_idx in zip(table.layer, table.neuron_idx)],
        index=table.index,
        dtype=bool,
    )


def build_top_score_mask(table: pd.DataFrame, score_column: str, ratio: float) -> pd.Series:
    if score_column not in table.columns:
        raise ValueError(f"Mapping score column {score_column!r} is absent.")
    if not 0.0 < ratio < 1.0:
        raise ValueError("mapping_protect_ratio must be strictly between zero and one.")
    selected = pd.Series(False, index=table.index)
    for _, group in table.groupby("layer", sort=True):
        count = max(1, int(torch.ceil(torch.tensor(len(group) * ratio)).item()))
        ranked = group.sort_values([score_column, "neuron_idx"], ascending=[False, True], kind="stable")
        selected.loc[ranked.index[:count]] = True
    return selected


def overlap_statistics(left: pd.Series, right: pd.Series) -> dict[str, float | int]:
    left_bool = left.astype(bool)
    right_bool = right.astype(bool)
    both = int((left_bool & right_bool).sum())
    left_only = int((left_bool & ~right_bool).sum())
    right_only = int((~left_bool & right_bool).sum())
    neither = int((~left_bool & ~right_bool).sum())
    union = both + left_only + right_only
    # Haldane-Anscombe correction keeps enrichment finite for sparse masks.
    odds_ratio = ((both + 0.5) * (neither + 0.5)) / ((left_only + 0.5) * (right_only + 0.5))
    return {
        "intersection": both,
        "left_count": both + left_only,
        "right_count": both + right_only,
        "jaccard": both / union if union else 1.0,
        "left_recall": both / (both + left_only) if both + left_only else 0.0,
        "right_recall": both / (both + right_only) if both + right_only else 0.0,
        "odds_ratio": odds_ratio,
    }


def build_overlap_outputs(
    neupat: pd.DataFrame,
    q_scores: pd.DataFrame,
    *,
    mapping_scores: pd.DataFrame | None,
    mapping_score_column: str,
    mapping_protect_ratio: float,
    qband_start: float,
    qband_end: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _validate_keys(neupat, "NeuPAT score table")
    _validate_keys(q_scores, "q/r score table")
    missing_roles = sorted(set(ROLE_COLUMNS) - set(neupat.columns))
    if missing_roles:
        raise ValueError(f"NeuPAT score table is missing role columns: {missing_roles}.")
    q_keep = [
        column for column in q_scores.columns if column not in neupat.columns or column in {"layer", "neuron_idx"}
    ]
    combined = _merge_one_to_one(neupat, q_scores[q_keep], "q/r score table")
    if mapping_scores is not None:
        _validate_keys(mapping_scores, "mapping score table")
        mapping_keep = ["layer", "neuron_idx", mapping_score_column]
        combined = _merge_one_to_one(combined, mapping_scores[mapping_keep], "mapping score table")

    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(combined, None)
    layer_dims = get_layer_dims(combined, layer_col, neuron_col)
    qband_spec = parse_ablation_spec(f"rank_band:multimodal:{qband_start}:{qband_end}", 42)
    qband_masks = build_type_mask(
        combined,
        qband_spec,
        layer_col,
        neuron_col,
        score_cols,
        activation_col,
        layer_dims,
        None,
        "per_layer",
        1.0,
        0.0,
    )
    combined["qband_delete_candidate"] = masks_to_column(combined, qband_masks)
    if mapping_scores is not None:
        combined["mapping_protect"] = build_top_score_mask(combined, mapping_score_column, mapping_protect_ratio)
    else:
        combined["mapping_protect"] = False
    combined["neupat_role_protect"] = combined["neupat_language"] | combined["neupat_shared"]
    combined["joint_protect"] = combined["neupat_role_protect"] | combined["mapping_protect"]
    combined["qband_joint_eligible"] = combined["qband_delete_candidate"] & ~combined["joint_protect"]

    correlations: dict[str, float | None] = {}
    candidate_columns = [
        "q_visual",
        "q_text",
        "q_multimodal",
        "q_unknown",
        "r_visual",
        "r_text",
        "r_multimodal",
        "r_unknown",
        mapping_score_column,
    ]
    importance_columns = ["neupat_text_importance", "neupat_vision_importance", "neupat_overall_importance"]
    for left in importance_columns:
        for right in candidate_columns:
            if left in combined.columns and right in combined.columns:
                value = combined[[left, right]].corr(method="spearman").iloc[0, 1]
                correlations[f"{left}__{right}"] = None if pd.isna(value) else float(value)

    overlaps: dict[str, Any] = {}
    reference_masks = {
        "qband_delete_candidate": combined["qband_delete_candidate"],
        "mapping_protect": combined["mapping_protect"],
    }
    for role_column in ROLE_COLUMNS:
        overlaps[role_column] = {
            name: overlap_statistics(combined[role_column], mask) for name, mask in reference_masks.items()
        }
    per_layer = {
        str(int(layer_idx)): {
            **{column: int(group[column].sum()) for column in ROLE_COLUMNS},
            "qband_delete_candidate": int(group["qband_delete_candidate"].sum()),
            "joint_protected_inside_qband": int((group["qband_delete_candidate"] & group["joint_protect"]).sum()),
            "qband_joint_eligible": int(group["qband_joint_eligible"].sum()),
        }
        for layer_idx, group in combined.groupby("layer", sort=True)
    }
    report = {
        "qband": {"start": qband_start, "end": qband_end},
        "mapping_protect_ratio": mapping_protect_ratio if mapping_scores is not None else None,
        "correlations": correlations,
        "overlaps": overlaps,
        "per_layer": per_layer,
        "global": {
            "neurons": len(combined),
            "qband_delete_candidates": int(combined["qband_delete_candidate"].sum()),
            "joint_protected_inside_qband": int(
                (combined["qband_delete_candidate"] & combined["joint_protect"]).sum()
            ),
            "qband_joint_eligible": int(combined["qband_joint_eligible"].sum()),
        },
    }
    return combined, report


def main() -> None:
    args = parse_args()
    neupat = read_score_table(args.neupat_score_file)
    q_scores = read_score_table(args.q_score_file)
    mapping_scores = read_score_table(args.mapping_score_file) if args.mapping_score_file else None
    combined, report = build_overlap_outputs(
        neupat,
        q_scores,
        mapping_scores=mapping_scores,
        mapping_score_column=args.mapping_score_column,
        mapping_protect_ratio=args.mapping_protect_ratio,
        qband_start=args.qband_start,
        qband_end=args.qband_end,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_dir / "neupat_combined_scores.parquet", index=False)
    (output_dir / "neupat_overlap.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["global"], indent=2))


if __name__ == "__main__":
    main()
