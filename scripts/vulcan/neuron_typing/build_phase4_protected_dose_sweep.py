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

"""Build equal-budget Phase-4 dose candidates with NeuPAT and mapping protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from phase3_structural_utils import masks_to_neuron_ids, sha256_file
from run_phase2_ablation import (
    build_score_vector,
    build_secondary_vector,
    get_layer_dims,
    infer_score_columns,
    read_score_table,
    select_k_indices_deterministic,
    summarize_masks,
)


LANGUAGE_PROTECT_COLUMN = "neupat_language_shared_protect"
MAPPING_PROTECT_COLUMN = "mapping_top1_protect"
COMBINED_PROTECT_COLUMN = "neupat_mapping_protect"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a language-protected Phase-4 deletion-dose sweep.")
    parser.add_argument("--phase4_score_file", required=True, help="Phase-4 score table containing mapping_signal.")
    parser.add_argument("--neupat_score_file", required=True, help="NeuPAT score table containing role columns.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--band_start", type=float, default=0.05)
    parser.add_argument(
        "--deletion_ratios",
        default="0.01,0.025,0.05,0.10",
        help="Comma-separated deletion budgets; each band is [band_start, band_start + ratio).",
    )
    parser.add_argument("--mapping_protect_ratio", type=float, default=0.01)
    parser.add_argument("--expected_num_layers", type=int, default=24)
    parser.add_argument("--expected_layer_width", type=int, default=3584)
    return parser.parse_args()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def ratio_slug(value: float) -> str:
    return f"p{round(value * 1000):03d}"


def _rank_boundary(width: int, ratio: float) -> int:
    """Apply ceil without promoting exact decimal boundaries due to float addition."""
    return math.ceil(width * ratio - 1e-12)


def condition_columns(ratio: float) -> dict[str, str]:
    slug = ratio_slug(ratio)
    return {
        "q_only": f"qband_d{slug}_delete",
        "mapping": f"qband_d{slug}_mapping_protected_delete",
        "combined": f"qband_d{slug}_neupat_mapping_protected_delete",
    }


def parse_ratios(value: str) -> list[float]:
    ratios = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not ratios or any(ratio <= 0.0 for ratio in ratios):
        raise ValueError("Deletion ratios must be a non-empty set of positive values.")
    return ratios


def _validate_unique_keys(table: pd.DataFrame, name: str) -> None:
    required = {"layer", "neuron_idx"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{name} is missing key columns: {missing}")
    if table.duplicated(["layer", "neuron_idx"]).any():
        raise ValueError(f"{name} contains duplicate (layer, neuron_idx) keys.")


def _merge_neupat_roles(phase4: pd.DataFrame, neupat: pd.DataFrame) -> pd.DataFrame:
    _validate_unique_keys(phase4, "Phase-4 score table")
    _validate_unique_keys(neupat, "NeuPAT score table")
    role_columns = {"neupat_language", "neupat_shared"}
    missing = sorted(role_columns - set(neupat.columns))
    if missing:
        raise ValueError(f"NeuPAT score table is missing role columns: {missing}")
    roles = neupat[["layer", "neuron_idx", *sorted(role_columns)]].copy()
    merged = phase4.merge(roles, on=["layer", "neuron_idx"], how="left", validate="one_to_one")
    if len(merged) != len(phase4) or merged[list(role_columns)].isna().any().any():
        raise ValueError("Phase-4 and NeuPAT score tables do not contain the same neuron keys.")
    return merged


def _protected_candidate(
    *,
    original_band: list[int],
    ranked_ids: list[int],
    band_end: int,
    protected: set[int],
    dead: set[int],
) -> tuple[list[int], list[int], int | None]:
    selected = [neuron for neuron in original_band if neuron not in protected]
    refill: list[int] = []
    if len(selected) == len(original_band):
        return selected, refill, None
    for rank, neuron in enumerate(ranked_ids[band_end:], start=band_end):
        if neuron in protected or neuron in selected or neuron in dead:
            continue
        refill.append(neuron)
        if len(selected) + len(refill) == len(original_band):
            return selected + refill, refill, rank
    raise RuntimeError("Could not refill an exact deletion budget from the lower same-layer q ranks.")


def build_protected_dose_sweep(
    phase4_scores: pd.DataFrame,
    neupat_scores: pd.DataFrame,
    *,
    band_start: float,
    deletion_ratios: list[float],
    mapping_protect_ratio: float,
    expected_num_layers: int,
    expected_layer_width: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 <= band_start < 1.0:
        raise ValueError("band_start must be in [0, 1).")
    if any(ratio <= 0.0 or band_start + ratio > 1.0 for ratio in deletion_ratios):
        raise ValueError("Every deletion ratio must be positive and end no later than rank 1.0.")
    if not 0.0 < mapping_protect_ratio <= 1.0:
        raise ValueError("mapping_protect_ratio must be in (0, 1].")

    table = _merge_neupat_roles(phase4_scores, neupat_scores)
    required = {"q_multimodal", "r_multimodal", "mapping_signal"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Phase-4 score table is missing columns: {missing}")
    layer_col, neuron_col, score_cols, _ = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    if sorted(layer_dims) != list(range(expected_num_layers)):
        raise ValueError("Score table does not contain the expected contiguous layer set.")
    if any(width != expected_layer_width for width in layer_dims.values()):
        raise ValueError(f"Expected width {expected_layer_width} in every layer, got {layer_dims}.")

    table[LANGUAGE_PROTECT_COLUMN] = table["neupat_language"].astype(bool) | table["neupat_shared"].astype(bool)
    table[MAPPING_PROTECT_COLUMN] = False
    table[COMBINED_PROTECT_COLUMN] = False
    columns_by_ratio = {ratio: condition_columns(ratio) for ratio in deletion_ratios}
    for columns in columns_by_ratio.values():
        for column in columns.values():
            table[column] = False

    per_layer: dict[str, Any] = {}
    masks: dict[str, dict[int, torch.Tensor]] = {
        column: {layer: torch.zeros(width, dtype=torch.bool) for layer, width in layer_dims.items()}
        for columns in columns_by_ratio.values()
        for column in columns.values()
    }
    for layer, group in table.groupby(layer_col, sort=True):
        layer = int(layer)
        width = layer_dims[layer]
        ordered = group.sort_values(neuron_col)
        neuron_ids = ordered[neuron_col].to_numpy(dtype=np.int64)
        if len(neuron_ids) != width or not np.array_equal(neuron_ids, np.arange(width)):
            raise ValueError(f"Layer {layer} does not have dense neuron ids [0, {width}).")
        row_index = dict(zip(ordered[neuron_col].astype(int), ordered.index))
        q_scores = build_score_vector(ordered, neuron_col, score_cols["multimodal"], width)
        q_secondary = build_secondary_vector(ordered, neuron_col, "multimodal", width)
        ranked, _ = select_k_indices_deterministic(
            q_scores,
            width,
            secondary_key=q_secondary,
            neuron_ids=torch.arange(width, dtype=torch.long),
        )
        ranked_ids = ranked.tolist()
        mapping_scores = build_score_vector(ordered, neuron_col, "mapping_signal", width)
        mapping_count = max(1, math.ceil(width * mapping_protect_ratio))
        mapping_selected, mapping_cutoff = select_k_indices_deterministic(
            mapping_scores,
            mapping_count,
            neuron_ids=torch.arange(width, dtype=torch.long),
        )
        mapping_protected = set(mapping_selected.tolist())
        language_protected = {
            int(row[neuron_col]) for _, row in ordered.iterrows() if bool(row[LANGUAGE_PROTECT_COLUMN])
        }
        combined_protected = language_protected | mapping_protected
        dead = {int(row[neuron_col]) for _, row in ordered.iterrows() if "is_dead" in ordered and bool(row["is_dead"])}
        table.loc[[row_index[neuron] for neuron in mapping_protected], MAPPING_PROTECT_COLUMN] = True
        table.loc[[row_index[neuron] for neuron in combined_protected], COMBINED_PROTECT_COLUMN] = True
        layer_result: dict[str, Any] = {
            "width": width,
            "language_shared_protected": len(language_protected),
            "mapping_protected": len(mapping_protected),
            "combined_protected": len(combined_protected),
            "mapping_unique_beyond_language_shared": len(mapping_protected - language_protected),
            "mapping_cutoff": mapping_cutoff,
            "doses": {},
        }
        start = _rank_boundary(width, band_start)
        for ratio in deletion_ratios:
            end = _rank_boundary(width, band_start + ratio)
            original = ranked_ids[start:end]
            if not original:
                raise ValueError(f"Deletion ratio {ratio} selects no neurons in layer {layer}.")
            mapping_selected_ids, mapping_refill, mapping_max_rank = _protected_candidate(
                original_band=original,
                ranked_ids=ranked_ids,
                band_end=end,
                protected=mapping_protected,
                dead=dead,
            )
            combined_selected_ids, combined_refill, combined_max_rank = _protected_candidate(
                original_band=original,
                ranked_ids=ranked_ids,
                band_end=end,
                protected=combined_protected,
                dead=dead,
            )
            selected_by_kind = {
                "q_only": original,
                "mapping": mapping_selected_ids,
                "combined": combined_selected_ids,
            }
            for kind, selected in selected_by_kind.items():
                column = columns_by_ratio[ratio][kind]
                if len(selected) != len(original):
                    raise RuntimeError(f"Layer {layer} condition {column} lost its exact deletion budget.")
                protection = (
                    set() if kind == "q_only" else mapping_protected if kind == "mapping" else combined_protected
                )
                if set(selected) & protection:
                    raise RuntimeError(f"Layer {layer} condition {column} intersects its protection set.")
                table.loc[[row_index[neuron] for neuron in selected], column] = True
                masks[column][layer][torch.tensor(selected, dtype=torch.long)] = True
            layer_result["doses"][ratio_slug(ratio)] = {
                "deletion_ratio": ratio,
                "rank_start": start,
                "rank_end_exclusive": end,
                "budget": len(original),
                "mapping_overlap": len(set(original) & mapping_protected),
                "language_shared_overlap": len(set(original) & language_protected),
                "combined_overlap": len(set(original) & combined_protected),
                "mapping_refilled": len(mapping_refill),
                "mapping_refill_max_rank": mapping_max_rank,
                "combined_refilled": len(combined_refill),
                "combined_refill_max_rank": combined_max_rank,
                "combined_refill_max_percentile": combined_max_rank / width if combined_max_rank is not None else None,
            }
        per_layer[str(layer)] = layer_result

    conditions: dict[str, Any] = {}
    for ratio, columns in columns_by_ratio.items():
        conditions[ratio_slug(ratio)] = {
            "deletion_ratio": ratio,
            "band_start": band_start,
            "band_end": band_start + ratio,
            "columns": columns,
            "mask_summaries": {kind: summarize_masks(masks[column]) for kind, column in columns.items()},
            "mask_hashes": {
                kind: _canonical_hash(masks_to_neuron_ids(masks[column])) for kind, column in columns.items()
            },
            "verification": {
                "equal_budget": len({int(table[column].sum()) for column in columns.values()}) == 1,
                "mapping_disjoint": not bool((table[columns["mapping"]] & table[MAPPING_PROTECT_COLUMN]).any()),
                "combined_disjoint": not bool((table[columns["combined"]] & table[COMBINED_PROTECT_COLUMN]).any()),
            },
        }
    metadata = {
        "artifact_version": 1,
        "name": "phase4_neupat_mapping_protected_dose_sweep",
        "status": "pre_hook_screen",
        "structural_pruning_allowed": False,
        "hypothesis": (
            "At an equal, reduced deletion budget, protecting causally language-sensitive NeuPAT language U shared "
            "neurons plus mapping top-1% neurons improves broad language retention without losing multimodal safety."
        ),
        "selection_rule": (
            "Run C4 first and choose the largest combined-protection dose with delta NLL <= 0.05 and no more than "
            "0.02 NLL degradation versus its equal-budget q-only control; then run Caption, POPE, and VQA-Med."
        ),
        "band_start": band_start,
        "deletion_ratios": deletion_ratios,
        "mapping_protect_ratio": mapping_protect_ratio,
        "protection_columns": {
            "language_shared": LANGUAGE_PROTECT_COLUMN,
            "mapping": MAPPING_PROTECT_COLUMN,
            "combined": COMBINED_PROTECT_COLUMN,
        },
        "conditions": conditions,
        "per_layer": per_layer,
        "totals": {
            "language_shared_protected": int(table[LANGUAGE_PROTECT_COLUMN].sum()),
            "mapping_protected": int(table[MAPPING_PROTECT_COLUMN].sum()),
            "combined_protected": int(table[COMBINED_PROTECT_COLUMN].sum()),
            "mapping_unique_beyond_language_shared": int(
                (table[MAPPING_PROTECT_COLUMN] & ~table[LANGUAGE_PROTECT_COLUMN]).sum()
            ),
        },
        "warning": "No candidate may be structurally materialized before every formal hook gate passes.",
    }
    return table, metadata


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    phase4_path = Path(args.phase4_score_file).resolve()
    neupat_path = Path(args.neupat_score_file).resolve()
    table, metadata = build_protected_dose_sweep(
        read_score_table(phase4_path),
        read_score_table(neupat_path),
        band_start=args.band_start,
        deletion_ratios=parse_ratios(args.deletion_ratios),
        mapping_protect_ratio=args.mapping_protect_ratio,
        expected_num_layers=args.expected_num_layers,
        expected_layer_width=args.expected_layer_width,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_output = output_dir / "dose_sweep_scores.parquet"
    metadata_output = output_dir / "dose_sweep_metadata.json"
    metadata["provenance"] = {
        "phase4_score_file": {"path": str(phase4_path), "sha256": sha256_file(phase4_path)},
        "neupat_score_file": {"path": str(neupat_path), "sha256": sha256_file(neupat_path)},
    }
    metadata["outputs"] = {
        "score_file": str(score_output.resolve()),
        "metadata_file": str(metadata_output.resolve()),
    }
    _atomic_parquet(table, score_output)
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"totals": metadata["totals"], "conditions": metadata["conditions"]}, indent=2))


if __name__ == "__main__":
    main()
