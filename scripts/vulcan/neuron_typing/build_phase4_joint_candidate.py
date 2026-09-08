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

"""Build an equal-budget q-band deletion mask protected by mapping top-1%."""

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


DELETE_COLUMN = "joint_qband_mapping_protected_delete"
PROTECT_COLUMN = "mapping_top1_protect"
ORIGINAL_COLUMN = "original_qband_delete"
REFILL_COLUMN = "joint_qband_refill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mapping-protected Phase-4 joint deletion candidate.")
    parser.add_argument("--score_file", required=True, help="Phase-4 augmented neuron score parquet.")
    parser.add_argument("--phase3_mask", required=True, help="Frozen Phase-3 q 5--20% mask JSON.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--band_start", type=float, default=0.05)
    parser.add_argument("--band_end", type=float, default=0.20)
    parser.add_argument("--mapping_protect_ratio", type=float, default=0.01)
    parser.add_argument("--expected_num_layers", type=int, default=24)
    parser.add_argument("--expected_layer_width", type=int, default=3584)
    return parser.parse_args()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_mask(path: str | Path) -> dict[int, set[int]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(layer): {int(neuron) for neuron in neurons} for layer, neurons in value.items()}


def build_joint_candidate(
    score_table: pd.DataFrame,
    *,
    frozen_phase3_mask: dict[int, set[int]],
    band_start: float,
    band_end: float,
    mapping_protect_ratio: float,
    expected_num_layers: int,
    expected_layer_width: int,
) -> tuple[pd.DataFrame, dict[int, torch.Tensor], dict[str, Any]]:
    if not 0.0 <= band_start < band_end <= 1.0:
        raise ValueError("Band must satisfy 0 <= start < end <= 1.")
    if not 0.0 < mapping_protect_ratio <= 1.0:
        raise ValueError("mapping_protect_ratio must be in (0, 1].")
    required = {"q_multimodal", "r_multimodal", "mapping_signal"}
    missing = sorted(required - set(score_table.columns))
    if missing:
        raise ValueError(f"Phase-4 score file is missing columns: {missing}")

    table = score_table.copy()
    layer_col, neuron_col, score_cols, _ = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    if sorted(layer_dims) != list(range(expected_num_layers)):
        raise ValueError("Score table does not contain the expected contiguous layer set.")
    if any(width != expected_layer_width for width in layer_dims.values()):
        raise ValueError(f"Expected width {expected_layer_width} in every layer, got {layer_dims}.")
    if set(frozen_phase3_mask) != set(layer_dims):
        raise ValueError("Frozen Phase-3 mask layers do not match the score table.")

    masks = {layer: torch.zeros(width, dtype=torch.bool) for layer, width in layer_dims.items()}
    table[ORIGINAL_COLUMN] = False
    table[PROTECT_COLUMN] = False
    table[REFILL_COLUMN] = False
    table[DELETE_COLUMN] = False
    per_layer: dict[str, Any] = {}
    all_overlap: list[tuple[int, int]] = []
    all_refill: list[tuple[int, int]] = []

    for layer, group in table.groupby(layer_col, sort=True):
        layer = int(layer)
        width = layer_dims[layer]
        ordered = group.sort_values(neuron_col)
        neuron_ids = ordered[neuron_col].to_numpy(dtype=np.int64)
        if len(neuron_ids) != width or not np.array_equal(neuron_ids, np.arange(width)):
            raise ValueError(f"Layer {layer} does not have dense neuron ids [0, {width}).")
        q_scores = build_score_vector(ordered, neuron_col, score_cols["multimodal"], width)
        q_secondary = build_secondary_vector(ordered, neuron_col, "multimodal", width)
        q_ranked, _ = select_k_indices_deterministic(
            q_scores,
            width,
            secondary_key=q_secondary,
            neuron_ids=torch.arange(width, dtype=torch.long),
        )
        q_ranked_ids = q_ranked.tolist()
        start = math.ceil(width * band_start)
        end = math.ceil(width * band_end)
        reconstructed_band = set(q_ranked_ids[start:end])
        frozen_band = frozen_phase3_mask[layer]
        if reconstructed_band != frozen_band:
            raise ValueError(f"Layer {layer} reconstructed q-band does not match the frozen Phase-3 mask.")

        mapping_scores = build_score_vector(ordered, neuron_col, "mapping_signal", width)
        mapping_count = max(1, math.ceil(width * mapping_protect_ratio))
        mapping_selected, mapping_cutoff = select_k_indices_deterministic(
            mapping_scores,
            mapping_count,
            neuron_ids=torch.arange(width, dtype=torch.long),
        )
        protected = set(mapping_selected.tolist())
        overlap = sorted(frozen_band & protected)
        selected = set(frozen_band - protected)
        refill = []
        # Refill only below the original 20% boundary. The protected top 0--5%
        # specialist region is never used as a replacement pool.
        for neuron in q_ranked_ids[end:]:
            if neuron in protected or neuron in selected:
                continue
            if "is_dead" in ordered.columns:
                dead_value = ordered.loc[ordered[neuron_col] == neuron, "is_dead"].iloc[0]
                if bool(dead_value):
                    continue
            refill.append(neuron)
            selected.add(neuron)
            if len(selected) == len(frozen_band):
                break
        if len(selected) != len(frozen_band):
            raise RuntimeError(f"Layer {layer} could not refill the exact deletion budget.")
        if selected & protected:
            raise RuntimeError(f"Layer {layer} joint deletion mask intersects mapping protection.")

        masks[layer][torch.tensor(sorted(selected), dtype=torch.long)] = True
        layer_rows = ordered.index.to_numpy()
        table.loc[layer_rows[list(reconstructed_band)], ORIGINAL_COLUMN] = True
        table.loc[layer_rows[list(protected)], PROTECT_COLUMN] = True
        table.loc[layer_rows[refill], REFILL_COLUMN] = True
        table.loc[layer_rows[list(selected)], DELETE_COLUMN] = True
        all_overlap.extend((layer, neuron) for neuron in overlap)
        all_refill.extend((layer, neuron) for neuron in refill)
        rank_lookup = {neuron: rank for rank, neuron in enumerate(q_ranked_ids)}
        per_layer[str(layer)] = {
            "width": width,
            "original_band_rank_start": start,
            "original_band_rank_end_exclusive": end,
            "original_budget": len(frozen_band),
            "mapping_protected": len(protected),
            "protected_inside_original_band": len(overlap),
            "retained_from_original_band": len(frozen_band) - len(overlap),
            "refilled": len(refill),
            "refill_rank_min": min((rank_lookup[n] for n in refill), default=None),
            "refill_rank_max": max((rank_lookup[n] for n in refill), default=None),
            "final_selected": len(selected),
            "mapping_cutoff": mapping_cutoff,
        }

    neuron_ids = masks_to_neuron_ids(masks)
    original_ids = {str(layer): sorted(neurons) for layer, neurons in frozen_phase3_mask.items()}
    protected_ids = {
        str(layer): sorted(table.loc[(table[layer_col] == layer) & table[PROTECT_COLUMN], neuron_col].astype(int))
        for layer in layer_dims
    }
    refill_ids = {
        str(layer): sorted(table.loc[(table[layer_col] == layer) & table[REFILL_COLUMN], neuron_col].astype(int))
        for layer in layer_dims
    }
    metadata = {
        "artifact_version": 1,
        "name": "qband_05_20_mapping_top1_protected",
        "status": "pre_hook_gate",
        "structural_pruning_allowed": False,
        "band_start": band_start,
        "band_end": band_end,
        "mapping_protect_ratio": mapping_protect_ratio,
        "ordering": ["q_multimodal desc", "r_multimodal desc", "neuron_idx asc"],
        "refill_policy": "same-layer q rank immediately below the original band; never refill from top 0--5%",
        "columns": {
            "original": ORIGINAL_COLUMN,
            "protected": PROTECT_COLUMN,
            "refill": REFILL_COLUMN,
            "joint_delete": DELETE_COLUMN,
        },
        "per_layer": per_layer,
        "totals": {
            "original_selected": sum(len(neurons) for neurons in frozen_phase3_mask.values()),
            "mapping_protected": int(table[PROTECT_COLUMN].sum()),
            "protected_inside_original_band": len(all_overlap),
            "refilled": len(all_refill),
            "final_selected": int(table[DELETE_COLUMN].sum()),
        },
        "mask_summary": summarize_masks(masks),
        "hashes": {
            "original_mask": _canonical_hash(original_ids),
            "mapping_protection": _canonical_hash(protected_ids),
            "refill": _canonical_hash(refill_ids),
            "joint_deletion": _canonical_hash(neuron_ids),
        },
        "verification": {
            "frozen_phase3_mask_reproduced": True,
            "same_budget_per_layer": all(
                row["final_selected"] == row["original_budget"] for row in per_layer.values()
            ),
            "joint_delete_disjoint_from_mapping_protection": not bool(
                (table[DELETE_COLUMN] & table[PROTECT_COLUMN]).any()
            ),
            "refill_same_layer_below_band": True,
        },
        "warning": (
            "This candidate must pass hook Caption, all POPE splits, C4 text-only, and VQA-Med gates before "
            "a structural checkpoint may be created."
        ),
    }
    return table, masks, metadata


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    score_path = Path(args.score_file).resolve()
    phase3_path = Path(args.phase3_mask).resolve()
    table, masks, metadata = build_joint_candidate(
        read_score_table(score_path),
        frozen_phase3_mask=_load_mask(phase3_path),
        band_start=args.band_start,
        band_end=args.band_end,
        mapping_protect_ratio=args.mapping_protect_ratio,
        expected_num_layers=args.expected_num_layers,
        expected_layer_width=args.expected_layer_width,
    )
    metadata["provenance"] = {
        "score_file": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "phase3_mask": {"path": str(phase3_path), "sha256": sha256_file(phase3_path)},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_output = output_dir / "neuron_scores_joint_candidate.parquet"
    mask_output = output_dir / "joint_candidate.mask.json"
    metadata_output = output_dir / "joint_candidate.metadata.json"
    _atomic_parquet(table, score_output)
    mask_output.write_text(
        json.dumps(masks_to_neuron_ids(masks), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata["outputs"] = {
        "score_file": str(score_output.resolve()),
        "mask_file": str(mask_output.resolve()),
        "metadata_file": str(metadata_output.resolve()),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"totals": metadata["totals"], "verification": metadata["verification"]}, indent=2))


if __name__ == "__main__":
    main()
