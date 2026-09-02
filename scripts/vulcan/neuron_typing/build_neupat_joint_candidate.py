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

"""Build a pre-causal-gate q-band candidate with NeuPAT/mapping protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from phase3_structural_utils import masks_to_neuron_ids
from run_phase2_ablation import (
    build_score_vector,
    build_secondary_vector,
    get_layer_dims,
    infer_score_columns,
    read_score_table,
    select_k_indices_deterministic,
    summarize_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a NeuPAT-protected q-band deletion candidate.")
    parser.add_argument("--score_file", required=True, help="neupat_combined_scores.parquet from overlap analysis.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--band_start", type=float, default=0.05)
    parser.add_argument("--band_end", type=float, default=0.20)
    parser.add_argument("--budget_per_layer", type=int, default=None)
    parser.add_argument(
        "--allow_backfill",
        action="store_true",
        help="Backfill below the q-band to restore the exact budget. Experimental; requires new causal gates.",
    )
    return parser.parse_args()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_joint_candidate(
    table: pd.DataFrame,
    *,
    band_start: float,
    band_end: float,
    budget_per_layer: int | None,
    allow_backfill: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    if not 0.0 <= band_start < band_end <= 1.0:
        raise ValueError("Band must satisfy 0 <= start < end <= 1.")
    required = {"q_multimodal", "joint_protect"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Combined NeuPAT score file is missing columns: {missing}.")
    layer_col, neuron_col, score_cols, _ = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    masks = {layer_idx: torch.zeros(width, dtype=torch.bool) for layer_idx, width in layer_dims.items()}
    per_layer: dict[str, Any] = {}
    for layer_idx, group in table.groupby(layer_col, sort=True):
        layer_idx = int(layer_idx)
        width = layer_dims[layer_idx]
        start = math.ceil(width * band_start)
        end = math.ceil(width * band_end)
        budget = budget_per_layer if budget_per_layer is not None else end - start
        if budget <= 0 or budget > width - start:
            raise ValueError(f"Invalid layer-{layer_idx} deletion budget {budget} for width {width}.")
        primary = build_score_vector(group, neuron_col, score_cols["multimodal"], width)
        secondary = build_secondary_vector(group, neuron_col, "multimodal", width)
        ranked, _ = select_k_indices_deterministic(
            primary,
            width,
            secondary_key=secondary,
            neuron_ids=torch.arange(width, dtype=torch.long),
        )
        ranked_ids = ranked.tolist()
        protected = {int(row[neuron_col]) for _, row in group.iterrows() if bool(row["joint_protect"])}
        band_ids = ranked_ids[start:end]
        if "qband_delete_candidate" in group.columns:
            recorded_band = {
                int(row[neuron_col]) for _, row in group.iterrows() if bool(row["qband_delete_candidate"])
            }
            if set(band_ids) != recorded_band:
                raise ValueError(
                    f"Layer {layer_idx} q-band does not match the recorded overlap mask. "
                    "Use the same --band_start/--band_end values as analyze_neupat_overlap.py."
                )
        selected = [neuron for neuron in band_ids if neuron not in protected]
        backfill: list[int] = []
        if len(selected) < budget and allow_backfill:
            for neuron in ranked_ids[end:]:
                if neuron not in protected and neuron not in selected:
                    backfill.append(neuron)
                    if len(selected) + len(backfill) == budget:
                        break
        selected.extend(backfill)
        if len(selected) < budget and budget_per_layer is not None:
            raise ValueError(
                f"Layer {layer_idx} has only {len(selected)} unprotected candidates for exact budget {budget}. "
                "Use --allow_backfill to draw candidates below the original q-band."
            )
        masks[layer_idx][torch.tensor(selected, dtype=torch.long)] = True
        per_layer[str(layer_idx)] = {
            "width": width,
            "target_budget": budget,
            "selected": len(selected),
            "protected_inside_original_band": sum(neuron in protected for neuron in band_ids),
            "backfilled": len(backfill),
            "backfill_rank_start": end if backfill else None,
        }
    metadata = {
        "artifact_version": 1,
        "name": "neupat_mapping_protected_qband_candidate",
        "status": "pre_causal_gate",
        "structural_pruning_allowed": False,
        "band_start": band_start,
        "band_end": band_end,
        "allow_backfill": allow_backfill,
        "per_layer": per_layer,
        "mask_summary": summarize_masks(masks),
        "warning": (
            "NeuPAT reserve/update roles are not pruning evidence. This candidate must pass Caption, text-only, "
            "and all POPE gates before any structural checkpoint is built."
        ),
    }
    return masks, metadata


def main() -> None:
    args = parse_args()
    table = read_score_table(args.score_file)
    masks, metadata = build_joint_candidate(
        table,
        band_start=args.band_start,
        band_end=args.band_end,
        budget_per_layer=args.budget_per_layer,
        allow_backfill=args.allow_backfill,
    )
    neuron_ids = masks_to_neuron_ids(masks)
    metadata["mask_sha256"] = _canonical_sha256(neuron_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "joint_candidate.mask.json").write_text(
        json.dumps(neuron_ids, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "joint_candidate.metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["mask_summary"], indent=2))


if __name__ == "__main__":
    main()
