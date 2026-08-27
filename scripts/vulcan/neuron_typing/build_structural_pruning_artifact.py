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

"""Freeze the formal Phase-3 q-band mask and structural pruning artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from phase3_structural_utils import (
    DEFAULT_ABLATION_SPEC,
    build_singleton_cluster_idx,
    canonical_json_sha256,
    masks_to_neuron_ids,
    resolve_existing_path,
    sha256_file,
    theoretical_mlp_parameter_reduction,
    validate_deletion_masks,
    validate_singleton_cluster_idx,
)
from run_phase2_ablation import (
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    parse_ablation_spec,
    read_score_table,
    summarize_cutoffs,
    summarize_masks,
)


MASK_NAME = "q_multimodal_band_05_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen Phase-3 structural q-band artifact.")
    parser.add_argument("--score_file", required=True, help="Exact score parquet used by formal Phase 2.")
    parser.add_argument("--phase2_result", required=True, help="Formal Phase-2 caption result containing q-band.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--typing_config", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--calibration_file", default=None)
    parser.add_argument("--band_start", type=float, default=0.05)
    parser.add_argument("--band_end", type=float, default=0.20)
    parser.add_argument("--expected_num_layers", type=int, default=24)
    parser.add_argument("--expected_layer_width", type=int, default=3584)
    parser.add_argument("--hidden_size", type=int, default=1024)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_phase1_artifacts(
    score_file: str | Path,
    *,
    typing_config: str | None,
    typing_manifest: str | None,
    calibration_manifest: str | None,
    calibration_file: str | None,
) -> dict[str, Path]:
    run_root = Path(score_file).resolve().parent.parent
    config_path = Path(typing_config).resolve() if typing_config else run_root / "activations/config.json"
    manifest_path = (
        Path(typing_manifest).resolve() if typing_manifest else run_root / "activations/sample_manifest.json"
    )
    calibration_manifest_path = (
        Path(calibration_manifest).resolve() if calibration_manifest else run_root / "calibration/sample_manifest.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Typing config does not exist: {config_path}")
    config = load_json(config_path)
    calibration_path = (
        Path(calibration_file).resolve()
        if calibration_file
        else resolve_existing_path(config["quantile_path"], relative_to=config_path)
    )
    artifacts = {
        "typing_config": config_path,
        "typing_manifest": manifest_path,
        "calibration_manifest": calibration_manifest_path,
        "calibration_file": calibration_path,
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required Phase-1 artifacts are missing: {missing}")
    return artifacts


def verify_phase2_reproduction(
    phase2_result: dict[str, Any],
    *,
    phase2_result_path: str | Path,
    score_file: str | Path,
    result_name: str,
    mask_summary: dict[str, Any],
    cutoff_summary: dict[str, Any],
) -> dict[str, Any]:
    config = phase2_result.get("config", {})
    if config.get("selection") != "per_layer":
        raise ValueError("Formal Phase-2 result did not use per-layer selection.")
    recorded_score = config.get("score_file")
    if not recorded_score:
        raise ValueError("Formal Phase-2 result does not record score_file.")
    recorded_score_path = resolve_existing_path(recorded_score, relative_to=phase2_result_path)
    score_path = Path(score_file).resolve()
    if not recorded_score_path.is_file():
        raise FileNotFoundError(f"Phase-2 recorded score file does not exist: {recorded_score_path}")
    if sha256_file(recorded_score_path) != sha256_file(score_path):
        raise ValueError("The requested score_file is not byte-identical to the score file used by formal Phase 2.")

    recorded_mask_summary = phase2_result.get("mask_summaries", {}).get(result_name)
    recorded_cutoffs = phase2_result.get("cutoff_summaries", {}).get(result_name)
    if recorded_mask_summary is None or recorded_cutoffs is None:
        raise ValueError(f"Formal Phase-2 result does not contain {result_name!r}.")
    if recorded_mask_summary != mask_summary:
        raise ValueError("Reconstructed q-band mask counts do not match formal Phase 2.")
    if recorded_cutoffs != cutoff_summary:
        raise ValueError("Reconstructed q-band cutoff summary does not match formal Phase 2.")
    return {
        "validated": True,
        "selection": "per_layer",
        "result_name": result_name,
        "same_score_sha256": True,
        "same_mask_summary": True,
        "same_cutoff_summary": True,
        "phase2_result_sha256": sha256_file(phase2_result_path),
    }


def build_artifacts(
    score_table: pd.DataFrame,
    *,
    band_start: float,
    band_end: float,
    expected_num_layers: int,
    expected_layer_width: int,
    hidden_size: int,
) -> tuple[dict[int, Any], dict[str, Any], list[list[dict[str, Any]]], dict[str, Any]]:
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(score_table, None)
    layer_dims = get_layer_dims(score_table, layer_col, neuron_col)
    spec = parse_ablation_spec(f"rank_band:multimodal:{band_start}:{band_end}", 42)
    masks = build_type_mask(
        score_table,
        spec,
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
    expected_pruned = math.ceil(expected_layer_width * band_end) - math.ceil(expected_layer_width * band_start)
    mask_validation = validate_deletion_masks(
        masks,
        expected_num_layers=expected_num_layers,
        expected_layer_width=expected_layer_width,
        expected_pruned_per_layer=expected_pruned,
    )

    if "is_dead" in score_table.columns:
        dead = score_table[score_table["is_dead"].astype(bool)]
        selected_dead = [
            (int(row[layer_col]), int(row[neuron_col]))
            for _, row in dead.iterrows()
            if bool(masks[int(row[layer_col])][int(row[neuron_col])])
        ]
        if selected_dead:
            raise ValueError(f"Frozen q-band unexpectedly selects dead neurons: {selected_dead}")
    else:
        selected_dead = []

    cluster_idx = build_singleton_cluster_idx(masks)
    cluster_validation = validate_singleton_cluster_idx(cluster_idx, masks)
    details = {
        "spec": spec.result_name,
        "score_columns": score_cols,
        "layer_column": layer_col,
        "neuron_column": neuron_col,
        "mask_validation": mask_validation,
        "cluster_validation": cluster_validation,
        "selected_dead_neurons": selected_dead,
        "theoretical_reduction": theoretical_mlp_parameter_reduction(masks, hidden_size),
    }
    return masks, summarize_masks(masks), cluster_idx, details


def main() -> None:
    args = parse_args()
    if not 0 <= args.band_start < args.band_end <= 1:
        raise ValueError("Band must satisfy 0 <= start < end <= 1.")
    result_name = f"rank_band:multimodal:{args.band_start:g}:{args.band_end:g}"
    if result_name != DEFAULT_ABLATION_SPEC:
        raise ValueError(
            f"Formal Phase 3 is frozen to {DEFAULT_ABLATION_SPEC}; got {result_name}. "
            "Use a separately named experiment for hardware-aligned variants."
        )

    score_path = Path(args.score_file).resolve()
    phase2_path = Path(args.phase2_result).resolve()
    score_table = read_score_table(score_path)
    masks, mask_summary, cluster_idx, details = build_artifacts(
        score_table,
        band_start=args.band_start,
        band_end=args.band_end,
        expected_num_layers=args.expected_num_layers,
        expected_layer_width=args.expected_layer_width,
        hidden_size=args.hidden_size,
    )
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(score_table, None)
    spec = parse_ablation_spec(result_name, 42)
    cutoff_summary = summarize_cutoffs(
        score_table,
        spec,
        masks,
        layer_col,
        neuron_col,
        score_cols,
        activation_col,
        1.0,
        0.0,
    )
    phase2 = load_json(phase2_path)
    phase2_reproduction = verify_phase2_reproduction(
        phase2,
        phase2_result_path=phase2_path,
        score_file=score_path,
        result_name=result_name,
        mask_summary=mask_summary,
        cutoff_summary=cutoff_summary,
    )
    phase1_artifacts = infer_phase1_artifacts(
        score_path,
        typing_config=args.typing_config,
        typing_manifest=args.typing_manifest,
        calibration_manifest=args.calibration_manifest,
        calibration_file=args.calibration_file,
    )

    deletion_ids = masks_to_neuron_ids(masks)
    mask_hash = canonical_json_sha256(deletion_ids)
    cluster_hash = canonical_json_sha256(cluster_idx)
    provenance_files = {"score_file": score_path, "phase2_result": phase2_path, **phase1_artifacts}
    provenance = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in provenance_files.items()}
    metadata = {
        "artifact_version": 1,
        "name": MASK_NAME,
        "frozen": True,
        "ablation_spec": result_name,
        "ordering": ["q_multimodal desc", "r_multimodal desc", "neuron_idx asc"],
        "mask_sha256": mask_hash,
        "cluster_idx_sha256": cluster_hash,
        "mask_summary": mask_summary,
        "cutoff_summary": cutoff_summary,
        "phase2_reproduction": phase2_reproduction,
        "provenance": provenance,
        **details,
    }
    metadata["metadata_sha256"] = canonical_json_sha256(metadata)

    output_dir = Path(args.output_dir).resolve() / "masks"
    mask_path = output_dir / f"{MASK_NAME}.mask.json"
    cluster_path = output_dir / f"{MASK_NAME}.cluster_idx.json"
    metadata_path = output_dir / f"{MASK_NAME}.metadata.json"
    write_json(mask_path, deletion_ids)
    write_json(cluster_path, cluster_idx)
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "mask": str(mask_path),
                "cluster_idx": str(cluster_path),
                "metadata": str(metadata_path),
                "mask_sha256": mask_hash,
                "summary": mask_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
