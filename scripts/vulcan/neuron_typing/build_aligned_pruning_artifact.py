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

"""Build a frozen 3,072-wide hardware-aligned q-multimodal pruning artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_structural_pruning_artifact import (
    infer_phase1_artifacts,
    load_json,
    verify_phase2_reproduction,
    write_json,
)
from phase3_structural_utils import (
    build_singleton_cluster_idx,
    canonical_json_sha256,
    masks_to_neuron_ids,
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


DEFAULT_NAME = "q_multimodal_aligned_3072"
DEFAULT_SPEC = "rank_window:multimodal:180:512"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase-3.4 hardware-aligned q-band artifact.")
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--hook_caption_result", required=True)
    parser.add_argument("--hook_quality_gate", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--rank_start", type=int, default=180)
    parser.add_argument("--rank_count", type=int, default=512)
    parser.add_argument("--typing_config", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--calibration_file", default=None)
    parser.add_argument("--expected_num_layers", type=int, default=24)
    parser.add_argument("--expected_layer_width", type=int, default=3584)
    parser.add_argument("--hidden_size", type=int, default=1024)
    return parser.parse_args()


def build_aligned_artifacts(
    score_table,
    *,
    rank_start: int,
    rank_count: int,
    expected_num_layers: int,
    expected_layer_width: int,
    hidden_size: int,
) -> tuple[dict[int, Any], dict[str, Any], list[list[dict[str, Any]]], dict[str, Any]]:
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(score_table, None)
    layer_dims = get_layer_dims(score_table, layer_col, neuron_col)
    spec = parse_ablation_spec(f"rank_window:multimodal:{rank_start}:{rank_count}", 42)
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
    mask_validation = validate_deletion_masks(
        masks,
        expected_num_layers=expected_num_layers,
        expected_layer_width=expected_layer_width,
        expected_pruned_per_layer=rank_count,
    )
    selected_dead = []
    if "is_dead" in score_table.columns:
        for _, row in score_table[score_table["is_dead"].astype(bool)].iterrows():
            layer, neuron = int(row[layer_col]), int(row[neuron_col])
            if bool(masks[layer][neuron]):
                selected_dead.append((layer, neuron))
    if selected_dead:
        raise ValueError(f"Aligned q-band unexpectedly selects dead neurons: {selected_dead}")

    cluster_idx = build_singleton_cluster_idx(masks)
    details = {
        "spec": spec.result_name,
        "score_columns": score_cols,
        "layer_column": layer_col,
        "neuron_column": neuron_col,
        "rank_start": rank_start,
        "rank_count": rank_count,
        "rank_end": rank_start + rank_count,
        "protected_prefix_ratio": rank_start / expected_layer_width,
        "pruned_ratio": rank_count / expected_layer_width,
        "target_intermediate_size": expected_layer_width - rank_count,
        "mask_validation": mask_validation,
        "cluster_validation": validate_singleton_cluster_idx(cluster_idx, masks),
        "selected_dead_neurons": selected_dead,
        "theoretical_reduction": theoretical_mlp_parameter_reduction(masks, hidden_size),
    }
    return masks, summarize_masks(masks), cluster_idx, details


def main() -> None:
    args = parse_args()
    if args.rank_start < 0 or args.rank_count <= 0:
        raise ValueError("rank_start must be non-negative and rank_count must be positive.")
    if args.expected_layer_width - args.rank_count != 3072:
        raise ValueError(
            f"P3.4 aligned artifact must target width 3072, got {args.expected_layer_width - args.rank_count}."
        )
    result_name = f"rank_window:multimodal:{args.rank_start}:{args.rank_count}"

    score_path = Path(args.score_file).resolve()
    caption_path = Path(args.hook_caption_result).resolve()
    hook_gate_path = Path(args.hook_quality_gate).resolve()
    hook_gate = load_json(hook_gate_path)
    if not hook_gate.get("passed") or hook_gate.get("ablation_spec") != result_name:
        raise ValueError("Aligned structural artifact requires a passing hook gate for the exact rank window.")

    score_table = read_score_table(score_path)
    masks, mask_summary, cluster_idx, details = build_aligned_artifacts(
        score_table,
        rank_start=args.rank_start,
        rank_count=args.rank_count,
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
    hook_caption = load_json(caption_path)
    hook_reproduction = verify_phase2_reproduction(
        hook_caption,
        phase2_result_path=caption_path,
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
    provenance_files = {
        "score_file": score_path,
        "hook_caption_result": caption_path,
        "hook_quality_gate": hook_gate_path,
        **phase1_artifacts,
    }
    provenance = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in provenance_files.items()}
    metadata = {
        "artifact_version": 1,
        "name": args.name,
        "frozen": True,
        "hardware_aligned": True,
        "ablation_spec": result_name,
        "ordering": ["q_multimodal desc", "r_multimodal desc", "neuron_idx asc"],
        "mask_sha256": mask_hash,
        "cluster_idx_sha256": cluster_hash,
        "mask_summary": mask_summary,
        "cutoff_summary": cutoff_summary,
        "hook_reproduction": hook_reproduction,
        "hook_quality_gate": hook_gate,
        "provenance": provenance,
        **details,
    }
    metadata["metadata_sha256"] = canonical_json_sha256(metadata)

    output_dir = Path(args.output_dir).resolve() / "masks"
    mask_path = output_dir / f"{args.name}.mask.json"
    cluster_path = output_dir / f"{args.name}.cluster_idx.json"
    metadata_path = output_dir / f"{args.name}.metadata.json"
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
                "target_intermediate_size": details["target_intermediate_size"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
