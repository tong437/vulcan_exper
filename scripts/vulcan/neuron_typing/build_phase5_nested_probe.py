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

"""Extend a physically validated Phase-5 mask along a static-saliency ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from phase3_structural_utils import (
    canonical_json_sha256,
    masks_to_neuron_ids,
    neuron_ids_to_masks,
    sha256_file,
    theoretical_mlp_parameter_reduction,
)
from phase5_structural_utils import (
    build_partial_singleton_cluster_idx,
    target_layer_dims,
    validate_partial_singleton_cluster_idx,
)
from run_phase5_single_sample_frontier import normalize_global_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nested Phase-5 physical-probe artifacts.")
    parser.add_argument("--base_artifact_dir", required=True)
    parser.add_argument("--saliency_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--deletion_budgets", required=True, help="Comma-separated budgets larger than the base.")
    parser.add_argument("--method", default="taylor")
    parser.add_argument("--global_normalization", default="layer_mean", choices=["none", "layer_mean", "layer_rank"])
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extend_deletion_masks(
    base_masks: dict[int, torch.Tensor],
    scores: dict[int, torch.Tensor],
    deletion_budget: int,
    *,
    normalization: str,
) -> tuple[dict[int, torch.Tensor], list[tuple[int, int]]]:
    if set(base_masks) != set(scores):
        raise ValueError("Base-mask and saliency layers do not match.")
    base_count = sum(int(mask.sum()) for mask in base_masks.values())
    total = sum(mask.numel() for mask in base_masks.values())
    if deletion_budget <= base_count or deletion_budget >= total:
        raise ValueError(f"Nested budget must lie in ({base_count}, {total}), got {deletion_budget}.")
    normalized = normalize_global_scores(scores, normalization)
    candidates = []
    for layer in sorted(base_masks):
        mask = base_masks[layer]
        values = normalized[layer]
        if values.ndim != 1 or values.numel() != mask.numel() or not bool(torch.isfinite(values).all()):
            raise ValueError(f"Invalid saliency vector for layer {layer}.")
        for neuron in (~mask).nonzero(as_tuple=False).flatten().tolist():
            candidates.append((float(values[neuron]), layer, neuron))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    additions = [(layer, neuron) for _, layer, neuron in candidates[: deletion_budget - base_count]]
    result = {layer: mask.clone() for layer, mask in base_masks.items()}
    for layer, neuron in additions:
        result[layer][neuron] = True
    return result, additions


def main() -> None:
    args = parse_args()
    base_artifact_dir = Path(args.base_artifact_dir).resolve()
    saliency_path = Path(args.saliency_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    base_metadata = load_json(base_artifact_dir / "metadata.json")
    base_neuron_ids = load_json(base_artifact_dir / "mask.json")
    layer_dims = {int(layer): int(width) for layer, width in base_metadata["layer_dims"].items()}
    base_masks = neuron_ids_to_masks(base_neuron_ids, layer_dims)
    saliency_artifact = torch.load(saliency_path, map_location="cpu", weights_only=False)
    if args.method not in saliency_artifact["scores"]:
        raise ValueError(f"Unknown saliency method {args.method!r}.")
    scores = {int(layer): values.float().cpu() for layer, values in saliency_artifact["scores"][args.method].items()}
    budgets = [int(value) for value in args.deletion_budgets.split(",") if value.strip()]
    if len(budgets) != len(set(budgets)):
        raise ValueError("Nested budgets must be unique.")

    summary = {"complete": False, "runs": {}}
    write_json(output_dir / "nested_frontier.json", summary)
    for budget in budgets:
        masks, additions = extend_deletion_masks(base_masks, scores, budget, normalization=args.global_normalization)
        neuron_ids = masks_to_neuron_ids(masks)
        cluster_idx = build_partial_singleton_cluster_idx(masks)
        validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
        run_dir = output_dir / f"delete_{budget}"
        artifact_dir = run_dir / "artifact"
        mask_path = artifact_dir / "mask.json"
        cluster_path = artifact_dir / "cluster_idx.json"
        metadata = {
            **base_metadata,
            "phase": "5E_nested_probe",
            "run": f"nested_{base_metadata['run']}_to_{budget}",
            "source_gate": "physical_probe",
            "source_strict_feasible": False,
            "source_behavioral_feasible": False,
            "source_physical_probe_eligible": True,
            "source_mask_file": str(mask_path),
            "mask_sha256": canonical_json_sha256(neuron_ids),
            "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
            "deletion_budget": budget,
            "target_layer_dims": target_layer_dims(masks),
            "validation": validation,
            "theoretical_reduction": theoretical_mlp_parameter_reduction(
                masks, base_metadata["theoretical_reduction"]["hidden_size"]
            ),
            "nested_probe": {
                "base_artifact_dir": str(base_artifact_dir),
                "base_metadata_sha256": sha256_file(base_artifact_dir / "metadata.json"),
                "base_mask_sha256": canonical_json_sha256(base_neuron_ids),
                "base_deletion_budget": int(base_metadata["deletion_budget"]),
                "saliency_file": str(saliency_path),
                "saliency_file_sha256": sha256_file(saliency_path),
                "method": args.method,
                "global_normalization": args.global_normalization,
                "added_neurons": len(additions),
                "added_by_layer": {
                    str(layer): sum(added_layer == layer for added_layer, _ in additions)
                    for layer in sorted(layer_dims)
                },
            },
        }
        write_json(mask_path, neuron_ids)
        write_json(cluster_path, cluster_idx)
        write_json(artifact_dir / "metadata.json", metadata)
        summary["runs"][str(budget)] = {
            "artifact_dir": str(artifact_dir),
            "mask_sha256": metadata["mask_sha256"],
            "deletion_budget": budget,
            "added_neurons": len(additions),
        }
        write_json(output_dir / "nested_frontier.json", summary)
    summary["complete"] = True
    write_json(output_dir / "nested_frontier.json", summary)
    print(json.dumps({"complete": True, "budgets": budgets, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
