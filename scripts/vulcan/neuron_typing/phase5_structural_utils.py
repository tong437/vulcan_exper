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

"""Utilities for materializing non-uniform Phase-5 deletion masks."""

from __future__ import annotations

from typing import Any

import torch


def validate_phase5_deletion_masks(masks_by_layer: dict[int, torch.Tensor]) -> dict[str, Any]:
    """Validate a global mask while allowing individual layers to remain unchanged."""
    if not masks_by_layer:
        raise ValueError("Deletion mask is empty.")
    layers = sorted(masks_by_layer)
    if layers != list(range(len(layers))):
        raise ValueError(f"Deletion mask layers must be contiguous from zero, got {layers}.")

    per_layer = {}
    for layer, mask in sorted(masks_by_layer.items()):
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError(f"Layer {layer} mask must be a one-dimensional bool tensor.")
        width = int(mask.numel())
        pruned = int(mask.sum())
        if pruned >= width:
            raise ValueError(f"Layer {layer} must keep at least one neuron, got {pruned}/{width}.")
        per_layer[str(layer)] = {"width": width, "pruned": pruned, "kept": width - pruned}

    total = sum(row["width"] for row in per_layer.values())
    total_pruned = sum(row["pruned"] for row in per_layer.values())
    if total_pruned == 0:
        raise ValueError("Deletion mask does not prune any neurons.")
    return {
        "num_layers": len(layers),
        "total_neurons": total,
        "total_pruned": total_pruned,
        "total_kept": total - total_pruned,
        "pruned_ratio": total_pruned / total,
        "per_layer": per_layer,
    }


def build_partial_singleton_cluster_idx(
    masks_by_layer: dict[int, torch.Tensor],
) -> list[list[dict[str, int | list[int]]] | None]:
    """Encode exact deletion, using ``None`` for layers whose width is unchanged."""
    validate_phase5_deletion_masks(masks_by_layer)
    cluster_idx = []
    for layer, mask in sorted(masks_by_layer.items()):
        if not bool(mask.any()):
            cluster_idx.append(None)
            continue
        keep_ids = (~mask).nonzero(as_tuple=False).flatten().tolist()
        cluster_idx.append([{"anchor": neuron, "neuron": [neuron]} for neuron in keep_ids])
    return cluster_idx


def validate_partial_singleton_cluster_idx(
    cluster_idx: list[list[dict[str, Any]] | None], masks_by_layer: dict[int, torch.Tensor]
) -> dict[str, Any]:
    """Require clusters to represent exactly the complement of a Phase-5 mask."""
    mask_summary = validate_phase5_deletion_masks(masks_by_layer)
    if len(cluster_idx) != len(masks_by_layer):
        raise ValueError(f"cluster_idx has {len(cluster_idx)} layers, expected {len(masks_by_layer)}.")

    per_layer = {}
    for layer, mask in sorted(masks_by_layer.items()):
        clusters = cluster_idx[layer]
        expected_keep = (~mask).nonzero(as_tuple=False).flatten().tolist()
        if not bool(mask.any()):
            if clusters is not None:
                raise ValueError(f"Unchanged layer {layer} must use null clusters.")
            actual_keep = expected_keep
        else:
            if clusters is None:
                raise ValueError(f"Pruned layer {layer} cannot use null clusters.")
            actual_keep = []
            for cluster in clusters:
                anchor = cluster.get("anchor")
                neurons = cluster.get("neuron")
                if not isinstance(anchor, int) or neurons != [anchor]:
                    raise ValueError(f"Layer {layer} contains a non-singleton cluster: {cluster}.")
                actual_keep.append(anchor)
            if actual_keep != expected_keep:
                raise ValueError(f"Layer {layer} clusters do not exactly cover kept neurons in order.")
        per_layer[str(layer)] = {"clusters": None if clusters is None else len(clusters), "kept": len(actual_keep)}
    return {"validated": True, "mask_summary": mask_summary, "per_layer": per_layer}


def target_layer_dims(
    masks_by_layer: dict[int, torch.Tensor],
) -> dict[int, int]:
    validate_phase5_deletion_masks(masks_by_layer)
    return {layer: int(mask.numel() - mask.sum()) for layer, mask in sorted(masks_by_layer.items())}


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"Logit shapes differ: {tuple(reference.shape)} vs {tuple(candidate.shape)}.")
    delta = candidate.float() - reference.float()
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
    }
