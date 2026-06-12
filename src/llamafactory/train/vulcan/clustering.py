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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

from ...extras.constants import IGNORE_INDEX
from .modeling import find_mlp_layers, get_intermediate_size
from .schema import ClusterIdx


@dataclass
class MultimodalActivationStats:
    r"""Per-layer VQA activation statistics with equal weight for each dataset group."""

    anchor_scores: torch.Tensor
    signatures: torch.Tensor
    signature_names: tuple[str, ...] = ("image", "question", "prediction")


def _validate_keep_ratio(keep_ratio: float) -> None:
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError("keep_ratio must be in (0, 1].")


def get_cluster_greedy_match(
    activation: np.ndarray | torch.Tensor,
    vectors: np.ndarray | torch.Tensor,
    n_clusters: int,
    activation_signatures: np.ndarray | torch.Tensor | None = None,
    activation_distance_weight: float = 0.0,
) -> list[dict[str, Any]]:
    r"""Group neurons by nearest weight vectors and choose the strongest activation as anchor."""
    if isinstance(activation, torch.Tensor):
        activation = activation.detach().float().cpu().numpy()
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().float().cpu().numpy()

    vectors = vectors.astype(np.float32)
    activation = activation.astype(np.float32)
    num_vectors = vectors.shape[0]
    if n_clusters <= 0 or n_clusters > num_vectors:
        raise ValueError(f"n_clusters must be in [1, {num_vectors}], got {n_clusters}.")

    if activation.shape[0] != num_vectors:
        raise ValueError(f"activation length {activation.shape[0]} does not match vectors {num_vectors}.")

    if activation_signatures is not None:
        if isinstance(activation_signatures, torch.Tensor):
            activation_signatures = activation_signatures.detach().float().cpu().numpy()
        activation_signatures = activation_signatures.astype(np.float32)
        if activation_signatures.shape[0] != num_vectors:
            raise ValueError(
                f"activation signature length {activation_signatures.shape[0]} does not match vectors {num_vectors}."
            )
        if activation_distance_weight < 0:
            raise ValueError("activation_distance_weight must be non-negative.")

        signature_mean = activation_signatures.mean(axis=0, keepdims=True)
        signature_std = activation_signatures.std(axis=0, keepdims=True)
        normalized_signatures = (activation_signatures - signature_mean) / np.maximum(signature_std, 1e-6)
        vectors = np.concatenate([vectors, activation_distance_weight * normalized_signatures], axis=1)

    dist = euclidean_distances(vectors).astype(np.float32)
    assigned = np.zeros(num_vectors, dtype=bool)
    cluster_list: list[dict[str, Any]] = []
    while not assigned.all():
        unassigned = np.where(~assigned)[0]
        remaining_clusters = n_clusters - len(cluster_list)
        if remaining_clusters == 1:
            act = activation[unassigned]
            max_idx_in_cluster = int(np.argmax(act))
            anchor_idx = int(unassigned[max_idx_in_cluster])
            cluster_list.append({"anchor": anchor_idx, "neuron": unassigned.astype(int).tolist()})
            assigned[unassigned] = True
            continue

        cluster_size = int(np.ceil(len(unassigned) / remaining_clusters))
        sub_dist = dist[np.ix_(unassigned, unassigned)]
        np.fill_diagonal(sub_dist, np.inf)
        nearest = sub_dist.min(axis=1)
        seed = int(unassigned[nearest.argmin()])
        seed_dists = dist[seed, unassigned]
        k = cluster_size
        nearest_local = np.argsort(seed_dists)[:k]
        members = unassigned[nearest_local]
        act = activation[members]
        max_idx_in_cluster = int(np.argmax(act))
        anchor_idx = int(members[max_idx_in_cluster])
        cluster_list.append({"anchor": anchor_idx, "neuron": members.astype(int).tolist()})
        assigned[members] = True

    return cluster_list


def get_mlp_weight_vectors(mlp: torch.nn.Module, normalize_rows: bool = False) -> torch.Tensor:
    up_weight = mlp.up_proj.weight.detach().float()
    gate_weight = mlp.gate_proj.weight.detach().float()
    if normalize_rows:
        up_weight = torch.nn.functional.normalize(up_weight, dim=1)
        gate_weight = torch.nn.functional.normalize(gate_weight, dim=1)

    return torch.cat([up_weight, gate_weight], dim=1)


def move_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if hasattr(batch, "to"):
        return batch.to(device)

    return batch


def get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def collect_mlp_activations(model: torch.nn.Module, dataloader, max_batches: int | None = None) -> torch.Tensor:
    r"""Collect mean absolute down_proj input activations for every MLP layer."""
    mlp_layers = find_mlp_layers(model)
    device = get_model_device(model)
    total_batches = "unknown" if max_batches is None else str(max_batches)
    print(f"Collecting Vulcan MLP activations on {device} for {total_batches} batches.", flush=True)
    activations = [
        torch.zeros(get_intermediate_size(layer_ref.mlp), dtype=torch.float64, device="cpu")
        for layer_ref in mlp_layers
    ]
    token_counts = [0 for _ in mlp_layers]
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden = inputs[0].detach().float()
            hidden = hidden.reshape(-1, hidden.shape[-1]).abs()
            activations[layer_idx] += hidden.sum(dim=0).cpu().double()
            token_counts[layer_idx] += hidden.shape[0]

        return hook_fn

    for layer_idx, layer_ref in enumerate(mlp_layers):
        hooks.append(layer_ref.mlp.down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = move_to_device(batch, device)
            model(**batch)
            if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
                print(f"Collected activation batch {batch_idx + 1}/{total_batches}.", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    for layer_idx, token_count in enumerate(token_counts):
        if token_count > 0:
            activations[layer_idx] /= token_count

    return torch.stack([activation.float() for activation in activations], dim=0)


@torch.no_grad()
def collect_multimodal_mlp_activations(
    model: torch.nn.Module,
    dataloaders: Mapping[str, Any],
    image_token_id: int,
    special_token_ids: set[int] | None = None,
    max_batches: int | None = None,
    image_weight: float = 0.4,
    question_weight: float = 0.4,
    prediction_weight: float = 0.2,
    weight_by_down_proj: bool = True,
) -> MultimodalActivationStats:
    r"""Collect sample-normalized image/question/prediction activation signatures.

    Each dataloader is treated as one task category. Statistics are averaged
    within a sample, then within a category, and finally across categories.
    This prevents image resolution and category size from dominating anchors.
    """
    weights = torch.tensor([image_weight, question_weight, prediction_weight], dtype=torch.float32)
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Multimodal activation weights must be non-negative and sum to a positive value.")
    weights /= weights.sum()

    mlp_layers = find_mlp_layers(model)
    device = get_model_device(model)
    special_token_ids = special_token_ids or set()
    current_masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    category_signatures: list[torch.Tensor] = []

    for category_name, dataloader in dataloaders.items():
        sums = [
            torch.zeros((3, get_intermediate_size(layer_ref.mlp)), dtype=torch.float64, device="cpu")
            for layer_ref in mlp_layers
        ]
        counts = [torch.zeros(3, dtype=torch.float64, device="cpu") for _ in mlp_layers]
        hooks = []

        def make_hook(layer_position: int):
            def hook_fn(module, inputs, output):
                if current_masks is None:
                    return

                hidden = inputs[0].detach().float().abs()
                for signature_idx, token_mask in enumerate(current_masks):
                    mask = token_mask[:, : hidden.shape[1]]
                    valid_samples = mask.any(dim=1)
                    if not valid_samples.any():
                        continue

                    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
                    pooled = pooled[valid_samples]
                    sums[layer_position][signature_idx] += pooled.sum(dim=0).cpu().double()
                    counts[layer_position][signature_idx] += valid_samples.sum().cpu().double()

            return hook_fn

        for layer_position, layer_ref in enumerate(mlp_layers):
            hooks.append(layer_ref.mlp.down_proj.register_forward_hook(make_hook(layer_position)))

        try:
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                batch = move_to_device(batch, device)
                input_ids = batch["input_ids"]
                labels = batch.get("labels")
                attention_mask = batch.get("attention_mask")
                valid_mask = (
                    attention_mask.bool()
                    if attention_mask is not None
                    else torch.ones_like(input_ids, dtype=torch.bool)
                )
                image_mask = (input_ids == image_token_id) & valid_mask
                non_visual_mask = (~image_mask) & valid_mask

                question_mask = non_visual_mask.clone()
                if labels is not None:
                    question_mask &= labels == IGNORE_INDEX
                for token_id in special_token_ids:
                    question_mask &= input_ids != token_id

                prediction_mask = torch.zeros_like(valid_mask)
                if labels is not None and labels.shape[1] > 1:
                    prediction_mask[:, :-1] = labels[:, 1:] != IGNORE_INDEX
                    prediction_mask &= valid_mask

                current_masks = (image_mask, question_mask, prediction_mask)
                model(**batch)
                current_masks = None
                if (batch_idx + 1) % 25 == 0 or batch_idx == 0:
                    print(f"Collected multimodal activation batch {batch_idx + 1} for {category_name}.", flush=True)
        finally:
            current_masks = None
            for hook in hooks:
                hook.remove()

        layer_signatures = []
        for layer_position, layer_ref in enumerate(mlp_layers):
            signature = sums[layer_position] / counts[layer_position].clamp_min(1.0).unsqueeze(1)
            if weight_by_down_proj:
                down_norm = layer_ref.mlp.down_proj.weight.detach().float().norm(dim=0).cpu().double()
                signature = signature * down_norm.unsqueeze(0)
            layer_signatures.append(signature.float().transpose(0, 1))

        category_signatures.append(torch.stack(layer_signatures, dim=0))

    if not category_signatures:
        raise ValueError("At least one category dataloader is required for multimodal activation collection.")

    signatures = torch.stack(category_signatures, dim=0).mean(dim=0)
    anchor_scores = (signatures * weights.view(1, 1, -1)).sum(dim=-1)
    return MultimodalActivationStats(anchor_scores=anchor_scores, signatures=signatures)


def build_uniform_cluster_idx(
    model: torch.nn.Module,
    activations: torch.Tensor,
    keep_ratio: float,
) -> ClusterIdx:
    mlp_layers = find_mlp_layers(model)
    return build_layerwise_cluster_idx(model, activations, keep_ratios=[keep_ratio] * len(mlp_layers))


def build_layerwise_cluster_idx(
    model: torch.nn.Module,
    activations: torch.Tensor,
    keep_ratios: list[float],
) -> ClusterIdx:
    cluster_idx: ClusterIdx = []
    mlp_layers = find_mlp_layers(model)
    if len(keep_ratios) != len(mlp_layers):
        raise ValueError(f"Expected {len(mlp_layers)} keep ratios, got {len(keep_ratios)}.")

    for layer_idx, (layer_ref, activation) in enumerate(zip(mlp_layers, activations)):
        keep_ratio = float(keep_ratios[layer_idx])
        _validate_keep_ratio(keep_ratio)
        intermediate_size = get_intermediate_size(layer_ref.mlp)
        target_size = max(1, int(intermediate_size * keep_ratio))
        if target_size == intermediate_size:
            cluster_idx.append(None)
            continue

        print(
            f"Building Vulcan cluster_idx for layer {layer_idx + 1}/{len(mlp_layers)} "
            f"({intermediate_size} -> {target_size}).",
            flush=True,
        )
        vectors = get_mlp_weight_vectors(layer_ref.mlp)
        cluster_idx.append(get_cluster_greedy_match(activation, vectors, target_size))

    return cluster_idx


def build_multimodal_cluster_idx(
    model: torch.nn.Module,
    stats: MultimodalActivationStats,
    keep_ratios: list[float],
    normalize_weight_rows: bool = True,
    activation_distance_weight: float = 0.25,
) -> ClusterIdx:
    r"""Build clusters using contribution anchors and optional modality signatures."""
    mlp_layers = find_mlp_layers(model)
    if len(keep_ratios) != len(mlp_layers):
        raise ValueError(f"Expected {len(mlp_layers)} keep ratios, got {len(keep_ratios)}.")
    if stats.anchor_scores.shape[0] != len(mlp_layers) or stats.signatures.shape[0] != len(mlp_layers):
        raise ValueError("Multimodal activation statistics do not match the model layer count.")

    cluster_idx: ClusterIdx = []
    for layer_position, layer_ref in enumerate(mlp_layers):
        keep_ratio = float(keep_ratios[layer_position])
        _validate_keep_ratio(keep_ratio)
        intermediate_size = get_intermediate_size(layer_ref.mlp)
        target_size = max(1, int(intermediate_size * keep_ratio))
        if target_size == intermediate_size:
            cluster_idx.append(None)
            continue

        vectors = get_mlp_weight_vectors(layer_ref.mlp, normalize_rows=normalize_weight_rows)
        cluster_idx.append(
            get_cluster_greedy_match(
                stats.anchor_scores[layer_position],
                vectors,
                target_size,
                activation_signatures=stats.signatures[layer_position],
                activation_distance_weight=activation_distance_weight,
            )
        )

    return cluster_idx


def build_third_keep_ratios(
    num_layers: int,
    first_keep_ratio: float,
    middle_keep_ratio: float,
    last_keep_ratio: float,
    first_layer_ratio: float = 1.0 / 3.0,
    last_layer_ratio: float = 1.0 / 3.0,
) -> list[float]:
    r"""Build front/middle/back keep ratios for decoder layers."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if first_layer_ratio < 0 or last_layer_ratio < 0 or first_layer_ratio + last_layer_ratio > 1:
        raise ValueError("Layer split ratios must be non-negative and sum to at most 1.")

    for keep_ratio in (first_keep_ratio, middle_keep_ratio, last_keep_ratio):
        _validate_keep_ratio(keep_ratio)

    first_count = int(num_layers * first_layer_ratio)
    last_count = int(num_layers * last_layer_ratio)
    middle_count = num_layers - first_count - last_count
    return [first_keep_ratio] * first_count + [middle_keep_ratio] * middle_count + [last_keep_ratio] * last_count
