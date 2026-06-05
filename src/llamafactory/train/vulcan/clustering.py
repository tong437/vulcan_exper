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
from typing import Any

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances

from .modeling import find_mlp_layers, get_intermediate_size
from .schema import ClusterIdx


def _validate_keep_ratio(keep_ratio: float) -> None:
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError("keep_ratio must be in (0, 1].")


def get_cluster_greedy_match(
    activation: np.ndarray | torch.Tensor,
    vectors: np.ndarray | torch.Tensor,
    n_clusters: int,
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

    dist = euclidean_distances(vectors).astype(np.float32)
    assigned = np.zeros(num_vectors, dtype=bool)
    cluster_list: list[dict[str, Any]] = []
    cluster_size = num_vectors // n_clusters

    while not assigned.all():
        unassigned = np.where(~assigned)[0]
        if len(unassigned) <= cluster_size:
            act = activation[unassigned]
            max_idx_in_cluster = int(np.argmax(act))
            anchor_idx = int(unassigned[max_idx_in_cluster])
            cluster_list.append({"anchor": anchor_idx, "neuron": unassigned.astype(int).tolist()})
            assigned[unassigned] = True
            continue

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


def get_mlp_weight_vectors(mlp: torch.nn.Module) -> torch.Tensor:
    return torch.cat([mlp.up_proj.weight.detach().float(), mlp.gate_proj.weight.detach().float()], dim=1)


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
