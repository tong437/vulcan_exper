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

from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .modeling import find_mlp_layers, get_intermediate_size
from .schema import ClusterIdx


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

    dist = cdist(vectors, vectors, metric="euclidean").astype(np.float32)
    assigned = np.zeros(num_vectors, dtype=bool)
    cluster_list: list[dict[str, Any]] = []

    while not assigned.all():
        unassigned = np.where(~assigned)[0]
        remaining_clusters = n_clusters - len(cluster_list)
        if remaining_clusters <= 1:
            anchor_idx = int(unassigned[np.argmax(activation[unassigned])])
            cluster_list.append({"anchor": anchor_idx, "neuron": unassigned.astype(int).tolist()})
            assigned[unassigned] = True
            break

        cluster_size = int(np.ceil(len(unassigned) / remaining_clusters))
        sub_dist = dist[np.ix_(unassigned, unassigned)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        nearest = sub_dist.min(axis=1)
        seed = int(unassigned[nearest.argmin()])
        seed_dists = dist[seed, unassigned]
        members = unassigned[np.argsort(seed_dists)[:cluster_size]]
        anchor_idx = int(members[np.argmax(activation[members])])
        cluster_list.append({"anchor": anchor_idx, "neuron": members.astype(int).tolist()})
        assigned[members] = True

    return cluster_list


def get_mlp_weight_vectors(mlp: torch.nn.Module) -> torch.Tensor:
    return torch.cat([mlp.up_proj.weight.detach().float(), mlp.gate_proj.weight.detach().float()], dim=1)


@torch.no_grad()
def collect_mlp_activations(model: torch.nn.Module, dataloader, max_batches: int | None = None) -> torch.Tensor:
    r"""Collect mean absolute down_proj input activations for every MLP layer."""
    mlp_layers = find_mlp_layers(model)
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

            model(**batch)
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
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError("keep_ratio must be in (0, 1].")

    cluster_idx: ClusterIdx = []
    for layer_ref, activation in zip(find_mlp_layers(model), activations):
        intermediate_size = get_intermediate_size(layer_ref.mlp)
        target_size = max(1, int(intermediate_size * keep_ratio))
        if target_size == intermediate_size:
            cluster_idx.append(None)
            continue

        vectors = get_mlp_weight_vectors(layer_ref.mlp)
        cluster_idx.append(get_cluster_greedy_match(activation, vectors, target_size))

    return cluster_idx
