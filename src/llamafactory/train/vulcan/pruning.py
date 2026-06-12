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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .modeling import find_mlp_layers, get_hidden_size, get_intermediate_size
from .schema import ClusterIdx


if TYPE_CHECKING:
    from torch import nn


@dataclass
class PruningSummary:
    original_intermediate_size: int
    pruned_intermediate_size: int | list[int]
    num_layers: int


def _new_linear_like(old_linear: "nn.Linear", in_features: int, out_features: int) -> "nn.Linear":
    return torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=old_linear.bias is not None,
        device=old_linear.weight.device,
        dtype=old_linear.weight.dtype,
    )


def _set_config_intermediate_sizes(model: "nn.Module", intermediate_sizes: list[int]) -> None:
    root_config = getattr(model, "config", None)
    configs = [root_config]
    if root_config is not None:
        configs.append(getattr(root_config, "text_config", None))
        setattr(root_config, "vulcan_intermediate_sizes", intermediate_sizes)

    for config in configs:
        if config is None:
            continue

        if len(set(intermediate_sizes)) == 1 and hasattr(config, "intermediate_size"):
            setattr(config, "intermediate_size", intermediate_sizes[0])


def validate_pruning(model: "nn.Module", cluster_idx: ClusterIdx) -> tuple[int, list[int]]:
    mlp_layers = find_mlp_layers(model)
    if len(cluster_idx) != len(mlp_layers):
        raise ValueError(f"cluster_idx has {len(cluster_idx)} layers, but model has {len(mlp_layers)} MLP layers.")

    original_sizes = {get_intermediate_size(layer_ref.mlp) for layer_ref in mlp_layers}
    if len(original_sizes) != 1:
        raise ValueError(f"Expected uniform original intermediate size, got {sorted(original_sizes)}.")

    original_size = original_sizes.pop()
    target_sizes = [original_size if layer_clusters is None else len(layer_clusters) for layer_clusters in cluster_idx]
    return original_size, target_sizes


@torch.no_grad()
def pruning_mlp(model: "nn.Module", cluster_idx: ClusterIdx) -> PruningSummary:
    r"""Replace every gated MLP with a narrower one according to cluster_idx."""
    original_size, target_sizes = validate_pruning(model, cluster_idx)
    mlp_layers = find_mlp_layers(model)

    for layer_ref, layer_clusters, target_size in zip(mlp_layers, cluster_idx, target_sizes):
        if layer_clusters is None:
            continue

        mlp = layer_ref.mlp
        hidden_size = get_hidden_size(mlp)
        new_up_proj = _new_linear_like(mlp.up_proj, hidden_size, target_size)
        new_gate_proj = _new_linear_like(mlp.gate_proj, hidden_size, target_size)
        new_down_proj = _new_linear_like(mlp.down_proj, target_size, hidden_size)

        for new_idx, cluster in enumerate(layer_clusters):
            anchor_idx = int(cluster["anchor"])
            neuron_idxs = torch.tensor(cluster["neuron"], device=mlp.up_proj.weight.device, dtype=torch.long)

            new_up_proj.weight[new_idx].copy_(mlp.up_proj.weight[anchor_idx])
            new_gate_proj.weight[new_idx].copy_(mlp.gate_proj.weight[anchor_idx])
            new_down_proj.weight[:, new_idx].copy_(mlp.down_proj.weight.index_select(1, neuron_idxs).sum(dim=1))

            if mlp.up_proj.bias is not None:
                new_up_proj.bias[new_idx].copy_(mlp.up_proj.bias[anchor_idx])
            if mlp.gate_proj.bias is not None:
                new_gate_proj.bias[new_idx].copy_(mlp.gate_proj.bias[anchor_idx])

        if mlp.down_proj.bias is not None:
            new_down_proj.bias.copy_(mlp.down_proj.bias)

        mlp.up_proj = new_up_proj
        mlp.gate_proj = new_gate_proj
        mlp.down_proj = new_down_proj
        if hasattr(mlp, "intermediate_size"):
            setattr(mlp, "intermediate_size", target_size)
        if hasattr(mlp, "config") and hasattr(mlp.config, "intermediate_size"):
            setattr(mlp.config, "intermediate_size", target_size)

    _set_config_intermediate_sizes(model, target_sizes)
    summary_size: int | list[int]
    if len(set(target_sizes)) == 1:
        summary_size = target_sizes[0]
    else:
        summary_size = target_sizes

    return PruningSummary(
        original_intermediate_size=original_size,
        pruned_intermediate_size=summary_size,
        num_layers=len(mlp_layers),
    )
