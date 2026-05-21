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

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .modeling import find_mlp_layers
from .schema import ClusterIdx


if TYPE_CHECKING:
    from torch import nn

    from ...hparams import FinetuningArguments


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        return -20.0

    return math.log(math.expm1(value))


def init_collapse_lambdas(model: "nn.Module", finetuning_args: "FinetuningArguments") -> None:
    if not finetuning_args.collapse_learnable_lambda:
        return

    if hasattr(model, "vulcan_raw_lambda1") or hasattr(model, "vulcan_raw_lambda2"):
        return

    model.register_parameter(
        "vulcan_raw_lambda1",
        torch.nn.Parameter(torch.tensor(_inverse_softplus(finetuning_args.collapse_lambda1), dtype=torch.float32)),
    )
    model.register_parameter(
        "vulcan_raw_lambda2",
        torch.nn.Parameter(torch.tensor(_inverse_softplus(finetuning_args.collapse_lambda2), dtype=torch.float32)),
    )


def get_collapse_lambdas(
    model: "nn.Module", finetuning_args: "FinetuningArguments"
) -> tuple[torch.Tensor, torch.Tensor]:
    if finetuning_args.collapse_learnable_lambda:
        lambda1 = F.softplus(getattr(model, "vulcan_raw_lambda1"))
        lambda2 = F.softplus(getattr(model, "vulcan_raw_lambda2"))
        return lambda1, lambda2

    device = next(model.parameters()).device
    return (
        torch.tensor(finetuning_args.collapse_lambda1, device=device, dtype=torch.float32),
        torch.tensor(finetuning_args.collapse_lambda2, device=device, dtype=torch.float32),
    )


def _make_grad_hook(up_proj_w: torch.Tensor, gate_proj_w: torch.Tensor, up_width: int):
    def grad_hook(grad: torch.Tensor) -> None:
        up_grad = grad[:, :up_width].to(device=up_proj_w.device, dtype=up_proj_w.dtype)
        gate_grad = grad[:, up_width:].to(device=gate_proj_w.device, dtype=gate_proj_w.dtype)

        if up_proj_w.grad is None:
            up_proj_w.grad = up_grad
        else:
            up_proj_w.grad = up_proj_w.grad + up_grad

        if gate_proj_w.grad is None:
            gate_proj_w.grad = gate_grad
        else:
            gate_proj_w.grad = gate_proj_w.grad + gate_grad

    return grad_hook


def _get_weight_proxy(
    up_proj_w: torch.Tensor,
    gate_proj_w: torch.Tensor,
    use_weight_proxy: bool,
) -> torch.Tensor:
    if not use_weight_proxy:
        return torch.cat([up_proj_w.float(), gate_proj_w.float()], dim=1)

    up_width = up_proj_w.shape[1]
    weight_proxy = torch.cat([up_proj_w.detach(), gate_proj_w.detach()], dim=1).float()
    weight_proxy.requires_grad_(True)
    weight_proxy.register_hook(_make_grad_hook(up_proj_w, gate_proj_w, up_width))
    return weight_proxy


def _get_cluster_tensor_cache(
    model: "nn.Module",
    cluster_idx: ClusterIdx,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
    cache_key = (id(cluster_idx), str(device))
    cached_key = getattr(model, "_vulcan_cluster_tensor_cache_key", None)
    cached_value = getattr(model, "_vulcan_cluster_tensor_cache", None)
    if cached_key == cache_key and cached_value is not None:
        return cached_value

    tensor_cache: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    for layer_clusters in cluster_idx:
        if not layer_clusters:
            tensor_cache.append(None)
            continue

        neuron_idxs: list[int] = []
        anchor_idxs: list[int] = []
        for cluster in layer_clusters:
            neurons = [int(idx) for idx in cluster["neuron"]]
            neuron_idxs.extend(neurons)
            anchor_idxs.extend([int(cluster["anchor"])] * len(neurons))

        tensor_cache.append(
            (
                torch.tensor(neuron_idxs, device=device, dtype=torch.long),
                torch.tensor(anchor_idxs, device=device, dtype=torch.long),
            )
        )

    setattr(model, "_vulcan_cluster_tensor_cache_key", cache_key)
    setattr(model, "_vulcan_cluster_tensor_cache", tensor_cache)
    return tensor_cache


def weight_collapse_loss(
    model: "nn.Module",
    cluster_idx: ClusterIdx,
    lambda1: torch.Tensor,
    lambda2: torch.Tensor,
    use_weight_proxy: bool = True,
) -> torch.Tensor:
    r"""Compute Vulcan collapse loss for Qwen/Llama-style gated MLP layers."""
    mlp_layers = find_mlp_layers(model)
    if len(cluster_idx) != len(mlp_layers):
        raise ValueError(f"cluster_idx has {len(cluster_idx)} layers, but model has {len(mlp_layers)} MLP layers.")

    loss = torch.zeros((), device=lambda1.device, dtype=torch.float32)
    cluster_tensor_cache = _get_cluster_tensor_cache(model, cluster_idx, lambda1.device)
    for layer_ref, layer_tensors in zip(mlp_layers, cluster_tensor_cache):
        if layer_tensors is None:
            continue

        up_proj_w = layer_ref.mlp.up_proj.weight
        gate_proj_w = layer_ref.mlp.gate_proj.weight
        weight_proxy = _get_weight_proxy(up_proj_w, gate_proj_w, use_weight_proxy)

        neuron_idxs, anchor_idxs = layer_tensors
        diff_w = weight_proxy.index_select(0, neuron_idxs) - weight_proxy.index_select(0, anchor_idxs)
        loss = loss + lambda1 * diff_w.abs().sum() + lambda2 * diff_w.pow(2).sum()

    return loss
