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


if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class MLPLayerRef:
    index: int
    name: str
    layer: "nn.Module"
    mlp: "nn.Module"


def is_vulcan_mlp(module: "nn.Module") -> bool:
    return all(
        isinstance(getattr(module, proj_name, None), torch.nn.Linear)
        for proj_name in ("up_proj", "gate_proj", "down_proj")
    )


def _get_nested_module(model: "nn.Module", path: str) -> "nn.Module | None":
    module = model
    for attr in path.split("."):
        if not hasattr(module, attr):
            return None

        module = getattr(module, attr)

    return module


def _find_decoder_mlp_layers(model: "nn.Module") -> list[MLPLayerRef]:
    candidate_paths = (
        "model.language_model.layers",
        "model.language_model.model.layers",
        "model.model.language_model.layers",
        "model.model.language_model.model.layers",
        "language_model.layers",
        "language_model.model.layers",
        "text_model.layers",
        "model.text_model.layers",
        "model.model.layers",
        "model.layers",
        "layers",
        "transformer.h",
    )
    for path in candidate_paths:
        layers_module = _get_nested_module(model, path)
        if layers_module is None:
            continue

        try:
            enumerated_layers = list(enumerate(layers_module))
        except TypeError:
            continue

        layer_refs = []
        for layer_idx, layer in enumerated_layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is not None and is_vulcan_mlp(mlp):
                layer_refs.append(MLPLayerRef(index=layer_idx, name=f"{path}.{layer_idx}", layer=layer, mlp=mlp))

        if layer_refs:
            return layer_refs

    return []


def find_mlp_layers(model: "nn.Module") -> list[MLPLayerRef]:
    r"""Find decoder layers that expose Qwen/Llama-style gated MLP projections."""
    decoder_mlp_layers = _find_decoder_mlp_layers(model)
    if decoder_mlp_layers:
        return decoder_mlp_layers

    layers: list[MLPLayerRef] = []
    seen_mlps: set[int] = set()
    for name, module in model.named_modules():
        mlp = getattr(module, "mlp", None)
        if mlp is None or id(mlp) in seen_mlps or not is_vulcan_mlp(mlp):
            continue

        layers.append(MLPLayerRef(index=len(layers), name=name, layer=module, mlp=mlp))
        seen_mlps.add(id(mlp))

    if not layers:
        raise ValueError(
            "Cannot find Qwen/Llama-style MLP layers with up_proj, gate_proj and down_proj. "
            "Please check the model architecture."
        )

    return layers


def get_intermediate_size(mlp: "nn.Module") -> int:
    return int(mlp.up_proj.weight.shape[0])


def get_hidden_size(mlp: "nn.Module") -> int:
    return int(mlp.up_proj.weight.shape[1])
