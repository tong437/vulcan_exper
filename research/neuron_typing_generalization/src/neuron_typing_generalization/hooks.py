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

"""Model-agnostic post-gating capture at decoder MLP down projections."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .importance import ModalityImportance, compute_modality_importance, validate_token_masks
from .interfaces import FFNBinding, PreparedInputs, TokenRoleMasks


_LAYER_PATTERN = re.compile(r"^(?P<prefix>.*?)(?:layers|h)\.(?P<layer>\d+)\.mlp\.down_proj$")


def discover_decoder_ffns(model: nn.Module) -> tuple[FFNBinding, ...]:
    """Discover language-decoder ``mlp.down_proj`` modules without family conditionals."""
    grouped: dict[str, list[FFNBinding]] = {}
    for name, module in model.named_modules():
        match = _LAYER_PATTERN.match(name)
        weight = getattr(module, "weight", None)
        if match is None or not isinstance(weight, Tensor) or weight.ndim != 2:
            continue
        d_model, d_ff = weight.shape
        grouped.setdefault(match.group("prefix"), []).append(
            FFNBinding(int(match.group("layer")), module, int(d_model), int(d_ff))
        )
    candidates = []
    for prefix, bindings in grouped.items():
        bindings.sort(key=lambda binding: binding.layer)
        if [binding.layer for binding in bindings] == list(range(len(bindings))):
            candidates.append((prefix, tuple(bindings)))
    if not candidates:
        raise ValueError("No decoder MLP down projections matched '*.layers.<index>.mlp.down_proj'.")
    maximum_layers = max(len(bindings) for _, bindings in candidates)
    longest = [(prefix, bindings) for prefix, bindings in candidates if len(bindings) == maximum_layers]
    if len(longest) != 1:
        raise ValueError(f"Decoder discovery is ambiguous across module roots: {[prefix for prefix, _ in longest]}")
    return longest[0][1]


class DownProjectionCapture(AbstractContextManager["DownProjectionCapture"]):
    """Capture the input to every down projection, which is the post-gating activation ``z``."""

    def __init__(self, bindings: Iterable[FFNBinding], callback: Callable[[int, Tensor, Tensor], None]) -> None:
        self._bindings = tuple(bindings)
        self._callback = callback
        self._handles: list[Any] = []

    def __enter__(self) -> DownProjectionCapture:
        if self._handles:
            raise RuntimeError("Capture hooks are already installed.")
        for binding in self._bindings:
            if not hasattr(binding.down_projection, "register_forward_pre_hook"):
                raise TypeError(f"Layer {binding.layer} down projection does not support forward pre-hooks.")

            def hook(module: nn.Module, inputs: tuple[Any, ...], *, layer: int = binding.layer) -> None:
                if not inputs or not isinstance(inputs[0], Tensor):
                    raise TypeError(f"Layer {layer} down projection did not receive a tensor input.")
                weight = getattr(module, "weight", None)
                if not isinstance(weight, Tensor):
                    raise TypeError(f"Layer {layer} down projection has no tensor weight.")
                self._callback(layer, inputs[0].detach(), weight.detach())

            self._handles.append(binding.down_projection.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()


@dataclass(frozen=True)
class LayerImportance:
    """Visual and text importance captured for one layer and forward pass."""

    visual: ModalityImportance
    text: ModalityImportance


class StreamingImportanceCollector:
    """Immediately reduce captured activations instead of retaining token-by-neuron tensors."""

    def __init__(self, token_roles: TokenRoleMasks, down_projection_norms: Mapping[int, Tensor] | None = None) -> None:
        validate_token_masks(token_roles.visual, token_roles.text, token_roles.control, token_roles.padding)
        self._token_roles = token_roles
        self._down_projection_norms = down_projection_norms or {}
        self._layers: dict[int, LayerImportance] = {}

    def update(self, layer: int, activations: Tensor, down_projection_weight: Tensor) -> None:
        if layer in self._layers:
            raise RuntimeError(f"Layer {layer} was captured more than once in one collector.")
        down_norm = self._down_projection_norms.get(layer)
        visual = compute_modality_importance(
            activations, down_projection_weight, self._token_roles.visual, down_projection_norm=down_norm
        )
        text = compute_modality_importance(
            activations, down_projection_weight, self._token_roles.text, down_projection_norm=down_norm
        )
        self._layers[layer] = LayerImportance(visual=visual, text=text)

    def finalize(self) -> Mapping[int, LayerImportance]:
        if not self._layers:
            raise RuntimeError("No decoder layers were captured.")
        return dict(sorted(self._layers.items()))


class InjectedModelAdapter:
    """CPU-testable adapter around an already constructed model and input preparation callback."""

    def __init__(
        self,
        family: str,
        model: nn.Module,
        prepare: Callable[[Mapping[str, Any]], PreparedInputs],
    ) -> None:
        self.family = family
        self.model = model
        self._prepare = prepare
        self._bindings = discover_decoder_ffns(model)

    def prepare_inputs(self, sample: Mapping[str, Any]) -> PreparedInputs:
        return self._prepare(sample)

    def iter_ffn_bindings(self) -> Iterable[FFNBinding]:
        return iter(self._bindings)


def masks_from_spans(
    attention_mask: Tensor,
    visual_spans: Iterable[tuple[int, int]],
    text_spans: Iterable[tuple[int, int]],
) -> TokenRoleMasks:
    """Construct auditable masks after a model processor has resolved expanded decoder positions."""
    if attention_mask.ndim != 1:
        raise ValueError("attention_mask must be one-dimensional for one prepared sample.")
    length = attention_mask.shape[0]
    visual = torch.zeros(length, dtype=torch.bool, device=attention_mask.device)
    text = torch.zeros_like(visual)
    for target, spans in ((visual, visual_spans), (text, text_spans)):
        for start, stop in spans:
            if not 0 <= start <= stop <= length:
                raise IndexError(f"Token span [{start}, {stop}) is outside sequence length {length}.")
            target[start:stop] = True
    padding = ~attention_mask.bool()
    control = ~(visual | text | padding)
    roles = TokenRoleMasks(visual=visual, text=text, control=control, padding=padding)
    validate_token_masks(roles.visual, roles.text, roles.control, roles.padding)
    return roles
