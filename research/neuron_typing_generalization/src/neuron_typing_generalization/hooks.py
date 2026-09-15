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


class DirectActivationCollector:
    """Retain one smoke sample's raw post-gating activations for an independent RMS check."""

    def __init__(self) -> None:
        self._layers: dict[int, Tensor] = {}

    def update(self, layer: int, activations: Tensor, _down_projection_weight: Tensor) -> None:
        if layer in self._layers:
            raise RuntimeError(f"Layer {layer} was captured more than once in one direct collector.")
        self._layers[layer] = activations.detach().to(device="cpu", dtype=torch.float32)

    def finalize(self) -> Mapping[int, Tensor]:
        if not self._layers:
            raise RuntimeError("No decoder layers were captured for the direct check.")
        return dict(sorted(self._layers.items()))


def _direct_masked_rms(activations: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Compute RMS by direct boolean indexing, independently of the streaming reduction implementation."""
    if activations.ndim == 2:
        activations = activations.unsqueeze(0)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    mask = mask.to(device="cpu", dtype=torch.bool)
    if activations.ndim != 3 or tuple(mask.shape) != tuple(activations.shape[:2]):
        raise ValueError("Direct-check mask must match activation batch and token dimensions.")
    counts = mask.sum(dim=1)
    rows = []
    for batch_index in range(activations.shape[0]):
        selected = activations[batch_index, mask[batch_index]]
        if selected.shape[0]:
            rows.append(torch.sqrt(selected.square().mean(dim=0)))
        else:
            rows.append(torch.full((activations.shape[-1],), torch.nan, dtype=torch.float32))
    return torch.stack(rows), counts, counts > 0


def validate_direct_vs_streaming(
    model: nn.Module,
    bindings: Iterable[FFNBinding],
    prepared: PreparedInputs,
    down_projection_norms: Mapping[int, Tensor],
    *,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare one real forward's retained activations with immediately reduced streaming statistics."""
    binding_tuple = tuple(bindings)
    streaming = StreamingImportanceCollector(prepared.token_roles, down_projection_norms)
    direct = DirectActivationCollector()

    def capture(layer: int, activations: Tensor, weight: Tensor) -> None:
        streaming.update(layer, activations, weight)
        direct.update(layer, activations, weight)

    with torch.inference_mode(), DownProjectionCapture(binding_tuple, capture):
        model(**prepared.model_inputs)
    streamed = streaming.finalize()
    retained = direct.finalize()
    expected_layers = tuple(binding.layer for binding in binding_tuple)
    if tuple(streamed) != expected_layers or tuple(retained) != expected_layers:
        raise RuntimeError("Direct/streaming smoke check did not capture every decoder layer exactly once.")

    role_masks = {"visual": prepared.token_roles.visual, "text": prepared.token_roles.text}
    layer_reports = []
    passed = True
    maximum_absolute_error = 0.0
    maximum_relative_error = 0.0
    for layer in expected_layers:
        modality_reports = {}
        for modality, mask in role_masks.items():
            direct_rms, direct_count, direct_valid = _direct_masked_rms(retained[layer], mask)
            streamed_values = getattr(streamed[layer], modality)
            streamed_rms = streamed_values.activation_rms.detach().to(device="cpu", dtype=torch.float32)
            if streamed_rms.ndim == 1:
                streamed_rms = streamed_rms.unsqueeze(0)
            streamed_count = streamed_values.token_count.detach().to(device="cpu").reshape(-1)
            streamed_valid = streamed_values.valid.detach().to(device="cpu").reshape(-1)
            norm = down_projection_norms[layer].detach().to(device="cpu", dtype=torch.float32).reshape(1, -1)
            direct_importance = direct_rms * norm
            streamed_importance = streamed_values.importance.detach().to(device="cpu", dtype=torch.float32)
            if streamed_importance.ndim == 1:
                streamed_importance = streamed_importance.unsqueeze(0)
            finite = torch.isfinite(direct_rms) & torch.isfinite(streamed_rms)
            absolute = torch.abs(direct_rms - streamed_rms)
            relative = absolute / torch.clamp(torch.abs(direct_rms), min=torch.finfo(torch.float32).eps)
            max_absolute = float(absolute[finite].max().item()) if bool(finite.any()) else 0.0
            max_relative = float(relative[finite].max().item()) if bool(finite.any()) else 0.0
            rms_close = torch.allclose(
                direct_rms, streamed_rms, atol=absolute_tolerance, rtol=relative_tolerance, equal_nan=True
            )
            importance_close = torch.allclose(
                direct_importance,
                streamed_importance,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
                equal_nan=True,
            )
            counts_equal = torch.equal(direct_count, streamed_count)
            validity_equal = torch.equal(direct_valid, streamed_valid)
            modality_passed = rms_close and importance_close and counts_equal and validity_equal
            passed &= modality_passed
            maximum_absolute_error = max(maximum_absolute_error, max_absolute)
            maximum_relative_error = max(maximum_relative_error, max_relative)
            modality_reports[modality] = {
                "passed": modality_passed,
                "token_count": direct_count.tolist(),
                "maximum_rms_absolute_error": max_absolute,
                "maximum_rms_relative_error": max_relative,
            }
        layer_reports.append({"layer": layer, "modalities": modality_reports})
    return {
        "status": "passed" if passed else "failed",
        "layers_checked": len(layer_reports),
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "maximum_rms_absolute_error": maximum_absolute_error,
        "maximum_rms_relative_error": maximum_relative_error,
        "raw_activations_retained_after_check": False,
        "layers": layer_reports,
    }


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
