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

"""NeuPAT neuron-role allocation and role-aware SFT constraints.

The implementation follows the FFN interpretation used by NeuPAT: one neuron
is one gated-MLP intermediate channel, represented by rows in ``gate_proj`` and
``up_proj`` and the matching column in ``down_proj``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .modeling import find_mlp_layers


NEUPAT_ROLES = ("language", "multimodal", "shared", "reserve")


def select_importance_mass(scores: torch.Tensor, coverage: float) -> torch.Tensor:
    """Select the smallest deterministic index set covering ``coverage`` mass."""
    if scores.ndim != 1:
        raise ValueError(f"NeuPAT importance scores must be one-dimensional, got {tuple(scores.shape)}.")
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"NeuPAT coverage must be in (0, 1], got {coverage}.")
    values = scores.detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("NeuPAT importance scores must be finite.")
    if bool((values < 0).any()):
        raise ValueError("NeuPAT importance scores must be non-negative.")
    total = float(values.sum().item())
    if total <= 0.0:
        return torch.empty(0, dtype=torch.long)

    neuron_ids = torch.arange(values.numel(), dtype=torch.long)
    order = torch.argsort(neuron_ids, stable=True)
    order = order[torch.argsort(values[order], descending=True, stable=True)]
    cumulative = torch.cumsum(values[order], dim=0)
    target = coverage * total
    count = int(torch.searchsorted(cumulative, torch.tensor(target), right=False).item()) + 1
    return order[:count]


def allocate_neupat_roles(
    text_scores: torch.Tensor,
    vision_scores: torch.Tensor,
    *,
    tau_text: float = 0.8,
    tau_vision: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Allocate language, multimodal, shared, and reserve neuron roles."""
    if text_scores.shape != vision_scores.shape:
        raise ValueError(
            f"Text and vision NeuPAT scores must have the same shape, got "
            f"{tuple(text_scores.shape)} and {tuple(vision_scores.shape)}."
        )
    text_important = select_importance_mass(text_scores, tau_text)
    vision_important = select_importance_mass(vision_scores, tau_vision)
    width = text_scores.numel()
    text_mask = torch.zeros(width, dtype=torch.bool)
    vision_mask = torch.zeros(width, dtype=torch.bool)
    text_mask[text_important] = True
    vision_mask[vision_important] = True
    role_masks = {
        "language": text_mask & ~vision_mask,
        "multimodal": vision_mask & ~text_mask,
        "shared": text_mask & vision_mask,
        "reserve": ~text_mask & ~vision_mask,
    }
    return {
        "text_important": text_important,
        "vision_important": vision_important,
        **{name: mask.nonzero(as_tuple=False).flatten() for name, mask in role_masks.items()},
    }


def validate_neupat_artifact(
    artifact: dict[str, Any],
    *,
    expected_layer_dims: dict[int, int] | None = None,
) -> dict[int, dict[str, list[int]]]:
    """Validate and normalize a NeuPAT role artifact."""
    if artifact.get("artifact_version") != 1 or artifact.get("method") != "neupat":
        raise ValueError("Expected a NeuPAT artifact with artifact_version=1 and method='neupat'.")
    raw_layers = artifact.get("layers")
    if not isinstance(raw_layers, dict) or not raw_layers:
        raise ValueError("NeuPAT artifact must contain a non-empty `layers` mapping.")

    normalized: dict[int, dict[str, list[int]]] = {}
    for layer_key, row in raw_layers.items():
        layer_idx = int(layer_key)
        if not isinstance(row, dict):
            raise ValueError(f"NeuPAT layer {layer_idx} must be a mapping.")
        width = int(row.get("intermediate_size", -1))
        if width <= 0:
            raise ValueError(f"NeuPAT layer {layer_idx} has invalid intermediate_size={width}.")
        if expected_layer_dims is not None:
            if layer_idx not in expected_layer_dims:
                raise ValueError(f"NeuPAT artifact contains unexpected layer {layer_idx}.")
            if expected_layer_dims[layer_idx] != width:
                raise ValueError(
                    f"NeuPAT layer {layer_idx} width {width} does not match model width "
                    f"{expected_layer_dims[layer_idx]}."
                )

        role_sets: dict[str, set[int]] = {}
        for role in NEUPAT_ROLES:
            indices = row.get(role)
            if not isinstance(indices, list) or not all(isinstance(index, int) for index in indices):
                raise ValueError(f"NeuPAT layer {layer_idx} role {role!r} must be a list of integers.")
            role_set = set(indices)
            if len(role_set) != len(indices):
                raise ValueError(f"NeuPAT layer {layer_idx} role {role!r} contains duplicate indices.")
            if role_set and (min(role_set) < 0 or max(role_set) >= width):
                raise ValueError(f"NeuPAT layer {layer_idx} role {role!r} contains out-of-range indices.")
            role_sets[role] = role_set

        for position, left in enumerate(NEUPAT_ROLES):
            for right in NEUPAT_ROLES[position + 1 :]:
                overlap = role_sets[left] & role_sets[right]
                if overlap:
                    raise ValueError(
                        f"NeuPAT layer {layer_idx} roles {left!r} and {right!r} overlap at {sorted(overlap)[:5]}."
                    )
        assigned = set().union(*(role_sets[role] for role in NEUPAT_ROLES))
        if assigned != set(range(width)):
            missing = sorted(set(range(width)) - assigned)
            raise ValueError(f"NeuPAT layer {layer_idx} roles are not exhaustive; missing {missing[:5]}.")
        normalized[layer_idx] = {role: sorted(role_sets[role]) for role in NEUPAT_ROLES}

    if expected_layer_dims is not None and set(normalized) != set(expected_layer_dims):
        missing = sorted(set(expected_layer_dims) - set(normalized))
        raise ValueError(f"NeuPAT artifact is missing model layers: {missing}.")
    return dict(sorted(normalized.items()))


def load_neupat_artifact(
    path: str | Path,
    *,
    expected_layer_dims: dict[int, int] | None = None,
) -> tuple[dict[str, Any], dict[int, dict[str, list[int]]]]:
    artifact_path = Path(path)
    with artifact_path.open(encoding="utf-8") as file:
        artifact = json.load(file)
    return artifact, validate_neupat_artifact(artifact, expected_layer_dims=expected_layer_dims)


class NeuPATController:
    """Apply NeuPAT language-gradient masks and shared-neuron regularization."""

    def __init__(
        self,
        model: torch.nn.Module,
        role_path: str | Path,
        *,
        lambda_in: float = 0.1,
        lambda_out: float = 0.1,
        reduction: Literal["sum", "mean"] = "sum",
    ) -> None:
        if lambda_in < 0 or lambda_out < 0:
            raise ValueError("NeuPAT regularization coefficients must be non-negative.")
        if reduction not in {"sum", "mean"}:
            raise ValueError("NeuPAT reduction must be `sum` or `mean`.")

        self.mlp_layers = find_mlp_layers(model)
        layer_dims = {layer.index: int(layer.mlp.up_proj.weight.shape[0]) for layer in self.mlp_layers}
        self.artifact, self.roles = load_neupat_artifact(role_path, expected_layer_dims=layer_dims)
        self.lambda_in = float(lambda_in)
        self.lambda_out = float(lambda_out)
        self.reduction = reduction
        self.handles: list[Any] = []
        self.references: dict[int, dict[str, torch.Tensor]] = {}
        self._register_language_gradient_masks()
        self._capture_shared_references()

    @staticmethod
    def _row_gradient_hook(indices: torch.Tensor):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            keep = torch.ones(gradient.shape[0], device=gradient.device, dtype=gradient.dtype)
            keep[indices.to(gradient.device)] = 0
            return gradient * keep[:, None]

        return hook

    @staticmethod
    def _column_gradient_hook(indices: torch.Tensor):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            keep = torch.ones(gradient.shape[1], device=gradient.device, dtype=gradient.dtype)
            keep[indices.to(gradient.device)] = 0
            return gradient * keep[None, :]

        return hook

    def _register_language_gradient_masks(self) -> None:
        for layer_ref in self.mlp_layers:
            indices = torch.tensor(self.roles[layer_ref.index]["language"], dtype=torch.long)
            if indices.numel() == 0:
                continue
            for projection in (layer_ref.mlp.gate_proj, layer_ref.mlp.up_proj):
                if projection.weight.requires_grad:
                    self.handles.append(projection.weight.register_hook(self._row_gradient_hook(indices)))
            if layer_ref.mlp.down_proj.weight.requires_grad:
                self.handles.append(layer_ref.mlp.down_proj.weight.register_hook(self._column_gradient_hook(indices)))

    def _capture_shared_references(self) -> None:
        for layer_ref in self.mlp_layers:
            indices = torch.tensor(
                self.roles[layer_ref.index]["shared"],
                device=layer_ref.mlp.up_proj.weight.device,
                dtype=torch.long,
            )
            self.references[layer_ref.index] = {
                "indices": indices,
                "gate": layer_ref.mlp.gate_proj.weight.detach().index_select(0, indices).clone(),
                "up": layer_ref.mlp.up_proj.weight.detach().index_select(0, indices).clone(),
                "down": layer_ref.mlp.down_proj.weight.detach().index_select(1, indices).clone(),
            }

    @staticmethod
    def _reference_on(
        layer_reference: dict[str, torch.Tensor],
        name: str,
        parameter: torch.Tensor,
    ) -> torch.Tensor:
        reference = layer_reference[name]
        if reference.device != parameter.device or reference.dtype != parameter.dtype:
            reference = reference.to(device=parameter.device, dtype=parameter.dtype)
            layer_reference[name] = reference
        return reference

    def regularization(self) -> tuple[torch.Tensor, dict[str, float]]:
        device = self.mlp_layers[0].mlp.up_proj.weight.device
        input_terms: list[torch.Tensor] = []
        output_terms: list[torch.Tensor] = []
        input_elements = 0
        output_neurons = 0
        for layer_ref in self.mlp_layers:
            reference = self.references[layer_ref.index]
            indices = reference["indices"].to(layer_ref.mlp.up_proj.weight.device)
            reference["indices"] = indices
            if indices.numel() == 0:
                continue
            gate = layer_ref.mlp.gate_proj.weight.index_select(0, indices)
            up = layer_ref.mlp.up_proj.weight.index_select(0, indices)
            down = layer_ref.mlp.down_proj.weight.index_select(1, indices)
            gate_ref = self._reference_on(reference, "gate", gate)
            up_ref = self._reference_on(reference, "up", up)
            down_ref = self._reference_on(reference, "down", down)
            input_terms.extend(
                [(gate.float() - gate_ref.float()).square().sum(), (up.float() - up_ref.float()).square().sum()]
            )
            down_unit = F.normalize(down.float().T, dim=1)
            down_ref_unit = F.normalize(down_ref.float().T, dim=1)
            # 0.5 * ||normalize(w) - normalize(w0)||^2 is the cosine
            # distance, but is exactly zero for identical slices and cannot
            # become negative through floating-point roundoff.
            output_terms.append(0.5 * (down_unit - down_ref_unit).square().sum())
            input_elements += gate.numel() + up.numel()
            output_neurons += indices.numel()

        input_raw = torch.stack(input_terms).sum() if input_terms else torch.zeros((), device=device)
        output_raw = torch.stack(output_terms).sum() if output_terms else torch.zeros((), device=device)
        if self.reduction == "mean":
            input_raw = input_raw / max(input_elements, 1)
            output_raw = output_raw / max(output_neurons, 1)
        weighted = self.lambda_in * input_raw + self.lambda_out * output_raw
        return weighted, {
            "neupat_input_raw": float(input_raw.detach().item()),
            "neupat_output_raw": float(output_raw.detach().item()),
            "neupat_loss": float(weighted.detach().item()),
        }

    def role_counts(self) -> dict[str, int]:
        return {role: sum(len(layer_roles[role]) for layer_roles in self.roles.values()) for role in NEUPAT_ROLES}

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
