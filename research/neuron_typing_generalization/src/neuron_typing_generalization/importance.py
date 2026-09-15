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

"""Sample-conditional neuron importance and four-way response classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


CLASS_LABELS = ("W", "V", "T", "M")
INVALID_CLASS = -1


@dataclass(frozen=True)
class ModalityImportance:
    """Per-sample importance statistics for one token modality."""

    activation_rms: Tensor
    down_projection_norm: Tensor
    importance: Tensor
    token_count: Tensor
    valid: Tensor


@dataclass(frozen=True)
class ImportanceShares:
    """Layer-normalized importance shares and validity mask."""

    values: Tensor
    valid: Tensor


@dataclass(frozen=True)
class NeuronClassification:
    """Four-way V/T/M/W classes derived from visual and text shares."""

    classes: Tensor
    visual_strong: Tensor
    text_strong: Tensor
    valid: Tensor
    threshold: float


def _batched_activations(activations: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor, bool]:
    if activations.ndim not in (2, 3):
        raise ValueError("activations must have shape [tokens, d_ff] or [batch, tokens, d_ff].")
    was_unbatched = activations.ndim == 2
    if was_unbatched:
        activations = activations.unsqueeze(0)
    if token_mask.ndim == 1:
        token_mask = token_mask.unsqueeze(0)
    if token_mask.ndim != 2 or tuple(token_mask.shape) != tuple(activations.shape[:2]):
        raise ValueError("token_mask must match the activation batch and token dimensions.")
    return activations, token_mask.to(device=activations.device, dtype=torch.bool), was_unbatched


def validate_token_masks(
    visual_mask: Tensor,
    text_mask: Tensor,
    control_mask: Tensor | None = None,
    padding_mask: Tensor | None = None,
) -> None:
    """Reject overlapping token roles before modality statistics are computed."""
    masks = [visual_mask.bool(), text_mask.bool()]
    if control_mask is not None:
        masks.append(control_mask.bool())
    if padding_mask is not None:
        masks.append(padding_mask.bool())
    if any(mask.shape != masks[0].shape for mask in masks[1:]):
        raise ValueError("All token role masks must have identical shapes.")
    occupancy = torch.stack(masks).to(torch.int8).sum(dim=0)
    if torch.any(occupancy > 1):
        raise ValueError("Visual, text, control, and padding token masks must be mutually exclusive.")


def compute_modality_importance(
    activations: Tensor,
    down_projection_weight: Tensor,
    token_mask: Tensor,
    down_projection_norm: Tensor | None = None,
) -> ModalityImportance:
    """Compute down-projection-weighted activation RMS for one modality.

    Args:
        activations: Post-gating MLP activations with final dimension ``d_ff``.
        down_projection_weight: Weight with shape ``[d_model, d_ff]``.
        token_mask: Boolean selection over the activation token dimensions.
        down_projection_norm: Optional precomputed per-neuron norm with shape ``[d_ff]``.
    """
    activations, token_mask, was_unbatched = _batched_activations(activations, token_mask)
    if down_projection_weight.ndim != 2 or down_projection_weight.shape[1] != activations.shape[-1]:
        raise ValueError("down_projection_weight must have shape [d_model, d_ff].")

    accumulation = activations.to(torch.float32)
    counts = token_mask.sum(dim=1)
    denominator = counts.clamp_min(1).to(torch.float32).unsqueeze(-1)
    squared_sum = (accumulation.square() * token_mask.unsqueeze(-1)).sum(dim=1)
    rms = torch.sqrt(squared_sum / denominator)
    valid = counts > 0
    rms = torch.where(valid.unsqueeze(-1), rms, torch.full_like(rms, torch.nan))

    down_norm = down_projection_norm
    if down_norm is None:
        down_norm = torch.linalg.vector_norm(down_projection_weight.to(torch.float32), ord=2, dim=0)
    elif down_norm.ndim != 1 or down_norm.shape[0] != activations.shape[-1]:
        raise ValueError("down_projection_norm must have shape [d_ff].")
    down_norm = down_norm.to(device=activations.device, dtype=torch.float32)
    importance = rms * down_norm.unsqueeze(0)
    if was_unbatched:
        return ModalityImportance(rms[0], down_norm, importance[0], counts[0], valid[0])
    return ModalityImportance(rms, down_norm, importance, counts, valid)


def normalize_importance(importance: Tensor) -> ImportanceShares:
    """Normalize importance across FFN neurons for each sample/layer slice."""
    if importance.ndim < 1 or importance.shape[-1] == 0:
        raise ValueError("importance must have a non-empty d_ff dimension.")
    denominator = importance.sum(dim=-1, keepdim=True)
    valid = torch.isfinite(importance).all(dim=-1) & torch.isfinite(denominator.squeeze(-1))
    valid &= denominator.squeeze(-1) > 0
    safe_denominator = torch.where(valid.unsqueeze(-1), denominator, torch.ones_like(denominator))
    shares = importance / safe_denominator
    shares = torch.where(valid.unsqueeze(-1), shares, torch.full_like(shares, torch.nan))
    return ImportanceShares(values=shares, valid=valid)


def classify_neuron_shares(
    visual_shares: Tensor,
    text_shares: Tensor,
    visual_valid: Tensor | None = None,
    text_valid: Tensor | None = None,
) -> NeuronClassification:
    """Classify neurons using the strict uniform-share threshold from the study specification."""
    if visual_shares.shape != text_shares.shape or visual_shares.ndim < 1:
        raise ValueError("visual_shares and text_shares must have the same non-scalar shape.")
    d_ff = visual_shares.shape[-1]
    if d_ff == 0:
        raise ValueError("The d_ff dimension must be non-empty.")

    inferred_visual_valid = torch.isfinite(visual_shares).all(dim=-1)
    inferred_text_valid = torch.isfinite(text_shares).all(dim=-1)
    if visual_valid is not None:
        inferred_visual_valid &= visual_valid.to(device=visual_shares.device, dtype=torch.bool)
    if text_valid is not None:
        inferred_text_valid &= text_valid.to(device=text_shares.device, dtype=torch.bool)
    valid_slice = inferred_visual_valid & inferred_text_valid
    threshold = 1.0 / d_ff
    visual_strong = visual_shares > threshold
    text_strong = text_shares > threshold
    classes = visual_strong.to(torch.int8) + 2 * text_strong.to(torch.int8)
    valid = valid_slice.unsqueeze(-1).expand_as(classes)
    classes = torch.where(valid, classes, torch.full_like(classes, INVALID_CLASS))
    return NeuronClassification(classes, visual_strong & valid, text_strong & valid, valid, threshold)


def class_ids_to_labels(classes: Tensor) -> list[str]:
    """Convert a one-dimensional class tensor to stable string labels."""
    if classes.ndim != 1:
        raise ValueError("classes must be one-dimensional.")
    return [CLASS_LABELS[value] if 0 <= value < len(CLASS_LABELS) else "INVALID" for value in classes.tolist()]
