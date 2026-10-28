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

"""Static aligned MLP plans and physical dense-matrix slicing for throughput."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .hooks import discover_decoder_ffns


def calibration_costs(root: Path) -> tuple[list[np.ndarray], dict]:
    """Average per-sample normalized costs, excluding padded FFN coordinates."""
    manifest = json.loads((root / "manifest.json").read_text())
    widths = manifest["d_ff_by_layer"]
    paths = sorted((root / "sample_statistics").glob("*.npz"))
    if manifest["status"] != "complete" or manifest["samples"] != 1000 or len(paths) != 1000:
        raise ValueError("Calibration requires all 1000 frozen samples")
    result = [np.zeros(w, dtype=np.float64) for w in widths]
    digest, ids = hashlib.sha256(), []
    for path in paths:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        with np.load(path, allow_pickle=False) as data:
            ids.append(str(data["sample_id"].item()))
            q = data["q"]
            if q.shape[:2] != (len(widths), 3):
                raise ValueError("Invalid q shape")
            for layer, width in enumerate(widths):
                values = q[layer, :, :width].astype(np.float64)
                if not np.isfinite(values).all() or np.any(values < 0):
                    raise ValueError("Invalid calibration scores")
                s = values.sum(axis=0)
                if s.sum() <= 0:
                    raise ValueError("Zero layer total")
                result[layer] += s / s.sum() / len(widths) / len(paths)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate calibration IDs")
    return result, {"manifest": manifest, "statistics_sha256": digest.hexdigest(), "sample_ids": ids}


def make_plan(costs: list[np.ndarray], rate: float, method: str) -> dict:
    """Equal count before alignment; restore highest-cost removed units to align widths."""
    if not 0 <= rate <= 0.5 or method not in {"original", "uniform", "nonuniform"}:
        raise ValueError("Unsupported method or pruning rate")
    if method == "original" and rate != 0:
        raise ValueError("Original model requires rate zero")
    widths = np.array([len(s) for s in costs], dtype=int)
    if np.any(widths % 8) or np.any(widths <= 0):
        raise ValueError("Original widths must be positive multiples of eight")
    if any(not np.isfinite(s).all() or np.any(s < 0) for s in costs):
        raise ValueError("Invalid costs")
    counts = np.floor(rate * widths).astype(int)
    if method == "nonuniform":
        order = np.argsort(np.concatenate(costs), kind="stable")
        layer_ids = np.repeat(np.arange(len(widths)), widths)
        counts = np.bincount(layer_ids[order[: int(counts.sum())]], minlength=len(widths))
    # Keep a minimum width of eight: a zero-width Linear is not the intended dense architecture.
    aligned = np.maximum(8, ((widths - counts + 7) // 8) * 8)
    kept = []
    for scores, width in zip(costs, aligned):
        ranked = np.argsort(scores, kind="stable")
        kept.append(np.sort(ranked[len(scores) - int(width) :]).tolist())
    plan = {
        "method": method,
        "target_rate": rate,
        "original_widths": widths.tolist(),
        "pre_alignment_deleted": counts.tolist(),
        "aligned_widths": aligned.tolist(),
        "kept": kept,
        "actual_rate": float(1 - aligned.sum() / widths.sum()),
        "minimum_width": 8,
        "alignment": 8,
    }
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def decoder_mlps(model):
    down = {id(b.down_projection): b for b in discover_decoder_ffns(model)}
    mlps = []
    for module in model.modules():
        binding = down.get(id(getattr(module, "down_proj", None)))
        if binding is not None:
            mlps.append((binding.layer, module))
    return [module for _, module in sorted(mlps)]


def slice_linear(linear: nn.Linear, indices: torch.Tensor, dim: int) -> nn.Linear:
    if not isinstance(linear, nn.Linear) or hasattr(linear, "_hf_hook"):
        raise TypeError("Expected ordinary dense Linear without dispatch hooks")
    indices = indices.to(linear.weight.device)
    weight = linear.weight.detach().index_select(dim, indices).contiguous()
    replacement = nn.Linear(
        weight.shape[1], weight.shape[0], bias=linear.bias is not None, device=weight.device, dtype=weight.dtype
    )
    replacement.weight = nn.Parameter(weight, requires_grad=False)
    if linear.bias is not None:
        bias = linear.bias.detach()
        if dim == 0:
            bias = bias.index_select(0, indices)
        replacement.bias = nn.Parameter(bias.clone(), requires_grad=False)
    return replacement


def physically_prune(model, plan: dict) -> dict:
    """Slice gate/up rows and down columns; preserve down bias and all other modules."""
    mlps = decoder_mlps(model)
    if [m.down_proj.in_features for m in mlps] != plan["original_widths"]:
        raise ValueError("Loaded decoder widths disagree with plan")
    for m in mlps:
        if not all(isinstance(getattr(m, key, None), nn.Linear) for key in ("gate_proj", "up_proj", "down_proj")):
            raise TypeError(f"Unsupported MLP layout {type(m).__name__}; requires explicit adapter")
    before = sum(p.numel() for p in model.parameters())
    for m, keep in zip(mlps, plan["kept"]):
        indices = torch.tensor(keep, dtype=torch.long)
        for key in ("gate_proj", "up_proj"):
            setattr(m, key, slice_linear(getattr(m, key), indices, 0))
        m.down_proj = slice_linear(m.down_proj, indices, 1)
        if hasattr(m, "intermediate_size"):
            m.intermediate_size = len(keep)
    return {
        "parameters_before": before,
        "parameters_after": sum(p.numel() for p in model.parameters()),
        "mlp_parameters_after": sum(p.numel() for m in mlps for p in m.parameters()),
    }


def removed_indices(plan):
    return {l: np.setdiff1d(np.arange(w), plan["kept"][l]) for l, w in enumerate(plan["original_widths"])}
