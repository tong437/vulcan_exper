# Copyright 2026 the LlamaFactory team.
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

"""Shared utilities for Phase-3 structural pruning artifacts and verification."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ABLATION_SPEC = "rank_band:multimodal:0.05:0.2"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def masks_to_neuron_ids(masks_by_layer: dict[int, torch.Tensor]) -> dict[str, list[int]]:
    return {
        str(layer): mask.nonzero(as_tuple=False).flatten().tolist() for layer, mask in sorted(masks_by_layer.items())
    }


def neuron_ids_to_masks(
    neuron_ids_by_layer: dict[str | int, list[int]], layer_dims: dict[int, int]
) -> dict[int, torch.Tensor]:
    masks = {layer: torch.zeros(dim, dtype=torch.bool) for layer, dim in layer_dims.items()}
    normalized = {int(layer): neuron_ids for layer, neuron_ids in neuron_ids_by_layer.items()}
    if set(normalized) != set(layer_dims):
        raise ValueError(
            f"Mask layers do not match expected layers: mask={sorted(normalized)}, expected={sorted(layer_dims)}."
        )
    for layer, neuron_ids in normalized.items():
        if len(neuron_ids) != len(set(neuron_ids)):
            raise ValueError(f"Layer {layer} mask contains duplicate neuron IDs.")
        if any(neuron < 0 or neuron >= layer_dims[layer] for neuron in neuron_ids):
            raise ValueError(f"Layer {layer} mask contains an out-of-range neuron ID.")
        masks[layer][torch.tensor(neuron_ids, dtype=torch.long)] = True
    return masks


def validate_deletion_masks(
    masks_by_layer: dict[int, torch.Tensor],
    *,
    expected_num_layers: int | None = None,
    expected_layer_width: int | None = None,
    expected_pruned_per_layer: int | None = None,
) -> dict[str, Any]:
    if not masks_by_layer:
        raise ValueError("Deletion mask is empty.")
    layers = sorted(masks_by_layer)
    if layers != list(range(len(layers))):
        raise ValueError(f"Deletion mask layers must be contiguous from zero, got {layers}.")
    if expected_num_layers is not None and len(layers) != expected_num_layers:
        raise ValueError(f"Expected {expected_num_layers} layers, got {len(layers)}.")

    per_layer: dict[str, dict[str, int]] = {}
    for layer in layers:
        mask = masks_by_layer[layer]
        if mask.ndim != 1 or mask.dtype != torch.bool:
            raise ValueError(f"Layer {layer} mask must be a one-dimensional bool tensor.")
        width = int(mask.numel())
        pruned = int(mask.sum().item())
        if expected_layer_width is not None and width != expected_layer_width:
            raise ValueError(f"Layer {layer} has width {width}, expected {expected_layer_width}.")
        if expected_pruned_per_layer is not None and pruned != expected_pruned_per_layer:
            raise ValueError(f"Layer {layer} prunes {pruned} neurons, expected {expected_pruned_per_layer}.")
        if pruned <= 0 or pruned >= width:
            raise ValueError(f"Layer {layer} must keep and prune at least one neuron, got {pruned}/{width}.")
        per_layer[str(layer)] = {"width": width, "pruned": pruned, "kept": width - pruned}

    total = sum(row["width"] for row in per_layer.values())
    total_pruned = sum(row["pruned"] for row in per_layer.values())
    return {
        "num_layers": len(layers),
        "total_neurons": total,
        "total_pruned": total_pruned,
        "total_kept": total - total_pruned,
        "pruned_ratio": total_pruned / total,
        "per_layer": per_layer,
    }


def build_singleton_cluster_idx(
    masks_by_layer: dict[int, torch.Tensor],
) -> list[list[dict[str, int | list[int]]]]:
    """Encode plain deletion for ``pruning_mlp`` with one singleton cluster per kept neuron."""
    validate_deletion_masks(masks_by_layer)
    cluster_idx = []
    for layer in sorted(masks_by_layer):
        keep_ids = (~masks_by_layer[layer]).nonzero(as_tuple=False).flatten().tolist()
        cluster_idx.append([{"anchor": neuron, "neuron": [neuron]} for neuron in keep_ids])
    return cluster_idx


def validate_singleton_cluster_idx(
    cluster_idx: list[list[dict[str, Any]] | None], masks_by_layer: dict[int, torch.Tensor]
) -> dict[str, Any]:
    if len(cluster_idx) != len(masks_by_layer):
        raise ValueError(f"cluster_idx has {len(cluster_idx)} layers, expected {len(masks_by_layer)}.")
    per_layer = {}
    for layer, mask in sorted(masks_by_layer.items()):
        clusters = cluster_idx[layer]
        if clusters is None:
            raise ValueError(f"Layer {layer} unexpectedly has null clusters.")
        expected_keep = (~mask).nonzero(as_tuple=False).flatten().tolist()
        actual_keep = []
        for cluster in clusters:
            anchor = cluster.get("anchor")
            neurons = cluster.get("neuron")
            if not isinstance(anchor, int) or neurons != [anchor]:
                raise ValueError(f"Layer {layer} contains a non-singleton deletion cluster: {cluster}.")
            actual_keep.append(anchor)
        if actual_keep != expected_keep:
            raise ValueError(f"Layer {layer} singleton clusters do not exactly cover kept neurons in order.")
        per_layer[str(layer)] = {"clusters": len(clusters), "kept": len(expected_keep)}
    return {"validated": True, "per_layer": per_layer}


def theoretical_mlp_parameter_reduction(
    masks_by_layer: dict[int, torch.Tensor], hidden_size: int, *, biases: bool = False
) -> dict[str, Any]:
    pruned = sum(int(mask.sum().item()) for mask in masks_by_layer.values())
    weights_per_neuron = 3 * hidden_size
    bias_parameters_per_neuron = 2 if biases else 0
    removed_parameters = pruned * (weights_per_neuron + bias_parameters_per_neuron)
    return {
        "hidden_size": hidden_size,
        "pruned_neurons": pruned,
        "weights_per_neuron": weights_per_neuron,
        "bias_parameters_per_neuron": bias_parameters_per_neuron,
        "removed_parameters": removed_parameters,
        "fp16_bf16_removed_mib": removed_parameters * 2 / (1024**2),
        "fp32_removed_mib": removed_parameters * 4 / (1024**2),
    }


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    storage_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    return {"total": int(total), "trainable": int(trainable), "storage_bytes": int(storage_bytes)}


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    return sum(file.stat().st_size for file in root.rglob("*") if file.is_file())


def model_weight_size_bytes(path: str | Path) -> int:
    root = Path(path)
    patterns = ("model*.safetensors", "pytorch_model*.bin")
    files = {file for pattern in patterns for file in root.glob(pattern) if file.is_file()}
    if not files:
        raise FileNotFoundError(f"Cannot find model weight files under {root}.")
    return sum(file.stat().st_size for file in files)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_measurements(values: list[float], *, bootstrap_samples: int = 2000, seed: int = 2026) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty measurement list.")
    array = np.asarray(values, dtype=np.float64)
    result: dict[str, Any] = {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": percentile(values, 95),
        "min": float(array.min()),
        "max": float(array.max()),
        "raw": [float(value) for value in values],
    }
    if bootstrap_samples > 0 and len(values) > 1:
        rng = np.random.default_rng(seed)
        samples = rng.choice(array, size=(bootstrap_samples, len(array)), replace=True).mean(axis=1)
        result["mean_ci95"] = np.percentile(samples, [2.5, 97.5]).tolist()
    return result


def timed_runs(
    function: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    synchronize: Callable[[], None] | None = None,
) -> list[float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive.")
    sync = synchronize or (lambda: None)
    for _ in range(warmup):
        function()
        sync()
    timings = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        function()
        sync()
        timings.append(time.perf_counter() - start)
    return timings


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_existing_path(recorded_path: str, *, relative_to: str | Path) -> Path:
    path = Path(recorded_path)
    if path.is_absolute() or path.exists():
        return path.resolve()
    return (Path(relative_to).resolve().parent / path).resolve()
