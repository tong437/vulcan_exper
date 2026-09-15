# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Executable activation collection with injected models and immediate reduction."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .hooks import DownProjectionCapture, InjectedModelAdapter, StreamingImportanceCollector
from .schema_validation import validate_document


ACTIVATION_COLUMNS = [
    "run_id",
    "sample_id",
    "model",
    "dataset",
    "layer",
    "neuron",
    "modality",
    "token_count",
    "activation_rms",
    "down_projection_norm",
    "valid",
]


def _write_parquet_atomic(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_npz_atomic(destination: Path, **arrays: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_npy_atomic(destination: Path, values: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npy", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.save(temporary_path, values, allow_pickle=False)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_atomic(destination: Path, value: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(value, temporary, ensure_ascii=False, indent=2)
        temporary.write("\n")
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_sample_vector(value: Tensor, name: str) -> Tensor:
    if value.ndim == 1:
        return value
    if value.ndim == 2 and value.shape[0] == 1:
        return value[0]
    raise ValueError(f"{name} must describe exactly one sample; got shape {tuple(value.shape)}.")


def collect_reduced_activations(
    adapter: InjectedModelAdapter,
    samples: pd.DataFrame,
    run_dir: Path,
    run_id: str,
    model_name: str,
    dataset_name: str,
) -> dict[str, Any]:
    """Run one injected model into resumable compact per-sample partitions."""
    if samples.empty or "sample_id" not in samples:
        raise ValueError("A nonempty sample manifest with sample_id is required.")
    bindings = tuple(adapter.iter_ffn_bindings())
    layers = len(bindings)
    d_ff_by_layer = tuple(binding.d_ff for binding in bindings)
    d_ff = max(d_ff_by_layer)
    activation_dir = run_dir / "activations"
    part_dir = activation_dir / "sample_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    down_norm_tensors = {
        binding.layer: torch.linalg.vector_norm(
            binding.down_projection.weight.detach().to(torch.float32), ord=2, dim=0
        )
        for binding in bindings
    }
    down_norms = np.zeros((layers, d_ff), dtype=np.float32)
    for layer, width in enumerate(d_ff_by_layer):
        down_norms[layer, :width] = down_norm_tensors[layer].cpu().numpy()
    if not np.isfinite(down_norms).all():
        raise RuntimeError("Down-projection norms contain non-finite values.")
    weight_path = activation_dir / "down_projection_norms.npy"
    if weight_path.exists():
        existing = np.load(weight_path, allow_pickle=False)
        if existing.shape != down_norms.shape or not np.array_equal(existing, down_norms):
            raise RuntimeError("Existing down-projection norms do not match the injected model.")
    else:
        _write_npy_atomic(weight_path, down_norms)

    parts = []
    resumed_samples = 0
    for sample_index, sample in enumerate(samples.to_dict(orient="records")):
        sample_id = str(sample["sample_id"])
        output = part_dir / f"{sample_index:06d}.npz"
        if output.exists():
            with np.load(output, allow_pickle=False) as existing:
                existing_id = str(existing["sample_id"].item())
                if existing_id != sample_id or existing["activation_rms"].shape != (layers, 2, d_ff):
                    raise RuntimeError(f"Existing activation partition conflicts with sample {sample_index}: {output}")
            resumed_samples += 1
            parts.append(
                {"sample_index": sample_index, "sample_id": sample_id, "file": str(output.relative_to(run_dir))}
            )
            continue
        prepared = adapter.prepare_inputs(sample)
        collector = StreamingImportanceCollector(prepared.token_roles, down_norm_tensors)
        with torch.inference_mode(), DownProjectionCapture(bindings, collector.update):
            adapter.model(**prepared.model_inputs)
        captured = collector.finalize()
        if tuple(captured) != tuple(binding.layer for binding in bindings):
            raise RuntimeError("Not every discovered decoder layer was captured exactly once.")
        activation_rms = np.zeros((layers, 2, d_ff), dtype=np.float32)
        token_count = np.empty((layers, 2), dtype=np.int32)
        valid = np.empty((layers, 2), dtype=np.bool_)
        for layer, statistics in captured.items():
            for modality_index, values in enumerate((statistics.visual, statistics.text)):
                rms = _single_sample_vector(values.activation_rms, "activation_rms").to(torch.float32).cpu()
                valid_tensor = values.valid
                is_valid = bool(valid_tensor.item()) if valid_tensor.ndim == 0 else bool(valid_tensor[0].item())
                count_tensor = values.token_count
                count = int(count_tensor.item()) if count_tensor.ndim == 0 else int(count_tensor[0].item())
                width = d_ff_by_layer[layer]
                activation_rms[layer, modality_index, :width] = rms.numpy()
                token_count[layer, modality_index] = count
                valid[layer, modality_index] = is_valid
        for layer, width in enumerate(d_ff_by_layer):
            if valid[layer].any() and not np.isfinite(activation_rms[layer, valid[layer], :width]).all():
                raise RuntimeError(f"Valid activation RMS statistics are non-finite for sample_id={sample_id}.")
        _write_npz_atomic(
            output,
            sample_id=np.asarray(sample_id),
            activation_rms=activation_rms,
            token_count=token_count,
            valid=valid,
        )
        parts.append({"sample_index": sample_index, "sample_id": sample_id, "file": str(output.relative_to(run_dir))})
        _write_json_atomic(
            activation_dir / "progress.json",
            {"run_id": run_id, "completed_samples": sample_index + 1, "total_samples": len(samples)},
        )

    part_digest = hashlib.sha256()
    for part in parts:
        path = run_dir / part["file"]
        digest = _sha256(path)
        part["sha256"] = digest
        part_digest.update(f"{part['sample_index']}:{part['sample_id']}:{digest}\n".encode())
    rows = len(samples) * sum(d_ff_by_layer) * 2
    manifest = {
        "run_id": run_id,
        "samples": len(samples),
        "layers": layers,
        "d_ff": d_ff,
        "d_ff_by_layer": list(d_ff_by_layer),
        "rows": rows,
        "storage_format": "partitioned_npz_v1",
        "statistics_directory": str(part_dir.relative_to(run_dir)),
        "parts": parts,
        "parts_sha256": part_digest.hexdigest(),
        "down_projection_norms_file": str(weight_path.relative_to(run_dir)),
        "down_projection_norms_sha256": _sha256(weight_path),
        "resumed_samples": resumed_samples,
        "raw_token_activations_retained": False,
    }
    validate_document(manifest, "activation_manifest.schema.json")
    manifest_path = run_dir / "activations/manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest


class _SyntheticMLP(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.down_proj = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(width))

    def forward(self, values: Tensor) -> Tensor:
        return self.down_proj(values)


class _SyntheticLayer(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = _SyntheticMLP(width)

    def forward(self, values: Tensor) -> Tensor:
        return self.mlp(values)


class _SyntheticDecoder(torch.nn.Module):
    def __init__(self, layers: int, width: int) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_SyntheticLayer(width) for _ in range(layers)])

    def forward(self, input_values: Tensor) -> Tensor:
        values = input_values
        for layer in self.layers:
            values = layer(values)
        return values


def collect_synthetic_fixture(
    run_dir: Path,
    layers: int = 2,
    d_ff: int = 4,
    run_id: str = "synthetic_cpu_fixture",
) -> dict[str, Any]:
    """Execute the production collector against deterministic CPU-only model and masks."""
    from .hooks import masks_from_spans
    from .interfaces import PreparedInputs

    samples = pd.read_parquet(run_dir / "samples/sample_manifest.parquet")
    model = _SyntheticDecoder(layers, d_ff)

    def prepare(sample: Mapping[str, Any]) -> PreparedInputs:
        stable = int(hashlib.sha256(str(sample["sample_id"]).encode()).hexdigest()[:8], 16)
        scale = 1.0 + (stable % 7) / 10
        base = torch.arange(1, 4 * d_ff + 1, dtype=torch.float32).reshape(4, d_ff)
        base = base.roll(shifts=stable % d_ff, dims=1) * scale
        roles = masks_from_spans(torch.ones(4), visual_spans=[(0, 2)], text_spans=[(2, 4)])
        return PreparedInputs(model_inputs={"input_values": base}, token_roles=roles)

    adapter = InjectedModelAdapter("synthetic", model, prepare)
    dataset_name = str(samples["dataset"].iloc[0]) if "dataset" in samples else "synthetic"
    return collect_reduced_activations(adapter, samples, run_dir, run_id, "synthetic", dataset_name)
