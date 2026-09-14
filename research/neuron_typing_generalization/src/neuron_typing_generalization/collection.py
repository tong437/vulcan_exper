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

import pandas as pd
import torch
from torch import Tensor

from .hooks import DownProjectionCapture, InjectedModelAdapter, StreamingImportanceCollector
from .schema_validation import validate_document, validate_frame


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
    """Run one injected model and persist only reduced RMS/weight statistics."""
    if samples.empty or "sample_id" not in samples:
        raise ValueError("A nonempty sample manifest with sample_id is required.")
    bindings = tuple(adapter.iter_ffn_bindings())
    d_ff_values = {binding.d_ff for binding in bindings}
    if len(d_ff_values) != 1:
        raise ValueError("This score schema requires a constant d_ff across decoder layers.")
    rows = []
    for sample in samples.to_dict(orient="records"):
        prepared = adapter.prepare_inputs(sample)
        collector = StreamingImportanceCollector(prepared.token_roles)
        with torch.inference_mode(), DownProjectionCapture(bindings, collector.update):
            adapter.model(**prepared.model_inputs)
        captured = collector.finalize()
        if tuple(captured) != tuple(binding.layer for binding in bindings):
            raise RuntimeError("Not every discovered decoder layer was captured exactly once.")
        for layer, statistics in captured.items():
            for modality, values in (("visual", statistics.visual), ("text", statistics.text)):
                rms = _single_sample_vector(values.activation_rms, "activation_rms").to(torch.float32).cpu()
                norm = values.down_projection_norm.to(torch.float32).cpu()
                valid_tensor = values.valid
                valid = bool(valid_tensor.item()) if valid_tensor.ndim == 0 else bool(valid_tensor[0].item())
                count_tensor = values.token_count
                token_count = int(count_tensor.item()) if count_tensor.ndim == 0 else int(count_tensor[0].item())
                for neuron in range(len(rms)):
                    rows.append(
                        {
                            "run_id": run_id,
                            "sample_id": str(sample["sample_id"]),
                            "model": model_name,
                            "dataset": dataset_name,
                            "layer": layer,
                            "neuron": neuron,
                            "modality": modality,
                            "token_count": token_count,
                            "activation_rms": float(rms[neuron]) if valid else None,
                            "down_projection_norm": float(norm[neuron]),
                            "valid": valid,
                        }
                    )
    frame = pd.DataFrame(rows, columns=ACTIVATION_COLUMNS)
    validate_frame(frame, "activation_stats.schema.json")
    output = run_dir / "activations/neuron_activation_stats.parquet"
    _write_parquet_atomic(frame, output)
    manifest = {
        "run_id": run_id,
        "samples": len(samples),
        "layers": len(bindings),
        "d_ff": next(iter(d_ff_values)),
        "rows": len(frame),
        "statistics_file": str(output.relative_to(run_dir)),
        "statistics_sha256": _sha256(output),
        "raw_token_activations_retained": False,
    }
    validate_document(manifest, "activation_manifest.schema.json")
    manifest_path = run_dir / "activations/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
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
