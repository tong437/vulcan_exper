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

"""Collect resumable five-way V/T/M/C/W counts for one model across datasets."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch
from neuron_typing_generalization.ablation_runner import evaluation_prompt
from neuron_typing_generalization.adapters import load_local_processor_adapter, load_model_layout
from neuron_typing_generalization.advisor_control_states import classify_control_shares, validate_state_counts
from neuron_typing_generalization.config import expand_study, load_study
from neuron_typing_generalization.gpu_runner import (
    _load_image_index,
    _load_replay_rows,
    _model_input_device,
    _move_model_inputs,
    _move_roles,
    _validate_environment,
)
from neuron_typing_generalization.hooks import DownProjectionCapture, discover_decoder_ffns
from neuron_typing_generalization.importance import compute_modality_importance, normalize_importance
from neuron_typing_generalization.runtime import build_run_id, resolve_artifact_root
from PIL import Image


DATASETS = ("mme", "mmbench", "textvqa", "chartqa")


def _write_npz_atomic(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, **values)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_dataset(
    spec: Any,
    model: torch.nn.Module,
    processor_adapter: Any,
    bindings: tuple[Any, ...],
    down_norms: dict[int, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    replay_dir: Path,
    output_root: Path,
    checkpoint_every: int,
    save_sample_statistics: bool,
) -> dict[str, Any]:
    run_id = build_run_id(spec)
    source_run = resolve_artifact_root(spec) / "runs" / spec.study / run_id
    samples = pd.read_parquet(source_run / "samples/sample_manifest.parquet")
    expected = int(spec.study_values["sampling"]["expected_samples"])
    if len(samples) != expected:
        raise ValueError(f"{run_id} has {len(samples)} frozen samples; expected {expected}.")
    image_index = _load_image_index(resolve_artifact_root(spec), spec.dataset.name)
    replay_rows = _load_replay_rows(replay_dir, spec.dataset.name)
    output_dir = output_root / f"{spec.model.name}__{spec.dataset.name}"
    checkpoint_path = output_dir / "checkpoint.npz"
    statistics_dir = output_dir / "sample_statistics"
    if save_sample_statistics:
        statistics_dir.mkdir(parents=True, exist_ok=True)
    widths = [binding.d_ff for binding in bindings]
    counts = np.zeros((5, len(bindings), max(widths)), dtype=np.int32)
    control_token_total = 0
    control_token_min = np.iinfo(np.int32).max
    control_token_max = 0
    completed = 0
    if checkpoint_path.is_file():
        with np.load(checkpoint_path, allow_pickle=False) as saved:
            counts = saved["counts"].astype(np.int32, copy=True)
            completed = int(saved["completed_samples"].item())
            control_token_total = int(saved["control_token_total"].item())
            control_token_min = int(saved["control_token_min"].item())
            control_token_max = int(saved["control_token_max"].item())
        validate_state_counts(counts, completed, widths)

    started = time.monotonic()
    for sample_index in range(completed, len(samples)):
        sample = samples.iloc[sample_index].to_dict()
        question = evaluation_prompt(
            spec.dataset.name,
            sample,
            replay_rows.get(str(sample["sample_id"])),
        )
        image_path = image_index[str(sample["image_id"])]
        with Image.open(image_path) as image:
            prepared = processor_adapter.prepare_inputs(
                {"question": question, "image": image.convert("RGB")}, add_generation_prompt=True
            )
        model_inputs = _move_model_inputs(prepared.model_inputs, device, dtype)
        roles = _move_roles(prepared.token_roles, device)
        token_count = int(roles.control.sum().item())
        if token_count <= 0:
            raise RuntimeError(f"Empty control mask for sample {sample['sample_id']}.")
        captured: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def capture(layer: int, activations: torch.Tensor, weight: torch.Tensor) -> None:
            q_values = []
            for mask in (roles.visual, roles.text, roles.control):
                importance = compute_modality_importance(
                    activations, weight, mask, down_projection_norm=down_norms[layer]
                ).importance
                normalized = normalize_importance(importance).values
                if normalized.ndim == 2:
                    normalized = normalized[0]
                q_values.append(normalized.detach().to(device="cpu", dtype=torch.float32).numpy())
            captured[layer] = (np.stack(q_values), classify_control_shares(*q_values))

        with torch.inference_mode(), DownProjectionCapture(bindings, capture):
            model(**model_inputs)
        if tuple(sorted(captured)) != tuple(range(len(bindings))):
            raise RuntimeError("Not every decoder layer was captured exactly once.")
        sample_q = np.full((len(bindings), 3, max(widths)), np.nan, dtype=np.float32)
        sample_classes = np.full((len(bindings), max(widths)), -1, dtype=np.int8)
        for layer, (q_values, classes) in captured.items():
            counts[:, layer, : widths[layer]] += np.eye(5, dtype=np.int32)[classes].T
            sample_q[layer, :, : widths[layer]] = q_values
            sample_classes[layer, : widths[layer]] = classes
        if save_sample_statistics:
            _write_npz_atomic(
                statistics_dir / f"{sample_index:06d}.npz",
                sample_id=np.asarray(str(sample["sample_id"])),
                q=sample_q,
                classes=sample_classes,
            )
        completed = sample_index + 1
        control_token_total += token_count
        control_token_min = min(control_token_min, token_count)
        control_token_max = max(control_token_max, token_count)
        if completed % checkpoint_every == 0 or completed == len(samples):
            validate_state_counts(counts, completed, widths)
            _write_npz_atomic(
                checkpoint_path,
                counts=counts,
                completed_samples=np.asarray(completed, dtype=np.int32),
                control_token_total=np.asarray(control_token_total, dtype=np.int64),
                control_token_min=np.asarray(control_token_min, dtype=np.int32),
                control_token_max=np.asarray(control_token_max, dtype=np.int32),
            )
            print(
                json.dumps({"model": spec.model.name, "dataset": spec.dataset.name, "completed": completed}),
                flush=True,
            )

    manifest = {
        "status": "complete",
        "model": spec.model.name,
        "dataset": spec.dataset.name,
        "run_id": run_id,
        "samples": completed,
        "layers": len(bindings),
        "d_ff_by_layer": widths,
        "state_order": ["V", "T", "M", "C", "W"],
        "threshold": "strictly greater than 1 / d_ff(layer)",
        "control_mask": "non-visual, non-question-text, non-padding decoder input positions",
        "input_policy": "advisor ablation evaluation_prompt with add_generation_prompt=True",
        "control_token_count": {
            "minimum": control_token_min,
            "maximum": control_token_max,
            "mean": control_token_total / completed,
        },
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "checkpoint": str(checkpoint_path),
        "sample_statistics": str(statistics_dir) if save_sample_statistics else None,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--save-sample-statistics", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")
    specs = [
        spec
        for spec in expand_study(load_study(args.study.resolve()))
        if spec.model.name == args.model and spec.dataset.name in args.datasets
    ]
    specs.sort(key=lambda spec: args.datasets.index(spec.dataset.name))
    if len(specs) != len(args.datasets):
        raise ValueError(f"Expected {len(args.datasets)} specs, found {len(specs)}.")
    _validate_environment(specs[0])
    import transformers

    layout = load_model_layout(specs[0].model.values)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[str(specs[0].model.values["dtype"])]
    model = getattr(transformers, layout.architecture).from_pretrained(
        layout.model_path,
        local_files_only=True,
        trust_remote_code=bool(specs[0].model.values.get("trust_remote_code", False)),
        dtype=dtype,
        device_map=specs[0].model.values.get("device_map", "auto"),
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = _model_input_device(model)
    processor = load_local_processor_adapter(specs[0].model.values, use_fast=False)
    bindings = discover_decoder_ffns(model)
    down_norms = {
        binding.layer: torch.linalg.vector_norm(
            binding.down_projection.weight.detach().to(torch.float32), ord=2, dim=0
        )
        for binding in bindings
    }
    results = [
        _run_dataset(
            spec,
            model,
            processor,
            bindings,
            down_norms,
            device,
            dtype,
            args.replay_dir,
            args.output_root,
            args.checkpoint_every,
            args.save_sample_statistics,
        )
        for spec in specs
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
