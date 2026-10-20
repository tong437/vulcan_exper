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

"""Resumable A1/A2 weak-neuron removal with matched frozen prompts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .ablation_runner import (
    _load_replay_rows,
    _move_generation_inputs,
    _read_completed,
    _select_spec,
    _selection_digest,
    _write_json_atomic,
    evaluation_prompt,
)
from .adapters import load_local_processor_adapter, load_model_layout
from .advisor_ablation import DownProjectionAblation, per_layer_deletion_count, score_prediction
from .control_ablation_runner import _load_statistics
from .gpu_runner import _load_image_index, _model_input_device, _move_roles, _validate_environment
from .hooks import DownProjectionCapture, discover_decoder_ffns
from .importance import compute_modality_importance, normalize_importance
from .runtime import build_run_id, resolve_artifact_root
from .weak_pruning import TARGET_PERCENTS, select_weak_indices


def _write_pooled_atomic(path: Path, sample_id: str, q_all: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez(temporary, sample_id=np.asarray(sample_id), q_all=q_all)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pooled_shares(
    model: torch.nn.Module,
    model_inputs: dict[str, Any],
    roles: Any,
    bindings: tuple[Any, ...],
    down_norms: dict[int, torch.Tensor],
    widths: tuple[int, ...],
) -> np.ndarray:
    """Normalize importance computed directly on V ∪ T ∪ C input positions."""
    pooled_mask = roles.visual | roles.text | roles.control
    if not torch.equal(pooled_mask, ~roles.padding):
        raise ValueError("Pooled token mask must equal all non-padding decoder positions.")
    captured: dict[int, np.ndarray] = {}

    def capture(layer: int, activations: torch.Tensor, weight: torch.Tensor) -> None:
        importance = compute_modality_importance(
            activations, weight, pooled_mask, down_projection_norm=down_norms[layer]
        ).importance
        normalized = normalize_importance(importance)
        if not torch.all(normalized.valid):
            raise ValueError(f"Layer {layer} has invalid pooled importance.")
        values = normalized.values
        if values.ndim == 2:
            values = values[0]
        captured[layer] = values.detach().to(device="cpu", dtype=torch.float32).numpy()

    capture_inputs = {**model_inputs, "use_cache": False, "output_attentions": False, "output_hidden_states": False}
    with torch.inference_mode(), DownProjectionCapture(bindings, capture):
        model(**capture_inputs)
    if tuple(sorted(captured)) != tuple(range(len(bindings))):
        raise RuntimeError("Not every decoder layer was captured for pooled importance.")
    q_all = np.full((len(bindings), max(widths)), np.nan, dtype=np.float32)
    for layer, width in enumerate(widths):
        if captured[layer].shape != (width,):
            raise ValueError(f"Pooled share width disagrees with layer {layer}.")
        q_all[layer, :width] = captured[layer]
    return q_all


def _load_or_compute_pooled(
    path: Path,
    sample_id: str,
    model: torch.nn.Module,
    model_inputs: dict[str, Any],
    roles: Any,
    bindings: tuple[Any, ...],
    down_norms: dict[int, torch.Tensor],
    widths: tuple[int, ...],
) -> np.ndarray:
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["sample_id"].item()) != sample_id:
                raise ValueError("Pooled statistics disagree with frozen sample order.")
            q_all = saved["q_all"].astype(np.float32, copy=True)
    else:
        q_all = _pooled_shares(model, model_inputs, roles, bindings, down_norms, widths)
        _write_pooled_atomic(path, sample_id, q_all)
    if q_all.shape != (len(widths), max(widths)):
        raise ValueError("Saved pooled-share shape disagrees with decoder widths.")
    for layer, width in enumerate(widths):
        values = q_all[layer, :width]
        if not np.isfinite(values).all() or not np.isclose(values.sum(dtype=np.float64), 1.0, atol=1e-4):
            raise ValueError(f"Layer {layer} pooled shares are invalid.")
    return q_all


def _summarize(rows: list[dict[str, Any]], expected_samples: int, widths: tuple[int, ...]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    expected_conditions = {"baseline"} | {
        f"{method}_{percent:02d}pct" for method in ("A1", "A2") for percent in TARGET_PERCENTS
    }
    if len(frame) != expected_samples * len(expected_conditions):
        raise ValueError("A1/A2 prediction matrix is incomplete.")
    counts = frame.groupby("condition").size()
    if set(counts.index) != expected_conditions or not (counts == expected_samples).all():
        raise ValueError("A1/A2 condition counts are incomplete.")
    if frame.duplicated(["sample_index", "condition"]).any():
        raise ValueError("Duplicate sample-condition predictions are present.")
    conditions = []
    for condition, group in frame.groupby("condition", sort=False):
        percent = int(group["removal_percent"].iloc[0])
        target = (
            0 if condition == "baseline" else sum(per_layer_deletion_count(width, percent / 100.0) for width in widths)
        )
        if not (group["target_neurons"] == target).all():
            raise ValueError(f"Target-neuron counts disagree for {condition}.")
        selected = group["selected_neurons"].to_numpy(dtype=np.int64)
        if np.any(selected > target):
            raise ValueError(f"Selected-neuron counts exceed the target for {condition}.")
        conditions.append(
            {
                "condition": condition,
                "method": group["method"].iloc[0],
                "removal_percent": percent,
                "score": float(group["score"].mean()),
                "actual_removal_percent": float((group["selected_neurons"] / sum(widths) * 100).mean()),
                "selected_neurons_mean": float(selected.mean()),
                "target_neurons": target,
                "shortfall_prediction_count": int(np.count_nonzero(selected < target)),
                "available_weak_neurons_mean": float(group["available_weak_neurons"].mean()),
            }
        )
    return {
        "expected_samples": expected_samples,
        "total_neurons": sum(widths),
        "target_percents": list(TARGET_PERCENTS),
        "conditions": conditions,
    }


def run_weak_pruning(
    study_path: Path,
    model_name: str,
    dataset_name: str,
    statistics_root: Path,
    output_dir: Path,
    *,
    replay_dir: Path,
    max_new_tokens: int = 32,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Execute baseline and A1/A2 × five target percentages for one run."""
    spec = _select_spec(study_path.resolve(), model_name, dataset_name)
    _validate_environment(spec)
    source_run = resolve_artifact_root(spec) / "runs" / spec.study / build_run_id(spec)
    samples = pd.read_parquet(source_run / "samples/sample_manifest.parquet")
    if max_samples is not None:
        samples = samples.iloc[:max_samples].copy()
    replay_rows = _load_replay_rows(replay_dir, dataset_name)
    image_index = _load_image_index(resolve_artifact_root(spec), dataset_name)
    frozen_manifest = json.loads(
        (statistics_root / f"{model_name}__{dataset_name}" / "manifest.json").read_text(encoding="utf-8")
    )
    if frozen_manifest["samples"] != 1000 or frozen_manifest["status"] != "complete":
        raise ValueError("A1 requires the complete frozen five-state statistics.")
    widths = tuple(int(value) for value in frozen_manifest["d_ff_by_layer"])

    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "study": str(study_path.resolve()),
        "model": model_name,
        "dataset": dataset_name,
        "statistics_root": str(statistics_root.resolve()),
        "max_new_tokens": max_new_tokens,
        "max_samples": max_samples,
        "target_percents": list(TARGET_PERCENTS),
        "input_policy": frozen_manifest["input_policy"],
        "selection": "lowest qV+qT+qC within A1 W; lowest direct pooled q within A2 W",
    }
    configuration_path = output_dir / "run_config.json"
    if configuration_path.is_file():
        if json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Existing output directory has a different A1/A2 run configuration.")
    else:
        _write_json_atomic(configuration_path, configuration)

    import transformers

    layout = load_model_layout(spec.model.values)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[str(spec.model.values["dtype"])]
    model = getattr(transformers, layout.architecture).from_pretrained(
        layout.model_path,
        local_files_only=True,
        trust_remote_code=bool(spec.model.values.get("trust_remote_code", False)),
        dtype=dtype,
        device_map=spec.model.values.get("device_map", "auto"),
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = _model_input_device(model)
    processor = load_local_processor_adapter(spec.model.values, use_fast=False)
    tokenizer = processor.processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    bindings = discover_decoder_ffns(model)
    if tuple(binding.d_ff for binding in bindings) != widths:
        raise RuntimeError("Loaded FFN widths disagree with frozen five-state statistics.")
    down_norms = {
        binding.layer: torch.linalg.vector_norm(
            binding.down_projection.weight.detach().to(torch.float32), ord=2, dim=0
        )
        for binding in bindings
    }

    prediction_path = output_dir / "predictions.jsonl"
    completed, rows = _read_completed(prediction_path)
    conditions = [("baseline", None, 0)] + [
        (f"{method}_{percent:02d}pct", method, percent) for method in ("A1", "A2") for percent in TARGET_PERCENTS
    ]
    expected = len(samples) * len(conditions)
    started = time.monotonic()
    with prediction_path.open("a", encoding="utf-8", buffering=1) as handle:
        for sample_index, sample in enumerate(samples.to_dict(orient="records")):
            sample_id = str(sample["sample_id"])
            unfinished = [item for item in conditions if (sample_index, item[0], 0) not in completed]
            if not unfinished:
                continue
            prompt = evaluation_prompt(dataset_name, sample, replay_rows.get(sample_id))
            with Image.open(image_index[str(sample["image_id"])]) as image:
                prepared = processor.prepare_inputs(
                    {"question": prompt, "image": image.convert("RGB")}, add_generation_prompt=True
                )
            model_inputs = _move_generation_inputs(prepared.model_inputs, device, dtype)
            roles = _move_roles(prepared.token_roles, device)
            input_length = int(model_inputs["input_ids"].shape[1])
            q_three, classes_three = _load_statistics(
                statistics_root, model_name, dataset_name, sample_index, sample_id
            )
            q_all = _load_or_compute_pooled(
                output_dir / "sample_pooled_statistics" / f"{sample_index:06d}.npz",
                sample_id,
                model,
                model_inputs,
                roles,
                bindings,
                down_norms,
                widths,
            )
            for condition, method, percent in unfinished:
                selected = None
                shortfall_layers = 0
                available = 0
                target = 0
                if method is not None:
                    selected, shortfall_layers, available = select_weak_indices(
                        q_three, classes_three, q_all, widths, method=method, removal_percent=percent
                    )
                    target = sum(per_layer_deletion_count(width, percent / 100.0) for width in widths)
                context = nullcontext() if selected is None else DownProjectionAblation(bindings, selected)
                with torch.inference_mode(), context:
                    generated = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=pad_token_id,
                    )
                prediction = tokenizer.decode(generated[0, input_length:].cpu(), skip_special_tokens=True).strip()
                row = {
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "condition": condition,
                    "method": method,
                    "removal_percent": percent,
                    "repeat": 0,
                    "selection_sha256": _selection_digest(selected),
                    "selected_neurons": 0 if selected is None else sum(map(len, selected.values())),
                    "target_neurons": target,
                    "available_weak_neurons": available,
                    "shortfall_layers": shortfall_layers,
                    "prediction": prediction,
                    "score": score_prediction(dataset_name, prediction, sample["answer"]),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                completed.add((sample_index, condition, 0))
            if (sample_index + 1) % 25 == 0 or len(completed) == expected:
                print(f"[A1/A2] {model_name} x {dataset_name}: {len(completed)}/{expected}", flush=True)
    summary = _summarize(rows, len(samples), widths)
    summary.update(
        {
            "model": model_name,
            "dataset": dataset_name,
            "insufficient_pool": "delete_all_available_without_cross_class_top_up",
            "elapsed_seconds_this_invocation": time.monotonic() - started,
        }
    )
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary
