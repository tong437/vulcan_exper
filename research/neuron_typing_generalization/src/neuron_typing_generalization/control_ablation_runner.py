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

"""Resumable W/C generation ablation using frozen three-role response shares."""

from __future__ import annotations

import json
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
    _stable_seed,
    _write_json_atomic,
    evaluation_prompt,
    summarize_ablation_rows,
)
from .adapters import load_local_processor_adapter, load_model_layout
from .advisor_ablation import (
    AblationCondition,
    DownProjectionAblation,
    per_layer_deletion_count,
    score_prediction,
)
from .gpu_runner import _load_image_index, _model_input_device, _validate_environment
from .hooks import discover_decoder_ffns
from .runtime import build_run_id, resolve_artifact_root


CONTROL_CLASS_IDS = {"C": 3, "W": 4}


def build_control_ablation_conditions(random_repeats: int = 10) -> tuple[AblationCondition, ...]:
    """Build baseline, W/C random/max/min, and global-random conditions."""
    if random_repeats < 2:
        raise ValueError("random_repeats must be at least two.")
    conditions = [AblationCondition("baseline", None, "baseline", 0, False)]
    for state in ("W", "C"):
        conditions.extend(
            AblationCondition(f"{state}_random", state, "random", repeat, True) for repeat in range(random_repeats)
        )
    for method in ("max", "min"):
        conditions.extend(AblationCondition(f"{state}_{method}", state, method, 0, False) for state in ("W", "C"))
    conditions.extend(
        AblationCondition("global_random", None, "random", repeat, True) for repeat in range(random_repeats)
    )
    return tuple(conditions)


def select_control_indices(
    q: np.ndarray,
    classes: np.ndarray,
    widths: tuple[int, ...],
    *,
    fraction: float,
    state: str | None,
    method: str,
    seed: int,
) -> tuple[dict[int, np.ndarray], int]:
    """Select 1% per layer, deleting the whole state pool when it is smaller."""
    if q.ndim != 3 or q.shape[1] != 3 or classes.shape != (q.shape[0], q.shape[2]):
        raise ValueError("Frozen q/classes shapes are invalid.")
    if state is not None and state not in CONTROL_CLASS_IDS:
        raise ValueError(f"Unsupported state: {state}")
    if method not in {"random", "max", "min"}:
        raise ValueError(f"Unsupported method: {method}")
    generator = np.random.default_rng(seed)
    selected = {}
    shortfall_layers = 0
    for layer, width in enumerate(widths):
        target = per_layer_deletion_count(width, fraction)
        candidates = (
            np.arange(width, dtype=np.int64)
            if state is None
            else np.flatnonzero(classes[layer, :width] == CONTROL_CLASS_IDS[state]).astype(np.int64)
        )
        count = min(target, len(candidates))
        shortfall_layers += int(count < target)
        if method == "random":
            indices = np.sort(generator.choice(candidates, size=count, replace=False).astype(np.int64))
        else:
            strength = q[layer, 2, :width] if state == "C" else q[layer, :, :width].sum(axis=0)
            candidate_strength = strength[candidates]
            if not np.isfinite(candidate_strength).all():
                raise ValueError(f"Layer {layer} contains non-finite response strengths.")
            primary = -candidate_strength if method == "max" else candidate_strength
            order = np.lexsort((candidates, primary))
            indices = np.sort(candidates[order[:count]])
        selected[layer] = indices
    return selected, shortfall_layers


def _load_statistics(
    root: Path, model: str, dataset: str, index: int, sample_id: str
) -> tuple[np.ndarray, np.ndarray]:
    path = root / f"{model}__{dataset}" / "sample_statistics" / f"{index:06d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["sample_id"].item()) != sample_id:
            raise ValueError("Frozen control statistics disagree with sample order.")
        return saved["q"].astype(np.float32, copy=True), saved["classes"].astype(np.int8, copy=True)


def run_control_ablation(
    study_path: Path,
    model_name: str,
    dataset_name: str,
    statistics_root: Path,
    output_dir: Path,
    *,
    replay_dir: Path,
    removal_fraction: float = 0.01,
    random_repeats: int = 10,
    max_new_tokens: int = 32,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the complete generation matrix for one model-dataset run."""
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
    widths = tuple(int(value) for value in frozen_manifest["d_ff_by_layer"])
    target_neurons = sum(per_layer_deletion_count(width, removal_fraction) for width in widths)

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
        raise RuntimeError("Loaded FFN widths disagree with frozen control statistics.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    completed, rows = _read_completed(prediction_path)
    conditions = build_control_ablation_conditions(random_repeats)
    expected = len(samples) * len(conditions)
    started = time.monotonic()
    with prediction_path.open("a", encoding="utf-8", buffering=1) as handle:
        for sample_index, sample in enumerate(samples.to_dict(orient="records")):
            sample_id = str(sample["sample_id"])
            unfinished = [
                condition
                for condition in conditions
                if (sample_index, condition.condition, condition.repeat) not in completed
            ]
            if not unfinished:
                continue
            replay_row = replay_rows.get(sample_id)
            prompt = evaluation_prompt(dataset_name, sample, replay_row)
            with Image.open(image_index[str(sample["image_id"])]) as image:
                prepared = processor.prepare_inputs(
                    {"question": prompt, "image": image.convert("RGB")}, add_generation_prompt=True
                )
            model_inputs = _move_generation_inputs(prepared.model_inputs, device, dtype)
            input_length = int(model_inputs["input_ids"].shape[1])
            q, classes = _load_statistics(statistics_root, model_name, dataset_name, sample_index, sample_id)
            for condition in unfinished:
                selected = None
                selection_seed = None
                shortfall_layers = 0
                if condition.method != "baseline":
                    selection_seed = _stable_seed(seed, sample_id, condition)
                    selected, shortfall_layers = select_control_indices(
                        q,
                        classes,
                        widths,
                        fraction=removal_fraction,
                        state=condition.state,
                        method=condition.method,
                        seed=selection_seed,
                    )
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
                    "condition": condition.condition,
                    "state": condition.state,
                    "method": condition.method,
                    "repeat": condition.repeat,
                    "random": condition.random,
                    "selection_seed": selection_seed,
                    "selection_sha256": _selection_digest(selected),
                    "selected_neurons": 0 if selected is None else sum(map(len, selected.values())),
                    "target_neurons": 0 if selected is None else target_neurons,
                    "shortfall_layers": shortfall_layers,
                    "prediction": prediction,
                    "score": score_prediction(dataset_name, prediction, sample["answer"]),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                completed.add((sample_index, condition.condition, condition.repeat))
            print(f"[W/C ablation] {model_name} x {dataset_name}: {len(completed)}/{expected}", flush=True)
    summary = summarize_ablation_rows(rows, len(samples), removal_fraction, random_repeats)
    summary.update(
        {
            "model": model_name,
            "dataset": dataset_name,
            "statistics_root": str(statistics_root),
            "insufficient_pool": "delete_all_available_without_cross-class_top_up",
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary
