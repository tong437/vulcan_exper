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

"""Resumable GPU generation for the advisor's response-state ablation study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .adapters import load_local_processor_adapter, load_model_layout
from .advisor_ablation import (
    AblationCondition,
    DownProjectionAblation,
    build_ablation_conditions,
    per_layer_deletion_count,
    score_prediction,
    select_ablation_indices,
)
from .advisor_prompts import mmbench_selection_prompt
from .config import ExperimentSpec, expand_study, load_study
from .gpu_runner import _load_image_index, _model_input_device, _validate_environment
from .hooks import discover_decoder_ffns
from .runtime import build_run_id, resolve_artifact_root


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


def _select_spec(study_path: Path, model_name: str, dataset_name: str) -> ExperimentSpec:
    matches = [
        spec
        for spec in expand_study(load_study(study_path))
        if spec.model.name == model_name and spec.dataset.name == dataset_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one model-dataset spec, found {len(matches)}.")
    return matches[0]


def _load_replay_rows(replay_dir: Path | None, dataset: str) -> dict[str, dict[str, Any]]:
    if replay_dir is None:
        return {}
    path = replay_dir / f"{dataset}_1000.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen replay rows are missing: {path}")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def evaluation_prompt(dataset: str, sample: Mapping[str, Any], replay_row: Mapping[str, Any] | None) -> str:
    """Build a concise metric-aligned generation prompt without changing sample identity."""
    question = str(sample["question"]).strip()
    if dataset == "mmbench":
        return mmbench_selection_prompt(sample, replay_row) + "\n\nAnswer with only the option letter."
    if dataset == "mme":
        return question + "\nAnswer only yes or no."
    if dataset in {"textvqa", "chartqa"}:
        return question + "\nAnswer with only the short answer."
    raise ValueError(f"Unsupported ablation dataset: {dataset}")


def _stable_seed(base_seed: int, sample_id: str, condition: AblationCondition) -> int:
    payload = f"{base_seed}|{sample_id}|{condition.condition}|{condition.repeat}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _selection_digest(selected: Mapping[int, np.ndarray] | None) -> str | None:
    if selected is None:
        return None
    digest = hashlib.sha256()
    for layer in sorted(selected):
        digest.update(f"{layer}:".encode())
        digest.update(np.asarray(selected[layer], dtype="<i8").tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _move_generation_inputs(values: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            target_dtype = dtype if value.is_floating_point() else value.dtype
            moved[key] = value.to(device=device, dtype=target_dtype, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _load_frozen_sample_statistics(
    run_dir: Path, sample_index: int, sample_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load the exact q values and classes frozen by the preceding response-state experiment."""
    score_path = run_dir / "scores/sample_parts" / f"{sample_index:06d}.npz"
    class_path = run_dir / "classifications/sample_parts" / f"{sample_index:06d}.npz"
    if not score_path.is_file() or not class_path.is_file():
        raise FileNotFoundError("Ablation requires completed importance and classification sample parts.")
    with np.load(score_path, allow_pickle=False) as arrays:
        if str(arrays["sample_id"].item()) != sample_id:
            raise ValueError("Importance partition disagrees with the frozen sample order.")
        q = arrays["q"].astype(np.float32, copy=True)
    with np.load(class_path, allow_pickle=False) as arrays:
        if str(arrays["sample_id"].item()) != sample_id:
            raise ValueError("Classification partition disagrees with the frozen sample order.")
        classes = arrays["classes"].astype(np.int8, copy=True)
    return q, classes


def _read_completed(path: Path) -> tuple[set[tuple[int, str, int]], list[dict[str, Any]]]:
    if not path.is_file():
        return set(), []
    completed = set()
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid ablation JSONL at line {line_number}: {path}") from error
            key = (int(row["sample_index"]), str(row["condition"]), int(row["repeat"]))
            if key in completed:
                raise ValueError(f"Duplicate ablation result: {key}")
            completed.add(key)
            rows.append(row)
    return completed, rows


def summarize_ablation_rows(
    rows: list[dict[str, Any]], expected_samples: int, removal_fraction: float, random_repeats: int
) -> dict[str, Any]:
    """Aggregate example scores within runs, then random-run means and sample SDs."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No ablation rows are available to summarize.")
    run_scores = (
        frame.groupby(["condition", "state", "method", "repeat"], dropna=False, sort=False)["score"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if (run_scores["count"] != expected_samples).any():
        incomplete = run_scores.loc[run_scores["count"] != expected_samples, ["condition", "repeat", "count"]]
        raise ValueError(f"Ablation results are incomplete: {incomplete.to_dict(orient='records')[:5]}")
    summaries = []
    for condition, group in run_scores.groupby("condition", sort=False):
        values = group["mean"].to_numpy(dtype=np.float64)
        condition_rows = frame.loc[frame["condition"] == condition]
        expected_repeats = random_repeats if bool(condition_rows["random"].iloc[0]) else 1
        if len(values) != expected_repeats:
            raise ValueError(f"Condition {condition} has {len(values)} runs; expected {expected_repeats}.")
        item = {
            "condition": condition,
            "state": None if pd.isna(group["state"].iloc[0]) else str(group["state"].iloc[0]),
            "method": str(group["method"].iloc[0]),
            "runs": len(values),
            "score_mean": float(values.mean()),
            "score_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
        if {"selected_neurons", "target_neurons"}.issubset(condition_rows.columns):
            actual = condition_rows["selected_neurons"].to_numpy(dtype=np.int64)
            target = condition_rows["target_neurons"].to_numpy(dtype=np.int64)
            item.update(
                {
                    "selected_neurons_mean": float(actual.mean()),
                    "selected_neurons_min": int(actual.min()),
                    "selected_neurons_max": int(actual.max()),
                    "target_neurons": int(target[0]),
                    "shortfall_prediction_count": int(np.count_nonzero(actual < target)),
                }
            )
        summaries.append(item)
    baseline = next(item for item in summaries if item["condition"] == "baseline")["score_mean"]
    for item in summaries:
        item["drop_mean"] = float(baseline - item["score_mean"])
        item["drop_std"] = item["score_std"]
    return {
        "expected_samples": expected_samples,
        "removal_fraction": removal_fraction,
        "random_repeats": random_repeats,
        "baseline_score": baseline,
        "conditions": summaries,
    }


def run_ablation_generation(
    study_path: Path,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
    *,
    replay_dir: Path | None,
    removal_fraction: float = 0.01,
    random_repeats: int = 10,
    max_new_tokens: int = 32,
    max_samples: int | None = None,
    seed: int = 42,
    insufficient_pool: str = "error",
) -> dict[str, Any]:
    """Load one model once and execute the complete resumable ablation matrix sample by sample."""
    spec = _select_spec(study_path.resolve(), model_name, dataset_name)
    _validate_environment(spec)
    run_dir = resolve_artifact_root(spec) / "runs" / spec.study / build_run_id(spec)
    samples = pd.read_parquet(run_dir / "samples/sample_manifest.parquet")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        samples = samples.iloc[:max_samples].copy()
    replay_rows = _load_replay_rows(replay_dir, dataset_name)
    image_index = _load_image_index(resolve_artifact_root(spec), dataset_name)

    score_manifest = json.loads((run_dir / "scores/manifest.json").read_text(encoding="utf-8"))
    class_manifest = json.loads((run_dir / "classifications/manifest.json").read_text(encoding="utf-8"))
    if score_manifest["parts_sha256"] != class_manifest["source_score_parts_sha256"]:
        raise RuntimeError("Classifications were not derived from the current frozen importance partitions.")
    d_ff_by_layer = tuple(int(value) for value in score_manifest["d_ff_by_layer"])
    target_neurons = sum(per_layer_deletion_count(width, removal_fraction) for width in d_ff_by_layer)

    import transformers

    layout = load_model_layout(spec.model.values)
    dtype_name = str(spec.model.values.get("dtype", "bfloat16"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported inference dtype: {dtype_name}")
    model_class = getattr(transformers, layout.architecture)
    model = model_class.from_pretrained(
        layout.model_path,
        local_files_only=True,
        trust_remote_code=bool(spec.model.values.get("trust_remote_code", False)),
        dtype=dtype,
        device_map=spec.model.values.get("device_map", "auto"),
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = _model_input_device(model)
    processor_adapter = load_local_processor_adapter(spec.model.values, use_fast=False)
    tokenizer = processor_adapter.processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise RuntimeError("Tokenizer defines neither pad_token_id nor eos_token_id.")
    bindings = discover_decoder_ffns(model)
    observed_widths = tuple(binding.d_ff for binding in bindings)
    if observed_widths != d_ff_by_layer:
        raise RuntimeError("Loaded FFN widths disagree with the frozen importance statistics.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    completed, rows = _read_completed(prediction_path)
    conditions = build_ablation_conditions(random_repeats)
    expected_results = len(samples) * len(conditions)
    started = time.monotonic()
    with prediction_path.open("a", encoding="utf-8", buffering=1) as output_handle:
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
            prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
            image_path = image_index.get(str(sample["image_id"]))
            if image_path is None:
                raise FileNotFoundError(f"No materialized image for sample {sample_id}.")
            selection_prompt_sha256 = (
                hashlib.sha256(mmbench_selection_prompt(sample, replay_row).encode()).hexdigest()
                if dataset_name == "mmbench"
                else None
            )
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                prepared = processor_adapter.prepare_inputs(
                    {"question": prompt, "image": rgb_image}, add_generation_prompt=True
                )
            model_inputs = _move_generation_inputs(prepared.model_inputs, device, dtype)
            input_length = int(model_inputs["input_ids"].shape[1])
            q, classes = _load_frozen_sample_statistics(run_dir, sample_index, sample_id)
            for condition in unfinished:
                selected = None
                selection_seed = None
                if condition.method != "baseline":
                    selection_seed = _stable_seed(seed, sample_id, condition)
                    selected = select_ablation_indices(
                        q,
                        classes,
                        d_ff_by_layer,
                        fraction=removal_fraction,
                        state=condition.state,
                        method=condition.method,
                        seed=selection_seed,
                        insufficient_pool=insufficient_pool,
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
                continuation = generated[0, input_length:]
                prediction = processor_adapter.processor.tokenizer.decode(
                    continuation.detach().cpu(), skip_special_tokens=True
                ).strip()
                score = score_prediction(dataset_name, prediction, sample["answer"])
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
                    "selected_neurons": 0 if selected is None else sum(len(indices) for indices in selected.values()),
                    "target_neurons": 0 if selected is None else target_neurons,
                    "selection_prompt_sha256": selection_prompt_sha256,
                    "prompt_sha256": prompt_sha256,
                    "prediction": prediction,
                    "score": score,
                }
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                completed.add((sample_index, condition.condition, condition.repeat))
            print(f"[ablation] {model_name} x {dataset_name}: {len(completed)}/{expected_results}", flush=True)

    summary = summarize_ablation_rows(rows, len(samples), removal_fraction, random_repeats)
    summary.update(
        {
            "model": model_name,
            "dataset": dataset_name,
            "run_dir": str(run_dir),
            "predictions": str(prediction_path),
            "selection_profile": "frozen_previous_response_state_experiment",
            "insufficient_pool": insufficient_pool,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    summary_path = output_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    return {"summary": str(summary_path), "predictions": str(prediction_path), **summary}
