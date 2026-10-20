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

"""Resumable sample-adaptive A1/A2 generation ablation on frozen V/T/C scores."""

from __future__ import annotations

import hashlib
import json
import time
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
from .adaptive_pruning import adaptive_selections, nonuniform_selection, single_layer_selections
from .advisor_ablation import DownProjectionAblation, score_prediction
from .control_ablation_runner import _load_statistics
from .gpu_runner import _load_image_index, _model_input_device, _validate_environment
from .hooks import discover_decoder_ffns
from .runtime import build_run_id, resolve_artifact_root


def _baseline_rows(
    baseline_dir: Path,
    model: str,
    dataset: str,
    input_policy: str,
    max_new_tokens: int,
    statistics_root: Path,
    study_path: Path,
) -> dict[int, dict[str, Any]]:
    """Reuse only the identical sixth-experiment clean-input generation setting."""
    config = json.loads((baseline_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
    expected = {
        "model": model,
        "dataset": dataset,
        "input_policy": input_policy,
        "max_new_tokens": max_new_tokens,
        "max_samples": None,
        "statistics_root": str(statistics_root.resolve()),
        "study": str(study_path.resolve()),
    }
    if any(config.get(key) != value for key, value in expected.items()) or summary.get("expected_samples") != 1000:
        raise ValueError("Sixth-experiment baseline has a different input, decoding setting, or coverage.")
    _, rows = _read_completed(baseline_dir / "predictions.jsonl")
    baseline_rows = [row for row in rows if row["condition"] == "baseline"]
    baseline = {int(row["sample_index"]): row for row in baseline_rows}
    if len(baseline_rows) != 1000 or len(baseline) != 1000 or set(baseline) != set(range(1000)):
        raise ValueError("Sixth-experiment baseline is incomplete or duplicated.")
    return baseline


def _group_id(sample: dict[str, Any]) -> str:
    pair = sample.get("pair_group_id")
    if pair is not None and not pd.isna(pair) and str(pair):
        return str(pair)
    return str(sample["image_id"])


def run_adaptive_pruning(
    study_path: Path,
    model_name: str,
    dataset_name: str,
    statistics_root: Path,
    baseline_root: Path,
    output_dir: Path,
    *,
    replay_dir: Path,
    epsilon: float = 0.05,
    max_new_tokens: int = 32,
    max_samples: int | None = None,
    methods: tuple[str, ...] = ("A1", "A2"),
    verify_clean: bool = False,
    uniform_source: Path | None = None,
    layer_scan: bool = False,
    scan_smoke: bool = False,
) -> dict[str, Any]:
    """Generate A1/A2 outputs and pair them with validated prior clean outputs."""
    if not methods or len(set(methods)) != len(methods) or set(methods) - {"A1", "A2", "N", "N_match"}:
        raise ValueError("methods must be a nonempty, unique subset of ('A1', 'A2', 'N', 'N_match').")
    conditions = ("baseline", *methods)
    spec = _select_spec(study_path.resolve(), model_name, dataset_name)
    _validate_environment(spec)
    source_run = resolve_artifact_root(spec) / "runs" / spec.study / build_run_id(spec)
    samples = pd.read_parquet(source_run / "samples/sample_manifest.parquet")
    if len(samples) != 1000:
        raise ValueError("Adaptive pruning requires the complete frozen 1,000-sample manifest.")
    split_manifest = None
    if layer_scan:
        groups = {}
        for idx, row in enumerate(samples.to_dict(orient="records")):
            groups.setdefault(_group_id(row), []).append(idx)
        ordered = sorted(
            groups, key=lambda g: hashlib.sha256(f"layer-calibration-20261013:{dataset_name}:{g}".encode()).hexdigest()
        )
        selected_indices = []
        for group in ordered:
            selected_indices.extend(groups[group])
            if len(selected_indices) >= 200:
                break
        selected_indices.sort()
        split_manifest = {
            "seed": 20261013,
            "calibration_indices": selected_indices,
            "heldout_indices": sorted(set(range(1000)) - set(selected_indices)),
            "calibration_sample_ids": samples.iloc[selected_indices].sample_id.astype(str).tolist(),
            "calibration_group_ids": [_group_id(r) for r in samples.iloc[selected_indices].to_dict(orient="records")],
            "policy": "whole-group hashed split of previously observed frozen 1000 samples",
        }
        samples = samples.iloc[selected_indices].copy()
    if max_samples is not None:
        if not 1 <= max_samples <= 1000:
            raise ValueError("max_samples must lie in [1, 1000].")
        samples = samples.iloc[:max_samples].copy()
    replay_rows = _load_replay_rows(replay_dir, dataset_name)
    image_index = _load_image_index(resolve_artifact_root(spec), dataset_name)
    frozen_manifest = json.loads(
        (statistics_root / f"{model_name}__{dataset_name}" / "manifest.json").read_text(encoding="utf-8")
    )
    if frozen_manifest["samples"] != 1000 or frozen_manifest["status"] != "complete":
        raise ValueError("Adaptive pruning requires complete frozen qV/qT/qC scores.")
    widths = tuple(int(value) for value in frozen_manifest["d_ff_by_layer"])
    if layer_scan:
        scan_layers = (0,) if scan_smoke else tuple(range(len(widths)))
        scan_rates = (0.0, 0.1, 0.3, 0.5) if scan_smoke else (0.1, 0.3, 0.5)
        methods = tuple(f"L{l:02d}_r{r:.2f}" for l in scan_layers for r in scan_rates)
        conditions = ("baseline", *methods)
    baseline_dir = baseline_root / f"{model_name}__{dataset_name}"
    baseline = _baseline_rows(
        baseline_dir,
        model_name,
        dataset_name,
        frozen_manifest["input_policy"],
        max_new_tokens,
        statistics_root,
        study_path,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "study": str(study_path.resolve()),
        "model": model_name,
        "dataset": dataset_name,
        "statistics_root": str(statistics_root.resolve()),
        "baseline_source": str(baseline_dir.resolve()),
        "source_run_id": frozen_manifest["run_id"],
        "input_policy": frozen_manifest["input_policy"],
        "epsilon": epsilon,
        "max_new_tokens": max_new_tokens,
        "max_samples": max_samples,
        "selection": "all neurons by qV+qT+qC; A1 exact knee, A2 equal-layer mean score budget",
    }
    if methods != ("A1", "A2"):
        configuration["methods"] = list(methods)
        configuration["selection"] = "all neurons by qV+qT+qC; A2 equal-layer mean score budget"
    if "N" in methods:
        configuration["selection"] = "N v1: global stable ascending s/layer_sum/L; maximum count within mean budget"
        configuration["d_ff_by_layer"] = list(widths)
        configuration["verify_clean"] = verify_clean
    uniform_rows = {}
    if "N_match" in methods:
        if uniform_source is None:
            raise ValueError("N_match requires a frozen uniform_source directory.")
        source_config = json.loads((uniform_source / "run_config.json").read_text())
        source_summary = json.loads((uniform_source / "summary.json").read_text())
        for key in (
            "study",
            "model",
            "dataset",
            "statistics_root",
            "baseline_source",
            "source_run_id",
            "input_policy",
            "epsilon",
            "max_new_tokens",
        ):
            if source_config.get(key) != configuration[key]:
                raise ValueError(f"Uniform reference differs on {key}.")
        if source_summary["expected_samples"] != 1000 or source_config.get("max_samples") is not None:
            raise ValueError("Uniform reference must cover all frozen samples.")
        source_bytes = (uniform_source / "predictions.jsonl").read_bytes()
        source_rows = [json.loads(line) for line in source_bytes.splitlines()]
        selected_rows = [r for r in source_rows if r["condition"] == "A2"]
        uniform_rows = {int(r["sample_index"]): r for r in selected_rows}
        clean_rows = [r for r in source_rows if r["condition"] == "baseline"]
        if {int(r["sample_index"]) for r in clean_rows} != set(range(1000)):
            raise ValueError("Uniform clean baseline has invalid or duplicated sample indices.")
        if len(selected_rows) != 1000 or set(uniform_rows) != set(range(1000)) or len(clean_rows) != 1000:
            raise ValueError("Uniform reference is incomplete or duplicated.")
        for row in clean_rows:
            clean = baseline[int(row["sample_index"])]
            if any(row[k] != clean[k] for k in ("sample_id", "prediction", "score")):
                raise ValueError("Uniform reference baseline differs from clean baseline.")
        configuration.update(
            selection="N_match v1: frozen per-sample U count; minimum equal-layer response cost",
            uniform_source=str(uniform_source.resolve()),
            uniform_predictions_sha256=hashlib.sha256(source_bytes).hexdigest(),
            d_ff_by_layer=list(widths),
            verify_clean=verify_clean,
        )
    config_path = output_dir / "run_config.json"
    if layer_scan:
        configuration.update(
            selection="single-layer v1: lowest s within target layer; other layers intact",
            scan_layers=list(scan_layers),
            scan_rates=list(scan_rates),
            verify_clean=verify_clean,
            split_manifest=split_manifest,
            d_ff_by_layer=list(widths),
        )
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != configuration:
            raise ValueError("Existing output directory has a different adaptive-pruning configuration.")
    else:
        _write_json_atomic(config_path, configuration)

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
        raise RuntimeError("Loaded FFN widths disagree with frozen scores.")

    prediction_path = output_dir / "predictions.jsonl"
    completed, rows = _read_completed(prediction_path)
    expected = len(samples) * len(conditions)
    started = time.monotonic()
    with prediction_path.open("a", encoding="utf-8", buffering=1) as handle:
        for sample_index, sample in zip(samples.index.tolist(), samples.to_dict(orient="records")):
            sample_id = str(sample["sample_id"])
            group_id = _group_id(sample)
            missing = [condition for condition in conditions if (sample_index, condition, 0) not in completed]
            if not missing:
                continue
            reference = baseline[sample_index]
            if reference["sample_id"] != sample_id:
                raise ValueError("Sixth-experiment baseline sample order differs from this run.")
            if not np.isclose(
                score_prediction(dataset_name, reference["prediction"], sample["answer"]), reference["score"]
            ):
                raise ValueError("Reused baseline answer does not reproduce its recorded score.")
            if "baseline" in missing:
                row = {
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "condition": "baseline",
                    "repeat": 0,
                    "selection_sha256": None,
                    "candidate_ratio": 0.0,
                    "candidate_score_share": 0.0,
                    "actual_ratio": 0.0,
                    "actual_score_share": 0.0,
                    "selected_neurons": 0,
                    "selected_per_layer": [0] * len(widths),
                    "prediction": reference["prediction"],
                    "score": float(reference["score"]),
                    "baseline_source": str(baseline_dir / "predictions.jsonl"),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                completed.add((sample_index, "baseline", 0))
            pending_ablation = [condition for condition in missing if condition != "baseline"]
            if not pending_ablation:
                continue
            q_three, _ = _load_statistics(statistics_root, model_name, dataset_name, sample_index, sample_id)
            plans = {} if methods in (("N",), ("N_match",)) else adaptive_selections(q_three, widths, epsilon=epsilon)
            if layer_scan:
                plans = single_layer_selections(q_three, widths, scan_layers, scan_rates)
            if "N" in methods:
                plans["N"] = nonuniform_selection(q_three, widths, epsilon=epsilon)
            if "N_match" in methods:
                frozen = uniform_rows[sample_index]
                if frozen["sample_id"] != sample_id or frozen["group_id"] != group_id:
                    raise ValueError("Uniform reference sample identity mismatch.")
                uniform = adaptive_selections(q_three, widths, epsilon=epsilon)["A2"]
                if (
                    list(uniform.selected_per_layer) != frozen["selected_per_layer"]
                    or _selection_digest(uniform.selected) != frozen["selection_sha256"]
                ):
                    raise ValueError("Recomputed uniform mask differs from frozen reference.")
                plans["N_match"] = nonuniform_selection(
                    q_three, widths, epsilon=epsilon, target_count=frozen["selected_neurons"]
                )
            prompt = evaluation_prompt(dataset_name, sample, replay_rows.get(sample_id))
            with Image.open(image_index[str(sample["image_id"])]) as image:
                prepared = processor.prepare_inputs(
                    {"question": prompt, "image": image.convert("RGB")}, add_generation_prompt=True
                )
            model_inputs = _move_generation_inputs(prepared.model_inputs, device, dtype)
            input_length = int(model_inputs["input_ids"].shape[1])

            def clean_prediction():
                with torch.inference_mode():
                    result = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=pad_token_id,
                    )
                return tokenizer.decode(result[0, input_length:].cpu(), skip_special_tokens=True).strip()

            if verify_clean and clean_prediction() != reference["prediction"]:
                raise ValueError("Fresh clean generation differs from the frozen baseline.")
            for method in pending_ablation:
                plan = plans[method]
                # The context removes all hooks before the next method or sample.
                with torch.inference_mode(), DownProjectionAblation(bindings, plan.selected):
                    generated = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=pad_token_id,
                    )
                prediction = tokenizer.decode(generated[0, input_length:].cpu(), skip_special_tokens=True).strip()
                if verify_clean and clean_prediction() != reference["prediction"]:
                    raise ValueError("Clean generation after hook removal differs from baseline.")
                if verify_clean and not sum(plan.selected_per_layer) and prediction != reference["prediction"]:
                    raise ValueError("Zero-neuron ablation differs from baseline.")
                row = {
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "condition": method,
                    "repeat": 0,
                    "selection_sha256": _selection_digest(plan.selected),
                    "candidate_ratio": plan.candidate_ratio,
                    "candidate_score_share": plan.candidate_score_share,
                    "actual_ratio": plan.actual_ratio,
                    "actual_score_share": plan.actual_score_share,
                    "selected_neurons": sum(plan.selected_per_layer),
                    "selected_per_layer": plan.selected_per_layer,
                    "prediction": prediction,
                    "score": score_prediction(dataset_name, prediction, sample["answer"]),
                }
                if method in ("N", "N_match"):
                    layer_scores = [q_three[l, :, :w].astype(np.float64).sum(axis=0) for l, w in enumerate(widths)]
                    row["actual_ratio_per_layer"] = [k / w for k, w in zip(plan.selected_per_layer, widths)]
                    row["actual_score_share_per_layer"] = [
                        float(s[plan.selected[l]].sum() / s.sum()) for l, s in enumerate(layer_scores)
                    ]
                    row["fully_ablated_layers"] = sum(k == w for k, w in zip(plan.selected_per_layer, widths))
                    row["clean_restore_verified"] = verify_clean
                    if method == "N_match":
                        row["uniform_selected_neurons"] = frozen["selected_neurons"]
                        row["uniform_actual_score_share"] = frozen["actual_score_share"]
                        row["uniform_selection_sha256"] = frozen["selection_sha256"]
                if layer_scan:
                    target_layer = int(method.split("_")[0][1:])
                    row.update(
                        target_layer=target_layer,
                        target_layer_rate=float(method.split("_r")[1]),
                        actual_layer_rate=plan.selected_per_layer[target_layer] / widths[target_layer],
                        layer_response_share=plan.actual_score_share * len(widths),
                        clean_restore_verified=verify_clean,
                    )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
                completed.add((sample_index, method, 0))
            if (sample_index + 1) % 25 == 0 or len(completed) == expected:
                print(f"[adaptive] {model_name} x {dataset_name}: {len(completed)}/{expected}", flush=True)
    frame = pd.DataFrame(rows)
    if len(frame) != expected or frame.duplicated(["sample_index", "condition"]).any():
        raise ValueError("Adaptive pruning prediction matrix is incomplete.")
    if set(frame["condition"]) != set(conditions):
        raise ValueError("Adaptive pruning conditions are incomplete.")
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "expected_samples": len(samples),
        "epsilon": epsilon,
        "total_neurons": sum(widths),
        "baseline_source": str(baseline_dir / "predictions.jsonl"),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "conditions": [
            {
                "condition": method,
                "score": float(group["score"].mean()),
                "mean_actual_ratio": float(group["actual_ratio"].mean()),
                "mean_actual_score_share": float(group["actual_score_share"].mean()),
            }
            for method, group in frame.groupby("condition", sort=False)
        ],
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary
