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

"""Configuration-driven, resumable GPU activation collection for frozen VLM runs."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .adapters import load_local_processor_adapter, load_model_layout
from .advisor_prompts import mmbench_selection_prompt
from .collection import collect_reduced_activations
from .config import ExperimentSpec, expand_study, load_study
from .hooks import InjectedModelAdapter, validate_direct_vs_streaming
from .interfaces import PreparedInputs, TokenRoleMasks
from .runtime import initialize_run, resolve_artifact_root


def _version_tuple(value: str) -> tuple[int, ...]:
    numeric = value.split("+", 1)[0].split(".dev", 1)[0]
    return tuple(int(part) for part in numeric.split(".") if part.isdigit())


def _validate_environment(spec: ExperimentSpec) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real activation collection requires a GPU.")
    import transformers

    minimum = spec.model.values.get("minimum_transformers_version")
    if minimum and _version_tuple(transformers.__version__) < _version_tuple(str(minimum)):
        raise RuntimeError(
            f"{spec.model.name} requires Transformers >= {minimum}; installed={transformers.__version__}."
        )
    architecture = str(spec.model.values["architecture"])
    if not hasattr(transformers, architecture):
        raise RuntimeError(f"Installed Transformers does not expose {architecture}.")


def _select_spec(study_path: Path, model_name: str, dataset_name: str) -> ExperimentSpec:
    specs = expand_study(load_study(study_path))
    matches = [spec for spec in specs if spec.model.name == model_name and spec.dataset.name == dataset_name]
    if len(matches) != 1:
        available = sorted({(spec.model.name, spec.dataset.name) for spec in specs})
        raise ValueError(
            f"Expected one run for model={model_name}, dataset={dataset_name}; "
            f"available={available}, matches={len(matches)}."
        )
    return matches[0]


def _model_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Could not determine a materialized model input device.")


def _move_model_inputs(values: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            target_dtype = dtype if value.is_floating_point() else value.dtype
            moved[key] = value.to(device=device, dtype=target_dtype, non_blocking=True)
        else:
            moved[key] = value
    moved.update({"use_cache": False, "output_attentions": False, "output_hidden_states": False})
    return moved


def _load_image_index(artifact_root: Path, dataset_name: str) -> dict[str, Path]:
    manifest_path = artifact_root / "images" / dataset_name / "image_manifest.parquet"
    frame = pd.read_parquet(manifest_path)
    image_root = artifact_root / "images" / dataset_name
    index = {str(row.image_id): image_root / str(row.relative_path) for row in frame.itertuples()}
    missing = [str(path) for path in index.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Materialized {dataset_name} images are incomplete; first missing file: {missing[0]}")
    return index


def _load_replay_rows(replay_dir: Path | None, dataset_name: str) -> dict[str, dict[str, Any]]:
    if dataset_name != "mmbench":
        return {}
    if replay_dir is None:
        raise ValueError("MMBench activation collection requires --replay-dir for frozen choices.")
    path = replay_dir / "mmbench_1000.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Frozen MMBench replay rows are missing: {path}")
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["sample_id"])] = row
    return rows


def _move_roles(roles: TokenRoleMasks, device: torch.device) -> TokenRoleMasks:
    return TokenRoleMasks(
        visual=roles.visual.to(device),
        text=roles.text.to(device),
        control=roles.control.to(device),
        padding=roles.padding.to(device),
    )


def run_gpu_collection(
    study_path: Path,
    model_name: str,
    dataset_name: str,
    max_samples: int | None = None,
    *,
    validate_streaming: bool = False,
    validation_only: bool = False,
    replay_dir: Path | None = None,
    down_norms_path: Path | None = None,
) -> dict[str, Any]:
    """Load one configured VLM and collect compact reduced activations with resume support."""
    spec = _select_spec(study_path.resolve(), model_name, dataset_name)
    _validate_environment(spec)
    context = initialize_run(spec)
    samples = pd.read_parquet(context.run_dir / "samples/sample_manifest.parquet")
    pair_path = context.run_dir / "samples/pair_manifest.parquet"
    pairs = pd.read_parquet(pair_path) if pair_path.is_file() else pd.DataFrame()
    sampling = spec.study_values.get("sampling", {})
    if "expected_samples" in sampling and len(samples) != int(sampling["expected_samples"]):
        raise ValueError("Frozen sample count disagrees with the study configuration.")
    if "expected_pairs" in sampling and len(pairs) != int(sampling["expected_pairs"]):
        raise ValueError("Frozen pair count disagrees with the study configuration.")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        samples = samples.iloc[:max_samples].copy()

    import transformers

    layout = load_model_layout(spec.model.values)
    dtype_name = str(spec.model.values.get("dtype", "bfloat16"))
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype_name)
    if dtype is None:
        raise ValueError(f"Unsupported inference dtype: {dtype_name}")
    model_class = getattr(transformers, layout.architecture)
    started = time.monotonic()
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
    image_index = _load_image_index(resolve_artifact_root(spec), spec.dataset.name)
    replay_rows = _load_replay_rows(replay_dir, spec.dataset.name)
    prompt_audit = []
    if spec.dataset.name == "mmbench":
        for sample in samples.to_dict(orient="records"):
            sample_id = str(sample["sample_id"])
            prompt = mmbench_selection_prompt(sample, replay_rows.get(sample_id))
            prompt_audit.append(
                {"sample_id": sample_id, "selection_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
            )
        (context.run_dir / "logs/mmbench_input_policy.json").write_text(
            json.dumps(
                {
                    "policy": "image_plus_question_plus_valid_choices_no_hint",
                    "generation_instruction_included": False,
                    "samples": prompt_audit,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def prepare(sample: Mapping[str, Any]) -> PreparedInputs:
        image_id = str(sample["image_id"])
        image_path = image_index.get(image_id)
        if image_path is None:
            raise FileNotFoundError(f"No materialized {spec.dataset.name} image for image_id={image_id}.")
        question = str(sample["question"])
        if spec.dataset.name == "mmbench":
            question = mmbench_selection_prompt(sample, replay_rows.get(str(sample["sample_id"])))
        with Image.open(image_path) as image:
            prepared = processor_adapter.prepare_inputs(
                {"question": question, "image": image.convert("RGB")}
            )
        return PreparedInputs(
            model_inputs=_move_model_inputs(prepared.model_inputs, device, dtype),
            token_roles=_move_roles(prepared.token_roles, device),
        )

    adapter = InjectedModelAdapter(layout.family, model, prepare)
    discovered = tuple(adapter.iter_ffn_bindings())
    expected_widths = layout.d_ff_by_layer or (layout.d_ff,) * layout.decoder_layers
    observed_widths = tuple(binding.d_ff for binding in discovered)
    if len(discovered) != layout.decoder_layers or observed_widths != expected_widths:
        raise RuntimeError(
            f"Discovered FFN layout disagrees with config: layers={len(discovered)}, d_ff={observed_widths}."
        )
    down_norms = None
    if down_norms_path is not None:
        stored = np.load(down_norms_path, allow_pickle=False)
        expected_shape = (len(discovered), max(observed_widths))
        if stored.shape != expected_shape:
            raise ValueError(f"Precomputed down-projection norms have shape {stored.shape}; expected {expected_shape}.")
        down_norms = {
            binding.layer: torch.from_numpy(stored[binding.layer, : binding.d_ff].astype(np.float32, copy=True))
            for binding in discovered
        }
    validation_report = None
    if validate_streaming or validation_only:
        validation_sample = samples.iloc[0].to_dict()
        validation_prepared = prepare(validation_sample)
        validation_down_norms = down_norms
        if validation_down_norms is None:
            validation_down_norms = {
                binding.layer: torch.linalg.vector_norm(
                    binding.down_projection.weight.detach().to(torch.float32), ord=2, dim=0
                )
                for binding in discovered
            }
        validation_report = validate_direct_vs_streaming(
            model,
            discovered,
            validation_prepared,
            validation_down_norms,
            # The direct boolean-index reduction and streaming masked reduction
            # sum hundreds of BF16-derived visual-token values in different
            # orders. These bounds remain tight while covering that expected
            # float32 accumulation-order error.
            absolute_tolerance=1e-5,
            relative_tolerance=5e-4,
        )
        validation_report.update(
            {
                "run_id": context.run_id,
                "model": spec.model.name,
                "model_revision": str(spec.model.values["revision"]),
                "dataset": spec.dataset.name,
                "dataset_revision": str(spec.dataset.values["revision"]),
                "sample_id": str(validation_sample["sample_id"]),
                "device": str(device),
                "dtype": dtype_name,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            }
        )
        validation_path = context.run_dir / "logs/direct_vs_streaming_validation.json"
        validation_path.write_text(
            json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if validation_report["status"] != "passed":
            raise RuntimeError(f"Direct-vs-streaming validation failed; see {validation_path}.")
        if validation_only:
            return {
                "run_dir": str(context.run_dir),
                "validation": validation_report,
                "validation_report": str(validation_path),
            }
    torch.cuda.reset_peak_memory_stats()
    manifest = collect_reduced_activations(
        adapter,
        samples,
        context.run_dir,
        context.run_id,
        spec.model.name,
        spec.dataset.name,
        down_projection_norms=down_norms,
    )
    runtime = {
        "run_id": context.run_id,
        "model": spec.model.name,
        "samples": len(samples),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "device": str(device),
        "dtype": dtype_name,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "input_policy": (
            "image_plus_question_plus_valid_choices_no_hint"
            if spec.dataset.name == "mmbench"
            else "frozen_sample_question"
        ),
        "down_projection_norms_source": str(down_norms_path) if down_norms_path is not None else "loaded_model",
    }
    (context.run_dir / "logs/gpu_collection_summary.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "run_dir": str(context.run_dir),
        "activation_manifest": manifest,
        "runtime": runtime,
        "validation": validation_report,
    }


def run_vqav2_gpu_collection(
    study_path: Path,
    model_name: str,
    max_samples: int | None = None,
    *,
    validate_streaming: bool = False,
    validation_only: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper for the original VQAv2-only command."""
    return run_gpu_collection(
        study_path,
        model_name,
        "vqav2",
        max_samples,
        validate_streaming=validate_streaming,
        validation_only=validation_only,
    )
