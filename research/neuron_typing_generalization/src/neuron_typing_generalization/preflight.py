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

"""No-weight-load CPU preflight for the first Qwen3.5-2B VQAv2 GPU smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from .adapters import load_local_processor_adapter, load_model_layout
from .config import expand_study, load_study
from .manifests import PAIR_COLUMNS, SAMPLE_COLUMNS, verify_frozen_manifest
from .runtime import build_run_id


def run_qwen_smoke_preflight(study_path: Path, experiment_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Validate configs, assets, processor masks, and storage shape without constructing a model."""
    specs = expand_study(load_study(study_path))
    if len(specs) != 1:
        raise ValueError("Smoke preflight requires exactly one expanded run.")
    spec = specs[0]
    if spec.model.name != "qwen3_5_2b" or spec.dataset.name != "vqav2":
        raise ValueError("This preflight is pinned to Qwen3.5-2B x VQAv2.")
    sampling = spec.study_values["sampling"]
    acceptance = spec.study_values["acceptance"]
    tier = str(sampling["manifest_tier"])
    manifest_dir = experiment_root / "manifests" / spec.dataset.name / tier
    sample_path = manifest_dir / "sample_manifest.parquet"
    pair_path = manifest_dir / "pair_manifest.parquet"
    sample_digest = verify_frozen_manifest(sample_path, SAMPLE_COLUMNS)
    pair_digest = verify_frozen_manifest(pair_path, PAIR_COLUMNS)
    samples = pd.read_parquet(sample_path)
    pairs = pd.read_parquet(pair_path)
    if len(samples) != int(sampling["expected_samples"]) or len(pairs) != int(sampling["pair_groups"]):
        raise ValueError("Frozen smoke manifest counts disagree with the study config.")

    layout = load_model_layout(spec.model.values)
    if layout.decoder_layers != int(acceptance["expected_decoder_layers"]):
        raise ValueError("Decoder layer acceptance check failed.")
    if layout.d_ff != int(acceptance["expected_d_ff"]):
        raise ValueError("d_ff acceptance check failed.")
    required_assets = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ]
    missing_assets = [name for name in required_assets if not (layout.model_path / name).is_file()]
    weight_files = sorted(layout.model_path.glob("*.safetensors"))
    if missing_assets or not weight_files:
        raise FileNotFoundError(f"Model assets are incomplete: missing={missing_assets}, weights={len(weight_files)}")

    image_manifest = pd.read_parquet(artifact_root / "images/vqav2/image_manifest.parquet")
    image_by_id = {
        str(row.image_id): artifact_root / "images/vqav2" / row.relative_path for row in image_manifest.itertuples()
    }
    processor_adapter = load_local_processor_adapter(spec.model.values, use_fast=False)
    prepared_rows = []
    for sample in samples.itertuples():
        image_id = str(sample.image_id)
        image_path = image_by_id.get(image_id)
        if image_path is None or not image_path.is_file():
            raise FileNotFoundError(f"Materialized smoke image is missing for image_id={image_id}.")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            prepared = processor_adapter.prepare_inputs(
                {"question": str(sample.question), "image": image.convert("RGB")}
            )
        roles = prepared.token_roles
        if not bool(roles.visual.any()) or not bool(roles.text.any()):
            raise ValueError(f"Empty visual/text token mask for sample_id={sample.sample_id}.")
        prepared_rows.append(
            {
                "sample_id": str(sample.sample_id),
                "sequence_tokens": int(roles.visual.numel()),
                "visual_tokens": int(roles.visual.sum()),
                "question_tokens": int(roles.text.sum()),
                "control_tokens": int(roles.control.sum()),
                "padding_tokens": int(roles.padding.sum()),
            }
        )

    score_rows = len(samples) * layout.decoder_layers * layout.d_ff * 2
    return {
        "status": "passed",
        "model_weights_loaded": False,
        "run_id": build_run_id(spec),
        "model": {
            "path": str(layout.model_path),
            "architecture": layout.architecture,
            "decoder_layers": layout.decoder_layers,
            "d_ff": layout.d_ff,
            "weight_files": [path.name for path in weight_files],
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
        },
        "manifest": {
            "tier": tier,
            "samples": len(samples),
            "pairs": len(pairs),
            "sample_sha256": sample_digest,
            "pair_sha256": pair_digest,
        },
        "processor": {
            "class": type(processor_adapter.processor).__name__,
            "samples_prepared": len(prepared_rows),
            "minimum_sequence_tokens": min(row["sequence_tokens"] for row in prepared_rows),
            "maximum_sequence_tokens": max(row["sequence_tokens"] for row in prepared_rows),
            "minimum_visual_tokens": min(row["visual_tokens"] for row in prepared_rows),
            "minimum_question_tokens": min(row["question_tokens"] for row in prepared_rows),
            "sample_token_audit": prepared_rows,
        },
        "storage_estimate": {
            "score_rows": score_rows,
            "minimum_dense_numeric_bytes": score_rows * 32,
            "note": "Parquet size varies with dictionary encoding; reserve at least 2x this numeric lower bound.",
        },
        "deferred_to_gpu": ["model_forward", "hook_layer_count", "streaming_vs_direct_activation_statistics"],
    }


def run_vqav2_three_model_preflight(study_path: Path, experiment_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Run processor, manifest, version, and storage checks for every VQAv2 smoke model."""
    import transformers
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    from .hooks import discover_decoder_ffns

    specs = expand_study(load_study(study_path))
    if not specs or any(spec.dataset.name != "vqav2" for spec in specs):
        raise ValueError("Three-model smoke preflight accepts only VQAv2 runs.")
    tiers = {str(spec.study_values["sampling"]["manifest_tier"]) for spec in specs}
    if len(tiers) != 1:
        raise ValueError("All smoke models must use the same frozen manifest tier.")
    tier = tiers.pop()
    manifest_dir = experiment_root / "manifests/vqav2" / tier
    sample_path = manifest_dir / "sample_manifest.parquet"
    pair_path = manifest_dir / "pair_manifest.parquet"
    sample_digest = verify_frozen_manifest(sample_path, SAMPLE_COLUMNS)
    pair_digest = verify_frozen_manifest(pair_path, PAIR_COLUMNS)
    samples = pd.read_parquet(sample_path)
    pairs = pd.read_parquet(pair_path)
    image_manifest = pd.read_parquet(artifact_root / "images/vqav2/image_manifest.parquet")
    image_by_id = {
        str(row.image_id): artifact_root / "images/vqav2" / row.relative_path for row in image_manifest.itertuples()
    }

    model_reports = []
    for spec in specs:
        report: dict[str, Any] = {"name": spec.model.name, "status": "failed", "model_weights_loaded": False}
        try:
            layout = load_model_layout(spec.model.values)
            minimum = spec.model.values.get("minimum_transformers_version")
            if minimum:
                from packaging.version import Version

                if Version(transformers.__version__) < Version(str(minimum)):
                    raise RuntimeError(f"Transformers {transformers.__version__} is below required version {minimum}.")
            if not hasattr(transformers, layout.architecture):
                raise RuntimeError(f"Transformers {transformers.__version__} does not expose {layout.architecture}.")
            model_config = AutoConfig.from_pretrained(layout.model_path, local_files_only=True)
            with init_empty_weights():
                empty_model = getattr(transformers, layout.architecture)(model_config)
            bindings = discover_decoder_ffns(empty_model)
            observed_widths = tuple(binding.d_ff for binding in bindings)
            expected_widths = layout.d_ff_by_layer or (layout.d_ff,) * layout.decoder_layers
            if len(bindings) != layout.decoder_layers or observed_widths != expected_widths:
                raise RuntimeError("Constructed decoder FFN layout disagrees with the frozen model configuration.")
            del empty_model
            weight_files = sorted(layout.model_path.glob("*.safetensors"))
            if not weight_files:
                raise FileNotFoundError(f"No safetensors weights under {layout.model_path}.")
            processor_adapter = load_local_processor_adapter(spec.model.values, use_fast=False)
            token_audit = []
            for sample in samples.itertuples():
                image_path = image_by_id.get(str(sample.image_id))
                if image_path is None or not image_path.is_file():
                    raise FileNotFoundError(f"Materialized image is missing for image_id={sample.image_id}.")
                with Image.open(image_path) as image:
                    prepared = processor_adapter.prepare_inputs(
                        {"question": str(sample.question), "image": image.convert("RGB")}
                    )
                roles = prepared.token_roles
                if not bool(roles.visual.any()) or not bool(roles.text.any()):
                    raise ValueError(f"Empty visual/text token mask for sample_id={sample.sample_id}.")
                token_audit.append(
                    {
                        "sample_id": str(sample.sample_id),
                        "sequence_tokens": int(roles.visual.numel()),
                        "visual_tokens": int(roles.visual.sum()),
                        "question_tokens": int(roles.text.sum()),
                    }
                )
            compact_bytes = len(samples) * sum(expected_widths) * 2 * 4
            report.update(
                {
                    "status": "passed",
                    "architecture": layout.architecture,
                    "decoder_layers": layout.decoder_layers,
                    "d_ff": layout.d_ff,
                    "maximum_d_ff": max(expected_widths),
                    "d_ff_by_layer": list(expected_widths),
                    "processor_class": type(processor_adapter.processor).__name__,
                    "weight_files": len(weight_files),
                    "weight_bytes": sum(path.stat().st_size for path in weight_files),
                    "minimum_sequence_tokens": min(row["sequence_tokens"] for row in token_audit),
                    "maximum_sequence_tokens": max(row["sequence_tokens"] for row in token_audit),
                    "minimum_visual_tokens": min(row["visual_tokens"] for row in token_audit),
                    "minimum_question_tokens": min(row["question_tokens"] for row in token_audit),
                    "compact_activation_bytes": compact_bytes,
                    "logical_long_form_rows": len(samples) * sum(expected_widths) * 2,
                }
            )
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
        model_reports.append(report)
    return {
        "status": "passed" if all(report["status"] == "passed" for report in model_reports) else "failed",
        "transformers": transformers.__version__,
        "manifest": {
            "tier": tier,
            "samples": len(samples),
            "pairs": len(pairs),
            "sample_sha256": sample_digest,
            "pair_sha256": pair_digest,
        },
        "models": model_reports,
        "deferred_to_gpu": ["model_forward", "hook_layer_count", "peak_vram", "streaming_vs_direct_statistics"],
    }
