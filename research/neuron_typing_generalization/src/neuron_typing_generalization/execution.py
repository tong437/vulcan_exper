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

"""Executable CPU stages after model activation/importance collection."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .schema_validation import validate_document, validate_frame
from .tables import (
    analyze_class_table,
    analyze_class_tensor,
    classify_score_table,
    controlled_pair_change_table,
    controlled_pair_change_tensor,
)


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


def _write_json_atomic(destination: Path, value: dict) -> None:
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


def _verify_parts(run_dir: Path, manifest: dict) -> list[Path]:
    parts = manifest.get("parts", [])
    if len(parts) != int(manifest["samples"]):
        raise ValueError("Compact manifest does not contain exactly one part per sample.")
    paths = []
    digest = hashlib.sha256()
    for expected_index, part in enumerate(parts):
        if int(part["sample_index"]) != expected_index:
            raise ValueError("Compact sample partitions are not contiguous and ordered.")
        path = run_dir / part["file"]
        observed = _sha256(path)
        if observed != part["sha256"]:
            raise RuntimeError(f"Compact partition digest mismatch: {path}")
        digest.update(f"{expected_index}:{part['sample_id']}:{observed}\n".encode())
        paths.append(path)
    if digest.hexdigest() != manifest["parts_sha256"]:
        raise RuntimeError("Compact partition-set digest does not match its manifest.")
    return paths


def _compact_importance_run(run_dir: Path, manifest: dict) -> dict[str, int | str]:
    weight_path = run_dir / manifest["down_projection_norms_file"]
    if _sha256(weight_path) != manifest["down_projection_norms_sha256"]:
        raise RuntimeError("Down-projection norm digest does not match its manifest.")
    weights = np.load(weight_path, allow_pickle=False).astype(np.float32, copy=False)
    expected_shape = (int(manifest["layers"]), int(manifest["d_ff"]))
    d_ff_by_layer = np.asarray(manifest["d_ff_by_layer"], dtype=np.int64)
    if d_ff_by_layer.shape != (expected_shape[0],) or np.any(d_ff_by_layer > expected_shape[1]):
        raise ValueError("Per-layer d_ff values disagree with the compact maximum width.")
    neuron_valid = np.arange(expected_shape[1])[None, :] < d_ff_by_layer[:, None]
    if weights.shape != expected_shape:
        raise ValueError(f"Down-projection norms have shape {weights.shape}; expected {expected_shape}.")

    score_dir = run_dir / "scores/sample_parts"
    score_parts = []
    for part, source in zip(manifest["parts"], _verify_parts(run_dir, manifest)):
        with np.load(source, allow_pickle=False) as arrays:
            sample_id = str(arrays["sample_id"].item())
            rms = arrays["activation_rms"].astype(np.float32, copy=False)
            token_count = arrays["token_count"].astype(np.int32, copy=False)
            valid = arrays["valid"].astype(np.bool_, copy=False)
        if sample_id != part["sample_id"] or rms.shape != (expected_shape[0], 2, expected_shape[1]):
            raise ValueError(f"Activation partition shape or identity mismatch: {source}")
        importance = rms * weights[:, None, :]
        denominator = importance.sum(axis=-1, keepdims=True)
        finite_importance = np.where(neuron_valid[:, None, :], np.isfinite(importance), True).all(axis=-1)
        slice_valid = valid & finite_importance & np.isfinite(denominator[..., 0])
        slice_valid &= denominator[..., 0] > 0
        q = np.divide(importance, denominator, out=np.full_like(importance, np.nan), where=denominator > 0)
        q[~slice_valid] = np.nan
        q[~np.broadcast_to(neuron_valid[:, None, :], q.shape)] = np.nan
        destination = score_dir / f"{int(part['sample_index']):06d}.npz"
        _write_npz_atomic(
            destination,
            sample_id=np.asarray(sample_id),
            activation_rms=rms,
            token_count=token_count,
            importance=importance,
            q=q,
            valid=slice_valid,
        )
        score_parts.append(
            {
                "sample_index": int(part["sample_index"]),
                "sample_id": sample_id,
                "file": str(destination.relative_to(run_dir)),
                "sha256": _sha256(destination),
            }
        )
    digest = hashlib.sha256()
    for part in score_parts:
        digest.update(f"{part['sample_index']}:{part['sample_id']}:{part['sha256']}\n".encode())
    score_manifest = {
        "run_id": manifest["run_id"],
        "samples": manifest["samples"],
        "layers": manifest["layers"],
        "d_ff": manifest["d_ff"],
        "d_ff_by_layer": manifest["d_ff_by_layer"],
        "rows": manifest["rows"],
        "storage_format": "partitioned_npz_v1",
        "parts": score_parts,
        "parts_sha256": digest.hexdigest(),
        "source_activation_parts_sha256": manifest["parts_sha256"],
    }
    output = run_dir / "scores/manifest.json"
    _write_json_atomic(output, score_manifest)
    return {"stage": "compute_importance", "score_rows": int(manifest["rows"]), "output": str(output)}


def compute_importance_run(run_dir: Path) -> dict[str, int | str]:
    """Convert reduced activation statistics into normalized sample-level importance scores."""
    activation_path = run_dir / "activations/neuron_activation_stats.parquet"
    manifest = json.loads((run_dir / "activations/manifest.json").read_text(encoding="utf-8"))
    validate_document(manifest, "activation_manifest.schema.json")
    if manifest.get("storage_format") == "partitioned_npz_v1":
        return _compact_importance_run(run_dir, manifest)
    if manifest["statistics_file"] != "activations/neuron_activation_stats.parquet":
        raise ValueError("Activation manifest points outside the fixed stage contract.")
    if _sha256(activation_path) != manifest["statistics_sha256"]:
        raise RuntimeError("Reduced activation statistics digest does not match its manifest.")
    statistics = pd.read_parquet(activation_path)
    required = {
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
    }
    missing = sorted(required - set(statistics.columns))
    if missing:
        raise ValueError(f"Activation statistics are missing columns: {missing}")
    validate_frame(statistics, "activation_stats.schema.json")
    activation_keys = ["run_id", "sample_id", "model", "dataset", "layer", "neuron", "modality"]
    if statistics.duplicated(activation_keys).any():
        raise ValueError("Activation statistics contain duplicate sample/layer/neuron/modality rows.")
    observed_run_ids = set(statistics["run_id"].astype(str))
    if observed_run_ids != {str(manifest["run_id"])}:
        raise ValueError("Activation rows do not have the run_id declared by their manifest.")
    observed_neurons = sorted(int(value) for value in statistics["neuron"].unique())
    if observed_neurons != list(range(int(manifest["d_ff"]))):
        raise ValueError("Activation neuron IDs do not match the manifest d_ff.")
    if (
        len(statistics) != int(manifest["rows"])
        or statistics["sample_id"].nunique() != int(manifest["samples"])
        or statistics["layer"].nunique() != int(manifest["layers"])
    ):
        raise ValueError("Activation row/sample/layer cardinality disagrees with its manifest.")
    scores = statistics.copy()
    scores["importance"] = scores["activation_rms"] * scores["down_projection_norm"]
    group_keys = ["run_id", "sample_id", "model", "dataset", "layer", "modality"]
    denominator = scores.groupby(group_keys, sort=False)["importance"].transform("sum")
    slice_valid = scores.groupby(group_keys, sort=False)["valid"].transform("all") & denominator.gt(0)
    scores["valid"] = scores["valid"].astype(bool) & slice_valid
    scores["importance"] = scores["importance"].where(scores["valid"])
    scores["q"] = scores["importance"].div(denominator).where(scores["valid"])
    validate_frame(scores, "neuron_scores.schema.json")
    output = run_dir / "scores/neuron_scores.parquet"
    _write_parquet_atomic(scores, output)
    return {"stage": "compute_importance", "score_rows": len(scores), "output": str(output)}


def classify_run(run_dir: Path) -> dict[str, int | str]:
    """Normalize importance shares and write sample-conditional neuron classes."""
    compact_manifest_path = run_dir / "scores/manifest.json"
    if compact_manifest_path.is_file():
        manifest = json.loads(compact_manifest_path.read_text(encoding="utf-8"))
        class_dir = run_dir / "classifications/sample_parts"
        class_parts = []
        d_ff_by_layer = np.asarray(manifest["d_ff_by_layer"], dtype=np.int64)
        neuron_valid = np.arange(int(manifest["d_ff"]))[None, :] < d_ff_by_layer[:, None]
        for part, source in zip(manifest["parts"], _verify_parts(run_dir, manifest)):
            with np.load(source, allow_pickle=False) as arrays:
                sample_id = str(arrays["sample_id"].item())
                q = arrays["q"].astype(np.float32, copy=False)
                valid = arrays["valid"].astype(np.bool_, copy=False)
            threshold = 1.0 / d_ff_by_layer[:, None]
            slice_valid = valid[:, 0] & valid[:, 1]
            visual_strong = q[:, 0] > threshold
            text_strong = q[:, 1] > threshold
            classes = visual_strong.astype(np.int8) + 2 * text_strong.astype(np.int8)
            classes[~slice_valid] = -1
            classes[~neuron_valid] = -1
            destination = class_dir / f"{int(part['sample_index']):06d}.npz"
            _write_npz_atomic(destination, sample_id=np.asarray(sample_id), classes=classes)
            class_parts.append(
                {
                    "sample_index": int(part["sample_index"]),
                    "sample_id": sample_id,
                    "file": str(destination.relative_to(run_dir)),
                    "sha256": _sha256(destination),
                }
            )
        digest = hashlib.sha256()
        for part in class_parts:
            digest.update(f"{part['sample_index']}:{part['sample_id']}:{part['sha256']}\n".encode())
        class_manifest = {
            "run_id": manifest["run_id"],
            "samples": manifest["samples"],
            "layers": manifest["layers"],
            "d_ff": manifest["d_ff"],
            "d_ff_by_layer": manifest["d_ff_by_layer"],
            "rows": int(manifest["samples"]) * int(d_ff_by_layer.sum()),
            "storage_format": "partitioned_npz_v1",
            "parts": class_parts,
            "parts_sha256": digest.hexdigest(),
            "source_score_parts_sha256": manifest["parts_sha256"],
        }
        output = run_dir / "classifications/manifest.json"
        _write_json_atomic(output, class_manifest)
        return {
            "stage": "classify_neurons",
            "score_rows": int(manifest["rows"]),
            "class_rows": int(class_manifest["rows"]),
            "output": str(output),
        }
    score_path = run_dir / "scores/neuron_scores.parquet"
    scores = pd.read_parquet(score_path)
    normalized, classes = classify_score_table(scores)
    validate_frame(normalized, "neuron_scores.schema.json")
    validate_frame(classes, "neuron_classes.schema.json")
    _write_parquet_atomic(normalized, score_path)
    output = run_dir / "classifications/neuron_classes.parquet"
    _write_parquet_atomic(classes, output)
    return {
        "stage": "classify_neurons",
        "score_rows": len(normalized),
        "class_rows": len(classes),
        "output": str(output),
    }


def compare_run(run_dir: Path, chunk_size: int = 64) -> dict[str, int | str]:
    """Write controlled pair changes, neuron profiles, transitions, and aggregate intermediates."""
    samples = pd.read_parquet(run_dir / "samples/sample_manifest.parquet")
    pairs = pd.read_parquet(run_dir / "samples/pair_manifest.parquet")
    compact_manifest_path = run_dir / "classifications/manifest.json"
    if compact_manifest_path.is_file():
        manifest = json.loads(compact_manifest_path.read_text(encoding="utf-8"))
        sample_ids = [str(value) for value in samples["sample_id"]]
        class_arrays = []
        for expected_id, source in zip(sample_ids, _verify_parts(run_dir, manifest)):
            with np.load(source, allow_pickle=False) as arrays:
                if str(arrays["sample_id"].item()) != expected_id:
                    raise ValueError("Classification partitions disagree with frozen sample order.")
                class_arrays.append(arrays["classes"].astype(np.int8, copy=True))
        tensor = np.stack(class_arrays)
        summary, profiles, transitions = analyze_class_tensor(
            tensor, sample_ids, str(manifest["run_id"]), pairs, chunk_size=chunk_size
        )
        changes = controlled_pair_change_tensor(tensor, sample_ids, str(manifest["run_id"]), pairs)
    else:
        classes = pd.read_parquet(run_dir / "classifications/neuron_classes.parquet")
        summary, profiles, transitions = analyze_class_table(classes, samples, pairs, chunk_size=chunk_size)
        changes = controlled_pair_change_table(classes, samples, pairs)
    validate_frame(changes, "pairwise_changes.schema.json")
    validate_frame(profiles, "neuron_profiles.schema.json")
    validate_document(summary, "summary_metrics.schema.json")
    _write_parquet_atomic(changes, run_dir / "distributions/pairwise_changes.parquet")
    _write_parquet_atomic(profiles, run_dir / "distributions/neuron_profiles.parquet")
    np.save(run_dir / "distributions/transition_matrices.npy", transitions, allow_pickle=False)
    (run_dir / "distributions/analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "stage": "compare_distributions",
        "controlled_pair_rows": len(changes),
        "profile_rows": len(profiles),
        "output": str(run_dir / "distributions"),
    }


def evaluate_run(run_dir: Path) -> dict[str, str]:
    """Promote validated CPU analysis outputs to final metric and figure manifests."""
    required = (
        run_dir / "distributions/pairwise_changes.parquet",
        run_dir / "distributions/neuron_profiles.parquet",
        run_dir / "distributions/transition_matrices.npy",
        run_dir / "distributions/analysis_summary.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation inputs are missing: {missing}")
    summary = json.loads(required[-1].read_text(encoding="utf-8"))
    validate_document(summary, "summary_metrics.schema.json")
    summary_path = run_dir / "metrics/summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    figure_path = run_dir / "figures/manifest.json"
    figure_path.write_text(
        json.dumps({"run_id": summary["run_id"], "figures": [], "status": "pending_gpu_results"}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"stage": "evaluate", "summary": str(summary_path), "figures": str(figure_path)}
