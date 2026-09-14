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
from .tables import analyze_class_table, classify_score_table, controlled_pair_change_table


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


def compute_importance_run(run_dir: Path) -> dict[str, int | str]:
    """Convert reduced activation statistics into normalized sample-level importance scores."""
    activation_path = run_dir / "activations/neuron_activation_stats.parquet"
    manifest = json.loads((run_dir / "activations/manifest.json").read_text(encoding="utf-8"))
    validate_document(manifest, "activation_manifest.schema.json")
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
    classes = pd.read_parquet(run_dir / "classifications/neuron_classes.parquet")
    samples = pd.read_parquet(run_dir / "samples/sample_manifest.parquet")
    pairs = pd.read_parquet(run_dir / "samples/pair_manifest.parquet")
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
