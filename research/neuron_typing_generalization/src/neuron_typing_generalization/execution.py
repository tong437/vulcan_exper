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
