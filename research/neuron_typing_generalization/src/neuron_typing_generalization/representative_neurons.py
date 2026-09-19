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

"""Deterministic representative-neuron selection and trace export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .synthesis import DATASET_LABELS, MODEL_LABELS
from .tracing import export_neuron_traces
from .unpaired_synthesis import DATASET_ORDER, MODEL_ORDER


CRITERION_ORDER = ("high_change", "high_entropy", "stable")
CRITERION_LABELS = {
    "high_change": "High change rate",
    "high_entropy": "High entropy",
    "stable": "Stable",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_representative_neurons(profiles: pd.DataFrame) -> pd.DataFrame:
    """Select one full-observation neuron for each deterministic criterion.

    Criteria are evaluated independently, so one neuron may legitimately represent both high change and high
    entropy. Stable neurons minimize change rate and entropy, then maximize their mean threshold margin.
    """
    required = {
        "layer",
        "neuron",
        "valid_samples",
        "invalid_samples",
        "change_rate_distinct_pairs",
        "normalized_entropy",
        "dominant_class",
        "dominant_proportion",
        "visual_margin_mean",
        "text_margin_mean",
        "count_W",
        "count_V",
        "count_T",
        "count_M",
    }
    missing = sorted(required.difference(profiles.columns))
    if missing:
        raise ValueError(f"Neuron profiles are missing selection columns: {missing}")
    candidates = profiles.loc[
        (profiles["valid_samples"] > 0)
        & (profiles["invalid_samples"] == 0)
        & np.isfinite(profiles["change_rate_distinct_pairs"])
        & np.isfinite(profiles["normalized_entropy"])
    ].copy()
    if candidates.empty:
        raise ValueError("No fully observed neuron is available for representative selection.")
    candidates["change_rate_percentile"] = candidates["change_rate_distinct_pairs"].rank(method="max", pct=True)
    candidates["entropy_percentile"] = candidates["normalized_entropy"].rank(method="max", pct=True)
    candidates["threshold_margin_strength"] = np.minimum(
        candidates["visual_margin_mean"].abs(), candidates["text_margin_mean"].abs()
    )
    sort_specs = {
        "high_change": (
            ["change_rate_distinct_pairs", "normalized_entropy", "layer", "neuron"],
            [False, False, True, True],
        ),
        "high_entropy": (
            ["normalized_entropy", "change_rate_distinct_pairs", "layer", "neuron"],
            [False, False, True, True],
        ),
        "stable": (
            [
                "change_rate_distinct_pairs",
                "normalized_entropy",
                "threshold_margin_strength",
                "dominant_proportion",
                "layer",
                "neuron",
            ],
            [True, True, False, False, True, True],
        ),
    }
    selected = []
    for criterion in CRITERION_ORDER:
        columns, ascending = sort_specs[criterion]
        row = candidates.sort_values(columns, ascending=ascending, kind="stable").iloc[0].to_dict()
        row["criterion"] = criterion
        row["criterion_label"] = CRITERION_LABELS[criterion]
        row["candidate_count"] = len(candidates)
        selected.append(row)
    return pd.DataFrame(selected)


def build_representative_neuron_exports(
    run_dirs: dict[str, dict[str, Path]], artifact_output_root: Path, report_dir: Path, *, analysis_commit: str
) -> dict[str, Any]:
    """Select representative neurons in all 12 runs and export their 1,000-sample traces."""
    artifact_output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    selection_rows: list[dict[str, Any]] = []
    unique_trace_rows: list[dict[str, Any]] = []
    source_profiles: list[dict[str, str]] = []
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            run_dir = run_dirs[dataset][model]
            profile_path = run_dir / "distributions/neuron_profiles.parquet"
            profiles = pd.read_parquet(profile_path)
            selected = select_representative_neurons(profiles)
            run_id = str(selected.iloc[0]["run_id"])
            if selected["run_id"].nunique() != 1 or run_id != run_dir.name:
                raise ValueError(f"Profile run identity does not match directory: {run_dir}")
            source_profiles.append({"run_id": run_id, "sha256": _sha256(profile_path)})
            unique_keys = list(dict.fromkeys(zip(selected["layer"].astype(int), selected["neuron"].astype(int))))
            trace_dir = artifact_output_root / run_id / "traces"
            trace_results = export_neuron_traces(run_dir, unique_keys, trace_dir)
            traces = {(int(item["layer"]), int(item["neuron"])): item for item in trace_results}
            for key, item in traces.items():
                paths = {name: Path(str(item[name])) for name in ("parquet", "csv", "figure")}
                if int(item["samples"]) != 1000:
                    raise ValueError(f"Representative trace does not contain 1,000 samples: {paths['parquet']}")
                unique_trace_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "run_id": run_id,
                        "layer": key[0],
                        "neuron": key[1],
                        "samples": int(item["samples"]),
                        **{
                            f"trace_{name}": str(path.relative_to(artifact_output_root))
                            for name, path in paths.items()
                        },
                        **{f"{name}_sha256": _sha256(path) for name, path in paths.items()},
                    }
                )
            for row in selected.to_dict(orient="records"):
                key = (int(row["layer"]), int(row["neuron"]))
                trace = next(
                    item
                    for item in unique_trace_rows
                    if item["run_id"] == run_id and key == (item["layer"], item["neuron"])
                )
                selection_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS[dataset],
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "run_id": run_id,
                        "criterion": row["criterion"],
                        "criterion_label": row["criterion_label"],
                        "layer": key[0],
                        "neuron": key[1],
                        "valid_samples": int(row["valid_samples"]),
                        "invalid_samples": int(row["invalid_samples"]),
                        "change_rate_distinct_pairs": float(row["change_rate_distinct_pairs"]),
                        "change_rate_percentile": float(row["change_rate_percentile"]),
                        "normalized_entropy": float(row["normalized_entropy"]),
                        "entropy_percentile": float(row["entropy_percentile"]),
                        "dominant_class": str(row["dominant_class"]),
                        "dominant_proportion": float(row["dominant_proportion"]),
                        "threshold_margin_strength": float(row["threshold_margin_strength"]),
                        **{f"count_{label}": int(row[f"count_{label}"]) for label in ("W", "V", "T", "M")},
                        "candidate_count": int(row["candidate_count"]),
                        "trace_parquet": trace["trace_parquet"],
                        "trace_csv": trace["trace_csv"],
                        "trace_figure": trace["trace_figure"],
                    }
                )
    selections = pd.DataFrame(selection_rows)
    if len(selections) != 36 or not selections["valid_samples"].eq(1000).all():
        raise ValueError("Expected three representative selections for each of 12 complete runs.")
    selections.to_csv(report_dir / "representative_neurons_v1_1000.csv", index=False)
    result = {
        "status": "complete",
        "analysis_code_commit": analysis_commit,
        "selection_method": {
            "scope": "one selection per criterion within each model × dataset run",
            "candidate_filter": "valid_samples > 0, invalid_samples = 0, finite change rate and entropy",
            "high_change": "maximum distinct-sample change rate; entropy is the first tie-breaker",
            "high_entropy": "maximum normalized four-state entropy; change rate is the first tie-breaker",
            "stable": (
                "minimum change rate, then minimum entropy, then maximum mean distance from the two class thresholds"
            ),
            "overlap_policy": "criteria are independent; overlapping selections share one exported trace",
        },
        "criterion_slots": len(selections),
        "unique_neurons": len(unique_trace_rows),
        "samples_per_trace": 1000,
        "artifact_root": str(artifact_output_root),
        "source_profiles": source_profiles,
        "selections": selection_rows,
        "traces": unique_trace_rows,
    }
    _write_json(report_dir / "representative_neurons_v1_1000.json", result)
    _write_json(artifact_output_root / "manifest.json", result)
    _write_markdown(result, report_dir / "representative_neurons_v1_1000.md")
    return result


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    rows = []
    for row in result["selections"]:
        rows.append(
            f"| {row['dataset_label']} | {row['model_label']} | {row['criterion_label']} | {row['layer']} | "
            f"{row['neuron']} | {row['change_rate_distinct_pairs']:.4f} | {row['normalized_entropy']:.4f} | "
            f"{row['dominant_class']} ({row['dominant_proportion']:.3f}) |"
        )
    text = f"""# Representative neurons and 1,000-sample traces

The deterministic selection contains {result["criterion_slots"]} criterion slots across 12 runs and resolves to
{result["unique_neurons"]} unique model-local neurons because high-change and high-entropy selections may overlap.
Every exported trace contains exactly {result["samples_per_trace"]} frozen samples.

## Selection rules

- High change: maximum all-distinct-sample response-state change rate; normalized entropy breaks ties.
- High entropy: maximum normalized W/V/T/M entropy; change rate breaks ties.
- Stable: minimum change rate, then minimum entropy, then maximum mean distance from the visual/text thresholds.
- Only fully observed neurons are eligible. Criteria are selected independently and overlap is preserved.

## Selected neurons

| Dataset | Model | Criterion | Layer | Neuron | Change rate | Normalized entropy | Dominant state |
|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## Reproducibility

- Analysis commit: `{result["analysis_code_commit"]}`
- Trace artifact root: `{result["artifact_root"]}`
- `representative_neurons_v1_1000.json` records source-profile and trace-file SHA-256 values.
"""
    path.write_text(text, encoding="utf-8")
