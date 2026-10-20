#!/usr/bin/env python3
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

"""Compute real A1/A2 per-sample removal rates without model generation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from neuron_typing_generalization.adaptive_pruning import adaptive_selections


MODELS = ("llava_1_5_7b", "qwen3_vl_2b_instruct", "qwen3_5_2b", "gemma_4_e2b_it")
DATASETS = ("mme", "mmbench", "textvqa", "chartqa")


def prepare_one(input_root: Path, runs_root: Path, output_dir: Path, model: str, dataset: str, epsilon: float) -> dict:
    run_dir = input_root / f"{model}__{dataset}"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("samples") != 1000:
        raise ValueError(f"Frozen response data are incomplete: {run_dir}")
    widths = tuple(int(width) for width in manifest["d_ff_by_layer"])
    samples = pd.read_parquet(runs_root / manifest["run_id"] / "samples/sample_manifest.parquet")
    if len(samples) != 1000:
        raise ValueError(f"Sample manifest has the wrong count: {run_dir}")
    rows = []
    for index, sample in enumerate(samples.to_dict(orient="records")):
        path = run_dir / "sample_statistics" / f"{index:06d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["sample_id"].item()) != str(sample["sample_id"]):
                raise ValueError(f"Frozen response sample order disagrees at {path}")
            plans = adaptive_selections(saved["q"], widths, epsilon=epsilon)
        group = sample.get("pair_group_id")
        if group is None or pd.isna(group) or not str(group):
            group = sample["image_id"]
        row = {"sample_index": index, "sample_id": str(sample["sample_id"]), "group_id": str(group)}
        for method in ("A1", "A2"):
            plan = plans[method]
            key = method.lower()
            row[f"{key}_candidate_ratio"] = plan.candidate_ratio
            row[f"{key}_actual_ratio"] = plan.actual_ratio
            row[f"{key}_candidate_score_share"] = plan.candidate_score_share
            row[f"{key}_actual_score_share"] = plan.actual_score_share
        rows.append(row)
        if (index + 1) % 100 == 0:
            print(f"{model}__{dataset}: {index + 1}/1000", flush=True)
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{model}__{dataset}.csv"
    with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "model": model,
        "dataset": dataset,
        "n_samples": len(frame),
        "epsilon": epsilon,
        "a1_mean_actual_ratio": float(frame["a1_actual_ratio"].mean()),
        "a1_p10_actual_ratio": float(frame["a1_actual_ratio"].quantile(0.1)),
        "a1_p90_actual_ratio": float(frame["a1_actual_ratio"].quantile(0.9)),
        "a2_mean_actual_ratio": float(frame["a2_actual_ratio"].mean()),
        "a2_p10_actual_ratio": float(frame["a2_actual_ratio"].quantile(0.1)),
        "a2_p90_actual_ratio": float(frame["a2_actual_ratio"].quantile(0.9)),
        "a2_max_actual_score_share": float(frame["a2_actual_score_share"].max()),
        "source_run_id": manifest["run_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("artifacts/neuron_typing_generalization/control_response_2026_10_frozen"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/neuron_typing_generalization/runs/advisor_2026_09_main"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "research/neuron_typing_generalization/reports/advisor_2026_09/tables/adaptive_pruning_selection"
        ),
    )
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--epsilon", type=float, default=0.05)
    args = parser.parse_args()
    results = []
    for model in MODELS:
        if args.model and model != args.model:
            continue
        for dataset in DATASETS:
            if args.dataset and dataset != args.dataset:
                continue
            result = prepare_one(args.input_root, args.runs_root, args.output_dir, model, dataset, args.epsilon)
            results.append(result)
            print(json.dumps(result), flush=True)
    if len(results) == 16:
        pd.DataFrame(results).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
