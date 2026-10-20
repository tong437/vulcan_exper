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

"""Run the advisor's Hellinger/average-linkage neuron-clustering pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from neuron_typing_generalization.advisor_clustering import (
    PROBABILITY_COLUMNS,
    average_linkage_clusters,
    condensed_distance_bytes,
    stable_run_seed,
    stratified_neuron_sample,
)
from neuron_typing_generalization.config import expand_study, load_study
from neuron_typing_generalization.runtime import build_run_id, resolve_artifact_root


MODEL_LABELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASET_LABELS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-neurons", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for spec in expand_study(load_study(args.study)):
        run_id = build_run_id(spec)
        run_dir = resolve_artifact_root(spec) / "runs" / spec.study / run_id
        profile_path = run_dir / "distributions/neuron_profiles.parquet"
        profile = pd.read_parquet(profile_path, columns=["layer", "neuron", *PROBABILITY_COLUMNS])
        unique_probability_count = len(profile[list(PROBABILITY_COLUMNS)].drop_duplicates())
        run_seed = stable_run_seed(args.seed, run_id)
        sample = stratified_neuron_sample(profile, args.maximum_neurons, run_seed)
        probabilities = sample[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        labels, linkage_matrix, cluster_rows = average_linkage_clusters(probabilities, args.threshold)
        sample.insert(0, "run_id", run_id)
        sample["cluster_id"] = labels

        destination = args.output_dir / run_id
        destination.mkdir(parents=True, exist_ok=True)
        assignment_path = destination / "sampled_neuron_assignments.parquet"
        cluster_path = destination / "cluster_summary.csv"
        linkage_path = destination / "average_linkage.npy"
        sample.to_parquet(assignment_path, index=False)
        pd.DataFrame(cluster_rows).to_csv(cluster_path, index=False)
        np.save(linkage_path, linkage_matrix, allow_pickle=False)
        largest_cluster = max(row["neuron_count"] for row in cluster_rows)
        singleton_clusters = sum(row["neuron_count"] == 1 for row in cluster_rows)
        row = {
            "model": spec.model.name,
            "model_label": MODEL_LABELS[spec.model.name],
            "dataset": spec.dataset.name,
            "dataset_label": DATASET_LABELS[spec.dataset.name],
            "run_id": run_id,
            "full_neuron_count": int(len(profile)),
            "unique_probability_count": unique_probability_count,
            "sampled_neuron_count": int(len(sample)),
            "sample_fraction": float(len(sample) / len(profile)),
            "layer_count": int(profile["layer"].nunique()),
            "global_seed": args.seed,
            "run_seed": run_seed,
            "hellinger_threshold": args.threshold,
            "cluster_count": len(cluster_rows),
            "singleton_cluster_count": singleton_clusters,
            "largest_cluster_neuron_count": largest_cluster,
            "largest_cluster_proportion": float(largest_cluster / len(sample)),
            "full_condensed_distance_gib": condensed_distance_bytes(len(profile)) / 1024**3,
            "unique_condensed_distance_gib": condensed_distance_bytes(unique_probability_count) / 1024**3,
            "sample_condensed_distance_mib": condensed_distance_bytes(len(sample)) / 1024**2,
            "source_profile_sha256": _sha256(profile_path),
            "assignment_sha256": _sha256(assignment_path),
            "cluster_summary_sha256": _sha256(cluster_path),
            "linkage_sha256": _sha256(linkage_path),
        }
        _write_json(
            destination / "manifest.json",
            {
                **row,
                "status": "complete",
                "scope": "exact average-linkage clustering within a deterministic layer-stratified neuron sample",
                "probability_order": ["V", "T", "M", "W"],
                "distance": "Hellinger",
                "linkage": "average",
                "cut_rule": f"SciPy fcluster cophenetic distance cut at {args.threshold:.2f}",
                "cluster_rows": cluster_rows,
            },
        )
        run_rows.append(row)
        print(f"{run_id}: {len(sample)} neurons -> {len(cluster_rows)} clusters", flush=True)

    summary = pd.DataFrame(run_rows)
    if len(summary) != 16:
        raise ValueError(f"Expected 16 runs, found {len(summary)}.")
    summary.to_csv(args.output_dir / "advisor_2026_09_hellinger_clustering_summary.csv", index=False)
    result = {
        "status": "complete",
        "runs": 16,
        "scope": "pilot; exact clustering of sampled neurons, not full-neuron clustering",
        "probability_order": ["V", "T", "M", "W"],
        "distance": "Hellinger",
        "linkage": "average",
        "threshold": args.threshold,
        "maximum_neurons_per_run": args.maximum_neurons,
        "sampling": "deterministic proportional layer-stratified sampling without replacement",
        "seed": args.seed,
        "why_not_full": (
            "Standard average linkage requires O(N^2) pairwise storage/time. Full runs contain 147,456–352,256 "
            "neurons, so even the float64 condensed distance vector alone requires tens to hundreds of GiB per run."
        ),
        "runs_detail": run_rows,
    }
    _write_json(args.output_dir / "advisor_2026_09_hellinger_clustering.json", result)
    table = "\n".join(
        f"| {row['model_label']} | {row['dataset_label']} | {row['sampled_neuron_count']:,} | "
        f"{row['cluster_count']:,} | {row['largest_cluster_proportion']:.1%} | "
        f"{row['full_condensed_distance_gib']:.1f} GiB |"
        for row in run_rows
    )
    markdown = f"""# Hellinger average-linkage neuron clustering pilot

This is an explicitly sampled pilot, not a full-neuron clustering result. Within each model × dataset run, at most
{args.maximum_neurons:,} real neurons are selected without replacement using deterministic proportional
layer-stratified sampling. Their exact all-pairs Hellinger distances are clustered by standard average linkage and cut
at `d_H = {args.threshold:.2f}`. Cluster IDs are canonicalized by decreasing cluster size.

| Model | Dataset | Sampled neurons | Clusters | Largest cluster | Full distance storage |
|---|---|---:|---:|---:|---:|
{table}

The last column is the float64 condensed distance vector alone and excludes linkage working memory. Per-run sampled
assignments, linkage matrices, cluster centroids and integrity hashes are stored in the sibling run directories.
Presentation and any full-scale approximation method remain deliberately unspecified.
"""
    (args.output_dir / "advisor_2026_09_hellinger_clustering.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "complete", "runs": 16}, indent=2))


if __name__ == "__main__":
    main()
