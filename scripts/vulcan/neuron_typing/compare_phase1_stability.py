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

"""Compare corrected Phase-1 score rankings and masks across sample sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCORES = [
    "q_visual",
    "q_text",
    "q_multimodal",
    "q_unknown",
    "r_visual",
    "r_text",
    "r_multimodal",
    "r_unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--ratios", default="0.05,0.2,0.3,0.5")
    parser.add_argument(
        "--bands",
        default="0.05:0.2",
        help="Comma-separated rank bands expressed as START:END (default: 0.05:0.2).",
    )
    parser.add_argument("--small_config", help="Activation config for the small run.")
    parser.add_argument("--large_config", help="Activation config for the large run.")
    parser.add_argument("--small_manifest", help="Typing manifest for the small run.")
    parser.add_argument("--large_manifest", help="Typing manifest for the large run.")
    parser.add_argument(
        "--require_prefix_validation",
        action="store_true",
        help="Require proof that both runs used identical calibration and that small is a strict prefix of large.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_run_file(score_file: str | Path, relative_path: str) -> Path:
    """Infer a run artifact from ``<run>/scores/<score-file>``."""
    return Path(score_file).parent.parent / relative_path


def resolve_recorded_path(value: str, config_path: str | Path) -> Path:
    """Resolve paths recorded relative to either cwd or the run directory."""
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path.resolve()

    run_relative = Path(config_path).resolve().parent.parent / path
    return run_relative.resolve()


def validate_prefix_experiment(
    small_config_path: str | Path,
    large_config_path: str | Path,
    small_manifest_path: str | Path,
    large_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify same calibration/configuration and an exact small-data prefix."""
    small_config = load_json(small_config_path)
    large_config = load_json(large_config_path)
    small_manifest = load_json(small_manifest_path)
    large_manifest = load_json(large_manifest_path)

    controlled_fields = [
        "model_name",
        "sample_offset",
        "threshold_mode",
        "t_visual",
        "t_text",
        "quantile_idx_visual",
        "quantile_idx_text",
        "visual_ratio",
        "visual_min_count",
        "text_ratio",
        "text_min_count",
        "top_k",
        "sample_score_top_m",
        "image_token_id",
        "num_mlp_layers",
        "intermediate_size",
    ]
    mismatched_fields = {
        field: {"small": small_config.get(field), "large": large_config.get(field)}
        for field in controlled_fields
        if small_config.get(field) != large_config.get(field)
    }
    if mismatched_fields:
        raise ValueError(f"Small/large activation configurations differ: {mismatched_fields}")

    small_calibration = resolve_recorded_path(small_config["quantile_path"], small_config_path)
    large_calibration = resolve_recorded_path(large_config["quantile_path"], large_config_path)
    if not small_calibration.is_file() or not large_calibration.is_file():
        raise FileNotFoundError(f"Calibration file missing: small={small_calibration}, large={large_calibration}")
    small_calibration_hash = sha256_file(small_calibration)
    large_calibration_hash = sha256_file(large_calibration)
    if small_calibration_hash != large_calibration_hash:
        raise ValueError("Small and large runs did not use byte-identical calibration files.")

    manifest_fields = ["dataset", "tokenized_path", "role"]
    manifest_mismatches = {
        field: {"small": small_manifest.get(field), "large": large_manifest.get(field)}
        for field in manifest_fields
        if small_manifest.get(field) != large_manifest.get(field)
    }
    if manifest_mismatches:
        raise ValueError(f"Small/large typing manifests differ: {manifest_mismatches}")

    small_count = int(small_manifest["num_rows"])
    large_count = int(large_manifest["num_rows"])
    if not 0 < small_count < large_count:
        raise ValueError(f"Expected a strict sample-size increase, got {small_count} and {large_count}.")
    for field in ["source_indices", "row_image_ids"]:
        small_values = small_manifest[field]
        large_values = large_manifest[field]
        if len(small_values) != small_count or len(large_values) != large_count:
            raise ValueError(f"Manifest {field!r} length does not match num_rows.")
        if small_values != large_values[:small_count]:
            raise ValueError(f"Small manifest is not an exact prefix of large manifest for {field!r}.")

    if int(small_config["actual_samples"]) != small_count:
        raise ValueError("Small config actual_samples does not match its manifest.")
    if int(large_config["actual_samples"]) != large_count:
        raise ValueError("Large config actual_samples does not match its manifest.")

    return {
        "validated": True,
        "same_controlled_configuration": True,
        "same_calibration_sha256": True,
        "calibration_sha256": small_calibration_hash,
        "calibration_path_small": str(small_calibration),
        "calibration_path_large": str(large_calibration),
        "strict_typing_prefix": True,
        "small_samples": small_count,
        "large_samples": large_count,
        "typing_offset": small_config["sample_offset"],
        "checked_prefix_fields": ["source_indices", "row_image_ids"],
    }


def ranked_mask(table: pd.DataFrame, score: str, ratio: float) -> set[tuple[int, int]]:
    selected: set[tuple[int, int]] = set()
    for layer, group in table.groupby("layer"):
        group = group[~group["is_dead"].astype(bool)].copy()
        secondary_name = "r_multimodal" if score != "r_multimodal" else "q_multimodal"
        secondary = group[secondary_name].fillna(-np.inf).to_numpy()
        primary = group[score].fillna(-np.inf).to_numpy()
        neuron_ids = group["neuron_idx"].to_numpy(dtype=int)
        order = np.lexsort((neuron_ids, -secondary, -primary))
        k = max(1, math.ceil(len(group) * ratio))
        selected.update((int(layer), int(neuron_ids[idx])) for idx in order[:k])
    return selected


def ranked_band(table: pd.DataFrame, score: str, start: float, end: float) -> set[tuple[int, int]]:
    """Select a per-layer rank interval using the same deterministic ordering as ranked_mask."""
    if not 0 <= start < end <= 1:
        raise ValueError(f"Invalid rank band {start}:{end}; expected 0 <= start < end <= 1.")

    selected: set[tuple[int, int]] = set()
    for layer, group in table.groupby("layer"):
        group = group[~group["is_dead"].astype(bool)].copy()
        secondary_name = "r_multimodal" if score != "r_multimodal" else "q_multimodal"
        secondary = group[secondary_name].fillna(-np.inf).to_numpy()
        primary = group[score].fillna(-np.inf).to_numpy()
        neuron_ids = group["neuron_idx"].to_numpy(dtype=int)
        order = np.lexsort((neuron_ids, -secondary, -primary))
        begin = math.ceil(len(group) * start)
        stop = math.ceil(len(group) * end)
        selected.update((int(layer), int(neuron_ids[idx])) for idx in order[begin:stop])
    return selected


def summarize_values(values: dict[int, float | None]) -> dict[str, Any]:
    finite = [value for value in values.values() if value is not None and np.isfinite(value)]
    if not finite:
        return {"mean": None, "median": None, "min": None, "max": None, "per_layer": values}

    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "per_layer": {str(layer): value for layer, value in values.items()},
    }


def per_layer_spearman(merged: pd.DataFrame, score: str) -> dict[str, Any]:
    correlations: dict[int, float | None] = {}
    for layer, group in merged.groupby("layer"):
        correlation = group[f"{score}_small"].corr(group[f"{score}_large"], method="spearman")
        correlations[int(layer)] = None if pd.isna(correlation) else float(correlation)
    return summarize_values(correlations)


def per_layer_jaccard(
    small_mask: set[tuple[int, int]], large_mask: set[tuple[int, int]], layers: list[int]
) -> dict[str, Any]:
    overlaps: dict[int, float | None] = {}
    for layer in layers:
        small_layer = {neuron for mask_layer, neuron in small_mask if mask_layer == layer}
        large_layer = {neuron for mask_layer, neuron in large_mask if mask_layer == layer}
        union = small_layer | large_layer
        overlaps[layer] = len(small_layer & large_layer) / len(union) if union else 1.0
    return summarize_values(overlaps)


def main() -> None:
    args = parse_args()
    small_path = Path(args.small)
    large_path = Path(args.large)
    small = pd.read_parquet(small_path)
    large = pd.read_parquet(large_path)
    keys = ["layer", "neuron_idx"]
    outer = small[keys].merge(large[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    if not bool((outer["_merge"] == "both").all()):
        raise ValueError("Small and large score tables do not contain identical neuron keys.")

    merged = small.merge(large, on=keys, suffixes=("_small", "_large"), validate="one_to_one")
    dead_mismatch = merged["is_dead_small"].astype(bool) != merged["is_dead_large"].astype(bool)
    if bool(dead_mismatch.any()):
        raise ValueError(f"Dead-neuron masks differ for {int(dead_mismatch.sum())} neurons.")
    alive = ~(merged["is_dead_small"].astype(bool) | merged["is_dead_large"].astype(bool))
    merged = merged[alive]

    ratios = [float(value) for value in args.ratios.split(",")]
    if any(not 0 < ratio <= 1 for ratio in ratios):
        raise ValueError("Each --ratios entry must satisfy 0 < ratio <= 1.")
    bands = []
    for value in args.bands.split(","):
        bounds = value.split(":")
        if len(bounds) != 2:
            raise ValueError("Each --bands entry must have the form START:END.")
        band = (float(bounds[0]), float(bounds[1]))
        if not 0 <= band[0] < band[1] <= 1:
            raise ValueError(f"Invalid rank band {value}; expected 0 <= START < END <= 1.")
        bands.append(band)

    config_paths = (
        Path(args.small_config) if args.small_config else infer_run_file(small_path, "activations/config.json"),
        Path(args.large_config) if args.large_config else infer_run_file(large_path, "activations/config.json"),
    )
    manifest_paths = (
        Path(args.small_manifest)
        if args.small_manifest
        else infer_run_file(small_path, "activations/sample_manifest.json"),
        Path(args.large_manifest)
        if args.large_manifest
        else infer_run_file(large_path, "activations/sample_manifest.json"),
    )
    integrity_paths = [*config_paths, *manifest_paths]
    if all(path.is_file() for path in integrity_paths):
        integrity = validate_prefix_experiment(*config_paths, *manifest_paths)
    elif args.require_prefix_validation:
        missing = [str(path) for path in integrity_paths if not path.is_file()]
        raise FileNotFoundError(f"Prefix validation artifacts are missing: {missing}")
    else:
        integrity = {"validated": False, "reason": "config or manifest artifacts were not available"}

    layers = sorted(int(layer) for layer in merged["layer"].unique())
    result = {
        "small": args.small,
        "large": args.large,
        "data_integrity": integrity,
        "matched_alive_neurons": int(len(merged)),
        "dead_neurons": int(len(small) - len(merged)),
        "scores": {},
    }
    for score in SCORES:
        correlation = merged[f"{score}_small"].corr(merged[f"{score}_large"], method="spearman")
        score_result = {
            "spearman": None if pd.isna(correlation) else float(correlation),
            "spearman_per_layer": per_layer_spearman(merged, score),
            "masks": {},
            "bands": {},
        }
        for ratio in ratios:
            small_mask = ranked_mask(small, score, ratio)
            large_mask = ranked_mask(large, score, ratio)
            union = small_mask | large_mask
            intersection = small_mask & large_mask
            score_result["masks"][str(ratio)] = {
                "small_count": len(small_mask),
                "large_count": len(large_mask),
                "intersection": len(intersection),
                "jaccard": len(intersection) / len(union) if union else 1.0,
            }
        for start, end in bands:
            small_mask = ranked_band(small, score, start, end)
            large_mask = ranked_band(large, score, start, end)
            union = small_mask | large_mask
            intersection = small_mask & large_mask
            score_result["bands"][f"{start}:{end}"] = {
                "small_count": len(small_mask),
                "large_count": len(large_mask),
                "intersection": len(intersection),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "jaccard_per_layer": per_layer_jaccard(small_mask, large_mask, layers),
            }
        result["scores"][score] = score_result

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
