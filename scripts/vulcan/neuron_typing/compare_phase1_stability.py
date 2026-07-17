"""Compare corrected Phase-1 score rankings and masks across sample sizes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SCORES = [
    "q_visual", "q_text", "q_multimodal", "q_unknown",
    "r_visual", "r_text", "r_multimodal", "r_unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--ratios", default="0.05,0.2,0.3,0.5")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    small = pd.read_parquet(args.small)
    large = pd.read_parquet(args.large)
    keys = ["layer", "neuron_idx"]
    merged = small.merge(large, on=keys, suffixes=("_small", "_large"), validate="one_to_one")
    alive = ~(merged["is_dead_small"].astype(bool) | merged["is_dead_large"].astype(bool))
    merged = merged[alive]

    ratios = [float(value) for value in args.ratios.split(",")]
    result = {
        "small": args.small,
        "large": args.large,
        "matched_alive_neurons": int(len(merged)),
        "scores": {},
    }
    for score in SCORES:
        correlation = merged[f"{score}_small"].corr(merged[f"{score}_large"], method="spearman")
        score_result = {"spearman": None if pd.isna(correlation) else float(correlation), "masks": {}}
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
        result["scores"][score] = score_result

    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
