# Copyright 2026 the LlamaFactory team.
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

"""Localize the causal effect of the Phase-4 mapping top-1% set by layer group."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
from run_phase2_ablation import (  # noqa: E402
    build_score_vector,
    infer_score_columns,
    read_score_table,
    select_k_indices_deterministic,
)
from run_phase44a_dose_response import _run_evaluation, _write_test_subset, parse_named_files  # noqa: E402


BLOCKS = tuple(tuple(range(start, start + 4)) for start in range(0, 24, 4))
FA_LAYERS = (3, 7, 11, 15, 19, 23)
GDN_LAYERS = tuple(layer for layer in range(24) if layer not in FA_LAYERS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mapping top-1% layer-group causal localization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mapping_metrics", required=True)
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument("--vqa", action="append", required=True, help="Repeatable NAME=POPE_FILE.")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2045)
    parser.add_argument("--stage", choices=["build", "evaluate", "summarize", "all"], default="all")
    return parser.parse_args()


def mask_column(score_name: str, group_name: str) -> str:
    return f"{score_name}_top1_{group_name}"


def mask_condition(score_name: str, group_name: str) -> str:
    return f"mask:{mask_column(score_name, group_name)}"


def group_layers() -> dict[str, tuple[int, ...]]:
    groups = {f"block{index + 1}": block for index, block in enumerate(BLOCKS)}
    groups.update(
        {
            "fa": FA_LAYERS,
            "gdn": GDN_LAYERS,
            "fa_without23": tuple(layer for layer in FA_LAYERS if layer != 23),
            "layer23": (23,),
            "all": tuple(range(24)),
        }
    )
    return groups


def _mask_hash(table: pd.DataFrame, columns: list[str]) -> str:
    digest = hashlib.sha256()
    for column in columns:
        digest.update(column.encode())
        digest.update(table[column].to_numpy(dtype=np.bool_).tobytes())
    return digest.hexdigest()


def build_localization_table(
    score_table: pd.DataFrame,
    *,
    ratio: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1].")
    table = score_table.copy()
    layer_col, neuron_col, _, _ = infer_score_columns(table, None)
    required = {"mapping_signal", "q_multimodal_rank_value"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Mapping score table is missing columns: {missing}")
    layers = sorted(int(layer) for layer in table[layer_col].unique())
    if layers != list(range(24)):
        raise ValueError(f"Localization expects layers 0..23, got {layers}.")

    top_masks: dict[str, dict[int, np.ndarray]] = {"mapping": {}, "q": {}}
    cutoffs: dict[str, dict[str, Any]] = {"mapping": {}, "q": {}}
    score_columns = {"mapping": "mapping_signal", "q": "q_multimodal_rank_value"}
    for layer, group in table.groupby(layer_col, sort=True):
        layer = int(layer)
        ordered = group.sort_values(neuron_col)
        neuron_ids = ordered[neuron_col].to_numpy(dtype=np.int64)
        width = int(neuron_ids.max()) + 1
        if len(neuron_ids) != width or not np.array_equal(neuron_ids, np.arange(width)):
            raise ValueError(f"Layer {layer} does not have dense neuron ids [0, {width}).")
        selected_count = max(1, math.ceil(width * ratio))
        for score_name, score_column in score_columns.items():
            scores = build_score_vector(ordered, neuron_col, score_column, width)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError(f"Layer {layer} has non-finite {score_column} values.")
            selected, cutoff = select_k_indices_deterministic(
                scores,
                selected_count,
                neuron_ids=torch.arange(width, dtype=torch.long),
            )
            mask = np.zeros(width, dtype=np.bool_)
            mask[selected.numpy()] = True
            top_masks[score_name][layer] = mask
            cutoffs[score_name][str(layer)] = cutoff

    groups = group_layers()
    generated_columns: list[str] = []
    counts: dict[str, dict[str, Any]] = {}
    for score_name in ("mapping", "q"):
        for group_name, allowed_layers in groups.items():
            column = mask_column(score_name, group_name)
            allowed = set(allowed_layers)
            values = np.zeros(len(table), dtype=np.bool_)
            for layer, group in table.groupby(layer_col, sort=True):
                layer = int(layer)
                if layer not in allowed:
                    continue
                ordered_indices = group.sort_values(neuron_col).index.to_numpy()
                values[ordered_indices] = top_masks[score_name][layer]
            table[column] = values
            generated_columns.append(column)
            per_layer = {
                str(int(layer)): int(group[column].sum()) for layer, group in table.groupby(layer_col, sort=True)
            }
            counts[column] = {
                "selected": int(table[column].sum()),
                "layers": list(allowed_layers),
                "per_layer": per_layer,
            }

    for score_name in ("mapping", "q"):
        all_column = mask_column(score_name, "all")
        block_columns = [mask_column(score_name, f"block{index + 1}") for index in range(6)]
        block_sum = table[block_columns].sum(axis=1)
        if not np.array_equal(block_sum.to_numpy(), table[all_column].astype(np.int64).to_numpy()):
            raise RuntimeError(f"{score_name} block masks do not form an exact partition of all top-1% neurons.")
        fa = table[mask_column(score_name, "fa")].to_numpy()
        gdn = table[mask_column(score_name, "gdn")].to_numpy()
        all_mask = table[all_column].to_numpy()
        if np.any(fa & gdn) or not np.array_equal(fa | gdn, all_mask):
            raise RuntimeError(f"{score_name} FA/GDN masks do not partition the full top-1% mask.")
        layer23 = table[mask_column(score_name, "layer23")].to_numpy()
        fa_without23 = table[mask_column(score_name, "fa_without23")].to_numpy()
        if np.any(layer23 & fa_without23) or not np.array_equal(layer23 | fa_without23, fa):
            raise RuntimeError(f"{score_name} layer-23 mask does not partition the FA mask.")

    metadata = {
        "ratio": ratio,
        "layer_groups": {name: list(layers) for name, layers in groups.items()},
        "score_columns": score_columns,
        "cutoffs": cutoffs,
        "counts": counts,
        "generated_columns": generated_columns,
        "combined_mask_sha256": _mask_hash(table, generated_columns),
        "verification": {
            "blocks_partition_all": True,
            "fa_gdn_partition_all": True,
            "layer23_partitions_fa": True,
        },
    }
    return table, metadata


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _delta(metrics: dict[str, Any], condition: str, name: str) -> float:
    key = f"delta_{name}"
    return float(metrics[condition].get(key, metrics[condition][name] - metrics["none"][name]))


def summarize_localization(
    evaluation_files: dict[str, Path],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups = group_layers()
    for task, path in evaluation_files.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete"):
            raise ValueError(f"Incomplete localization evaluation: {path}")
        metrics = payload["metrics"]
        rows = {}
        for group_name in groups:
            mapping_name = mask_condition("mapping", group_name)
            q_name = mask_condition("q", group_name)
            selected = metadata["counts"][mask_column("mapping", group_name)]["selected"]
            mapping_delta = _delta(metrics, mapping_name, "accuracy")
            q_delta = _delta(metrics, q_name, "accuracy")
            rows[group_name] = {
                "selected_neurons": selected,
                "mapping_delta_accuracy": mapping_delta,
                "mapping_delta_f1": _delta(metrics, mapping_name, "f1"),
                "mapping_accuracy_ci95": metrics[mapping_name].get("delta_accuracy_ci95"),
                "q_delta_accuracy": q_delta,
                "mapping_minus_q_delta_accuracy": mapping_delta - q_delta,
                "mapping_damage_per_100_neurons": (-mapping_delta * 100 / selected if selected else None),
            }
        block_delta_sum = sum(rows[f"block{index}"]["mapping_delta_accuracy"] for index in range(1, 7))
        rows["block_additivity"] = {
            "sum_individual_block_delta_accuracy": block_delta_sum,
            "all_layers_delta_accuracy": rows["all"]["mapping_delta_accuracy"],
            "interaction_delta": rows["all"]["mapping_delta_accuracy"] - block_delta_sum,
        }
        output[task] = rows
    return {
        "complete": True,
        "tasks": output,
        "interpretation": (
            "Layer-group deltas localize the frozen mapping top-1% set. Per-neuron normalization is descriptive "
            "because FFN ablation is non-additive; no localization result authorizes pruning."
        ),
        "structural_pruning_allowed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    mapping_metrics = json.loads(Path(args.mapping_metrics).read_text(encoding="utf-8"))
    if not mapping_metrics["gates"]["phase4_group_causal_ablation_allowed"]:
        raise RuntimeError("Phase-4 Gate A/B did not pass; localization is forbidden.")
    source_score = Path(mapping_metrics["outputs"]["augmented_score_file"])
    output_dir = Path(args.output_dir)
    score_file = output_dir / "masks" / "neuron_scores_top1_localization.parquet"
    metadata_file = output_dir / "masks" / "top1_localization_masks.json"

    if args.stage in {"build", "all"}:
        if score_file.exists() or metadata_file.exists():
            if not (score_file.exists() and metadata_file.exists()):
                raise FileExistsError("Localization mask outputs are only partially present.")
            existing = json.loads(metadata_file.read_text(encoding="utf-8"))
            if float(existing["ratio"]) != args.ratio or existing["source_score_file"] != str(source_score):
                raise ValueError("Existing localization mask configuration mismatch.")
            print(f"Skipping existing localization masks: {score_file}", flush=True)
        else:
            table, metadata = build_localization_table(read_score_table(source_score), ratio=args.ratio)
            metadata["source_score_file"] = str(source_score)
            _atomic_parquet(table, score_file)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            metadata_file.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if not score_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("Build localization masks before evaluation.")

    named_vqa = parse_named_files(args.vqa)
    split_payload = json.loads((Path(args.activation_dir) / "splits.json").read_text(encoding="utf-8"))
    test_ids = {image for image, split in split_payload["image_to_split"].items() if split == "test"}
    subset_files = {}
    for name, source in named_vqa.items():
        subset = output_dir / "test_subsets" / f"{name}.jsonl"
        _write_test_subset(source, args.image_root, test_ids, subset)
        subset_files[name] = subset

    groups = list(group_layers())
    conditions = [mask_condition(score_name, group) for score_name in ("mapping", "q") for group in groups]
    evaluation_files = {name: output_dir / "evaluations" / f"{name}.json" for name in named_vqa}
    if args.stage in {"evaluate", "all"}:
        for index, name in enumerate(named_vqa):
            _run_evaluation(
                config=args.config,
                score_file=score_file,
                subset_file=subset_files[name],
                output_file=evaluation_files[name],
                conditions=conditions,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
                calibration_manifest=args.calibration_manifest,
                typing_manifest=args.typing_manifest,
                seed=args.seed + index,
            )

    if args.stage in {"summarize", "all"}:
        missing = [str(path) for path in evaluation_files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Cannot summarize localization; missing: {missing}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        result = {
            "config": vars(args),
            "score_file": str(score_file),
            "mask_metadata": str(metadata_file),
            "test_image_count": len(test_ids),
            "evaluation_files": {name: str(path) for name, path in evaluation_files.items()},
            **summarize_localization(evaluation_files, metadata),
        }
        output_path = output_dir / "top1_localization.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    result = {
        "complete": True,
        "stage": args.stage,
        "output_dir": str(output_dir),
        "structural_pruning_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    result = run(parse_args())
    if "tasks" not in result:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
