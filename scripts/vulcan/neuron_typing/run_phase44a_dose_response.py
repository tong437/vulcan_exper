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

"""P4.4a causal dose response with global and q-stratified random controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
from dataset_guard import normalize_image_id  # noqa: E402
from evaluate_vqa import load_binary_records  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    build_score_vector,
    infer_score_columns,
    read_score_table,
    select_k_indices_deterministic,
)


DEFAULT_RATIOS = (0.01, 0.025, 0.05, 0.10, 0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run P4.4a mapping-high causal dose response.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mapping_metrics", required=True)
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument(
        "--main_vqa",
        action="append",
        required=True,
        help="Repeatable NAME=POPE_FILE for the cross-task main dose curve.",
    )
    parser.add_argument("--control_vqa", required=True, help="NAME=POPE_FILE for 20-seed control distributions.")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--ratios", default="0.01,0.025,0.05,0.10,0.15")
    parser.add_argument("--control_seed_count", type=int, default=20)
    parser.add_argument("--q_bins", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--control_bootstrap_samples", type=int, default=0)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.01)
    parser.add_argument("--max_yes_ratio_shift", type=float, default=0.05)
    parser.add_argument("--min_dose_spearman", type=float, default=0.8)
    parser.add_argument("--min_enriched_ratios", type=int, default=3)
    parser.add_argument("--max_empirical_p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2044)
    parser.add_argument("--stage", choices=["build", "main", "controls", "summarize", "all"], default="all")
    return parser.parse_args()


def parse_ratios(value: str) -> list[float]:
    ratios = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not ratios or any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError("ratios must be comma-separated values in (0, 1].")
    return ratios


def parse_named_files(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=FILE, got {value!r}.")
        name, path = value.split("=", maxsplit=1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid task name: {name!r}")
        if name in result:
            raise ValueError(f"Duplicate task name: {name}")
        result[name] = path
    return result


def ratio_label(ratio: float) -> str:
    """Return an identifier-safe ratio label in basis points."""
    return f"p{round(ratio * 10000):04d}"


def qstrat_column(ratio: float, seed: int) -> str:
    return f"qstrat_{ratio_label(ratio)}_s{seed:02d}"


def q_condition(ratio: float) -> str:
    return f"multimodal:{ratio:g}"


def mapping_condition(ratio: float) -> str:
    return f"matched_score:mapping_signal:highest:{q_condition(ratio)}"


def global_random_condition(ratio: float, seed: int) -> str:
    return f"matched_random:{q_condition(ratio)}:seed{seed}"


def qstrat_condition(ratio: float, seed: int) -> str:
    return f"matched_score:{qstrat_column(ratio, seed)}:highest:{q_condition(ratio)}"


def build_control_conditions(ratios: list[float], control_seeds: list[int]) -> list[str]:
    """Build controls together with every referenced exact-count q mask."""
    conditions = []
    for ratio in ratios:
        conditions.extend((q_condition(ratio), mapping_condition(ratio)))
        conditions.extend(global_random_condition(ratio, seed) for seed in control_seeds)
        conditions.extend(qstrat_condition(ratio, seed) for seed in control_seeds)
    return conditions


def _q_rank_bins(q_values: np.ndarray, q_bins: int) -> np.ndarray:
    neuron_ids = np.arange(len(q_values))
    order = np.lexsort((neuron_ids, q_values))
    bins = np.empty(len(q_values), dtype=np.int64)
    bins[order] = np.minimum(q_bins - 1, np.arange(len(q_values)) * q_bins // len(q_values))
    return bins


def _stable_rng_seed(base_seed: int, ratio: float, seed: int, layer: int, bin_index: int) -> int:
    payload = f"{base_seed}|{ratio:.12g}|{seed}|{layer}|{bin_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def build_qstratified_control_table(
    score_table: pd.DataFrame,
    *,
    ratios: list[float],
    control_seeds: list[int],
    q_bins: int,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add exact-count controls matching mapping-high's layerwise q-rank histogram."""
    if q_bins < 2:
        raise ValueError("q_bins must be at least two.")
    table = score_table.copy()
    layer_col, neuron_col, _, _ = infer_score_columns(table, None)
    required = {"mapping_signal", "q_multimodal_rank_value"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Phase-4 score table is missing columns: {missing}")

    control_columns = [qstrat_column(ratio, seed) for ratio in ratios for seed in control_seeds]
    for column in control_columns:
        if column in table:
            raise ValueError(f"Refusing to replace an existing q-stratified column: {column}")
    control_frame = pd.DataFrame(False, index=table.index, columns=control_columns, dtype=bool)
    table = pd.concat((table, control_frame), axis=1)

    metadata: dict[str, Any] = {
        "ratios": ratios,
        "control_seeds": control_seeds,
        "q_bins": q_bins,
        "base_seed": base_seed,
        "controls": {},
    }
    for layer_value, group in table.groupby(layer_col, sort=True):
        layer = int(layer_value)
        ordered_group = group.sort_values(neuron_col)
        neuron_ids = ordered_group[neuron_col].to_numpy(dtype=np.int64)
        dim = int(neuron_ids.max()) + 1
        if len(neuron_ids) != dim or not np.array_equal(neuron_ids, np.arange(dim)):
            raise ValueError(f"Layer {layer} does not have dense neuron ids [0, {dim}).")
        mapping_scores = build_score_vector(ordered_group, neuron_col, "mapping_signal", dim)
        if not bool(torch.isfinite(mapping_scores).all()):
            raise ValueError(f"Layer {layer} has non-finite mapping_signal.")
        q_values = ordered_group["q_multimodal_rank_value"].to_numpy(dtype=np.float64)
        if not np.isfinite(q_values).all():
            raise ValueError(f"Layer {layer} has non-finite q ranking values.")
        q_bin_ids = _q_rank_bins(q_values, q_bins)

        for ratio in ratios:
            selected_count = max(1, math.ceil(dim * ratio))
            mapping_selected, _ = select_k_indices_deterministic(
                mapping_scores,
                selected_count,
                neuron_ids=torch.arange(dim, dtype=torch.long),
            )
            mapping_selected_np = mapping_selected.numpy()
            target_bin_counts = np.bincount(q_bin_ids[mapping_selected_np], minlength=q_bins)
            ratio_meta = metadata["controls"].setdefault(
                ratio_label(ratio),
                {
                    "ratio": ratio,
                    "per_layer_selected": {},
                    "per_layer_mapping_q_bin_counts": {},
                    "columns": {},
                },
            )
            ratio_meta["per_layer_selected"][str(layer)] = selected_count
            ratio_meta["per_layer_mapping_q_bin_counts"][str(layer)] = target_bin_counts.tolist()

            layer_index = ordered_group.index.to_numpy()
            for seed in control_seeds:
                selected_parts = []
                for bin_index, count in enumerate(target_bin_counts.tolist()):
                    if count == 0:
                        continue
                    candidates = np.flatnonzero(q_bin_ids == bin_index)
                    if count > len(candidates):
                        raise RuntimeError(
                            f"q-stratified request exceeds bin size: layer={layer}, bin={bin_index}."
                        )
                    rng = np.random.default_rng(
                        _stable_rng_seed(base_seed, ratio, seed, layer, bin_index)
                    )
                    selected_parts.append(rng.choice(candidates, size=count, replace=False))
                selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
                if len(selected) != selected_count or len(np.unique(selected)) != selected_count:
                    raise RuntimeError("q-stratified control did not produce the requested unique count.")
                actual_counts = np.bincount(q_bin_ids[selected], minlength=q_bins)
                if not np.array_equal(actual_counts, target_bin_counts):
                    raise RuntimeError("q-stratified control does not match the target q-bin histogram.")
                column = qstrat_column(ratio, seed)
                table.loc[layer_index[selected], column] = True
                column_meta = ratio_meta["columns"].setdefault(
                    column,
                    {"per_layer_q_bin_counts": {}, "per_layer_selected": {}},
                )
                column_meta["per_layer_q_bin_counts"][str(layer)] = actual_counts.tolist()
                column_meta["per_layer_selected"][str(layer)] = selected_count

    metadata["verification"] = {}
    for ratio in ratios:
        expected_by_layer = metadata["controls"][ratio_label(ratio)]["per_layer_selected"]
        for seed in control_seeds:
            column = qstrat_column(ratio, seed)
            actual = {
                str(int(layer)): int(group[column].sum())
                for layer, group in table.groupby(layer_col, sort=True)
            }
            matches = actual == expected_by_layer
            metadata["verification"][column] = {
                "counts_match": matches,
                "per_layer_selected": actual,
            }
            if not matches:
                raise RuntimeError(f"Control column {column} failed exact-count verification.")
    return table, metadata


def _atomic_parquet(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    table.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _write_test_subset(
    source_path: str,
    image_root: str | None,
    test_image_ids: set[str],
    output_path: Path,
) -> int:
    records = load_binary_records(source_path, image_root)
    selected = [
        record for record in records if normalize_image_id(record["images"][0]) in test_image_ids
    ]
    selected_ids = {normalize_image_id(record["images"][0]) for record in selected}
    missing = test_image_ids - selected_ids
    if missing:
        raise ValueError(
            f"{source_path} is missing {len(missing)} Phase-4 test images; examples: {sorted(missing)[:5]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in selected:
            output.write(
                json.dumps(
                    {
                        "question_id": record["question_id"],
                        "image": record["images"][0],
                        "question": record["question"],
                        "answer": record["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(selected)


def _evaluation_complete(path: Path, required_conditions: set[str]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("complete")) and required_conditions.issubset(payload.get("metrics", {}))


def _run_evaluation(
    *,
    config: str,
    score_file: Path,
    subset_file: Path,
    output_file: Path,
    conditions: list[str],
    batch_size: int,
    bootstrap_samples: int,
    calibration_manifest: str | None,
    typing_manifest: str | None,
    seed: int,
) -> None:
    required = {"none", *conditions}
    if _evaluation_complete(output_file, required):
        print(f"Skipping complete P4.4a evaluation: {output_file}", flush=True)
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pope.py"),
        "--config",
        config,
        "--score_file",
        str(score_file),
        "--pope_file",
        str(subset_file),
        "--output_file",
        str(output_file),
        "--batch_size",
        str(batch_size),
        "--bootstrap_samples",
        str(bootstrap_samples),
        "--seed",
        str(seed),
        "--max_image_repeat",
        "6",
        "--allow_excessive_image_repeats",
    ]
    for condition in conditions:
        command.extend(("--ablation", condition))
    if calibration_manifest:
        command.extend(("--calibration_manifest", calibration_manifest))
    if typing_manifest:
        command.extend(("--typing_manifest", typing_manifest))
    if calibration_manifest and typing_manifest:
        command.append("--require_data_isolation")
    if output_file.exists():
        command.append("--resume")
    print(f"Running P4.4a evaluation: {output_file}", flush=True)
    subprocess.run(command, check=True)


def _condition_delta(metrics: dict[str, Any], condition: str, metric: str) -> float:
    condition_row = metrics[condition]
    delta_key = f"delta_{metric}"
    if delta_key in condition_row:
        return float(condition_row[delta_key])
    return float(condition_row[metric] - metrics["none"][metric])


def _control_distribution(
    metrics: dict[str, Any],
    conditions: list[str],
    mapping_delta: float,
) -> dict[str, Any]:
    deltas = np.asarray([_condition_delta(metrics, condition, "accuracy") for condition in conditions])
    empirical_p = float((1 + np.sum(deltas <= mapping_delta)) / (len(deltas) + 1))
    return {
        "conditions": conditions,
        "delta_accuracy_values": deltas.tolist(),
        "mean_delta_accuracy": float(deltas.mean()),
        "median_delta_accuracy": float(np.median(deltas)),
        "min_delta_accuracy": float(deltas.min()),
        "max_delta_accuracy": float(deltas.max()),
        "mapping_minus_control_mean": float(mapping_delta - deltas.mean()),
        "causal_enrichment": float(deltas.mean() - mapping_delta),
        "empirical_p_mapping_more_damaging": empirical_p,
        "yes_collapse_frequency": float(
            np.mean(
                [
                    metrics[condition]["yes_ratio"] >= 0.95 or metrics[condition]["yes_ratio"] <= 0.05
                    for condition in conditions
                ]
            )
        ),
    }


def summarize_results(
    *,
    main_files: dict[str, Path],
    control_file: Path,
    ratios: list[float],
    control_seeds: list[int],
    max_accuracy_drop: float,
    max_yes_ratio_shift: float,
    min_dose_spearman: float,
    min_enriched_ratios: int,
    max_empirical_p: float,
) -> dict[str, Any]:
    main_payloads = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in main_files.items()
    }
    control_payload = json.loads(control_file.read_text(encoding="utf-8"))
    main_summary: dict[str, Any] = {}
    dose_correlations = []
    for task_name, payload in main_payloads.items():
        metrics = payload["metrics"]
        task_rows = {}
        mapping_damage = []
        for ratio in ratios:
            mapping_name = mapping_condition(ratio)
            q_name = q_condition(ratio)
            mapping_delta = _condition_delta(metrics, mapping_name, "accuracy")
            q_delta = _condition_delta(metrics, q_name, "accuracy")
            yes_shift = float(metrics[mapping_name]["yes_ratio"] - metrics["none"]["yes_ratio"])
            task_rows[f"{ratio:g}"] = {
                "mapping": {
                    "delta_accuracy": mapping_delta,
                    "delta_f1": _condition_delta(metrics, mapping_name, "f1"),
                    "yes_ratio": float(metrics[mapping_name]["yes_ratio"]),
                    "delta_yes_ratio": yes_shift,
                    "accuracy_ci95": metrics[mapping_name].get("delta_accuracy_ci95"),
                },
                "q_multimodal": {
                    "delta_accuracy": q_delta,
                    "delta_f1": _condition_delta(metrics, q_name, "f1"),
                    "yes_ratio": float(metrics[q_name]["yes_ratio"]),
                },
            }
            mapping_damage.append(-mapping_delta)
        correlation = float(spearmanr(ratios, mapping_damage).statistic)
        dose_correlations.append(correlation)
        unsafe_ratios = [
            ratio
            for ratio in ratios
            if (
                task_rows[f"{ratio:g}"]["mapping"]["delta_accuracy"] < -max_accuracy_drop
                or abs(task_rows[f"{ratio:g}"]["mapping"]["delta_yes_ratio"]) > max_yes_ratio_shift
            )
        ]
        collapse_ratios = [
            ratio
            for ratio in ratios
            if (
                task_rows[f"{ratio:g}"]["mapping"]["yes_ratio"] >= 0.95
                or task_rows[f"{ratio:g}"]["mapping"]["yes_ratio"] <= 0.05
            )
        ]
        main_summary[task_name] = {
            "ratios": task_rows,
            "dose_spearman_accuracy_damage": correlation,
            "first_unsafe_ratio": min(unsafe_ratios) if unsafe_ratios else None,
            "first_yes_collapse_ratio": min(collapse_ratios) if collapse_ratios else None,
        }

    control_metrics = control_payload["metrics"]
    control_summary = {}
    enriched_ratios = []
    for ratio in ratios:
        mapping_name = mapping_condition(ratio)
        mapping_delta = _condition_delta(control_metrics, mapping_name, "accuracy")
        global_conditions = [global_random_condition(ratio, seed) for seed in control_seeds]
        qstrat_conditions = [qstrat_condition(ratio, seed) for seed in control_seeds]
        global_result = _control_distribution(control_metrics, global_conditions, mapping_delta)
        qstrat_result = _control_distribution(control_metrics, qstrat_conditions, mapping_delta)
        enriched = (
            global_result["empirical_p_mapping_more_damaging"] <= max_empirical_p
            and qstrat_result["empirical_p_mapping_more_damaging"] <= max_empirical_p
        )
        if enriched:
            enriched_ratios.append(ratio)
        control_summary[f"{ratio:g}"] = {
            "mapping_delta_accuracy": mapping_delta,
            "global_random": global_result,
            "q_stratified_random": qstrat_result,
            "enriched_against_both_controls": enriched,
        }

    median_dose_spearman = float(np.median(dose_correlations))
    return {
        "main_dose_curve": main_summary,
        "control_distributions": control_summary,
        "gates": {
            "monotonic_dose_response": {
                "passed": median_dose_spearman >= min_dose_spearman,
                "per_task_spearman": {
                    task: row["dose_spearman_accuracy_damage"] for task, row in main_summary.items()
                },
                "median_spearman": median_dose_spearman,
                "minimum_median_spearman": min_dose_spearman,
            },
            "mapping_causal_enrichment": {
                "passed": len(enriched_ratios) >= min_enriched_ratios,
                "enriched_ratios": enriched_ratios,
                "enriched_ratio_count": len(enriched_ratios),
                "minimum_enriched_ratios": min_enriched_ratios,
                "maximum_empirical_p": max_empirical_p,
            },
            "structural_pruning_allowed": False,
        },
        "interpretation": (
            "P4.4a tests causal localization only. Passing its gates does not turn inverse mapping "
            "predictability into a redundancy score and never authorizes structural pruning."
        ),
    }


def run_phase44a(args: argparse.Namespace) -> dict[str, Any]:
    ratios = parse_ratios(args.ratios)
    if args.control_seed_count < 1:
        raise ValueError("control_seed_count must be positive.")
    minimum_controls = math.ceil(1.0 / args.max_empirical_p) - 1
    if args.control_seed_count < minimum_controls:
        raise ValueError(
            f"At least {minimum_controls} controls are required to resolve max_empirical_p={args.max_empirical_p}."
        )
    control_seeds = list(range(1, args.control_seed_count + 1))
    main_vqa = parse_named_files(args.main_vqa)
    control_vqa = parse_named_files([args.control_vqa])
    control_name, control_source = next(iter(control_vqa.items()))
    if control_name not in main_vqa:
        raise ValueError("The control_vqa task must also appear in main_vqa.")

    mapping_metrics = json.loads(Path(args.mapping_metrics).read_text(encoding="utf-8"))
    if not mapping_metrics["gates"]["phase4_group_causal_ablation_allowed"]:
        raise RuntimeError("Phase-4 Gate A/B did not pass; P4.4a is forbidden.")
    source_score_file = Path(mapping_metrics["outputs"]["augmented_score_file"])
    split_payload = json.loads((Path(args.activation_dir) / "splits.json").read_text(encoding="utf-8"))
    test_image_ids = {
        image_id for image_id, split in split_payload["image_to_split"].items() if split == "test"
    }

    output_dir = Path(args.output_dir)
    score_file = output_dir / "controls" / "neuron_scores_phase44a.parquet"
    control_metadata_file = output_dir / "controls" / "qstratified_controls.json"
    expected_control_config = {
        "ratios": ratios,
        "control_seeds": control_seeds,
        "q_bins": args.q_bins,
        "base_seed": args.seed,
        "source_score_file": str(source_score_file),
    }
    if args.stage in {"build", "all"}:
        if score_file.exists() and control_metadata_file.exists():
            existing_metadata = json.loads(control_metadata_file.read_text(encoding="utf-8"))
            existing_config = {
                key: existing_metadata.get(key) for key in expected_control_config
            }
            if existing_config != expected_control_config:
                differing = sorted(
                    key
                    for key in expected_control_config
                    if existing_config.get(key) != expected_control_config[key]
                )
                raise ValueError(f"Existing P4.4a control table configuration mismatch: {differing}")
            print(f"Skipping existing P4.4a control table: {score_file}", flush=True)
        else:
            augmented = read_score_table(source_score_file)
            control_table, control_metadata = build_qstratified_control_table(
                augmented,
                ratios=ratios,
                control_seeds=control_seeds,
                q_bins=args.q_bins,
                base_seed=args.seed,
            )
            control_metadata["source_score_file"] = str(source_score_file)
            _atomic_parquet(control_table, score_file)
            control_metadata_file.parent.mkdir(parents=True, exist_ok=True)
            control_metadata_file.write_text(
                json.dumps(control_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    if not score_file.exists():
        raise FileNotFoundError(f"P4.4a control score file is missing: {score_file}")

    subset_files = {}
    for name, source in main_vqa.items():
        subset = output_dir / "test_subsets" / f"{name}.jsonl"
        _write_test_subset(source, args.image_root, test_image_ids, subset)
        subset_files[name] = subset
    if Path(control_source) != Path(main_vqa[control_name]):
        raise ValueError("control_vqa and the same-named main_vqa must reference the same source file.")

    main_conditions = [
        condition
        for ratio in ratios
        for condition in (q_condition(ratio), mapping_condition(ratio))
    ]
    main_files = {name: output_dir / "main" / f"{name}.json" for name in main_vqa}
    if args.stage in {"main", "all"}:
        for task_index, name in enumerate(main_vqa):
            _run_evaluation(
                config=args.config,
                score_file=score_file,
                subset_file=subset_files[name],
                output_file=main_files[name],
                conditions=main_conditions,
                batch_size=args.batch_size,
                bootstrap_samples=args.bootstrap_samples,
                calibration_manifest=args.calibration_manifest,
                typing_manifest=args.typing_manifest,
                seed=args.seed + task_index,
            )

    control_conditions = build_control_conditions(ratios, control_seeds)
    control_file = output_dir / "controls" / f"{control_name}.json"
    if args.stage in {"controls", "all"}:
        _run_evaluation(
            config=args.config,
            score_file=score_file,
            subset_file=subset_files[control_name],
            output_file=control_file,
            conditions=control_conditions,
            batch_size=args.batch_size,
            bootstrap_samples=args.control_bootstrap_samples,
            calibration_manifest=args.calibration_manifest,
            typing_manifest=args.typing_manifest,
            seed=args.seed + 100,
        )

    if args.stage in {"summarize", "all"}:
        missing_outputs = [str(path) for path in (*main_files.values(), control_file) if not path.exists()]
        if missing_outputs:
            raise FileNotFoundError(f"Cannot summarize P4.4a; missing outputs: {missing_outputs}")
        summary = summarize_results(
            main_files=main_files,
            control_file=control_file,
            ratios=ratios,
            control_seeds=control_seeds,
            max_accuracy_drop=args.max_accuracy_drop,
            max_yes_ratio_shift=args.max_yes_ratio_shift,
            min_dose_spearman=args.min_dose_spearman,
            min_enriched_ratios=args.min_enriched_ratios,
            max_empirical_p=args.max_empirical_p,
        )
        result = {
            "complete": True,
            "config": vars(args),
            "ratios": ratios,
            "control_seeds": control_seeds,
            "test_image_count": len(test_image_ids),
            "score_file": str(score_file),
            "control_metadata": str(control_metadata_file),
            "main_files": {name: str(path) for name, path in main_files.items()},
            "control_file": str(control_file),
            **summary,
        }
        output_path = output_dir / "phase44a_dose_response.json"
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result["gates"], indent=2, ensure_ascii=False))
        return result

    result = {
        "complete": True,
        "stage": args.stage,
        "output_dir": str(output_dir),
        "structural_pruning_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "phase44a_pipeline.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def main() -> None:
    result = run_phase44a(parse_args())
    if "gates" not in result:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
