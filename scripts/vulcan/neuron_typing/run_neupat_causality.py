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

"""Run matched-count NeuPAT role causality on Caption, text-only, and POPE."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
COMPONENT_ROLES = ("language", "multimodal", "shared", "reserve")
PRIMARY_TARGET = "language_protection"
TARGET_COLUMNS = {
    PRIMARY_TARGET: "neupat_role_protect",
    **{role: f"neupat_{role}" for role in COMPONENT_ROLES},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuPAT role matched-count causal evaluations.")
    parser.add_argument("--caption_config", required=True)
    parser.add_argument("--text_config", required=True)
    parser.add_argument("--score_file", required=True, help="Combined score parquet containing NeuPAT booleans.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pope", action="append", default=[], help="Repeat NAME=FILE for POPE splits.")
    parser.add_argument(
        "--skip_pope",
        action="store_true",
        help="Run only the primary Caption/text causal gate; POPE can be added by resuming the same output directory.",
    )
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--calibration_manifest", required=True)
    parser.add_argument("--typing_manifest", required=True)
    parser.add_argument("--probe_manifest", action="append", default=[])
    parser.add_argument("--caption_max_samples", type=int, default=500)
    parser.add_argument("--caption_sample_offset", type=int, default=2500)
    parser.add_argument("--text_max_samples", type=int, default=500)
    parser.add_argument("--text_sample_offset", type=int, default=0)
    parser.add_argument("--pope_max_images", type=int, default=None)
    parser.add_argument("--control_seed_count", type=int, default=5)
    parser.add_argument(
        "--include_component_roles",
        action="store_true",
        help="Also evaluate the four exploratory component roles. The primary target remains language U shared.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_named_files(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=FILE, got {value!r}.")
        name, path = (part.strip() for part in value.split("=", maxsplit=1))
        if not name or not path or name in result:
            raise ValueError(f"Invalid or duplicate NAME=FILE value: {value!r}.")
        result[name] = path
    return result


def get_targets(include_component_roles: bool) -> tuple[str, ...]:
    return (PRIMARY_TARGET, *COMPONENT_ROLES) if include_component_roles else (PRIMARY_TARGET,)


def target_condition(target: str) -> str:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown NeuPAT causal target: {target!r}.")
    return f"mask:{TARGET_COLUMNS[target]}"


def build_conditions(control_seed_count: int, targets: tuple[str, ...] = (PRIMARY_TARGET,)) -> list[str]:
    if control_seed_count <= 0:
        raise ValueError("control_seed_count must be positive.")
    conditions = [target_condition(target) for target in targets]
    conditions.extend(
        f"matched_random:{target_condition(target)}:seed{seed}"
        for target in targets
        for seed in range(1, control_seed_count + 1)
    )
    return conditions


def run_command(
    command: list[str],
    output_path: Path,
    *,
    dry_run: bool,
    force: bool,
    resume_existing: bool = False,
) -> None:
    if output_path.exists() and not force:
        if not resume_existing:
            print(f"Skipping existing result: {output_path}", flush=True)
            return
        command = [*command, "--resume"]
    print(" ".join(command), flush=True)
    if dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def summarize_metric(
    payload: dict[str, Any],
    target: str,
    metric: str,
    *,
    control_seed_count: int,
) -> dict[str, Any]:
    """Compare one target's causal effect against exact-count random controls."""
    condition = target_condition(target)
    metrics = payload.get("metrics", {})
    observed = metrics.get(condition, {}).get(metric)
    controls = [
        metrics.get(f"matched_random:{condition}:seed{seed}", {}).get(metric)
        for seed in range(1, control_seed_count + 1)
    ]
    controls = [float(value) for value in controls if value is not None]
    control_mean = statistics.fmean(controls) if controls else None
    return {
        "observed": observed,
        "matched_random": controls,
        "random_mean": control_mean,
        "random_population_std": statistics.pstdev(controls) if len(controls) > 1 else None,
        "excess_vs_random": float(observed) - control_mean
        if observed is not None and control_mean is not None
        else None,
    }


def paired_excess_nll_bootstrap(
    payload: dict[str, Any],
    target: str,
    *,
    control_seed_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Bootstrap target-minus-control NLL using aligned examples and token weights."""
    condition = target_condition(target)
    metrics = payload.get("metrics", {})
    target_rows = metrics.get(condition, {}).get("per_example", [])
    control_rows = [
        metrics.get(f"matched_random:{condition}:seed{seed}", {}).get("per_example", [])
        for seed in range(1, control_seed_count + 1)
    ]
    if not target_rows or any(not rows for rows in control_rows):
        return {"available": False}

    def index_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        return {int(row["source_index"]): row for row in rows}

    target_by_index = index_rows(target_rows)
    controls_by_index = [index_rows(rows) for rows in control_rows]
    common = set(target_by_index)
    for indexed in controls_by_index:
        common &= set(indexed)
    ordered = sorted(common)
    if len(ordered) != len(target_rows) or any(len(ordered) != len(rows) for rows in control_rows):
        raise ValueError("Target and matched-random per-example results are not exactly aligned.")

    excesses = []
    weights = []
    for source_index in ordered:
        target_row = target_by_index[source_index]
        control_delta = statistics.fmean(indexed[source_index]["delta_nll"] for indexed in controls_by_index)
        excesses.append(float(target_row["delta_nll"]) - control_delta)
        weights.append(int(target_row["token_count"]))

    def weighted_mean(indices: list[int]) -> float:
        total_weight = sum(weights[index] for index in indices)
        return sum(excesses[index] * weights[index] for index in indices) / total_weight

    point = weighted_mean(list(range(len(ordered))))
    rng = random.Random(bootstrap_seed)
    draws = sorted(weighted_mean([rng.randrange(len(ordered)) for _ in ordered]) for _ in range(bootstrap_samples))
    low_index = max(0, int(0.025 * bootstrap_samples))
    high_index = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))
    return {
        "available": True,
        "num_examples": len(ordered),
        "bootstrap_samples": bootstrap_samples,
        "excess_nll": point,
        "ci_low": draws[low_index],
        "ci_high": draws[high_index],
    }


def add_yes_ratio_deltas(payload: dict[str, Any]) -> None:
    """Derive yes-ratio deltas because evaluate_vqa stores only the absolute ratio."""
    metrics = payload.get("metrics", {})
    baseline = metrics.get("none", {}).get("yes_ratio")
    if baseline is None:
        return
    for name, values in metrics.items():
        if name != "none" and values.get("yes_ratio") is not None:
            values["delta_yes_ratio"] = float(values["yes_ratio"]) - float(baseline)


def summarize_results(
    output_dir: Path,
    pope_names: list[str],
    *,
    targets: tuple[str, ...],
    control_seed_count: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"complete": True, "tasks": {}}
    for task_name in ("caption", "text_only"):
        path = output_dir / f"{task_name}.json"
        if not path.exists():
            summary["complete"] = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"none", *build_conditions(control_seed_count, targets)}
        if not required.issubset(payload.get("metrics", {})):
            summary["complete"] = False
        summary["tasks"][task_name] = {
            target: summarize_metric(payload, target, "delta_nll", control_seed_count=control_seed_count)
            for target in targets
        }
    for name in pope_names:
        path = output_dir / f"pope_{name}.json"
        if not path.exists():
            summary["complete"] = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"none", *build_conditions(control_seed_count, targets)}
        if payload.get("complete") is False or not required.issubset(payload.get("metrics", {})):
            summary["complete"] = False
        add_yes_ratio_deltas(payload)
        summary["tasks"][f"pope_{name}"] = {
            target: {
                key: summarize_metric(payload, target, key, control_seed_count=control_seed_count)
                for key in ("delta_accuracy", "delta_f1", "delta_yes_ratio")
            }
            for target in targets
        }

    primary_evidence = {}
    for task_name in ("caption", "text_only"):
        path = output_dir / f"{task_name}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            primary_evidence[task_name] = paired_excess_nll_bootstrap(
                payload,
                PRIMARY_TARGET,
                control_seed_count=control_seed_count,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
            primary_evidence[task_name]["data_isolated"] = bool(payload.get("data_isolation", {}).get("is_isolated"))
    text_evidence = primary_evidence.get("text_only", {})
    gate_passed = bool(
        summary["complete"]
        and text_evidence.get("available")
        and text_evidence.get("data_isolated")
        and text_evidence.get("ci_low", float("-inf")) > 0
    )
    summary["primary_hypothesis"] = {
        "target": "language U shared",
        "mask_column": TARGET_COLUMNS[PRIMARY_TARGET],
        "claim": "The NeuPAT protection set is causally enriched for language ability versus layerwise matched random masks.",
        "primary_endpoint": "text-only target-minus-random excess NLL with paired 95% bootstrap CI",
        "evidence": primary_evidence,
        "gate_passed": gate_passed,
        "gate_rule": "complete run, isolated text data, and paired-bootstrap CI lower bound > 0",
        "scope": "This gate tests causal language sensitivity; post-SFT preservation requires the baseline matrix.",
    }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pope_files = parse_named_files(args.pope)
    if not args.skip_pope and not pope_files:
        raise ValueError("Provide at least one --pope NAME=FILE value, or explicitly pass --skip_pope.")
    if args.skip_pope and pope_files:
        raise ValueError("Do not combine --skip_pope with --pope inputs.")
    targets = get_targets(args.include_component_roles)
    conditions = build_conditions(args.control_seed_count, targets)
    common = ["--score_file", args.score_file]
    for condition in conditions:
        common.extend(["--ablation", condition])
    isolation = [
        "--calibration_manifest",
        args.calibration_manifest,
        "--typing_manifest",
        args.typing_manifest,
        "--require_data_isolation",
    ]
    for manifest in args.probe_manifest:
        isolation.extend(["--exclude_manifest", manifest])

    caption_path = output_dir / "caption.json"
    caption_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_phase2_ablation.py"),
        "--config",
        args.caption_config,
        "--output_file",
        str(caption_path),
        "--max_samples",
        str(args.caption_max_samples),
        "--sample_offset",
        str(args.caption_sample_offset),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.seed),
        "--seed",
        str(args.seed),
        *common,
        *isolation,
    ]
    run_command(caption_command, caption_path, dry_run=args.dry_run, force=args.force)

    text_path = output_dir / "text_only.json"
    text_command = [
        sys.executable,
        str(SCRIPT_DIR / "run_phase2_ablation.py"),
        "--config",
        args.text_config,
        "--output_file",
        str(text_path),
        "--dataset_stage",
        "pt",
        "--max_samples",
        str(args.text_max_samples),
        "--sample_offset",
        str(args.text_sample_offset),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.seed),
        "--seed",
        str(args.seed),
        *common,
        *isolation,
    ]
    run_command(text_command, text_path, dry_run=args.dry_run, force=args.force)

    for name, pope_file in pope_files.items():
        output_path = output_dir / f"pope_{name}.json"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_pope.py"),
            "--config",
            args.caption_config,
            "--score_file",
            args.score_file,
            "--pope_file",
            pope_file,
            "--output_file",
            str(output_path),
            "--batch_size",
            str(args.batch_size),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
            "--bootstrap_seed",
            str(args.seed),
            "--seed",
            str(args.seed),
            "--calibration_manifest",
            args.calibration_manifest,
            "--typing_manifest",
            args.typing_manifest,
            "--filter_manifest_overlaps",
            "--require_data_isolation",
        ]
        if args.image_root:
            command.extend(["--image_root", args.image_root])
        if args.pope_max_images is not None:
            command.extend(["--max_images", str(args.pope_max_images)])
        for manifest in args.probe_manifest:
            command.extend(["--exclude_manifest", manifest])
        for condition in conditions:
            command.extend(["--ablation", condition])
        run_command(
            command,
            output_path,
            dry_run=args.dry_run,
            force=args.force,
            resume_existing=True,
        )

    plan = {
        "primary_hypothesis": "NeuPAT language U shared protection set is causally enriched for language ability.",
        "targets": list(targets),
        "target_columns": {target: TARGET_COLUMNS[target] for target in targets},
        "conditions": conditions,
        "control_seed_count": args.control_seed_count,
        "score_file": str(Path(args.score_file).resolve()),
        "pope_deferred": args.skip_pope,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neupat_causality_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.dry_run:
        summary = summarize_results(
            output_dir,
            list(pope_files),
            targets=targets,
            control_seed_count=args.control_seed_count,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.seed,
        )
        (output_dir / "neupat_causality.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
