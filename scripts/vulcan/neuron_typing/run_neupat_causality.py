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
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROLES = ("language", "multimodal", "shared", "reserve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuPAT role matched-count causal evaluations.")
    parser.add_argument("--caption_config", required=True)
    parser.add_argument("--text_config", required=True)
    parser.add_argument("--score_file", required=True, help="Combined score parquet containing NeuPAT booleans.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pope", action="append", required=True, help="Repeat NAME=FILE for POPE splits.")
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


def build_conditions(control_seed_count: int) -> list[str]:
    if control_seed_count <= 0:
        raise ValueError("control_seed_count must be positive.")
    conditions = [f"mask:neupat_{role}" for role in ROLES]
    conditions.extend(
        f"matched_random:mask:neupat_{role}:seed{seed}" for role in ROLES for seed in range(1, control_seed_count + 1)
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
    role: str,
    metric: str,
    *,
    control_seed_count: int,
) -> dict[str, Any]:
    """Compare one role's causal effect against exact-count random controls."""
    condition = f"mask:neupat_{role}"
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


def add_yes_ratio_deltas(payload: dict[str, Any]) -> None:
    """Derive yes-ratio deltas because evaluate_vqa stores only the absolute ratio."""
    metrics = payload.get("metrics", {})
    baseline = metrics.get("none", {}).get("yes_ratio")
    if baseline is None:
        return
    for name, values in metrics.items():
        if name != "none" and values.get("yes_ratio") is not None:
            values["delta_yes_ratio"] = float(values["yes_ratio"]) - float(baseline)


def summarize_results(output_dir: Path, pope_names: list[str], *, control_seed_count: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"complete": True, "tasks": {}}
    for task_name in ("caption", "text_only"):
        path = output_dir / f"{task_name}.json"
        if not path.exists():
            summary["complete"] = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"none", *build_conditions(control_seed_count)}
        if not required.issubset(payload.get("metrics", {})):
            summary["complete"] = False
        summary["tasks"][task_name] = {
            role: summarize_metric(payload, role, "delta_nll", control_seed_count=control_seed_count) for role in ROLES
        }
    for name in pope_names:
        path = output_dir / f"pope_{name}.json"
        if not path.exists():
            summary["complete"] = False
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        required = {"none", *build_conditions(control_seed_count)}
        if payload.get("complete") is False or not required.issubset(payload.get("metrics", {})):
            summary["complete"] = False
        add_yes_ratio_deltas(payload)
        summary["tasks"][f"pope_{name}"] = {
            role: {
                key: summarize_metric(payload, role, key, control_seed_count=control_seed_count)
                for key in ("delta_accuracy", "delta_f1", "delta_yes_ratio")
            }
            for role in ROLES
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pope_files = parse_named_files(args.pope)
    conditions = build_conditions(args.control_seed_count)
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
        "roles": list(ROLES),
        "conditions": conditions,
        "control_seed_count": args.control_seed_count,
        "score_file": str(Path(args.score_file).resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neupat_causality_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.dry_run:
        summary = summarize_results(output_dir, list(pope_files), control_seed_count=args.control_seed_count)
        (output_dir / "neupat_causality.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
