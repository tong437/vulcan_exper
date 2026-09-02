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

"""Phase-2.6 fixed-budget completeness sweep across q rankings and tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

from run_phase2_ablation import (  # noqa: E402
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    parse_ablation_spec,
    read_score_table,
    summarize_masks,
)


REQUIRED_POPE_SPLITS = ("random", "popular", "adversarial")
DEFAULT_TAIL_STARTS = tuple(round(value / 100, 2) for value in range(20, 90, 5))
QBAND_CONDITION = "rank_band:multimodal:0.05:0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase-2.6 fixed-budget completeness sweep.")
    parser.add_argument("--caption_config", required=True)
    parser.add_argument("--text_config", required=True, help="LlamaFactory config for a held-out text-only dataset.")
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pope",
        action="append",
        required=True,
        help="Repeat random=FILE, popular=FILE, and adversarial=FILE.",
    )
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--text_dataset", default=None)
    parser.add_argument("--text_eval_dataset", default=None)
    parser.add_argument("--tail_starts", default=",".join(f"{value:g}" for value in DEFAULT_TAIL_STARTS))
    parser.add_argument("--window_ratio", type=float, default=0.15)
    parser.add_argument("--control_seed_count", type=int, default=20)
    parser.add_argument("--max_finalists", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--screen_caption_max_samples", type=int, default=256)
    parser.add_argument("--screen_caption_sample_offset", type=int, default=2500)
    parser.add_argument("--screen_text_max_samples", type=int, default=256)
    parser.add_argument("--screen_text_sample_offset", type=int, default=0)
    parser.add_argument("--screen_max_images", type=int, default=64)
    parser.add_argument("--screen_bootstrap_samples", type=int, default=200)
    parser.add_argument("--final_caption_max_samples", type=int, default=None)
    parser.add_argument("--final_caption_sample_offset", type=int, default=2500)
    parser.add_argument("--final_text_max_samples", type=int, default=None)
    parser.add_argument("--final_text_sample_offset", type=int, default=0)
    parser.add_argument("--final_bootstrap_samples", type=int, default=2000)
    parser.add_argument("--max_caption_delta_nll", type=float, default=0.05)
    parser.add_argument("--max_text_delta_nll", type=float, default=0.05)
    parser.add_argument("--max_pope_accuracy_drop", type=float, default=0.01)
    parser.add_argument("--max_yes_ratio_shift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--stage",
        choices=["build", "screen", "shortlist", "final", "summarize", "all"],
        default="all",
    )
    return parser.parse_args()


def parse_named_files(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=FILE, got {value!r}.")
        name, path = (part.strip() for part in value.split("=", maxsplit=1))
        if not name or not path:
            raise ValueError(f"Expected non-empty NAME=FILE, got {value!r}.")
        if name in result:
            raise ValueError(f"Duplicate task name: {name}.")
        result[name] = path
    missing = sorted(set(REQUIRED_POPE_SPLITS) - set(result))
    if missing:
        raise ValueError(f"Phase 2.6 requires all POPE splits; missing: {missing}.")
    return result


def parse_tail_starts(value: str, window_ratio: float) -> list[float]:
    starts = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not 0.0 < window_ratio < 1.0:
        raise ValueError("window_ratio must be strictly between zero and one.")
    if not starts or any(start < 0.20 or start + window_ratio > 1.0 + 1e-9 for start in starts):
        raise ValueError("tail_starts must begin at or after 0.20 and every window must end at or before 1.0.")
    return starts


def ratio_label(value: float) -> str:
    return f"p{round(value * 10000):04d}"


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_candidate_registry(
    *,
    intermediate_size: int,
    tail_starts: list[float],
    window_ratio: float,
) -> tuple[dict[str, dict[str, Any]], int]:
    qband_start = math.ceil(intermediate_size * 0.05)
    qband_end = math.ceil(intermediate_size * 0.20)
    budget = qband_end - qband_start
    candidates: dict[str, dict[str, Any]] = {
        "q_multimodal_p05_p20": {
            "family": "q_multimodal",
            "condition": QBAND_CONDITION,
            "description": "Frozen Phase-2 q_multimodal 5-20% reference band.",
            "rank_start": qband_start,
            "rank_count": budget,
            "nominal_start": 0.05,
        }
    }
    for start in tail_starts:
        rank_start = math.ceil(intermediate_size * start)
        if rank_start + budget > intermediate_size:
            raise ValueError(
                f"Tail window at {start:g} exceeds width {intermediate_size} with fixed budget {budget}."
            )
        candidate_id = f"q_multimodal_{ratio_label(start)}_{ratio_label(start + window_ratio)}"
        candidates[candidate_id] = {
            "family": "q_multimodal",
            "condition": f"rank_window:multimodal:{rank_start}:{budget}",
            "description": "Fixed-count q_multimodal tail window.",
            "rank_start": rank_start,
            "rank_count": budget,
            "nominal_start": start,
            "nominal_end": start + window_ratio,
            "actual_start": rank_start / intermediate_size,
            "actual_end": (rank_start + budget) / intermediate_size,
        }

    for type_name in ("visual", "text", "unknown"):
        score_column = f"q_{type_name}"
        candidates[f"{score_column}_top15"] = {
            "family": score_column,
            "condition": f"matched_score:{score_column}:highest:{QBAND_CONDITION}",
            "description": f"Top fixed-budget neurons ranked explicitly by {score_column}.",
            "rank_start": 0,
            "rank_count": budget,
            "nominal_start": 0.0,
            "nominal_end": window_ratio,
        }
    return candidates, budget


def build_experiment_plan(args: argparse.Namespace) -> dict[str, Any]:
    table = read_score_table(args.score_file)
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    required_columns = {"q_visual", "q_text", "q_multimodal", "q_unknown"}
    missing = sorted(required_columns - set(table.columns))
    if missing:
        raise ValueError(f"Phase 2.6 requires explicit q score columns; missing: {missing}.")
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    unique_dims = sorted(set(layer_dims.values()))
    if len(unique_dims) != 1:
        raise ValueError(f"Phase 2.6 fixed rank windows require uniform FFN widths, got {unique_dims}.")
    intermediate_size = unique_dims[0]
    tail_starts = parse_tail_starts(args.tail_starts, args.window_ratio)
    candidates, budget = build_candidate_registry(
        intermediate_size=intermediate_size,
        tail_starts=tail_starts,
        window_ratio=args.window_ratio,
    )

    conditions = list(dict.fromkeys(row["condition"] for row in candidates.values()))
    parsed = [parse_ablation_spec(condition, args.seed) for condition in conditions]
    masks = {
        spec.result_name: build_type_mask(
            table,
            spec,
            layer_col,
            neuron_col,
            score_cols,
            activation_col,
            layer_dims,
            None,
            "per_layer",
            1.0,
            0.0,
        )
        for spec in parsed
    }
    mask_summaries = {name: summarize_masks(mask) for name, mask in masks.items()}
    for name, summary in mask_summaries.items():
        counts = set(summary["per_layer_selected"].values())
        if counts != {budget}:
            raise RuntimeError(f"Candidate {name} violates the fixed per-layer budget: {sorted(counts)}.")

    plan: dict[str, Any] = {
        "phase": "2.6",
        "objective": "Fixed-budget q-ranking completeness sweep with cross-task safety gates.",
        "score_file": str(Path(args.score_file).resolve()),
        "score_columns": score_cols,
        "layer_column": layer_col,
        "neuron_column": neuron_col,
        "num_layers": len(layer_dims),
        "intermediate_size": intermediate_size,
        "per_layer_budget": budget,
        "actual_budget_ratio": budget / intermediate_size,
        "tail_starts": tail_starts,
        "nominal_window_ratio": args.window_ratio,
        "candidates": candidates,
        "mask_summaries": mask_summaries,
        "screen_tasks": ["caption_nll", "text_only_nll", "pope_random"],
        "formal_tasks": ["caption_nll", "text_only_nll", *[f"pope_{name}" for name in REQUIRED_POPE_SPLITS]],
        "thresholds": {
            "max_caption_delta_nll": args.max_caption_delta_nll,
            "max_text_delta_nll": args.max_text_delta_nll,
            "max_pope_accuracy_drop": args.max_pope_accuracy_drop,
            "max_yes_ratio_shift": args.max_yes_ratio_shift,
        },
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def _evaluation_complete(path: Path, conditions: list[str]) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {"none", *conditions}.issubset(metrics)


def _append_common_args(command: list[str], args: argparse.Namespace) -> None:
    if args.model_name_or_path:
        command.extend(("--model_name_or_path", args.model_name_or_path))
    if args.calibration_manifest:
        command.extend(("--calibration_manifest", args.calibration_manifest))
    if args.typing_manifest:
        command.extend(("--typing_manifest", args.typing_manifest))
    if args.calibration_manifest and args.typing_manifest:
        command.append("--require_data_isolation")


def _run_label_nll(
    *,
    config: str,
    output_file: Path,
    conditions: list[str],
    max_samples: int | None,
    sample_offset: int,
    bootstrap_samples: int,
    args: argparse.Namespace,
    text_only: bool,
) -> None:
    if _evaluation_complete(output_file, conditions):
        print(f"Skipping complete Phase-2.6 evaluation: {output_file}", flush=True)
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_phase2_ablation.py"),
        "--config",
        config,
        "--score_file",
        args.score_file,
        "--output_file",
        str(output_file),
        "--sample_offset",
        str(sample_offset),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--bootstrap_samples",
        str(bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    if max_samples is not None:
        command.extend(("--max_samples", str(max_samples)))
    if text_only and args.text_dataset:
        command.extend(("--dataset", args.text_dataset))
    if text_only and args.text_eval_dataset:
        command.extend(("--eval_dataset", args.text_eval_dataset))
    if text_only:
        command.extend(("--dataset_stage", "pt"))
    for condition in conditions:
        command.extend(("--ablation", condition))
    _append_common_args(command, args)
    print(f"Running Phase-2.6 label-NLL evaluation: {output_file}", flush=True)
    subprocess.run(command, check=True)


def _run_pope(
    *,
    split: str,
    source_file: str,
    output_file: Path,
    conditions: list[str],
    max_images: int | None,
    bootstrap_samples: int,
    args: argparse.Namespace,
) -> None:
    if _evaluation_complete(output_file, conditions):
        print(f"Skipping complete Phase-2.6 evaluation: {output_file}", flush=True)
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pope.py"),
        "--config",
        args.caption_config,
        "--score_file",
        args.score_file,
        "--pope_file",
        source_file,
        "--output_file",
        str(output_file),
        "--batch_size",
        str(args.batch_size),
        "--bootstrap_samples",
        str(bootstrap_samples),
        "--seed",
        str(args.seed),
        "--max_image_repeat",
        "6",
        "--allow_excessive_image_repeats",
    ]
    if args.image_root:
        command.extend(("--image_root", args.image_root))
    if max_images is not None:
        command.extend(("--max_images", str(max_images)))
    comparison_manifests = [path for path in (args.calibration_manifest, args.typing_manifest) if path]
    for manifest_path in comparison_manifests:
        command.extend(("--exclude_manifest", manifest_path))
    if comparison_manifests:
        command.append("--filter_manifest_overlaps")
    for condition in conditions:
        command.extend(("--ablation", condition))
    _append_common_args(command, args)
    if output_file.exists():
        command.append("--resume")
    print(f"Running Phase-2.6 POPE-{split}: {output_file}", flush=True)
    subprocess.run(command, check=True)


def _delta_nll(payload: dict[str, Any], condition: str) -> float:
    row = payload["metrics"][condition]
    return float(row.get("delta_nll", row["nll"] - payload["metrics"]["none"]["nll"]))


def _delta_accuracy(payload: dict[str, Any], condition: str) -> float:
    row = payload["metrics"][condition]
    return float(row.get("delta_accuracy", row["accuracy"] - payload["metrics"]["none"]["accuracy"]))


def summarize_screen(plan: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    caption = json.loads((output_dir / "screen" / "caption.json").read_text(encoding="utf-8"))
    text_only = json.loads((output_dir / "screen" / "text_only.json").read_text(encoding="utf-8"))
    pope = json.loads((output_dir / "screen" / "pope_random.json").read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    passing = []
    for candidate_id, candidate in plan["candidates"].items():
        condition = candidate["condition"]
        caption_delta = _delta_nll(caption, condition)
        text_delta = _delta_nll(text_only, condition)
        accuracy_delta = _delta_accuracy(pope, condition)
        yes_shift = float(pope["metrics"][condition]["yes_ratio"] - pope["metrics"]["none"]["yes_ratio"])
        gates = {
            "caption": caption_delta <= args.max_caption_delta_nll,
            "text_only": text_delta <= args.max_text_delta_nll,
            "pope_random_accuracy": accuracy_delta >= -args.max_pope_accuracy_drop,
            "pope_random_yes_ratio": abs(yes_shift) <= args.max_yes_ratio_shift,
        }
        safety_score = max(
            max(0.0, caption_delta) / args.max_caption_delta_nll,
            max(0.0, text_delta) / args.max_text_delta_nll,
            max(0.0, -accuracy_delta) / args.max_pope_accuracy_drop,
            abs(yes_shift) / args.max_yes_ratio_shift,
        )
        rows[candidate_id] = {
            **candidate,
            "screen_metrics": {
                "caption_delta_nll": caption_delta,
                "text_delta_nll": text_delta,
                "pope_random_delta_accuracy": accuracy_delta,
                "pope_random_yes_ratio": float(pope["metrics"][condition]["yes_ratio"]),
                "pope_random_delta_yes_ratio": yes_shift,
            },
            "gates": gates,
            "passed": all(gates.values()),
            "safety_score": safety_score,
        }
        if all(gates.values()):
            passing.append(candidate_id)
    passing.sort(key=lambda candidate_id: (rows[candidate_id]["safety_score"], candidate_id))
    finalists = passing[: args.max_finalists]
    result = {
        "plan_sha256": plan["plan_sha256"],
        "screen_complete": True,
        "candidate_count": len(rows),
        "passing_candidate_count": len(passing),
        "passing_candidates": passing,
        "finalists": finalists,
        "max_finalists": args.max_finalists,
        "candidates": rows,
    }
    _write_json(output_dir / "screen_summary.json", result)
    return result


def common_random_controls(seed_count: int) -> list[str]:
    if seed_count < 1:
        raise ValueError("control_seed_count must be positive.")
    return [f"matched_random:{QBAND_CONDITION}:seed{seed}" for seed in range(1, seed_count + 1)]


def _control_comparison(candidate_delta: float, control_deltas: list[float], *, lower_is_safer: bool) -> dict[str, Any]:
    controls = np.asarray(control_deltas, dtype=np.float64)
    if lower_is_safer:
        extreme_count = int(np.sum(controls <= candidate_delta))
    else:
        extreme_count = int(np.sum(controls >= candidate_delta))
    return {
        "candidate_delta": candidate_delta,
        "control_deltas": controls.tolist(),
        "control_mean": float(controls.mean()),
        "control_median": float(np.median(controls)),
        "candidate_minus_control_mean": float(candidate_delta - controls.mean()),
        "candidate_safer_than_control_median": bool(
            candidate_delta <= np.median(controls) if lower_is_safer else candidate_delta >= np.median(controls)
        ),
        "empirical_p_candidate_safer": float((1 + extreme_count) / (len(controls) + 1)),
    }


def summarize_final(
    plan: dict[str, Any],
    screen_summary: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    final_dir = output_dir / "final"
    payloads = {
        "caption": json.loads((final_dir / "caption.json").read_text(encoding="utf-8")),
        "text_only": json.loads((final_dir / "text_only.json").read_text(encoding="utf-8")),
        **{
            f"pope_{split}": json.loads((final_dir / f"pope_{split}.json").read_text(encoding="utf-8"))
            for split in REQUIRED_POPE_SPLITS
        },
    }
    controls = common_random_controls(args.control_seed_count)
    candidates: dict[str, Any] = {}
    safe_candidates = []
    for candidate_id in screen_summary["finalists"]:
        condition = plan["candidates"][candidate_id]["condition"]
        caption_delta = _delta_nll(payloads["caption"], condition)
        text_delta = _delta_nll(payloads["text_only"], condition)
        task_rows: dict[str, Any] = {
            "caption": {
                "delta_nll": caption_delta,
                "matched_control": _control_comparison(
                    caption_delta,
                    [_delta_nll(payloads["caption"], control) for control in controls],
                    lower_is_safer=True,
                ),
            },
            "text_only": {
                "delta_nll": text_delta,
                "matched_control": _control_comparison(
                    text_delta,
                    [_delta_nll(payloads["text_only"], control) for control in controls],
                    lower_is_safer=True,
                ),
            },
        }
        pope_gates = []
        for split in REQUIRED_POPE_SPLITS:
            payload = payloads[f"pope_{split}"]
            accuracy_delta = _delta_accuracy(payload, condition)
            yes_ratio = float(payload["metrics"][condition]["yes_ratio"])
            yes_shift = yes_ratio - float(payload["metrics"]["none"]["yes_ratio"])
            split_gate = accuracy_delta >= -args.max_pope_accuracy_drop and abs(yes_shift) <= args.max_yes_ratio_shift
            pope_gates.append(split_gate)
            task_rows[f"pope_{split}"] = {
                "delta_accuracy": accuracy_delta,
                "delta_f1": float(payload["metrics"][condition]["delta_f1"]),
                "yes_ratio": yes_ratio,
                "delta_yes_ratio": yes_shift,
                "passed": split_gate,
                "matched_control": _control_comparison(
                    accuracy_delta,
                    [_delta_accuracy(payload, control) for control in controls],
                    lower_is_safer=False,
                ),
            }
        gates = {
            "caption": caption_delta <= args.max_caption_delta_nll,
            "text_only": text_delta <= args.max_text_delta_nll,
            "all_pope": all(pope_gates),
        }
        passed = all(gates.values())
        if passed:
            safe_candidates.append(candidate_id)
        candidates[candidate_id] = {
            **plan["candidates"][candidate_id],
            "tasks": task_rows,
            "gates": gates,
            "formal_safety_passed": passed,
            "matched_control_median_wins": sum(
                row["matched_control"]["candidate_safer_than_control_median"] for row in task_rows.values()
            ),
        }
    safe_candidates.sort(
        key=lambda candidate_id: (-candidates[candidate_id]["matched_control_median_wins"], candidate_id)
    )
    result = {
        "complete": True,
        "plan_sha256": plan["plan_sha256"],
        "control_seed_count": args.control_seed_count,
        "common_matched_controls": controls,
        "formal_safety_passed": bool(safe_candidates),
        "safe_candidates": safe_candidates,
        "recommended_candidate": safe_candidates[0] if safe_candidates else None,
        "structural_followup_allowed": bool(safe_candidates),
        "candidates": candidates,
    }
    _write_json(output_dir / "phase26_completeness.json", result)
    return result


def run_phase26(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    pope_files = parse_named_files(args.pope)
    plan = build_experiment_plan(args)
    _write_json(output_dir / "phase26_plan.json", plan)
    if args.stage == "build":
        return plan

    candidate_conditions = list(dict.fromkeys(row["condition"] for row in plan["candidates"].values()))
    if args.stage in {"screen", "all"}:
        screen_dir = output_dir / "screen"
        _run_label_nll(
            config=args.caption_config,
            output_file=screen_dir / "caption.json",
            conditions=candidate_conditions,
            max_samples=args.screen_caption_max_samples,
            sample_offset=args.screen_caption_sample_offset,
            bootstrap_samples=args.screen_bootstrap_samples,
            args=args,
            text_only=False,
        )
        _run_label_nll(
            config=args.text_config,
            output_file=screen_dir / "text_only.json",
            conditions=candidate_conditions,
            max_samples=args.screen_text_max_samples,
            sample_offset=args.screen_text_sample_offset,
            bootstrap_samples=args.screen_bootstrap_samples,
            args=args,
            text_only=True,
        )
        _run_pope(
            split="random",
            source_file=pope_files["random"],
            output_file=screen_dir / "pope_random.json",
            conditions=candidate_conditions,
            max_images=args.screen_max_images,
            bootstrap_samples=args.screen_bootstrap_samples,
            args=args,
        )
    if args.stage in {"screen", "shortlist", "all"}:
        screen_summary = summarize_screen(plan, output_dir, args)
        if args.stage in {"screen", "shortlist"}:
            return screen_summary
    else:
        screen_summary = json.loads((output_dir / "screen_summary.json").read_text(encoding="utf-8"))

    finalists = screen_summary["finalists"]
    if not finalists:
        raise RuntimeError("No Phase-2.6 candidate passed screening; formal evaluation is forbidden.")
    finalist_conditions = [plan["candidates"][candidate_id]["condition"] for candidate_id in finalists]
    controls = common_random_controls(args.control_seed_count)
    final_conditions = list(dict.fromkeys([*finalist_conditions, QBAND_CONDITION, *controls]))
    if args.stage in {"final", "all"}:
        final_dir = output_dir / "final"
        _run_label_nll(
            config=args.caption_config,
            output_file=final_dir / "caption.json",
            conditions=final_conditions,
            max_samples=args.final_caption_max_samples,
            sample_offset=args.final_caption_sample_offset,
            bootstrap_samples=args.final_bootstrap_samples,
            args=args,
            text_only=False,
        )
        _run_label_nll(
            config=args.text_config,
            output_file=final_dir / "text_only.json",
            conditions=final_conditions,
            max_samples=args.final_text_max_samples,
            sample_offset=args.final_text_sample_offset,
            bootstrap_samples=args.final_bootstrap_samples,
            args=args,
            text_only=True,
        )
        for split in REQUIRED_POPE_SPLITS:
            _run_pope(
                split=split,
                source_file=pope_files[split],
                output_file=final_dir / f"pope_{split}.json",
                conditions=final_conditions,
                max_images=None,
                bootstrap_samples=args.final_bootstrap_samples,
                args=args,
            )
        if args.stage == "final":
            return {"final_evaluations_complete": True, "finalists": finalists}
    result = summarize_final(plan, screen_summary, output_dir, args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["formal_safety_passed"]:
        raise RuntimeError(f"Phase-2.6 formal safety gate failed; see {output_dir / 'phase26_completeness.json'}")
    return result


def main() -> None:
    run_phase26(parse_args())


if __name__ == "__main__":
    main()
