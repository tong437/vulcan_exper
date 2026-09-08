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

"""Evaluate the equal-budget q-band + mapping-protection joint hook mask."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL = "mask:original_qband_delete"
JOINT = "mask:joint_qband_mapping_protected_delete"
CONDITIONS = (ORIGINAL, JOINT)
REQUIRED_POPE_SPLITS = ("random", "popular", "adversarial")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-4 joint-mask safety and generalization gates.")
    parser.add_argument("--caption_config", required=True)
    parser.add_argument("--c4_config", required=True)
    parser.add_argument("--vqa_med_config", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--original_condition", default=ORIGINAL)
    parser.add_argument("--joint_condition", default=JOINT)
    parser.add_argument(
        "--additional_condition",
        action="append",
        default=[],
        help="Evaluate an exploratory mask condition but do not include it in the primary gate.",
    )
    parser.add_argument(
        "--pope", action="append", required=True, help="Repeat random=FILE, popular=FILE, adversarial=FILE."
    )
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=1,
        help="Tokenizer preprocessing workers; one avoids multiprocessing stalls on local image datasets.",
    )
    parser.add_argument("--caption_max_samples", type=int, default=500)
    parser.add_argument("--caption_sample_offset", type=int, default=2500)
    parser.add_argument(
        "--c4_max_samples",
        type=int,
        default=500,
        help="Frozen packed C4 development blocks (drawn from the 700-document source).",
    )
    parser.add_argument("--vqa_med_max_samples", type=int, default=1501)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2046)
    parser.add_argument("--max_caption_delta_nll", type=float, default=0.05)
    parser.add_argument("--max_c4_delta_nll", type=float, default=0.05)
    parser.add_argument("--max_vqa_med_delta_nll", type=float, default=0.05)
    parser.add_argument("--max_joint_minus_qband_nll", type=float, default=0.02)
    parser.add_argument("--max_pope_accuracy_drop", type=float, default=0.01)
    parser.add_argument("--max_pope_yes_ratio_shift", type=float, default=0.05)
    parser.add_argument("--max_joint_minus_qband_accuracy_drop", type=float, default=0.005)
    parser.add_argument("--stage", choices=["hook", "extended", "summarize", "all"], default="all")
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
    missing = sorted(set(REQUIRED_POPE_SPLITS) - set(result))
    extra = sorted(set(result) - set(REQUIRED_POPE_SPLITS))
    if missing or extra:
        raise ValueError(f"POPE files must be exactly {REQUIRED_POPE_SPLITS}; missing={missing}, extra={extra}.")
    return result


def _conditions(args: argparse.Namespace) -> tuple[str, ...]:
    values = [args.original_condition, args.joint_condition, *args.additional_condition]
    if any(not value.startswith("mask:") for value in values):
        raise ValueError("Every joint-gate condition must use the mask:COLUMN form.")
    if len(values) != len(set(values)):
        raise ValueError("Joint-gate conditions must be unique.")
    return tuple(values)


def _complete(path: Path, conditions: tuple[str, ...] = CONDITIONS) -> bool:
    if not path.is_file():
        return False
    metrics = json.loads(path.read_text(encoding="utf-8")).get("metrics", {})
    return {"none", *conditions}.issubset(metrics)


def _common(command: list[str], args: argparse.Namespace) -> None:
    if args.calibration_manifest:
        command.extend(("--calibration_manifest", args.calibration_manifest))
    if args.typing_manifest:
        command.extend(("--typing_manifest", args.typing_manifest))
    if args.calibration_manifest and args.typing_manifest:
        command.append("--require_data_isolation")


def _run_label_nll(
    *,
    config: str,
    output: Path,
    max_samples: int,
    sample_offset: int,
    dataset_stage: str,
    args: argparse.Namespace,
) -> None:
    conditions = _conditions(args)
    if _complete(output, conditions):
        print(f"Skipping complete evaluation: {output}", flush=True)
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_phase2_ablation.py"),
        "--config",
        config,
        "--score_file",
        args.score_file,
        "--output_file",
        str(output),
        "--max_samples",
        str(max_samples),
        "--sample_offset",
        str(sample_offset),
        "--dataset_stage",
        dataset_stage,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--preprocessing_num_workers",
        str(args.preprocessing_num_workers),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed),
    ]
    for condition in conditions:
        command.extend(("--ablation", condition))
    _common(command, args)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def _run_pope(split: str, source: str, output: Path, args: argparse.Namespace) -> None:
    conditions = _conditions(args)
    if _complete(output, conditions):
        print(f"Skipping complete evaluation: {output}", flush=True)
        return
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pope.py"),
        "--config",
        args.caption_config,
        "--score_file",
        args.score_file,
        "--pope_file",
        source,
        "--output_file",
        str(output),
        "--batch_size",
        str(args.batch_size),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--seed",
        str(args.seed + REQUIRED_POPE_SPLITS.index(split)),
        "--max_image_repeat",
        "6",
        "--allow_excessive_image_repeats",
    ]
    if args.image_root:
        command.extend(("--image_root", args.image_root))
    for manifest in (args.calibration_manifest, args.typing_manifest):
        if manifest:
            command.extend(("--exclude_manifest", manifest))
    if args.calibration_manifest or args.typing_manifest:
        command.append("--filter_manifest_overlaps")
    for condition in conditions:
        command.extend(("--ablation", condition))
    _common(command, args)
    if output.exists():
        command.append("--resume")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def _nll_gate(
    path: Path,
    threshold: float,
    noninferiority: float,
    original_condition: str = ORIGINAL,
    joint_condition: str = JOINT,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = payload["metrics"]["none"]
    qband = payload["metrics"][original_condition]
    joint = payload["metrics"][joint_condition]
    joint_delta = float(joint.get("delta_nll", joint["nll"] - base["nll"]))
    qband_delta = float(qband.get("delta_nll", qband["nll"] - base["nll"]))
    checks = {
        "joint_absolute_safety": joint_delta <= threshold,
        "joint_noninferior_to_qband": joint_delta - qband_delta <= noninferiority,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_nll": float(base["nll"]),
        "qband_delta_nll": qband_delta,
        "joint_delta_nll": joint_delta,
        "joint_minus_qband_delta_nll": joint_delta - qband_delta,
        "joint_paired_ci95": [joint.get("paired_ci_lo"), joint.get("paired_ci_hi")],
        "absolute_threshold": threshold,
        "noninferiority_margin": noninferiority,
    }


def _pope_gate(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    original_condition = getattr(args, "original_condition", ORIGINAL)
    joint_condition = getattr(args, "joint_condition", JOINT)
    base, qband, joint = metrics["none"], metrics[original_condition], metrics[joint_condition]
    joint_drop = float(base["accuracy"] - joint["accuracy"])
    qband_drop = float(base["accuracy"] - qband["accuracy"])
    yes_shift = abs(float(joint["yes_ratio"] - base["yes_ratio"]))
    checks = {
        "joint_accuracy_safety": joint_drop <= args.max_pope_accuracy_drop,
        "joint_yes_ratio_safety": yes_shift <= args.max_pope_yes_ratio_shift,
        "joint_noninferior_to_qband": joint_drop - qband_drop <= args.max_joint_minus_qband_accuracy_drop,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_accuracy": float(base["accuracy"]),
        "qband_accuracy_drop": qband_drop,
        "joint_accuracy_drop": joint_drop,
        "joint_minus_qband_accuracy_drop": joint_drop - qband_drop,
        "joint_yes_ratio_shift": yes_shift,
        "joint_delta_accuracy_ci95": joint.get("delta_accuracy_ci95"),
    }


def summarize(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    conditions = _conditions(args)
    paths = {
        "caption": output_dir / "hook" / "caption_nll.json",
        "c4": output_dir / "extended" / "c4_nll.json",
        "vqa_med": output_dir / "extended" / "vqa_med_nll.json",
        **{f"pope_{split}": output_dir / "hook" / f"pope_{split}.json" for split in REQUIRED_POPE_SPLITS},
    }
    missing = [str(path) for path in paths.values() if not _complete(path, conditions)]
    if missing:
        raise FileNotFoundError(f"Cannot summarize; missing or incomplete evaluations: {missing}")
    gates = {
        "caption": _nll_gate(
            paths["caption"],
            args.max_caption_delta_nll,
            args.max_joint_minus_qband_nll,
            args.original_condition,
            args.joint_condition,
        ),
        "pope": {split: _pope_gate(paths[f"pope_{split}"], args) for split in REQUIRED_POPE_SPLITS},
        "c4_text_only": _nll_gate(
            paths["c4"],
            args.max_c4_delta_nll,
            args.max_joint_minus_qband_nll,
            args.original_condition,
            args.joint_condition,
        ),
        "vqa_med_cross_domain": _nll_gate(
            paths["vqa_med"],
            args.max_vqa_med_delta_nll,
            args.max_joint_minus_qband_nll,
            args.original_condition,
            args.joint_condition,
        ),
    }
    passed = (
        gates["caption"]["passed"]
        and all(row["passed"] for row in gates["pope"].values())
        and gates["c4_text_only"]["passed"]
        and gates["vqa_med_cross_domain"]["passed"]
    )
    result = {
        "complete": True,
        "passed": passed,
        "structural_checkpoint_allowed": passed,
        "score_file": str(Path(args.score_file).resolve()),
        "conditions": {
            "equal_budget_reference": args.original_condition,
            "primary_protected_candidate": args.joint_condition,
            "exploratory": args.additional_condition,
        },
        "gates": gates,
        "evaluation_files": {name: str(path.resolve()) for name, path in paths.items()},
        "interpretation": (
            "VQA-Med uses teacher-forced answer NLL on all 1,501 classification examples. It measures "
            "cross-domain multimodal retention under the hook mask; it is not a generated-answer EM claim."
        ),
    }
    path = output_dir / "quality_gate.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    pope = parse_named_files(args.pope)
    if args.stage in {"hook", "all"}:
        _run_label_nll(
            config=args.caption_config,
            output=output_dir / "hook" / "caption_nll.json",
            max_samples=args.caption_max_samples,
            sample_offset=args.caption_sample_offset,
            dataset_stage="sft",
            args=args,
        )
        for split in REQUIRED_POPE_SPLITS:
            _run_pope(split, pope[split], output_dir / "hook" / f"pope_{split}.json", args)
    if args.stage in {"extended", "all"}:
        _run_label_nll(
            config=args.c4_config,
            output=output_dir / "extended" / "c4_nll.json",
            max_samples=args.c4_max_samples,
            sample_offset=0,
            dataset_stage="pt",
            args=args,
        )
        _run_label_nll(
            config=args.vqa_med_config,
            output=output_dir / "extended" / "vqa_med_nll.json",
            max_samples=args.vqa_med_max_samples,
            sample_offset=0,
            dataset_stage="sft",
            args=args,
        )
    if args.stage in {"summarize", "all"}:
        return summarize(output_dir, args)
    return {"complete": True, "stage": args.stage, "structural_checkpoint_allowed": False}


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
