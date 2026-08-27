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

"""Run held-out group ablations after the Phase-4 mapping gates pass."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
from dataset_guard import normalize_image_id  # noqa: E402
from evaluate_vqa import load_binary_records  # noqa: E402


QBAND = "rank_band:multimodal:0.05:0.2"
MAPPING_HIGH = "matched_score:mapping_signal:highest:rank_band:multimodal:0.05:0.2"
COMBINED_LOW = "matched_score:combined_protection:lowest:rank_band:multimodal:0.05:0.2"
RANDOM_CONTROLS = tuple(
    f"matched_random:rank_band:multimodal:0.05:0.2:seed{seed}" for seed in (1, 2, 3)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-4 held-out group causal and pruning-prior gates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mapping_metrics", required=True)
    parser.add_argument("--activation_dir", required=True)
    parser.add_argument(
        "--vqa",
        action="append",
        required=True,
        help="Repeatable NAME=POPE_FILE entry; test images from Phase-4 splits are selected automatically.",
    )
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--max_accuracy_drop", type=float, default=0.01)
    parser.add_argument("--max_f1_drop", type=float, default=0.015)
    parser.add_argument("--max_yes_ratio_shift", type=float, default=0.05)
    parser.add_argument("--min_causal_enrichment", type=float, default=0.0)
    parser.add_argument("--combined_vs_qband_tolerance", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _parse_named_files(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=FILE for --vqa, got {value!r}.")
        name, path = value.split("=", maxsplit=1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid VQA split name: {name!r}")
        if name in result:
            raise ValueError(f"Duplicate VQA split name: {name}")
        result[name] = path
    return result


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


def _run(command: list[str], stage: str) -> None:
    print(f"Running {stage}", flush=True)
    subprocess.run(command, check=True)


def _quality(metrics: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    baseline = metrics["none"]
    result = {}
    for condition in (QBAND, COMBINED_LOW):
        row = metrics[condition]
        delta_accuracy = float(row["accuracy"] - baseline["accuracy"])
        delta_f1 = float(row["f1"] - baseline["f1"])
        yes_shift = float(row["yes_ratio"] - baseline["yes_ratio"])
        result[condition] = {
            "delta_accuracy": delta_accuracy,
            "delta_f1": delta_f1,
            "delta_yes_ratio": yes_shift,
            "passed": (
                delta_accuracy >= -args.max_accuracy_drop
                and delta_f1 >= -args.max_f1_drop
                and abs(yes_shift) <= args.max_yes_ratio_shift
            ),
        }
    return result


def evaluate_group_causality(args: argparse.Namespace) -> dict[str, Any]:
    mapping_metrics = json.loads(Path(args.mapping_metrics).read_text(encoding="utf-8"))
    if not mapping_metrics["gates"]["phase4_group_causal_ablation_allowed"]:
        raise RuntimeError("Phase-4 Gate A/B failed; group causal ablation is forbidden.")
    score_file = mapping_metrics["outputs"]["augmented_score_file"]
    split_payload = json.loads((Path(args.activation_dir) / "splits.json").read_text(encoding="utf-8"))
    test_image_ids = {
        image_id for image_id, split in split_payload["image_to_split"].items() if split == "test"
    }
    if len(test_image_ids) < 2:
        raise ValueError("Phase-4 test split has fewer than two images.")

    named_files = _parse_named_files(args.vqa)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = {}
    for name, source_path in named_files.items():
        subset_path = output_dir / "test_subsets" / f"{name}.jsonl"
        selected_rows = _write_test_subset(source_path, args.image_root, test_image_ids, subset_path)
        output_file = output_dir / f"{name}.json"
        command = [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_pope.py"),
            "--config",
            args.config,
            "--score_file",
            score_file,
            "--pope_file",
            str(subset_path),
            "--output_file",
            str(output_file),
            "--batch_size",
            str(args.batch_size),
            "--bootstrap_samples",
            str(args.bootstrap_samples),
            "--seed",
            str(args.seed),
            "--max_image_repeat",
            "6",
            "--allow_excessive_image_repeats",
            "--ablation",
            QBAND,
            "--ablation",
            MAPPING_HIGH,
            "--ablation",
            COMBINED_LOW,
        ]
        for control in RANDOM_CONTROLS:
            command.extend(("--ablation", control))
        if args.calibration_manifest:
            command.extend(("--calibration_manifest", args.calibration_manifest))
        if args.typing_manifest:
            command.extend(("--typing_manifest", args.typing_manifest))
        if args.calibration_manifest and args.typing_manifest:
            command.append("--require_data_isolation")
        if args.resume and output_file.exists():
            command.append("--resume")
        _run(command, f"Phase-4 group ablation: {name}")
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        evaluations[name] = {
            "selected_rows": selected_rows,
            "output_file": str(output_file),
            "quality": _quality(payload["metrics"], args),
            "metrics": payload["metrics"],
        }

    causal_enrichment = []
    combined_advantage = []
    all_safe = True
    per_task = {}
    for name, evaluation in evaluations.items():
        metrics = evaluation["metrics"]
        baseline_accuracy = float(metrics["none"]["accuracy"])
        mapping_delta = float(metrics[MAPPING_HIGH]["accuracy"] - baseline_accuracy)
        random_deltas = [
            float(metrics[condition]["accuracy"] - baseline_accuracy) for condition in RANDOM_CONTROLS
        ]
        enrichment = float(np.mean(random_deltas) - mapping_delta)
        qband_delta = float(metrics[QBAND]["accuracy"] - baseline_accuracy)
        combined_delta = float(metrics[COMBINED_LOW]["accuracy"] - baseline_accuracy)
        advantage = combined_delta - qband_delta
        causal_enrichment.append(enrichment)
        combined_advantage.append(advantage)
        task_safe = all(row["passed"] for row in evaluation["quality"].values())
        all_safe &= task_safe
        per_task[name] = {
            "mapping_high_delta_accuracy": mapping_delta,
            "matched_random_mean_delta_accuracy": float(np.mean(random_deltas)),
            "mapping_causal_enrichment": enrichment,
            "qband_delta_accuracy": qband_delta,
            "combined_low_delta_accuracy": combined_delta,
            "combined_advantage_over_qband": advantage,
            "qband_and_combined_safe": task_safe,
        }

    mean_causal_enrichment = float(np.mean(causal_enrichment))
    mean_combined_advantage = float(np.mean(combined_advantage))
    causal_gate = mean_causal_enrichment >= args.min_causal_enrichment
    pruning_gate = (
        all_safe and mean_combined_advantage >= -args.combined_vs_qband_tolerance
    )
    result = {
        "complete": True,
        "config": vars(args),
        "mapping_metrics": args.mapping_metrics,
        "test_image_count": len(test_image_ids),
        "tasks": per_task,
        "gates": {
            "mapping_causal_enrichment": {
                "passed": causal_gate,
                "mean_enrichment": mean_causal_enrichment,
                "minimum_enrichment": args.min_causal_enrichment,
            },
            "combined_pruning_prior": {
                "passed": pruning_gate,
                "all_qband_and_combined_conditions_safe": all_safe,
                "mean_combined_advantage_over_qband": mean_combined_advantage,
                "negative_tolerance": args.combined_vs_qband_tolerance,
            },
            "phase4_structural_candidate_allowed": bool(causal_gate and pruning_gate),
        },
        "evaluation_files": {name: evaluation["output_file"] for name, evaluation in evaluations.items()},
    }
    output_path = output_dir / "phase4_group_causality.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if not result["gates"]["phase4_structural_candidate_allowed"]:
        raise RuntimeError(f"Phase-4 group causal/pruning gate failed; see {output_path}")
    return result


def main() -> None:
    result = evaluate_group_causality(parse_args())
    print(json.dumps(result["gates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
