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

"""Orchestrate the frozen-mask Phase-3 structural pruning pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VULCAN_DIR = SCRIPT_DIR.parent
MASK_NAME = "q_multimodal_band_05_20"
QBAND_RESULT = "rank_band:multimodal:0.05:0.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-3 structural q-band pruning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--phase2_caption_result", required=True)
    parser.add_argument("--phase2_pope_result", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smoke_pope_file", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--smoke_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--reuse_pruned_model", action="store_true")
    parser.add_argument("--skip_equivalence", action="store_true")
    parser.add_argument("--run_formal_evaluation", action="store_true")
    parser.add_argument(
        "--reuse_formal_evaluation",
        action="store_true",
        help="Reuse existing caption/POPE output files and only recompute the formal quality gate.",
    )
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument("--benchmark_warmup", type=int, default=10)
    parser.add_argument("--benchmark_repeats", type=int, default=50)
    parser.add_argument("--benchmark_batch_sizes", default="1,4")
    parser.add_argument("--benchmark_max_new_tokens", type=int, default=64)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_command(command: list[str], *, stage: str) -> None:
    print(f"[{stage}] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def option(command: list[str], name: str, value: Any) -> None:
    if value is not None:
        command.extend([name, str(value)])


def flag(command: list[str], name: str, enabled: bool) -> None:
    if enabled:
        command.append(name)


def compare_caption_quality(
    reference: dict[str, Any],
    structural: dict[str, Any],
    *,
    hook_result_name: str = QBAND_RESULT,
    hook_nll_tolerance: float = 0.05,
    original_nll_degradation_tolerance: float = 0.05,
) -> dict[str, Any]:
    hook = reference["metrics"][hook_result_name]
    original = reference["metrics"]["none"]
    candidate = structural["metrics"]["none"]
    hook_delta = float(candidate["nll"] - hook["nll"])
    original_delta = float(candidate["nll"] - original["nll"])
    checks = {
        "hook_nll": abs(hook_delta) <= hook_nll_tolerance,
        "original_nll_degradation": original_delta <= original_nll_degradation_tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "original_nll": original["nll"],
        "hook_nll": hook["nll"],
        "structural_nll": candidate["nll"],
        "structural_minus_hook_nll": hook_delta,
        "structural_minus_original_nll": original_delta,
        "hook_nll_tolerance": hook_nll_tolerance,
        "original_nll_degradation_tolerance": original_nll_degradation_tolerance,
    }


def compare_pope_quality(
    reference: dict[str, Any],
    structural: dict[str, Any],
    *,
    hook_result_name: str = QBAND_RESULT,
    prediction_mismatch_tolerance: float = 0.005,
    mismatch_boundary_margin_tolerance: float = 0.125,
    hook_accuracy_drift_tolerance: float = 0.002,
    hook_yes_ratio_drift_tolerance: float = 0.002,
    accuracy_drop_tolerance: float = 0.01,
    yes_ratio_drift_tolerance: float = 0.02,
) -> dict[str, Any]:
    hook = reference["metrics"][hook_result_name]
    original = reference["metrics"]["none"]
    candidate = structural["metrics"]["none"]
    hook_predictions = hook["predictions"]
    candidate_predictions = candidate["predictions"]
    if len(hook_predictions) != len(candidate_predictions):
        raise ValueError("Structural and hook POPE results contain different numbers of predictions.")
    mismatches = []
    for hook_row, candidate_row in zip(hook_predictions, candidate_predictions):
        if hook_row["source_index"] != candidate_row["source_index"]:
            raise ValueError("Structural and hook POPE predictions are not aligned by source_index.")
        if hook_row["prediction"] == candidate_row["prediction"]:
            continue
        hook_margin = float(hook_row["margin"])
        structural_margin = float(candidate_row["margin"])
        minimum_absolute_margin = min(abs(hook_margin), abs(structural_margin))
        mismatches.append(
            {
                "source_index": hook_row["source_index"],
                "question_id": hook_row.get("question_id"),
                "answer": hook_row.get("answer"),
                "hook_prediction": hook_row["prediction"],
                "structural_prediction": candidate_row["prediction"],
                "hook_margin": hook_margin,
                "structural_margin": structural_margin,
                "minimum_absolute_margin": minimum_absolute_margin,
                "near_decision_boundary": minimum_absolute_margin <= mismatch_boundary_margin_tolerance + 1e-8,
            }
        )
    prediction_match_ratio = 1.0 - len(mismatches) / len(hook_predictions) if hook_predictions else 1.0
    prediction_mismatch_ratio = 1.0 - prediction_match_ratio
    hook_accuracy_drift = abs(float(candidate["accuracy"] - hook["accuracy"]))
    hook_yes_ratio_drift = abs(float(candidate["yes_ratio"] - hook["yes_ratio"]))
    accuracy_drop = float(original["accuracy"] - candidate["accuracy"])
    yes_ratio_drift = abs(float(candidate["yes_ratio"] - original["yes_ratio"]))
    checks = {
        "prediction_agreement": prediction_mismatch_ratio <= prediction_mismatch_tolerance + 1e-12,
        "hook_accuracy_drift": hook_accuracy_drift <= hook_accuracy_drift_tolerance + 1e-12,
        "hook_yes_ratio_drift": hook_yes_ratio_drift <= hook_yes_ratio_drift_tolerance + 1e-12,
        "accuracy_drop": accuracy_drop <= accuracy_drop_tolerance,
        "yes_ratio_drift": yes_ratio_drift <= yes_ratio_drift_tolerance,
    }
    outside_boundary_count = sum(not row["near_decision_boundary"] for row in mismatches)
    boundary_diagnostic = {
        "reference_margin": mismatch_boundary_margin_tolerance,
        "all_mismatches_within_reference_margin": outside_boundary_count == 0,
        "outside_reference_margin_count": outside_boundary_count,
        "max_minimum_absolute_margin": max(
            (row["minimum_absolute_margin"] for row in mismatches),
            default=0.0,
        ),
        "is_hard_gate": False,
        "reason": "BF16 wide/narrow GEMM accumulation drift is shape-dependent; aggregate task gates are authoritative.",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "original": {key: original[key] for key in ("accuracy", "f1", "yes_ratio")},
        "hook": {key: hook[key] for key in ("accuracy", "f1", "yes_ratio")},
        "structural": {key: candidate[key] for key in ("accuracy", "f1", "yes_ratio")},
        "exact_prediction_match": not mismatches,
        "prediction_match_ratio": prediction_match_ratio,
        "prediction_mismatch_ratio": prediction_mismatch_ratio,
        "prediction_mismatch_count": len(mismatches),
        "prediction_mismatches": mismatches,
        "boundary_diagnostic": boundary_diagnostic,
        "hook_accuracy_drift": hook_accuracy_drift,
        "hook_yes_ratio_drift": hook_yes_ratio_drift,
        "accuracy_drop_from_original": accuracy_drop,
        "yes_ratio_drift_from_original": yes_ratio_drift,
        "prediction_mismatch_tolerance": prediction_mismatch_tolerance,
        "hook_accuracy_drift_tolerance": hook_accuracy_drift_tolerance,
        "hook_yes_ratio_drift_tolerance": hook_yes_ratio_drift_tolerance,
        "accuracy_drop_tolerance": accuracy_drop_tolerance,
        "yes_ratio_drift_tolerance": yes_ratio_drift_tolerance,
    }


def build_caption_command(
    args: argparse.Namespace, reference: dict[str, Any], pruned_model: Path, output_file: Path
) -> list[str]:
    config = reference["config"]
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_phase2_ablation.py"),
        "--config",
        args.config,
        "--model_name_or_path",
        str(pruned_model),
        "--score_file",
        args.score_file,
        "--output_file",
        str(output_file),
        "--ablation",
        "none",
        "--sample_offset",
        str(config["sample_offset"]),
        "--batch_size",
        str(args.batch_size),
    ]
    option(command, "--max_samples", config.get("max_samples"))
    option(command, "--dataset", config.get("dataset"))
    typing_manifest = Path(args.score_file).resolve().parent.parent / "activations/sample_manifest.json"
    calibration_manifest = Path(args.score_file).resolve().parent.parent / "calibration/sample_manifest.json"
    command.extend(
        [
            "--typing_manifest",
            str(typing_manifest),
            "--calibration_manifest",
            str(calibration_manifest),
            "--require_data_isolation",
        ]
    )
    return command


def build_pope_command(
    args: argparse.Namespace, reference: dict[str, Any], pruned_model: Path, output_file: Path
) -> list[str]:
    config = reference["config"]
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate_pope.py"),
        "--config",
        config["config"],
        "--model_name_or_path",
        str(pruned_model),
        "--score_file",
        config["score_file"],
        "--pope_file",
        config["pope_file"],
        "--output_file",
        str(output_file),
        "--sample_offset",
        str(config["sample_offset"]),
        "--image_offset",
        str(config["image_offset"]),
        "--seed",
        str(config["seed"]),
        "--bootstrap_samples",
        str(config["bootstrap_samples"]),
        "--bootstrap_seed",
        str(config["bootstrap_seed"]),
        "--batch_size",
        str(config["batch_size"]),
        "--max_image_repeat",
        str(config["max_image_repeat"]),
    ]
    for name in ("image_root", "max_samples", "max_images", "typing_manifest", "calibration_manifest"):
        option(command, f"--{name}", config.get(name))
    for manifest in config.get("exclude_manifest", []):
        command.extend(["--exclude_manifest", manifest])
    for name in (
        "filter_manifest_overlaps",
        "require_data_isolation",
        "allow_excessive_image_repeats",
        "include_shuffled_image_control",
    ):
        flag(command, f"--{name}", bool(config.get(name)))
    return command


def main() -> None:
    args = parse_args()
    if not args.skip_equivalence and not args.smoke_pope_file:
        raise ValueError("--smoke_pope_file is required unless --skip_equivalence is set.")
    if args.run_benchmark and not args.smoke_pope_file:
        raise ValueError("--smoke_pope_file is required for benchmarking.")
    if args.run_formal_evaluation and len(args.phase2_pope_result) != 3:
        raise ValueError("Formal evaluation requires exactly three --phase2_pope_result files.")
    if args.reuse_formal_evaluation and not args.run_formal_evaluation:
        raise ValueError("--reuse_formal_evaluation requires --run_formal_evaluation.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    model_dir = output_dir / "model"
    mask_path = mask_dir / f"{MASK_NAME}.mask.json"
    cluster_path = mask_dir / f"{MASK_NAME}.cluster_idx.json"
    metadata_path = mask_dir / f"{MASK_NAME}.metadata.json"
    state_path = output_dir / "config.json"
    previous_state = load_json(state_path) if state_path.is_file() else {}
    state: dict[str, Any] = {"config": vars(args), "stages": previous_state.get("stages", {})}
    state["complete"] = False

    build_command = [
        sys.executable,
        str(SCRIPT_DIR / "build_structural_pruning_artifact.py"),
        "--score_file",
        args.score_file,
        "--phase2_result",
        args.phase2_caption_result,
        "--output_dir",
        str(output_dir),
    ]
    run_command(build_command, stage="P3.0 frozen artifact")
    metadata = load_json(metadata_path)
    state["stages"]["artifact"] = {"passed": True, "mask_sha256": metadata["mask_sha256"]}
    write_json(state_path, state)

    if args.reuse_pruned_model:
        if not (model_dir / "pruning_summary.json").is_file():
            raise FileNotFoundError("--reuse_pruned_model requires model/pruning_summary.json.")
    else:
        save_command = [
            sys.executable,
            str(VULCAN_DIR / "save_pruned_model.py"),
            "--model_name_or_path",
            args.model_name_or_path,
            "--cluster_idx_path",
            str(cluster_path),
            "--output_dir",
            str(model_dir),
            "--config",
            args.config,
            "--provenance_path",
            str(metadata_path),
            "--expected_mask_sha256",
            metadata["mask_sha256"],
        ]
        run_command(save_command, stage="P3.1 save structural model")
    state["stages"]["model_save"] = {"passed": True, "path": str(model_dir)}
    write_json(state_path, state)

    if not args.skip_equivalence:
        equivalence_output = output_dir / "equivalence/equivalence_metrics.json"
        equivalence_command = [
            sys.executable,
            str(SCRIPT_DIR / "verify_structural_equivalence.py"),
            "--config",
            args.config,
            "--model_name_or_path",
            args.model_name_or_path,
            "--pruned_model_path",
            str(model_dir),
            "--mask_file",
            str(mask_path),
            "--cluster_idx_path",
            str(cluster_path),
            "--metadata_path",
            str(metadata_path),
            "--pope_file",
            args.smoke_pope_file,
            "--output_file",
            str(equivalence_output),
            "--max_samples",
            str(args.smoke_samples),
            "--batch_size",
            str(args.batch_size),
        ]
        option(equivalence_command, "--image_root", args.image_root)
        run_command(equivalence_command, stage="P3.1 equivalence gate")
        state["stages"]["equivalence"] = load_json(equivalence_output)
        write_json(state_path, state)

    if args.run_formal_evaluation:
        evaluation_dir = output_dir / "evaluation"
        caption_reference = load_json(args.phase2_caption_result)
        caption_output = evaluation_dir / "caption_nll.json"
        if args.reuse_formal_evaluation:
            if not caption_output.is_file():
                raise FileNotFoundError(f"Missing reusable formal caption result: {caption_output}")
            print(f"[P3.2 caption NLL] Reusing {caption_output}", flush=True)
        else:
            run_command(
                build_caption_command(args, caption_reference, model_dir, caption_output),
                stage="P3.2 caption NLL",
            )
        quality = {
            "caption": compare_caption_quality(caption_reference, load_json(caption_output)),
            "pope": {},
        }
        for reference_path in args.phase2_pope_result:
            reference = load_json(reference_path)
            split = Path(reference["config"]["pope_file"]).stem.removeprefix("coco_pope_")
            output = evaluation_dir / f"pope_{split}.json"
            if args.reuse_formal_evaluation:
                if not output.is_file():
                    raise FileNotFoundError(f"Missing reusable formal POPE result: {output}")
                print(f"[P3.2 POPE {split}] Reusing {output}", flush=True)
            else:
                run_command(build_pope_command(args, reference, model_dir, output), stage=f"P3.2 POPE {split}")
            quality["pope"][split] = compare_pope_quality(reference, load_json(output))
        quality["passed"] = quality["caption"]["passed"] and all(row["passed"] for row in quality["pope"].values())
        write_json(evaluation_dir / "quality_gate.json", quality)
        state["stages"]["formal_quality"] = quality
        write_json(state_path, state)
        if not quality["passed"]:
            raise RuntimeError("Phase-3 formal quality gate failed.")

    if args.run_benchmark:
        benchmark_output = output_dir / "benchmark/summary.json"
        benchmark_command = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_structural_pruning.py"),
            "--config",
            args.config,
            "--original_model_path",
            args.model_name_or_path,
            "--pruned_model_path",
            str(model_dir),
            "--pope_file",
            args.smoke_pope_file,
            "--output_file",
            str(benchmark_output),
            "--batch_sizes",
            args.benchmark_batch_sizes,
            "--warmup",
            str(args.benchmark_warmup),
            "--repeats",
            str(args.benchmark_repeats),
            "--max_new_tokens",
            str(args.benchmark_max_new_tokens),
        ]
        option(benchmark_command, "--image_root", args.image_root)
        run_command(benchmark_command, stage="P3.3 efficiency benchmark")
        state["stages"]["benchmark"] = load_json(benchmark_output)["comparison"]
        write_json(state_path, state)

    state["complete"] = True
    write_json(state_path, state)
    print(json.dumps({"complete": True, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
