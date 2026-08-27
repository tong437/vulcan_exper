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

"""Run the Phase-3.4 3,072-wide hardware-aligned q-band experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from build_aligned_pruning_artifact import DEFAULT_NAME, DEFAULT_SPEC
from phase3_structural_utils import sha256_file
from run_phase3_structural import (
    build_caption_command,
    build_pope_command,
    compare_caption_quality,
    compare_pope_quality,
    load_json,
    option,
    run_command,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
VULCAN_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase-3.4 aligned q-band validation and pruning.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--phase2_caption_result", required=True)
    parser.add_argument("--phase2_pope_result", action="append", default=[])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smoke_pope_file", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--smoke_samples", type=int, default=16)
    parser.add_argument("--reuse_hook_evaluation", action="store_true")
    parser.add_argument("--run_structural", action="store_true")
    parser.add_argument("--reuse_pruned_model", action="store_true")
    parser.add_argument("--skip_equivalence", action="store_true")
    parser.add_argument("--reuse_structural_evaluation", action="store_true")
    parser.add_argument("--run_benchmark", action="store_true")
    parser.add_argument(
        "--unaligned_model_path",
        default=None,
        help="Optional 3,047-wide checkpoint to rerun with the corrected benchmark for direct comparison.",
    )
    parser.add_argument("--benchmark_warmup", type=int, default=10)
    parser.add_argument("--benchmark_repeats", type=int, default=50)
    parser.add_argument("--benchmark_batch_sizes", default="1,4")
    parser.add_argument("--benchmark_max_new_tokens", type=int, default=64)
    parser.add_argument("--benchmark_bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


def prediction_match_ratio(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    if len(left) != len(right):
        raise ValueError("Prediction lists have different lengths.")
    matches = 0
    for left_row, right_row in zip(left, right):
        if left_row["source_index"] != right_row["source_index"]:
            raise ValueError("Prediction lists are not aligned by source_index.")
        matches += left_row["prediction"] == right_row["prediction"]
    return matches / len(left) if left else 1.0


def compare_aligned_hook_caption(
    reference: dict[str, Any],
    aligned: dict[str, Any],
    *,
    baseline_reproduction_tolerance: float = 0.01,
    original_nll_degradation_tolerance: float = 0.05,
    frozen_qband_nll_degradation_tolerance: float = 0.05,
) -> dict[str, Any]:
    reference_original = reference["metrics"]["none"]
    reference_qband = reference["metrics"]["rank_band:multimodal:0.05:0.2"]
    rerun_original = aligned["metrics"]["none"]
    candidate = aligned["metrics"][DEFAULT_SPEC]
    baseline_delta = float(rerun_original["nll"] - reference_original["nll"])
    original_delta = float(candidate["nll"] - rerun_original["nll"])
    qband_delta = float(candidate["nll"] - reference_qband["nll"])
    checks = {
        "baseline_reproduction": abs(baseline_delta) <= baseline_reproduction_tolerance,
        "original_nll_degradation": original_delta <= original_nll_degradation_tolerance,
        "frozen_qband_nll_degradation": qband_delta <= frozen_qband_nll_degradation_tolerance,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reference_original_nll": reference_original["nll"],
        "rerun_original_nll": rerun_original["nll"],
        "frozen_qband_nll": reference_qband["nll"],
        "aligned_nll": candidate["nll"],
        "baseline_reproduction_delta": baseline_delta,
        "aligned_minus_original_nll": original_delta,
        "aligned_minus_frozen_qband_nll": qband_delta,
        "baseline_reproduction_tolerance": baseline_reproduction_tolerance,
        "original_nll_degradation_tolerance": original_nll_degradation_tolerance,
        "frozen_qband_nll_degradation_tolerance": frozen_qband_nll_degradation_tolerance,
    }


def compare_aligned_hook_pope(reference: dict[str, Any], aligned: dict[str, Any]) -> dict[str, Any]:
    reference_original = reference["metrics"]["none"]
    rerun_original = aligned["metrics"]["none"]
    candidate = aligned["metrics"][DEFAULT_SPEC]
    baseline_match = prediction_match_ratio(reference_original["predictions"], rerun_original["predictions"])
    accuracy_drop = float(rerun_original["accuracy"] - candidate["accuracy"])
    yes_ratio_drift = abs(float(candidate["yes_ratio"] - rerun_original["yes_ratio"]))
    checks = {
        "baseline_prediction_reproduction": baseline_match == 1.0,
        "accuracy_drop": accuracy_drop <= 0.01,
        "yes_ratio_drift": yes_ratio_drift <= 0.02,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reference_original": {key: reference_original[key] for key in ("accuracy", "f1", "yes_ratio")},
        "rerun_original": {key: rerun_original[key] for key in ("accuracy", "f1", "yes_ratio")},
        "aligned": {key: candidate[key] for key in ("accuracy", "f1", "yes_ratio")},
        "baseline_prediction_match_ratio": baseline_match,
        "accuracy_drop_from_original": accuracy_drop,
        "yes_ratio_drift_from_original": yes_ratio_drift,
        "accuracy_drop_tolerance": 0.01,
        "yes_ratio_drift_tolerance": 0.02,
    }


def require_reusable(path: Path, stage: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"[{stage}] Missing reusable result: {path}")
    print(f"[{stage}] Reusing {path}", flush=True)


def main() -> None:
    args = parse_args()
    if len(args.phase2_pope_result) != 3:
        raise ValueError("P3.4 requires exactly three --phase2_pope_result files.")
    if args.run_structural and not args.skip_equivalence and not args.smoke_pope_file:
        raise ValueError("Structural equivalence requires --smoke_pope_file.")
    if args.run_benchmark and (not args.run_structural or not args.smoke_pope_file):
        raise ValueError("Benchmarking requires --run_structural and --smoke_pope_file.")
    if args.reuse_pruned_model and not args.run_structural:
        raise ValueError("--reuse_pruned_model requires --run_structural.")
    if args.reuse_structural_evaluation and not args.run_structural:
        raise ValueError("--reuse_structural_evaluation requires --run_structural.")

    output_dir = Path(args.output_dir).resolve()
    hook_dir = output_dir / "hook_evaluation"
    structural_dir = output_dir / "structural_evaluation"
    model_dir = output_dir / "model"
    mask_dir = output_dir / "masks"
    state_path = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    previous_state = load_json(state_path) if state_path.is_file() else {}
    state: dict[str, Any] = {"config": vars(args), "stages": previous_state.get("stages", {}), "complete": False}

    caption_reference = load_json(args.phase2_caption_result)
    caption_hook_output = hook_dir / "caption_nll.json"
    if args.reuse_hook_evaluation:
        require_reusable(caption_hook_output, "P3.4 hook caption")
    else:
        command = build_caption_command(args, caption_reference, Path(args.model_name_or_path), caption_hook_output)
        command.extend(["--ablation", DEFAULT_SPEC])
        run_command(command, stage="P3.4 hook caption")

    hook_quality: dict[str, Any] = {
        "ablation_spec": DEFAULT_SPEC,
        "caption": compare_aligned_hook_caption(caption_reference, load_json(caption_hook_output)),
        "pope": {},
        "inputs": {
            "score_file": {"path": str(Path(args.score_file).resolve()), "sha256": sha256_file(args.score_file)},
            "caption_result": {
                "path": str(caption_hook_output),
                "sha256": sha256_file(caption_hook_output),
            },
        },
    }
    hook_pope_outputs: dict[str, Path] = {}
    for reference_path in args.phase2_pope_result:
        reference = load_json(reference_path)
        split = Path(reference["config"]["pope_file"]).stem.removeprefix("coco_pope_")
        output = hook_dir / f"pope_{split}.json"
        hook_pope_outputs[split] = output
        if args.reuse_hook_evaluation:
            require_reusable(output, f"P3.4 hook POPE {split}")
        else:
            command = build_pope_command(args, reference, Path(args.model_name_or_path), output)
            command.extend(["--ablation", DEFAULT_SPEC])
            run_command(command, stage=f"P3.4 hook POPE {split}")
        hook_quality["pope"][split] = compare_aligned_hook_pope(reference, load_json(output))
        hook_quality["inputs"][f"pope_{split}_result"] = {
            "path": str(output),
            "sha256": sha256_file(output),
        }
    hook_quality["passed"] = hook_quality["caption"]["passed"] and all(
        row["passed"] for row in hook_quality["pope"].values()
    )
    hook_gate_path = hook_dir / "quality_gate.json"
    write_json(hook_gate_path, hook_quality)
    state["stages"]["aligned_hook_quality"] = hook_quality
    write_json(state_path, state)
    if not hook_quality["passed"]:
        raise RuntimeError("P3.4 aligned hook safety gate failed; structural pruning is forbidden.")

    artifact_command = [
        sys.executable,
        str(SCRIPT_DIR / "build_aligned_pruning_artifact.py"),
        "--score_file",
        args.score_file,
        "--hook_caption_result",
        str(caption_hook_output),
        "--hook_quality_gate",
        str(hook_gate_path),
        "--output_dir",
        str(output_dir),
    ]
    run_command(artifact_command, stage="P3.4 frozen aligned artifact")
    metadata_path = mask_dir / f"{DEFAULT_NAME}.metadata.json"
    mask_path = mask_dir / f"{DEFAULT_NAME}.mask.json"
    cluster_path = mask_dir / f"{DEFAULT_NAME}.cluster_idx.json"
    metadata = load_json(metadata_path)
    state["stages"]["aligned_artifact"] = {
        "passed": True,
        "mask_sha256": metadata["mask_sha256"],
        "target_intermediate_size": metadata["target_intermediate_size"],
    }
    write_json(state_path, state)

    if not args.run_structural:
        state["complete"] = True
        write_json(state_path, state)
        print(json.dumps({"complete": True, "hook_only": True, "output_dir": str(output_dir)}, indent=2))
        return

    if args.reuse_pruned_model:
        require_reusable(model_dir / "pruning_summary.json", "P3.4 structural model")
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
        run_command(save_command, stage="P3.4 save aligned structural model")
    state["stages"]["model_save"] = {"passed": True, "path": str(model_dir)}
    write_json(state_path, state)

    if not args.skip_equivalence:
        equivalence_output = output_dir / "equivalence/equivalence_metrics.json"
        command = [
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
        option(command, "--image_root", args.image_root)
        run_command(command, stage="P3.4 structural equivalence")
        state["stages"]["equivalence"] = load_json(equivalence_output)
        write_json(state_path, state)

    caption_structural_output = structural_dir / "caption_nll.json"
    if args.reuse_structural_evaluation:
        require_reusable(caption_structural_output, "P3.4 structural caption")
    else:
        run_command(
            build_caption_command(args, load_json(caption_hook_output), model_dir, caption_structural_output),
            stage="P3.4 structural caption",
        )
    structural_quality: dict[str, Any] = {
        "caption": compare_caption_quality(
            load_json(caption_hook_output),
            load_json(caption_structural_output),
            hook_result_name=DEFAULT_SPEC,
        ),
        "pope": {},
    }
    for split, hook_output in hook_pope_outputs.items():
        structural_output = structural_dir / f"pope_{split}.json"
        if args.reuse_structural_evaluation:
            require_reusable(structural_output, f"P3.4 structural POPE {split}")
        else:
            run_command(
                build_pope_command(args, load_json(hook_output), model_dir, structural_output),
                stage=f"P3.4 structural POPE {split}",
            )
        structural_quality["pope"][split] = compare_pope_quality(
            load_json(hook_output),
            load_json(structural_output),
            hook_result_name=DEFAULT_SPEC,
        )
    structural_quality["passed"] = structural_quality["caption"]["passed"] and all(
        row["passed"] for row in structural_quality["pope"].values()
    )
    structural_gate_path = structural_dir / "quality_gate.json"
    write_json(structural_gate_path, structural_quality)
    state["stages"]["structural_quality"] = structural_quality
    write_json(state_path, state)
    if not structural_quality["passed"]:
        raise RuntimeError("P3.4 aligned structural quality gate failed.")

    if args.run_benchmark:
        benchmark_output = output_dir / "benchmark/summary.json"

        def benchmark_command(candidate_path: str | Path, output_file: Path) -> list[str]:
            command = [
                sys.executable,
                str(SCRIPT_DIR / "benchmark_structural_pruning.py"),
                "--config",
                args.config,
                "--original_model_path",
                args.model_name_or_path,
                "--pruned_model_path",
                str(candidate_path),
                "--pope_file",
                args.smoke_pope_file,
                "--output_file",
                str(output_file),
                "--batch_sizes",
                args.benchmark_batch_sizes,
                "--warmup",
                str(args.benchmark_warmup),
                "--repeats",
                str(args.benchmark_repeats),
                "--max_new_tokens",
                str(args.benchmark_max_new_tokens),
                "--bootstrap_samples",
                str(args.benchmark_bootstrap_samples),
            ]
            option(command, "--image_root", args.image_root)
            return command

        run_command(
            benchmark_command(model_dir, benchmark_output),
            stage="P3.4 aligned efficiency benchmark",
        )
        state["stages"]["benchmark"] = load_json(benchmark_output)["comparison"]
        if args.unaligned_model_path:
            unaligned_output = output_dir / "benchmark/unaligned_3047_summary.json"
            run_command(
                benchmark_command(args.unaligned_model_path, unaligned_output),
                stage="P3.4 corrected 3,047 benchmark",
            )
            state["stages"]["benchmark_unaligned_3047"] = load_json(unaligned_output)["comparison"]
        write_json(state_path, state)

    state["complete"] = True
    write_json(state_path, state)
    print(json.dumps({"complete": True, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
