#!/usr/bin/env python3
"""POPE evaluation using the shared image-aware binary VQA evaluator."""

from __future__ import annotations

import argparse

from evaluate_vqa import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POPE typed-neuron ablation evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        help="Optional model-path override, used by Phase-3 structural checkpoint evaluation.",
    )
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--pope_file", required=True, help="POPE JSON or JSONL file.")
    parser.add_argument("--image_root", default=None, help="Root directory for relative COCO image paths.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=None, help="Number of question rows after filtering.")
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Select complete image groups instead of truncating question rows (recommended).",
    )
    parser.add_argument("--image_offset", type=int, default=0)
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--filter_manifest_overlaps", action="store_true")
    parser.add_argument("--require_data_isolation", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=6)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--include_shuffled_image_control", action="store_true")
    parser.add_argument("--shuffled_image_seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true", help="Resume completed conditions from output_file.")
    args = parser.parse_args()
    args.vqa_file = args.pope_file
    return args


def main() -> None:
    result = run_evaluation(parse_args(), task_name="pope")
    for name, metrics in result["metrics"].items():
        print(
            f"{name:35s} accuracy={metrics['accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} yes_ratio={metrics['yes_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
