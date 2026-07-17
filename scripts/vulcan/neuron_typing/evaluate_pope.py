#!/usr/bin/env python3
"""POPE evaluation using the shared image-aware binary VQA evaluator."""

from __future__ import annotations

import argparse

from evaluate_vqa import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POPE typed-neuron ablation evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--pope_file", required=True, help="POPE JSON or JSONL file.")
    parser.add_argument("--image_root", default=None, help="Root directory for relative COCO image paths.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--ablation", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--require_data_isolation", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=5)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
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
