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

"""Evaluate the post-SFT matrix on the held-out VQA-RAD test set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate_neupat_sft_language import DEFAULT_MATRIX_DIR, DEFAULT_MODEL, RUN_NAMES, completed_training
from evaluate_vqa import paired_binary_analysis


ROOT_DIR = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT_DIR / "scripts" / "vulcan" / "neuron_typing" / "evaluate_vqa.py"
DEFAULT_SCORE_FILE = "saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the NeuPAT SFT matrix on VQA-RAD.")
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--matrix_dir", default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--output_dir", default=f"{DEFAULT_MATRIX_DIR}/vqa_rad_eval")
    parser.add_argument("--score_file", default=DEFAULT_SCORE_FILE)
    parser.add_argument("--vqa_file", default="datasets/vqa_rad/test.jsonl")
    parser.add_argument("--image_root", default="datasets/vqa_rad")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260902)
    parser.add_argument("--noninferiority_margin", type=float, default=0.02)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def evaluation_command(args: argparse.Namespace, name: str, output_file: Path) -> list[str]:
    config = {
        "base": "examples/vulcan/qwen35_08b_vqa_rad_vanilla_full_neupat_matrix.yaml",
        "vanilla_full": "examples/vulcan/qwen35_08b_vqa_rad_vanilla_full_neupat_matrix.yaml",
        "lora": "examples/vulcan/qwen35_08b_vqa_rad_lora_neupat_matrix.yaml",
        "neupat": "examples/vulcan/qwen35_08b_vqa_rad_neupat_sft.yaml",
    }[name]
    command = [
        sys.executable,
        str(EVALUATOR),
        "--config",
        config,
        "--score_file",
        args.score_file,
        "--vqa_file",
        args.vqa_file,
        "--image_root",
        args.image_root,
        "--output_file",
        str(output_file),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.bootstrap_seed),
    ]
    matrix_dir = Path(args.matrix_dir)
    if name == "base":
        command.extend(("--model_name_or_path", args.base_model))
    elif name == "lora":
        command.extend(
            (
                "--model_name_or_path",
                args.base_model,
                "--adapter_name_or_path",
                str(matrix_dir / "lora"),
            )
        )
    else:
        command.extend(("--model_name_or_path", str(matrix_dir / name)))
    return command


def load_metrics(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("complete"):
        raise ValueError(f"Incomplete evaluation: {path}")
    return result["metrics"]["none"]


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 1000:
        raise ValueError("Use at least 1000 image-cluster bootstrap resamples for the formal comparison.")
    matrix_dir = Path(args.matrix_dir)
    if not args.dry_run:
        for name in RUN_NAMES[1:]:
            if not completed_training(matrix_dir / name, adapter=name == "lora"):
                raise FileNotFoundError(f"Incomplete matrix checkpoint: {matrix_dir / name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {name: output_dir / f"{name}.json" for name in RUN_NAMES}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["WANDB_DISABLED"] = "true"
    for name, output_file in output_files.items():
        if output_file.is_file():
            print(f"Skipping completed evaluation {name}: {output_file}", flush=True)
            continue
        command = evaluation_command(args, name, output_file)
        print(" ".join(map(str, command)), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)
    if args.dry_run:
        return

    loaded = {name: load_metrics(path) for name, path in output_files.items()}
    comparisons = {}
    for index, name in enumerate(RUN_NAMES[1:]):
        comparisons[f"{name}_minus_base"] = paired_binary_analysis(
            loaded["base"]["predictions"],
            loaded[name]["predictions"],
            num_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed + index,
        )
    comparisons["neupat_minus_vanilla_full"] = paired_binary_analysis(
        loaded["vanilla_full"]["predictions"],
        loaded["neupat"]["predictions"],
        num_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed + 3,
    )
    comparison = comparisons["neupat_minus_vanilla_full"]
    ci_low = comparison["delta_accuracy_ci95"][0]
    summary = {
        "complete": True,
        "criterion": (
            "NeuPAT is non-inferior to vanilla full-SFT when the lower 95% image-cluster bootstrap bound "
            f"for accuracy(NeuPAT)-accuracy(vanilla) exceeds -{args.noninferiority_margin:.3f}."
        ),
        "noninferiority_passed": ci_low > -args.noninferiority_margin,
        "metrics": {
            name: {key: value for key, value in metrics.items() if key != "predictions"}
            for name, metrics in loaded.items()
        },
        "comparisons": comparisons,
    }
    (output_dir / "vqa_rad_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
