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

"""Evaluate post-SFT language retention on the fixed 500-example C4 slice."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT_DIR / "scripts" / "vulcan" / "neuron_typing" / "run_phase2_ablation.py"
DEFAULT_MODEL = "/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b"
DEFAULT_MATRIX_DIR = "saves/qwen35-0_8b-vqa-rad/neupat_matrix_20260902"
RUN_NAMES = ("base", "vanilla_full", "lora", "neupat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the NeuPAT SFT matrix for language retention.")
    parser.add_argument(
        "--config",
        default="scripts/vulcan/neuron_typing/configs/neupat_language_eval.formal.yaml",
    )
    parser.add_argument(
        "--score_file",
        default="saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet",
    )
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--matrix_dir", default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--output_dir", default=f"{DEFAULT_MATRIX_DIR}/language_eval_c4_500")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260902)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def completed_training(model_dir: Path, *, adapter: bool = False) -> bool:
    model_marker = model_dir / ("adapter_config.json" if adapter else "config.json")
    return model_marker.is_file() and (model_dir / "all_results.json").is_file()


def evaluation_command(args: argparse.Namespace, name: str, output_file: Path) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--config",
        args.config,
        "--score_file",
        args.score_file,
        "--output_file",
        str(output_file),
        "--ablation",
        "none",
        "--dataset_stage",
        "pt",
        "--max_samples",
        str(args.max_samples),
        "--sample_offset",
        "0",
        "--batch_size",
        "1",
        "--num_workers",
        "0",
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.bootstrap_seed),
    ]
    matrix_dir = Path(args.matrix_dir)
    if name == "base":
        command.extend(("--model_name_or_path", args.base_model))
    elif name == "lora":
        adapter_dir = matrix_dir / name
        command.extend(
            ("--model_name_or_path", args.base_model, "finetuning_type=lora", f"adapter_name_or_path={adapter_dir}")
        )
    else:
        command.extend(("--model_name_or_path", str(matrix_dir / name)))
    return command


def load_per_example(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["metrics"]["none"]
    rows = metrics["per_example"]
    if not rows:
        raise ValueError(f"No per-example metrics in {path}.")
    return metrics, rows


def paired_weighted_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Estimate token-weighted NLL(left - right) with paired resampling."""
    if len(left) != len(right):
        raise ValueError("Paired evaluations have different sample counts.")
    keys_left = [(row["source_index"], row["token_count"]) for row in left]
    keys_right = [(row["source_index"], row["token_count"]) for row in right]
    if keys_left != keys_right:
        raise ValueError("Paired evaluations are not aligned by source index and token count.")

    differences = [float(a["nll_sum"]) - float(b["nll_sum"]) for a, b in zip(left, right)]
    token_counts = [int(row["token_count"]) for row in left]
    observed = sum(differences) / sum(token_counts)
    rng = random.Random(seed)
    bootstraps = []
    for _ in range(samples):
        indices = [rng.randrange(len(left)) for _ in left]
        bootstraps.append(sum(differences[index] for index in indices) / sum(token_counts[index] for index in indices))
    bootstraps.sort()
    return {
        "delta_nll": observed,
        "ci_low": bootstraps[int(0.025 * (samples - 1))],
        "ci_high": bootstraps[int(0.975 * (samples - 1))],
        "num_examples": len(left),
        "num_tokens": sum(token_counts),
    }


def main() -> None:
    args = parse_args()
    if args.max_samples != 500:
        raise ValueError("The formal language-retention evaluation requires exactly 500 examples.")
    if args.bootstrap_samples < 1000:
        raise ValueError("Use at least 1000 paired bootstrap resamples for the formal comparison.")

    matrix_dir = Path(args.matrix_dir)
    if not args.dry_run:
        for name in RUN_NAMES[1:]:
            if not completed_training(matrix_dir / name, adapter=name == "lora"):
                raise FileNotFoundError(f"Incomplete matrix checkpoint: {matrix_dir / name}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["WANDB_DISABLED"] = "true"
    output_files = {name: output_dir / f"{name}.json" for name in RUN_NAMES}
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

    loaded = {name: load_per_example(path) for name, path in output_files.items()}
    metrics = {
        name: {key: value for key, value in data[0].items() if key != "per_example"} for name, data in loaded.items()
    }
    comparisons = {}
    for name in RUN_NAMES[1:]:
        comparisons[f"{name}_minus_base"] = paired_weighted_bootstrap(
            loaded[name][1],
            loaded["base"][1],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
    comparisons["neupat_minus_vanilla_full"] = paired_weighted_bootstrap(
        loaded["neupat"][1],
        loaded["vanilla_full"][1],
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1,
    )
    primary = comparisons["neupat_minus_vanilla_full"]
    summary = {
        "complete": True,
        "primary_hypothesis": "NeuPAT retains more held-out language ability than matched vanilla full-SFT.",
        "primary_criterion": "95% paired bootstrap CI for NLL(NeuPAT)-NLL(vanilla full-SFT) is below zero.",
        "gate_passed": primary["ci_high"] < 0.0,
        "metrics": metrics,
        "comparisons": comparisons,
    }
    (output_dir / "language_retention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
