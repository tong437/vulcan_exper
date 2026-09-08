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

"""Evaluate and select a vanilla full-SFT language-forgetting stress setting."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate_neupat_sft_language import completed_training, load_per_example, paired_weighted_bootstrap
from evaluate_vqa import paired_binary_analysis
from run_neupat_forgetting_stress import DEFAULT_CONFIG, DEFAULT_OUTPUT_ROOT, STRESS_LADDER


ROOT_DIR = Path(__file__).resolve().parents[3]
LANGUAGE_EVALUATOR = ROOT_DIR / "scripts" / "vulcan" / "neuron_typing" / "run_phase2_ablation.py"
VQA_EVALUATOR = ROOT_DIR / "scripts" / "vulcan" / "neuron_typing" / "evaluate_vqa.py"
DEFAULT_MODEL = "/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b"
DEFAULT_LANGUAGE_CONFIG = "scripts/vulcan/neuron_typing/configs/neupat_language_stress_dev.yaml"
DEFAULT_SCORE_FILE = "saves/neuron_typing/neupat_overlap_formal_2048/neupat_combined_scores.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the NeuPAT vanilla forgetting-stress ladder on dev only.")
    parser.add_argument("--base_model", default=DEFAULT_MODEL)
    parser.add_argument("--training_config", default=DEFAULT_CONFIG)
    parser.add_argument("--language_config", default=DEFAULT_LANGUAGE_CONFIG)
    parser.add_argument("--score_file", default=DEFAULT_SCORE_FILE)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate", action="append", choices=tuple(STRESS_LADDER))
    parser.add_argument("--max_language_samples", type=int, default=500)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260904)
    parser.add_argument("--minimum_ppl_increase", type=float, default=0.05)
    parser.add_argument("--minimum_vqa_accuracy_gain", type=float, default=0.02)
    parser.add_argument("--base_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def language_command(args: argparse.Namespace, model: str, output_file: Path) -> list[str]:
    return [
        sys.executable,
        str(LANGUAGE_EVALUATOR),
        "--config",
        args.language_config,
        "--score_file",
        args.score_file,
        "--output_file",
        str(output_file),
        "--ablation",
        "none",
        "--dataset_stage",
        "pt",
        "--max_samples",
        str(args.max_language_samples),
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
        "--model_name_or_path",
        model,
    ]


def vqa_command(args: argparse.Namespace, model: str, output_file: Path) -> list[str]:
    return [
        sys.executable,
        str(VQA_EVALUATOR),
        "--config",
        args.training_config,
        "--score_file",
        args.score_file,
        "--vqa_file",
        "datasets/vqa_rad/stress_dev.jsonl",
        "--image_root",
        "datasets/vqa_rad",
        "--allow_excessive_image_repeats",
        "--output_file",
        str(output_file),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--bootstrap_seed",
        str(args.bootstrap_seed),
        "--model_name_or_path",
        model,
    ]


def run_once(command: list[str], output_file: Path, env: dict[str, str], *, dry_run: bool) -> None:
    if output_file.is_file():
        print(f"Skipping completed dev evaluation: {output_file}", flush=True)
        return
    print(" ".join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)


def load_vqa_metrics(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("complete") or "none" not in result.get("metrics", {}):
        raise ValueError(f"Incomplete VQA dev evaluation: {path}")
    return result["metrics"]["none"]


def main() -> None:
    args = parse_args()
    if args.max_language_samples != 500:
        raise ValueError("Stress discovery is frozen to exactly 500 packed language-dev examples.")
    if args.bootstrap_samples < 1000:
        raise ValueError("Use at least 1000 paired bootstrap samples.")
    if not 0.0 < args.minimum_ppl_increase < 1.0:
        raise ValueError("minimum_ppl_increase must be a fraction strictly between zero and one.")

    output_root = Path(args.output_root)
    eval_root = output_root / "dev_eval"
    language_root = eval_root / "language"
    vqa_root = eval_root / "vqa"
    language_root.mkdir(parents=True, exist_ok=True)
    vqa_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["WANDB_DISABLED"] = "true"

    language_files = {"base": language_root / "base.json"}
    vqa_files = {"base": vqa_root / "base.json"}
    run_once(
        language_command(args, args.base_model, language_files["base"]),
        language_files["base"],
        env,
        dry_run=args.dry_run,
    )
    run_once(vqa_command(args, args.base_model, vqa_files["base"]), vqa_files["base"], env, dry_run=args.dry_run)

    names = [] if args.base_only else (args.candidate or list(STRESS_LADDER))
    for name in names:
        model_dir = output_root / name
        if not completed_training(model_dir):
            print(f"Candidate is not complete; skipping dev evaluation: {name}", flush=True)
            continue
        language_files[name] = language_root / f"{name}.json"
        vqa_files[name] = vqa_root / f"{name}.json"
        run_once(
            language_command(args, str(model_dir), language_files[name]),
            language_files[name],
            env,
            dry_run=args.dry_run,
        )
        run_once(vqa_command(args, str(model_dir), vqa_files[name]), vqa_files[name], env, dry_run=args.dry_run)
    if args.dry_run:
        return

    base_language, base_language_rows = load_per_example(language_files["base"])
    base_vqa = load_vqa_metrics(vqa_files["base"])
    nll_threshold = math.log1p(args.minimum_ppl_increase)
    candidates: dict[str, Any] = {}
    available = [
        name
        for name in STRESS_LADDER
        if (language_root / f"{name}.json").is_file() and (vqa_root / f"{name}.json").is_file()
    ]
    for name in available:
        language_files[name] = language_root / f"{name}.json"
        vqa_files[name] = vqa_root / f"{name}.json"
        language, language_rows = load_per_example(language_files[name])
        vqa = load_vqa_metrics(vqa_files[name])
        language_delta = paired_weighted_bootstrap(
            language_rows,
            base_language_rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        vqa_delta = paired_binary_analysis(
            base_vqa["predictions"],
            vqa["predictions"],
            num_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        )
        forgetting_passed = language_delta["delta_nll"] >= nll_threshold and language_delta["ci_low"] > 0.0
        adaptation_passed = (
            vqa_delta["delta_accuracy"] >= args.minimum_vqa_accuracy_gain and vqa_delta["delta_f1"] >= 0.0
        )
        candidates[name] = {
            "hyperparameters": STRESS_LADDER[name],
            "language": {key: value for key, value in language.items() if key != "per_example"},
            "vqa": {key: value for key, value in vqa.items() if key != "predictions"},
            "language_minus_base": language_delta,
            "vqa_minus_base": vqa_delta,
            "forgetting_passed": forgetting_passed,
            "multimodal_adaptation_passed": adaptation_passed,
            "qualifies_as_stress_regime": forgetting_passed and adaptation_passed,
        }

    first_qualifying = next(
        (name for name in STRESS_LADDER if candidates.get(name, {}).get("qualifies_as_stress_regime")),
        None,
    )
    first_missing = next((name for name in STRESS_LADDER if name not in candidates), None)
    summary = {
        "artifact_version": 1,
        "complete_for_available_candidates": True,
        "selection_scope": "development data only; the language lockbox was not accessed",
        "selection_order": list(STRESS_LADDER),
        "criteria": {
            "language_forgetting": (
                f"NLL increase >= log(1 + {args.minimum_ppl_increase}) = {nll_threshold:.9f} "
                "and paired 95% CI lower bound > 0"
            ),
            "multimodal_adaptation": (
                f"VQA dev accuracy increase >= {args.minimum_vqa_accuracy_gain} and F1 does not decrease"
            ),
        },
        "base": {
            "language": {key: value for key, value in base_language.items() if key != "per_example"},
            "vqa": {key: value for key, value in base_vqa.items() if key != "predictions"},
        },
        "candidates": candidates,
        "first_qualifying_candidate": first_qualifying,
        "recommended_next_candidate": None if first_qualifying else first_missing,
        "stress_regime_found": first_qualifying is not None,
    }
    summary_path = eval_root / "stress_selection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
