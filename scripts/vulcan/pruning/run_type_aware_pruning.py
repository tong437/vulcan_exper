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

"""Phase 3: Type-Aware Pruning Experiments.

Compare type-aware pruning against baseline methods.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: Type-Aware Pruning.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--neuron_scores", required=True, help="Path to neuron_type_scores.parquet.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--pruning_ratios", default="0.05,0.10,0.15,0.20,0.30",
                        help="Comma-separated pruning ratios.")
    parser.add_argument("--eval_tasks", default="caption,vqa,text",
                        help="Comma-separated evaluation tasks.")
    parser.add_argument("--skip_pruning_score", action="store_true", help="Skip pruning score computation.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation.")
    parser.add_argument("--skip_plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def run_command(cmd: list[str], desc: str):
    """Run a subprocess command."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with return code {result.returncode}")
        sys.exit(1)


def plot_pruning_comparison(results_path: Path, output_path: Path):
    """Plot pruning ratio vs score for all methods."""
    with open(results_path) as f:
        results = json.load(f)

    methods = set()
    ratios = set()
    for entry in results:
        methods.add(entry["method"])
        ratios.add(entry["ratio"])

    methods = sorted(methods)
    ratios = sorted(ratios)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    tasks = ["caption", "vqa", "text"]
    task_metrics = {"caption": "cider", "vqa": "accuracy", "text": "perplexity"}

    for ax, task in zip(axes, tasks):
        for method in methods:
            scores = []
            for ratio in ratios:
                entry = next((e for e in results if e["method"] == method and e["ratio"] == ratio), None)
                if entry and task in entry.get("evaluations", {}):
                    metric = task_metrics[task]
                    scores.append(entry["evaluations"][task].get(metric, 0))
                else:
                    scores.append(None)

            valid_ratios = [r for r, s in zip(ratios, scores) if s is not None]
            valid_scores = [s for s in scores if s is not None]
            if valid_scores:
                ax.plot(valid_ratios, valid_scores, marker="o", label=method)

        ax.set_xlabel("Pruning Ratio")
        ax.set_ylabel(task_metrics[task])
        ax.set_title(f"{task.upper()} Performance")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def plot_type_aware_vs_baselines(results_path: Path, output_path: Path, target_ratio: float = 0.2):
    """Bar chart comparing methods at a fixed pruning ratio."""
    with open(results_path) as f:
        results = json.load(f)

    filtered = [e for e in results if e["ratio"] == target_ratio]
    if not filtered:
        print(f"No results for ratio={target_ratio}")
        return

    methods = [e["method"] for e in filtered]
    tasks = ["caption", "vqa", "text"]
    task_metrics = {"caption": "cider", "vqa": "accuracy", "text": "perplexity"}

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(methods))
    width = 0.25

    for i, task in enumerate(tasks):
        scores = []
        for entry in filtered:
            if task in entry.get("evaluations", {}):
                metric = task_metrics[task]
                scores.append(entry["evaluations"][task].get(metric, 0))
            else:
                scores.append(0)

        ax.bar(x + i * width, scores, width, label=task.upper())

    ax.set_xlabel("Pruning Method")
    ax.set_ylabel("Score")
    ax.set_title(f"Pruning Method Comparison (ratio={target_ratio})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path.name}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent

    if not args.skip_pruning_score:
        cmd = [
            sys.executable,
            str(script_dir / "compute_pruning_score.py"),
            "--neuron_scores", args.neuron_scores,
            "--output_dir", str(output_dir / "pruning_scores"),
        ]
        run_command(cmd, "Compute Pruning Scores")

    print(f"\n{'='*60}")
    print("Phase 3 Pruning Framework Ready")
    print(f"{'='*60}")

    print(f"\nOutputs in: {output_dir}")
    print(f"\nTo run evaluation, implement model loading and evaluation tasks")
    print(f"in the evaluation script and run with appropriate configs.")

    config = {
        "config": args.config,
        "neuron_scores": args.neuron_scores,
        "pruning_ratios": [float(r) for r in args.pruning_ratios.split(",")],
        "eval_tasks": [t.strip() for t in args.eval_tasks.split(",")],
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
