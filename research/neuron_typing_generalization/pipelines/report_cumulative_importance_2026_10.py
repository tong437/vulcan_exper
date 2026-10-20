#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Report observed dataset-average cumulative importance for the frozen 16 runs."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vulcan_cumulative_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}


def layer_mean_curve(q: np.ndarray, widths: list[int], u: np.ndarray) -> np.ndarray:
    """Compute equal-layer mean of exact ascending-score Lorenz curves."""
    if q.ndim != 3 or q.shape[0] != len(widths) or q.shape[1] != 3:
        raise ValueError("Expected q shape [layers, 3, max_d_ff].")
    result = np.zeros_like(u, dtype=np.float64)
    for width in sorted(set(widths)):
        layers = [i for i, d_ff in enumerate(widths) if d_ff == width]
        scores = q[layers, :, :width].astype(np.float64).sum(axis=1)
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("Scores must be finite and nonnegative.")
        totals = scores.sum(axis=1)
        if np.any(totals <= 0):
            raise ValueError("Every layer must have a positive total score.")
        cumulative = np.cumsum(np.sort(scores, axis=1), axis=1)
        # Grid points fall between neuron counts floor(u*d_ff) and ceil(u*d_ff).
        position = u * width
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, width)
        fraction = position - lower
        lower_values = np.zeros((len(layers), len(u)), dtype=np.float64)
        valid = lower > 0
        lower_values[:, valid] = cumulative[:, lower[valid] - 1]
        upper_values = cumulative[:, upper - 1]
        interpolated = (lower_values * (1 - fraction) + upper_values * fraction) / totals[:, None]
        result += interpolated.sum(axis=0)
    result /= len(widths)
    result[0], result[-1] = 0.0, 1.0
    return result


def candidate_knee(u: np.ndarray, curve: np.ndarray) -> int:
    """First maximizer of the vertical gap below the uniform-score diagonal."""
    gap = u - curve
    return int(np.flatnonzero(np.isclose(gap, gap.max(), rtol=1e-10, atol=1e-12))[0])


def plot_curve(u, mean, p10, p90, model, dataset, count, output):
    knee = candidate_knee(u, mean)
    x, y = u[knee], mean[knee]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.77)
    fig.suptitle("Dataset-average cumulative importance", fontsize=17, fontweight="bold", y=0.97)
    fig.text(0.55, 0.89, f"{model} | {dataset} | N = {count}", ha="center", color="#475569")
    fig.text(0.55, 0.847, "Computed from observed qV + qT + qC neuron scores", ha="center", color="#64748B", fontsize=10)
    ax.fill_between(u, p10, p90, color="#80BEE7", alpha=0.65, label="10–90% range across samples")
    ax.plot(u, mean, color="#397DB5", lw=2.7, label=r"Dataset mean $G_{\mathcal{D}}(u)$")
    ax.plot(u, u, color="#94A3B8", lw=1.4, ls="--", label="Uniform-score reference")
    ax.plot([x, x], [y, x], color="#D58A49", lw=2, ls=":")
    ax.scatter([x], [y], s=65, color="#D58A49", zorder=5, label="Candidate knee of dataset mean")
    note_x = 0.1 if x >= 0.45 else 0.55
    ax.annotate(
        f"Candidate removal: {x:.1%}\nCumulative score share: {y:.1%}",
        xy=(x, y), xytext=(note_x, 0.67), textcoords="axes fraction", fontsize=10,
        color="#334155", bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#E2E8F0"),
        arrowprops=dict(arrowstyle="->", color="#D58A49", lw=1.2),
    )
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Lowest-score neurons selected per layer (u)", ylabel="Cumulative importance share")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(color="#E4EAF0", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(length=0, pad=6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, edgecolor="#E2E8F0")
    fig.text(0.13, 0.064, "Narrow 10–90% band = similar sample-level curves; not a confidence interval.", fontsize=8.5, color="#64748B")
    fig.text(0.13, 0.026, "The mean-curve knee is descriptive; safe pruning requires an accuracy test.", fontsize=8.5, color="#64748B")
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240, facecolor="white")
    plt.close(fig)


def report_run(run_dir: Path, output_dir: Path, grid_size: int, limit: int | None) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("samples") != 1000:
        raise ValueError(f"Frozen run is not complete: {run_dir}")
    widths = [int(v) for v in manifest["d_ff_by_layer"]]
    source = run_dir / "sample_statistics"
    files = sorted(source.glob("*.npz"))
    if len(files) != manifest["samples"]:
        raise ValueError(f"Expected {manifest['samples']} sample files in {source}; found {len(files)}")
    if limit is not None:
        files = files[:limit]
    u = np.linspace(0, 1, grid_size)
    sample_curves = np.empty((len(files), grid_size), dtype=np.float64)
    for idx, path in enumerate(files):
        with np.load(path, allow_pickle=False) as data:
            sample_curves[idx] = layer_mean_curve(data["q"], widths, u)
        if (idx + 1) % 100 == 0:
            print(f"{run_dir.name}: {idx + 1}/{len(files)}", flush=True)
    mean = sample_curves.mean(axis=0)
    p10, p90 = np.percentile(sample_curves, (10, 90), axis=0)
    if not (np.all(np.diff(mean) >= -1e-10) and np.all(mean <= u + 1e-10)):
        raise ValueError(f"Cumulative curve invariant failed: {run_dir.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / run_dir.name
    with stem.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("u", "dataset_mean", "sample_p10", "sample_p90"))
        writer.writerows(zip(u, mean, p10, p90))
    plot_curve(u, mean, p10, p90, MODELS[manifest["model"]], DATASETS[manifest["dataset"]], len(files), stem)
    knee = candidate_knee(u, mean)
    summary = {
        "model": manifest["model"], "dataset": manifest["dataset"], "n_samples": len(files),
        "n_layers": len(widths), "grid_size": grid_size, "source_manifest": str(run_dir / "manifest.json"),
        "source_run_id": manifest["run_id"], "candidate_knee_u": float(u[knee]),
        "candidate_knee_score_share": float(mean[knee]),
        "mean_share_at_1pct": float(np.interp(0.01, u, mean)),
        "mean_share_at_5pct": float(np.interp(0.05, u, mean)),
        "mean_share_at_10pct": float(np.interp(0.10, u, mean)),
        "mean_share_at_20pct": float(np.interp(0.20, u, mean)),
    }
    stem.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("artifacts/neuron_typing_generalization/control_response_2026_10_frozen"))
    parser.add_argument("--output-dir", type=Path, default=Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/cumulative_importance"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--limit", type=int, help="Smoke test only; never use for final results.")
    parser.add_argument("--replot-only", action="store_true", help="Redraw completed figures from saved CSVs.")
    parser.add_argument("--summarize-only", action="store_true", help="Combine all 16 completed run summaries.")
    args = parser.parse_args()
    if args.grid_size < 2 or (args.limit is not None and args.limit < 1):
        parser.error("Grid size must be >=2 and limit, when given, must be >=1.")
    if args.replot_only:
        for model in MODELS:
            if args.model and model != args.model:
                continue
            for dataset in DATASETS:
                if args.dataset and dataset != args.dataset:
                    continue
                stem = args.output_dir / f"{model}__{dataset}"
                metadata = json.loads(stem.with_suffix(".json").read_text())
                if metadata["n_samples"] != 1000:
                    raise ValueError(f"Expected 1000 observed samples: {stem}")
                rows = np.genfromtxt(stem.with_suffix(".csv"), delimiter=",", skip_header=1)
                if rows.shape != (metadata["grid_size"], 4) or not np.isfinite(rows).all():
                    raise ValueError(f"Invalid saved curve: {stem}")
                plot_curve(*rows.T, MODELS[model], DATASETS[dataset], metadata["n_samples"], stem)
                print(f"Redrew {stem.name}", flush=True)
        return
    if args.summarize_only:
        results = []
        for model in MODELS:
            for dataset in DATASETS:
                path = args.output_dir / f"{model}__{dataset}.json"
                result = json.loads(path.read_text())
                if result["n_samples"] != 1000 or not path.with_suffix(".png").exists():
                    raise ValueError(f"Incomplete run: {path}")
                results.append(result)
        with (args.output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Verified and summarized {len(results)} complete model–dataset figures.")
        return
    results = []
    for model in MODELS:
        if args.model and model != args.model:
            continue
        for dataset in DATASETS:
            if args.dataset and dataset != args.dataset:
                continue
            run_dir = args.input_root / f"{model}__{dataset}"
            results.append(report_run(run_dir, args.output_dir, args.grid_size, args.limit))
            print(json.dumps(results[-1]), flush=True)
    if args.limit is None and len(results) == 16:
        with (args.output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main()
