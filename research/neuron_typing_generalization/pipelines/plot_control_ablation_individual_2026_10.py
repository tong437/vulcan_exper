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

"""Create one W/C ablation figure for each completed model-dataset run."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
METRICS = {"mme": "accuracy", "mmbench": "accuracy", "textvqa": "soft accuracy", "chartqa": "relaxed accuracy"}
CONDITIONS = ("baseline", "W_random", "C_random", "W_max", "C_max", "W_min", "C_min", "global_random")
STATES = ("W", "C")
METHODS = ("random", "max", "min")
METHOD_COLORS = {"random": "#4C78A8", "max": "#DC8456", "min": "#64A18C"}


def plot_run(rows: pd.DataFrame, model: str, dataset: str, output_dir: Path) -> Path:
    data = rows.set_index("condition")
    means = np.array([[float(data.loc[f"{state}_{method}", "drop_pp"]) for method in METHODS] for state in STATES])
    standard_deviations = np.array(
        [[float(data.loc[f"{state}_{method}", "sd_pp"]) for method in METHODS] for state in STATES]
    )
    reference = float(data.loc["global_random", "drop_pp"])
    reference_std = float(data.loc["global_random", "sd_pp"])
    baseline = float(data.loc["baseline", "baseline_percent"])
    c_selected = float(data.loc["C_random", "selected_mean"])
    c_target = int(data.loc["C_random", "target_neurons"])

    fig, axis = plt.subplots(figsize=(9, 5.5), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.23, top=0.73)
    fig.suptitle("Response-state ablation", fontsize=19, fontweight="bold", y=0.98)
    fig.text(
        0.545,
        0.901,
        f"{MODELS[model]} | {DATASETS[dataset]} | 1% per layer, all layers",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    fig.text(0.545, 0.855, f"Baseline {METRICS[dataset]}: {baseline:.2f}%", ha="center", fontsize=10, color="#64748B")

    x = np.arange(len(STATES))
    width = 0.23
    bottom = min(0.0, float((means - standard_deviations).min()), reference - reference_std)
    top = max(0.0, float((means + standard_deviations).max()), reference + reference_std)
    span = max(top - bottom, 1.0)
    axis.axhspan(reference - reference_std, reference + reference_std, color="#796A85", alpha=0.11, zorder=0)
    axis.axhline(reference, color="#796A85", linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
    axis.axhline(0, color="#334155", linewidth=1.1, zorder=2)
    for method_index, method in enumerate(METHODS):
        positions = x + (method_index - 1) * width
        axis.bar(positions, means[:, method_index], width=width * 0.9, color=METHOD_COLORS[method], zorder=3)
        if (standard_deviations[:, method_index] > 0).any():
            axis.errorbar(
                positions,
                means[:, method_index],
                yerr=standard_deviations[:, method_index],
                fmt="none",
                ecolor="#334155",
                capsize=3,
                linewidth=1,
                zorder=4,
            )
        for state_index, position in enumerate(positions):
            value = means[state_index, method_index]
            offset = standard_deviations[state_index, method_index] + span * 0.025
            axis.text(
                position,
                value + (offset if value >= 0 else -offset),
                f"{value:+.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=10,
                color="#334155",
            )
    handles = [Patch(facecolor=METHOD_COLORS[method], label=f"{method.title()} within state") for method in METHODS]
    handles.append(
        Line2D([0], [0], color="#796A85", linestyle="--", linewidth=1.6, label=f"Global random ({reference:+.2f} pp)")
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.545, 0.824),
        ncol=2,
        frameon=False,
        fontsize=10,
        columnspacing=2,
    )
    axis.set_xticks(x, STATES)
    axis.set_xlabel("Response state of ablated neurons", labelpad=10)
    axis.set_ylabel("Score drop from baseline (percentage points)", labelpad=10)
    axis.set_ylim(bottom - span * 0.13, top + span * 0.22)
    axis.set_xlim(-0.6, 1.6)
    axis.grid(axis="y", color="#E4EAF0", linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(length=0, pad=7)
    fig.text(
        0.12,
        0.11,
        "Zero line = baseline. Positive drop = worse performance; negative drop = improvement.",
        fontsize=9,
        color="#64748B",
    )
    fig.text(0.12, 0.073, "Error bars / band: +/- 1 SD across 10 random runs.", fontsize=9, color="#64748B")
    fig.text(
        0.12, 0.036, f"C neurons removed per sample: {c_selected:.1f}/{c_target} target.", fontsize=9, color="#64748B"
    )
    path = output_dir / f"{model}__{dataset}.png"
    fig.savefig(path, dpi=250, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    plt.switch_backend("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.input_csv)
    if len(rows) != 128:
        raise ValueError(f"Expected 128 condition rows, got {len(rows)}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    paths = []
    for model in MODELS:
        for dataset in DATASETS:
            subset = rows[(rows["model"] == model) & (rows["dataset"] == dataset)]
            if len(subset) != 8 or set(subset["condition"]) != set(CONDITIONS):
                raise ValueError(f"Incomplete conditions for {model} x {dataset}")
            paths.append(plot_run(subset, model, dataset, args.output_dir))
    with ZipFile(args.output_dir / "wc_ablation_16_figures_png.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
    index = ["# W/C ablation: 16 individual figures", ""]
    for model in MODELS:
        index.extend([f"## {MODELS[model]}", ""])
        for dataset in DATASETS:
            filename = f"{model}__{dataset}.png"
            index.extend([f"### {DATASETS[dataset]}", "", f"![{MODELS[model]} {DATASETS[dataset]}]({filename})", ""])
    (args.output_dir / "index.md").write_text("\n".join(index), encoding="utf-8")
    print(f"Created {len(paths)} PNGs, {len(paths)} PDFs, and one ZIP in {args.output_dir}")


if __name__ == "__main__":
    main()
