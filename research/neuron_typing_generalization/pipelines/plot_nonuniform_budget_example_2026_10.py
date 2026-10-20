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

"""Render an explicitly simulated U/N comparison; no experimental data are read."""

from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot_comparison(scores, ratios, budget, *, baseline_score, subtitle, metric, output, simulated=False) -> None:
    """Render the common layout from supplied percentages."""
    eps = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    colors = ["#397DB5", "#D98A49"]
    labels = ["U: uniform across layers", "N: nonuniform across layers"]
    markers = ["o", "s"]
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[4.0, 1.05], hspace=0.63, wspace=0.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    fig.subplots_adjust(left=0.075, right=0.97, top=0.70, bottom=0.14)
    fig.suptitle("Uniform vs nonuniform pruning at the same response budget", y=0.975, fontsize=23, fontweight="bold")
    fig.text(0.52, 0.915, subtitle, ha="center", fontsize=16, color="#475569")
    fig.text(
        0.52,
        0.865,
        "SIMULATED EXAMPLE — not experimental results"
        if simulated
        else "Measured results | N = 1000 paired samples per condition",
        ha="center",
        fontsize=14,
        color="#A94E49" if simulated else "#64748B",
        fontweight="bold",
    )
    handles = []
    for i in range(2):
        (line,) = axes[0].plot(eps, scores[i], color=colors[i], marker=markers[i], lw=2.6, ms=7, label=labels[i])
        handles.append(line)
        axes[1].plot(eps, ratios[i], color=colors[i], marker=markers[i], lw=2.6, ms=7)
        for x, y in zip(eps, ratios[i]):
            axes[1].annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 10 if i else -18),
                ha="center",
                fontsize=10,
                color=colors[i],
            )
    baseline = axes[0].axhline(
        baseline_score, color="#8B99AB", ls="--", lw=1.8, label=f"Baseline: {baseline_score:.1f}%"
    )
    fig.legend(
        handles=[*handles, baseline],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.825),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    axes[0].set_title("Task score", pad=14, fontsize=16)
    axes[1].set_title("Actual mean pruning rate", pad=14, fontsize=16)
    axes[0].set_ylabel(f"{metric} (%)")
    axes[1].set_ylabel("Removed / all real FFN neurons (%)")
    values = np.r_[np.asarray(scores).ravel(), baseline_score]
    spread = max(float(np.ptp(values)), 2.0)
    axes[0].set_ylim(max(0, values.min() - spread * 0.15), min(100, values.max() + spread * 0.15))
    axes[1].set_ylim(0, 63)
    for ax in axes:
        ax.set_xlim(0.035, 0.315)
        ax.set_xticks(eps, [f"{e:.2f}" for e in eps])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=1 if ax is axes[0] else 0))
        ax.set_xlabel("Mean response-score budget ε", labelpad=12)
        ax.grid(color="#E4EAF0", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(length=0, pad=7)
    table_ax = fig.add_subplot(grid[1, :])
    table_ax.axis("off")
    table_ax.text(
        0,
        1.16,
        "Actual mean response-score share removed (B)",
        transform=table_ax.transAxes,
        fontsize=12,
        color="#475569",
    )
    table = table_ax.table(
        cellText=[[name, *[f"{v:.4f}%" for v in row]] for name, row in zip(["U: uniform", "N: nonuniform"], budget)],
        colLabels=["Method", *[f"ε={e:.2f}" for e in eps]],
        cellLoc="center",
        colWidths=[0.19, *[0.135] * 6],
        bbox=(0, 0, 1, 0.95),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDE5EC")
        if row == 0:
            cell.set_facecolor("#F3F5F8")
    fig.text(
        0.075,
        0.073,
        "Budget = equal-layer mean response share. Pruning rate = total removed / total neurons.",
        fontsize=11,
        color="#64748B",
    )
    fig.text(
        0.075,
        0.035,
        ("All values are illustrative. " if simulated else "")
        + "Task-score axis is zoomed. More removal does not guarantee better accuracy.",
        fontsize=11,
        color="#64748B",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=200, facecolor="white")
    plt.close(fig)
    print(output.with_suffix(".png"))


def main() -> None:
    # Invented layout values, not experimental observations.
    plot_comparison(
        ([79.5, 78.8, 77.5, 75.6, 72.8, 69.5], [79.6, 79.0, 77.8, 75.3, 71.9, 67.8]),
        ([10.5, 19.5, 27.5, 34.5, 41.0, 47.0], [13.0, 24.0, 33.5, 42.0, 49.0, 55.0]),
        ([4.97, 9.96, 14.95, 19.95, 24.94, 29.94], [4.99, 9.99, 14.99, 19.99, 24.99, 29.99]),
        baseline_score=80.0,
        subtitle="Qwen3-VL-2B | MMBench | Sample-adaptive selection",
        metric="MMBench accuracy",
        simulated=True,
        output=Path(__file__).resolve().parents[1]
        / "reports/advisor_2026_09/figures/nonuniform_budget/nonuniform_budget_SIMULATED_example",
    )


if __name__ == "__main__":
    main()
