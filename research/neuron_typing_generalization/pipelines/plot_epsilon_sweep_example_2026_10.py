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

"""Draw a clearly simulated layout example for the A2 epsilon scan."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vulcan_epsilon_example_mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPSILONS = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
# Every value below is invented solely to demonstrate the proposed plot layout.
BASELINE = 80.0
SCORES = np.array([79.6, 79.0, 77.5, 75.4, 72.0, 67.0])
ACTUAL_PRUNING = np.array([11.0, 20.0, 27.5, 34.0, 39.5, 44.5])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/epsilon_sweep"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "epsilon_sweep_SIMULATED_example"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(9.0, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 0.9), hspace=0.60)
    ax = fig.add_subplot(grid[0])
    table_ax = fig.add_subplot(grid[1])
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.14, top=0.73)
    fig.suptitle("Effect of response-score budget ε", fontsize=19, fontweight="bold", y=0.97)
    fig.text(0.545, 0.89, "Qwen3-VL-2B | MMBench | illustrative layout", ha="center", color="#475569")
    fig.text(
        0.545,
        0.835,
        "SIMULATED EXAMPLE — not experimental results",
        ha="center",
        color="#A94E49",
        fontsize=11,
        fontweight="bold",
    )

    ax.axhline(BASELINE, color="#8B99AB", ls="--", lw=1.8, label="Unpruned baseline (simulated)")
    ax.fill_between(EPSILONS, SCORES, BASELINE, color="#80BEE7", alpha=0.20)
    ax.plot(EPSILONS, SCORES, color="#397DB5", marker="o", ms=7, lw=2.8, label="A2 task score (simulated)")
    for epsilon, score in zip(EPSILONS, SCORES):
        ax.annotate(f"{score:.1f}", (epsilon, score), xytext=(0, -17), textcoords="offset points", ha="center", color="#334155", fontsize=9)
    ax.set_xlim(0.035, 0.315)
    ax.set_ylim(62.0, 82.0)
    ax.set_xticks(EPSILONS, [f"{value:.2f}" for value in EPSILONS])
    ax.set_yticks(np.arange(65, 81, 5), [f"{value:.0f}%" for value in np.arange(65, 81, 5)])
    ax.set_xlabel("Response-score budget ε", labelpad=9)
    ax.set_ylabel("MMBench accuracy (%)", labelpad=8)
    ax.grid(color="#E4EAF0", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(length=0, pad=6)
    ax.legend(loc="lower left", fontsize=9, frameon=True, facecolor="white", edgecolor="#E2E8F0")

    table_ax.axis("off")
    table_ax.text(0.0, 1.12, "Illustrative mean actual pruning rate", transform=table_ax.transAxes, color="#475569", fontsize=9)
    table = table_ax.table(
        cellText=[[f"{value:.1f}%" for value in ACTUAL_PRUNING]],
        colLabels=[f"ε={value:.2f}" for value in EPSILONS],
        cellLoc="center",
        loc="center",
        bbox=(0.0, 0.0, 1.0, 0.90),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#DDE5EC")
        cell.set_linewidth(0.8)
    fig.text(0.13, 0.055, "ε limits cumulative response share, not the fraction of neurons removed. Y-axis is zoomed.", color="#64748B", fontsize=9)

    for suffix in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=240, facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
