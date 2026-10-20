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

"""Readable, numerically annotated single-layer sensitivity figures."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox


def label_layers(ax, subset):
    """Place every layer label in free display space, without moving data points."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent()
    occupied = []
    points = ax.transData.transform(subset[["response_percent", "drop_pp"]].to_numpy())
    markers = [Bbox.from_bounds(x - 5, y - 5, 10, 10) for x, y in points]
    font = FontProperties(size=9)
    offsets = [(dx, dy) for dx in range(-150, 151, 15) for dy in range(-105, 106, 15)]
    offsets.sort(key=lambda d: d[0] ** 2 + d[1] ** 2)
    for (_, row), (x, y) in zip(subset.iterrows(), points):
        text = f"L{int(row.layer)}"
        width, height, _ = renderer.get_text_width_height_descent(text, font, False)
        for dx, dy in offsets:
            box = Bbox.from_bounds(x + dx - width / 2 - 3, y + dy - height / 2 - 3, width + 6, height + 6)
            if not bounds.contains(box.x0, box.y0) or not bounds.contains(box.x1, box.y1):
                continue
            if any(box.overlaps(other) for other in occupied + markers):
                continue
            occupied.append(box)
            ax.annotate(
                text,
                (row.response_percent, row.drop_pp),
                xytext=ax.transData.inverted().transform((x + dx, y + dy)),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=9,
                color="#253449",
                arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": 0.65},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
            )
            break
        else:
            raise RuntimeError(f"Could not place {text} without overlaps; enlarge the figure.")


def plot_detail(frame, title, output, bound=4.0):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(17, 14))
    grid = fig.add_gridspec(3, 2, width_ratios=(0.8, 1.8), hspace=0.40, wspace=0.32)
    fig.subplots_adjust(left=0.075, right=0.97, top=0.86, bottom=0.11)
    fig.suptitle("Single-layer pruning: response share vs task sensitivity", fontsize=23, fontweight="bold", y=0.97)
    fig.text(0.52, 0.928, title, ha="center", color="#475569", fontsize=14)
    heat = fig.add_subplot(grid[:, 0])
    matrix = frame.pivot(index="layer", columns="rate", values="drop_pp").sort_index()
    img = heat.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    heat.set_xticks(range(3), [f"{r:.0%}" for r in matrix.columns])
    heat.set_yticks(range(len(matrix)), [f"L{l}" for l in matrix.index], fontsize=10)
    heat.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    heat.set_yticks(np.arange(-0.5, len(matrix), 1), minor=True)
    heat.grid(which="minor", color="white", lw=1)
    heat.tick_params(which="minor", length=0)
    heat.set_xlabel("Removal within target layer")
    heat.set_ylabel("Decoder layer (0-based)")
    heat.set_title("Score drop (pp): exact values", pad=14)
    for i, row in enumerate(matrix.to_numpy()):
        for j, value in enumerate(row):
            heat.text(
                j,
                i,
                f"{value:+.2f}" if abs(value) >= 0.005 else "0.00",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if abs(value) > bound * 0.58 else "#172536",
            )
    bar = fig.colorbar(img, ax=heat, orientation="horizontal", pad=0.065, fraction=0.025)
    bar.set_label("Negative = improvement  |  Positive = worse", fontsize=10)
    ylim = (min(-0.5, frame.drop_pp.min() - 0.65), max(0.5, frame.drop_pp.max() + 0.65))
    for index, (rate, color, marker) in enumerate(
        [(0.1, "#397DB5", "o"), (0.3, "#D98A49", "s"), (0.5, "#5B9D8C", "^")]
    ):
        ax = fig.add_subplot(grid[index, 1])
        subset = frame[np.isclose(frame.rate, rate)].sort_values("layer")
        ax.scatter(subset.response_percent, subset.drop_pp, color=color, marker=marker, s=42, zorder=3)
        lo, hi = subset.response_percent.min(), subset.response_percent.max()
        padding = max((hi - lo) * 0.18, 1.0)
        ax.set_xlim(lo - padding, hi + padding)
        ax.set_ylim(*ylim)
        ax.axhline(0, color="#94A3B8", ls="--", lw=1)
        ax.grid(color="#E4EAF0", lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"{rate:.0%} layer removal — every point labeled by layer", fontsize=13, loc="left")
        ax.set_ylabel("Score drop (pp)")
        ax.set_xlabel("Mean response share removed in target layer (%)", fontsize=10)
        label_layers(ax, subset)
    fig.text(
        0.075,
        0.045,
        "Shared heatmap scale across all 8 figures. Scatter y-scales match within each figure; x-scales vary by removal rate.",
        color="#64748B",
        fontsize=11,
    )
    fig.text(
        0.075,
        0.022,
        "Leader lines identify labels; data points are not shifted. Exploratory calibration; paired 95% intervals are in the CSV.",
        color="#64748B",
        fontsize=11,
    )
    for extension in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=180, facecolor="white")
    plt.close(fig)
