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

"""Explore the distribution of the V/T/C importance sum from frozen five-state data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


STATES = ("V", "T", "M", "C", "W")
COLORS = {"V": "#397DB5", "T": "#D78A49", "M": "#5A9E88", "C": "#A579AC", "W": "#CBD5E1"}
PLOT_ORDER = ("W", "C", "V", "T", "M")
LOWER, UPPER, BINS = 0.05, 20.0, 72


def collect(run_dir: Path) -> dict[str, object]:
    """Stream all frozen samples into equal-log-width bins without retaining neuron rows."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["samples"] != 1000 or tuple(manifest["state_order"]) != STATES:
        raise ValueError(f"Expected 1,000 complete five-state samples: {run_dir}")
    widths = np.asarray(manifest["d_ff_by_layer"], dtype=np.int64)
    real_neurons = np.arange(int(widths.max()))[None, :] < widths[:, None]
    edges = np.geomspace(LOWER, UPPER, BINS + 1)
    counts = np.zeros((len(STATES), BINS), dtype=np.int64)
    total = below = above = weak_above_one = under_one = nonweak_under_one = 0
    importance_sum = 0.0
    log_lower = np.log(LOWER)
    log_bin_scale = BINS / np.log(UPPER / LOWER)

    for sample_index in range(1000):
        path = run_dir / "sample_statistics" / f"{sample_index:06d}.npz"
        with np.load(path, allow_pickle=False) as saved:
            q, classes = saved["q"], saved["classes"]
        if q.shape != (len(widths), 3, int(widths.max())) or classes.shape != real_neurons.shape:
            raise ValueError(f"Unexpected frozen-statistics shape: {path}")
        response = (q.sum(axis=1) * (widths[:, None] / 3.0))[real_neurons]
        state = classes[real_neurons]
        if not np.isfinite(response).all() or (response < 0).any() or not np.isin(state, range(5)).all():
            raise ValueError(f"Invalid normalized importance or state: {path}")
        total += len(response)
        importance_sum += response.sum(dtype=np.float64)
        below += int((response < LOWER).sum())
        above += int((response >= UPPER).sum())
        weak_above_one += int(((state == 4) & (response > 1.00001)).sum())
        threshold_mask = response <= 1.0
        under_one += int(threshold_mask.sum())
        nonweak_under_one += int(((state != 4) & threshold_mask).sum())
        in_range = (response >= LOWER) & (response < UPPER)
        bin_index = ((np.log(response[in_range]) - log_lower) * log_bin_scale).astype(np.int64)
        bin_index = np.minimum(bin_index, BINS - 1)
        joint_index = state[in_range].astype(np.int64) * BINS + bin_index
        counts += np.bincount(joint_index, minlength=len(STATES) * BINS).reshape(len(STATES), BINS)
        if (sample_index + 1) % 100 == 0:
            print(f"{run_dir.name}: {sample_index + 1}/1000 samples", flush=True)

    if total != 1000 * int(widths.sum()) or counts.sum() + below + above != total or weak_above_one:
        raise ValueError("Distribution accounting failed or a W response exceeded normalized sum 1.")
    if not np.isclose(importance_sum / total, 1.0, atol=1e-4):
        raise ValueError("The layer-width-adjusted mean should equal one.")
    return {
        "model": manifest["model"],
        "dataset": manifest["dataset"],
        "samples": 1000,
        "neurons_per_sample": int(widths.sum()),
        "observations": total,
        "normalized_sum_mean": importance_sum / total,
        "r_min_plotted": LOWER,
        "r_max_plotted": UPPER,
        "below_plot": below,
        "above_plot": above,
        "r_at_most_one": under_one,
        "nonweak_with_r_at_most_one": nonweak_under_one,
        "edges": edges.tolist(),
        "counts_by_state": {name: counts[index].tolist() for index, name in enumerate(STATES)},
    }


def plot(summary: dict[str, object], output_stem: Path) -> None:
    """Show the distribution and its state composition on a shared log-scale axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(summary["edges"], dtype=float)
    total = int(summary["observations"])
    counts = {state: np.asarray(summary["counts_by_state"][state], dtype=float) for state in STATES}
    in_bin = sum(counts.values())
    figure, (density, composition) = plt.subplots(
        2, 1, figsize=(10.4, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.3]}, facecolor="white"
    )
    figure.subplots_adjust(left=0.12, right=0.97, top=0.81, bottom=0.19, hspace=0.16)
    figure.suptitle("Modality-aware importance sum", fontsize=18, fontweight="bold", y=0.98)
    figure.text(
        0.545,
        0.92,
        f"{summary['model']} | {str(summary['dataset']).upper()} | 1,000 frozen inputs",
        ha="center",
        fontsize=11,
        color="#475569",
    )
    figure.text(
        0.545,
        0.87,
        "r = d_ff (qV + qT + qC) / 3; layer mean r = 1 by construction",
        ha="center",
        fontsize=10,
        color="#64748B",
    )
    bottom_mass = np.zeros(BINS)
    bottom_mix = np.zeros(BINS)
    for state in PLOT_ORDER:
        mass = 100 * counts[state] / total
        mix = np.divide(100 * counts[state], in_bin, out=np.zeros(BINS), where=in_bin > 0)
        density.bar(
            edges[:-1],
            mass,
            np.diff(edges),
            bottom=bottom_mass,
            align="edge",
            color=COLORS[state],
            linewidth=0,
            label=state,
        )
        composition.bar(
            edges[:-1], mix, np.diff(edges), bottom=bottom_mix, align="edge", color=COLORS[state], linewidth=0
        )
        bottom_mass += mass
        bottom_mix += mix
    for axis in (density, composition):
        axis.set_xscale("log")
        axis.set_xlim(LOWER, UPPER)
        axis.axvline(1, color="#334155", lw=1.5, ls="--")
        axis.grid(axis="y", color="#E4EAF0", lw=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#CBD5E1")
        axis.tick_params(length=0, pad=5)
    density.set_ylabel("All observations per log bin (%)")
    density.legend(loc="upper right", ncol=5, framealpha=0.97, edgecolor="#E2E8F0")
    composition.set_ylabel("State mix within bin (%)")
    composition.set_ylim(0, 100)
    composition.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    composition.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    composition.set_xlabel("Normalized importance sum r (log scale)", labelpad=8)
    outside = 100 * (int(summary["below_plot"]) + int(summary["above_plot"])) / total
    figure.text(
        0.12,
        0.105,
        "Dashed line: r = 1. W implies r ≤ 1, but r ≤ 1 does not imply W; individual V/T/C thresholds define W.",
        fontsize=9,
        color="#64748B",
    )
    figure.text(
        0.12,
        0.075,
        f"Each real neuron × sample counted once; layers weighted by true width. {outside:.3f}% outside x-range.",
        fontsize=9,
        color="#64748B",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=250, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    summary = collect(args.run_dir)
    plot(summary, args.output_stem)
    args.output_stem.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in summary.items() if key not in {"edges", "counts_by_state"}}, indent=2)
    )


if __name__ == "__main__":
    main()
