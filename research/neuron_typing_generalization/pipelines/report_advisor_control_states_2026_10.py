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

"""Build the 4 x 4 layerwise V/T/M/C/W response-state report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from neuron_typing_generalization.advisor_control_states import CONTROL_STATE_ORDER, validate_state_counts


MODEL_LABELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASET_ORDER = ("mme", "mmbench", "textvqa", "chartqa")
DATASET_LABELS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
STATE_COLORS = {"V": "#397DB5", "T": "#DB9743", "M": "#599E8B", "C": "#8B6FB1", "W": "#D8DFE7"}


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    return plt


def _ticks(layers: np.ndarray, maximum: int = 13) -> np.ndarray:
    step = max(1, int(np.ceil(len(layers) / maximum)))
    return np.unique(np.r_[layers[::step], layers[-1]])


def _load_distribution(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or int(manifest["samples"]) != 1000:
        raise ValueError(f"Incomplete control-state run: {run_dir}")
    with np.load(run_dir / "checkpoint.npz", allow_pickle=False) as saved:
        counts = saved["counts"].astype(np.int64, copy=False)
        completed = int(saved["completed_samples"].item())
    widths = [int(value) for value in manifest["d_ff_by_layer"]]
    validate_state_counts(counts, completed, widths)
    rows = []
    for layer, width in enumerate(widths):
        state_totals = counts[:, layer, :width].sum(axis=1)
        shares = state_totals / (completed * width)
        row = {"layer": layer, "neuron_count": width, "d_ff": width, "sample_count": completed}
        row.update({f"share_{state}": float(shares[index]) for index, state in enumerate(CONTROL_STATE_ORDER)})
        row["share_sum"] = float(shares.sum())
        row["share_sum_residual"] = float(shares.sum() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows), manifest


def _save_figure(frame: pd.DataFrame, model: str, dataset: str, path: Path) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(16.0, 10.5), facecolor="white")
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.14, top=0.79)
    figure.suptitle("Layerwise response-state distribution", y=0.975, fontsize=21, fontweight="bold")
    figure.text(
        0.54,
        0.90,
        f"{MODEL_LABELS[model]} | {DATASET_LABELS[dataset]} | generation-aligned prompt",
        ha="center",
        fontsize=13,
        color="#64748B",
    )
    layers = frame["layer"].to_numpy(dtype=np.int64)
    bottom = np.zeros(len(frame), dtype=np.float64)
    labels_and_bars = []
    for state in CONTROL_STATE_ORDER:
        shares = frame[f"share_{state}"].to_numpy(dtype=np.float64)
        bars = axis.bar(
            layers,
            shares,
            bottom=bottom,
            width=0.88,
            color=STATE_COLORS[state],
            label=state,
            edgecolor="none",
            zorder=3,
        )
        centers = bottom + shares / 2.0
        label_color = "#334155" if state == "W" else "#FFFFFF"
        for layer, center, share, bar in zip(layers, centers, shares, bars):
            if share < 0.03:
                axis.annotate(
                    f"{share:.1%}",
                    xy=(layer, center),
                    xytext=(5 if layer % 2 == 0 else -5, 0),
                    textcoords="offset points",
                    ha="left" if layer % 2 == 0 else "right",
                    va="center",
                    fontsize=5.5,
                    color="#334155",
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
                    zorder=5,
                    clip_on=True,
                )
                continue
            label = axis.text(
                layer,
                center,
                f"{share:.1%}",
                ha="center",
                va="center",
                rotation=90,
                rotation_mode="anchor",
                fontsize=8.0,
                color=label_color,
                zorder=4,
                clip_on=True,
            )
            labels_and_bars.append((label, bar))
        bottom += shares
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5, frameon=False, columnspacing=2, handlelength=1.6
    )
    axis.set(xlim=(float(layers.min()) - 0.7, float(layers.max()) + 0.7), ylim=(0.0, 1.0))
    axis.set_xticks(_ticks(layers))
    axis.set_xlabel("Decoder layer index (0-based)", labelpad=11)
    axis.set_ylabel("Mean response-state share", labelpad=11)
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axis.grid(axis="y", color="#E9EEF4", zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(length=0, pad=6)
    figure.text(
        0.10,
        0.025,
        "Each layer sums to 100%; average over 1,000 inputs and real neurons within that layer.",
        fontsize=10.5,
        color="#64748B",
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for font_size in np.arange(8.0, 3.8, -0.2):
        for label, _ in labels_and_bars:
            label.set_fontsize(float(font_size))
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        if all(
            label.get_window_extent(renderer=renderer).y0 >= bar.get_window_extent(renderer=renderer).y0 + 1.0
            and label.get_window_extent(renderer=renderer).y1 <= bar.get_window_extent(renderer=renderer).y1 - 1.0
            for label, bar in labels_and_bars
        ):
            break
    else:
        raise RuntimeError(f"Percentage labels do not fit for {model} x {dataset}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=250, facecolor="white")
    figure.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    figure_dir = args.output_dir / "figures/layerwise_control_state_distributions"
    table_dir = args.output_dir / "tables/layerwise_control_state_distributions"
    table_dir.mkdir(parents=True, exist_ok=True)
    summaries, long_frames = [], []
    figure_rows = []
    for model in MODEL_LABELS:
        for dataset in DATASET_ORDER:
            frame, manifest = _load_distribution(args.input_root / f"{model}__{dataset}")
            stem = f"{model}__{dataset}__layerwise_control_state_distribution"
            csv_path = table_dir / f"{stem}.csv"
            png_path = figure_dir / f"{stem}.png"
            frame.to_csv(csv_path, index=False)
            _save_figure(frame, model, dataset, png_path)
            annotated = frame.copy()
            annotated.insert(0, "dataset", dataset)
            annotated.insert(0, "model", model)
            long_frames.append(annotated)
            weights = frame["neuron_count"].to_numpy(dtype=np.float64)
            summary = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "samples": int(manifest["samples"]),
                "layers": len(frame),
                "neurons": int(frame["neuron_count"].sum()),
                "maximum_layer_sum_residual": float(frame["share_sum_residual"].abs().max()),
                "control_token_count": manifest["control_token_count"],
            }
            for state in CONTROL_STATE_ORDER:
                summary[f"overall_share_{state}"] = float(np.average(frame[f"share_{state}"], weights=weights))
            summaries.append(summary)
            figure_rows.append({"model": model, "dataset": dataset, "png": str(png_path), "csv": str(csv_path)})
    long_frame = pd.concat(long_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_frame.to_csv(args.output_dir / "advisor_2026_10_control_states_long.csv", index=False)
    summary_frame.to_csv(args.output_dir / "advisor_2026_10_control_states_summary.csv", index=False)
    result = {
        "status": "complete",
        "runs": len(summaries),
        "figures": len(figure_rows),
        "state_order": list(CONTROL_STATE_ORDER),
        "input_policy": "advisor ablation evaluation_prompt with add_generation_prompt=True",
        "classification": {
            "V": "qV > 1/d_ff and qT <= 1/d_ff",
            "T": "qV <= 1/d_ff and qT > 1/d_ff",
            "M": "qV > 1/d_ff and qT > 1/d_ff",
            "C": "qV <= 1/d_ff and qT <= 1/d_ff and qC > 1/d_ff",
            "W": "qV <= 1/d_ff and qT <= 1/d_ff and qC <= 1/d_ff",
        },
        "maximum_layer_sum_residual": max(row["maximum_layer_sum_residual"] for row in summaries),
        "summaries": summaries,
        "individual_figures": figure_rows,
    }
    (args.output_dir / "advisor_2026_10_control_states.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    header = "| Model | Dataset | V | T | M | C | W |"
    lines = [header, "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        values = " | ".join(f"{row[f'overall_share_{state}']:.2%}" for state in CONTROL_STATE_ORDER)
        lines.append(f"| {row['model_label']} | {row['dataset_label']} | {values} |")
    markdown = (
        """# Layerwise V/T/M/C/W response-state distributions

All 16 runs use the exact answer-generation prompts from the preceding ablation experiment and
`add_generation_prompt=True`. `C` contains non-visual, non-question-text, non-padding decoder positions. Every layer
is averaged over 1,000 frozen inputs and its real FFN neurons; no layers or neurons are padded.

"""
        + "\n".join(lines)
        + "\n"
    )
    (args.output_dir / "advisor_2026_10_control_states.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"status": "complete", "runs": len(summaries), "figures": len(figure_rows)}, indent=2))


if __name__ == "__main__":
    main()
