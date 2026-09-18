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

"""On-demand sample traces for one model-local decoder MLP neuron."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .importance import CLASS_LABELS, INVALID_CLASS


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def export_neuron_trace(
    run_dir: Path, layer: int, neuron: int, output_dir: Path | None = None
) -> dict[str, str | int]:
    """Export q, importance, RMS, class, and sample metadata for one neuron across all samples."""
    score_manifest = json.loads((run_dir / "scores/manifest.json").read_text(encoding="utf-8"))
    class_manifest = json.loads((run_dir / "classifications/manifest.json").read_text(encoding="utf-8"))
    d_ff_by_layer = [int(value) for value in score_manifest["d_ff_by_layer"]]
    if not 0 <= layer < len(d_ff_by_layer):
        raise IndexError(f"layer={layer} is outside [0, {len(d_ff_by_layer)}).")
    if not 0 <= neuron < d_ff_by_layer[layer]:
        raise IndexError(f"neuron={neuron} is outside layer {layer}'s d_ff={d_ff_by_layer[layer]}.")
    if score_manifest["samples"] != class_manifest["samples"]:
        raise ValueError("Score and classification manifests disagree on sample count.")

    samples = pd.read_parquet(run_dir / "samples/sample_manifest.parquet").reset_index(drop=True)
    score_parts = score_manifest["parts"]
    class_parts = class_manifest["parts"]
    if len(samples) != len(score_parts) or len(score_parts) != len(class_parts):
        raise ValueError("Trace inputs do not contain one aligned partition per frozen sample.")

    rows = []
    for sample_index, (sample, score_part, class_part) in enumerate(
        zip(samples.to_dict(orient="records"), score_parts, class_parts)
    ):
        expected_id = str(sample["sample_id"])
        if str(score_part["sample_id"]) != expected_id or str(class_part["sample_id"]) != expected_id:
            raise ValueError(f"Trace partition identity mismatch at sample index {sample_index}.")
        with np.load(run_dir / score_part["file"], allow_pickle=False) as scores:
            score_id = str(scores["sample_id"].item())
            q = scores["q"][layer, :, neuron].astype(np.float64)
            importance = scores["importance"][layer, :, neuron].astype(np.float64)
            rms = scores["activation_rms"][layer, :, neuron].astype(np.float64)
            token_count = scores["token_count"][layer].astype(np.int64)
        with np.load(run_dir / class_part["file"], allow_pickle=False) as classes:
            class_id = int(classes["classes"][layer, neuron])
        if score_id != expected_id:
            raise ValueError(f"Trace score identity mismatch at sample index {sample_index}.")
        rows.append(
            {
                "sample_index": sample_index,
                "sample_id": expected_id,
                "image_id": str(sample["image_id"]),
                "question": str(sample["question"]),
                "answer": sample.get("answer"),
                "stratum": sample.get("stratum"),
                "q_visual": float(q[0]),
                "q_text": float(q[1]),
                "importance_visual": float(importance[0]),
                "importance_text": float(importance[1]),
                "activation_rms_visual": float(rms[0]),
                "activation_rms_text": float(rms[1]),
                "visual_token_count": int(token_count[0]),
                "text_token_count": int(token_count[1]),
                "class": CLASS_LABELS[class_id] if class_id != INVALID_CLASS else "INVALID",
            }
        )

    frame = pd.DataFrame(rows)
    destination = output_dir or run_dir / "neuron_traces"
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"layer_{layer:03d}_neuron_{neuron:06d}"
    parquet_path = destination / f"{stem}.parquet"
    csv_path = destination / f"{stem}.csv"
    figure_path = destination / f"{stem}.png"
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    _save_trace_figure(frame, 1.0 / d_ff_by_layer[layer], layer, neuron, figure_path)
    return {
        "layer": layer,
        "neuron": neuron,
        "samples": len(frame),
        "parquet": str(parquet_path),
        "csv": str(csv_path),
        "figure": str(figure_path),
    }


def _save_trace_figure(frame: pd.DataFrame, threshold: float, layer: int, neuron: int, destination: Path) -> None:
    plt = _load_pyplot()
    colors = {"W": "#9aa0a6", "V": "#4285f4", "T": "#f4b400", "M": "#0f9d58", "INVALID": "#000000"}
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2.2, 0.45, 2.2]})
    axes[0].plot(frame["sample_index"], frame["q_visual"], label="q visual", linewidth=0.8)
    axes[0].plot(frame["sample_index"], frame["q_text"], label="q text", linewidth=0.8)
    axes[0].axhline(threshold, color="black", linestyle="--", linewidth=0.8, label="1 / d_ff")
    axes[0].set(ylabel="Importance share q", title=f"Layer {layer}, neuron {neuron}: frozen-sample trace")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.2)

    class_values = np.asarray(
        [[CLASS_LABELS.index(value) if value in CLASS_LABELS else 4 for value in frame["class"]]]
    )
    from matplotlib.colors import ListedColormap

    axes[1].imshow(class_values, aspect="auto", interpolation="nearest", cmap=ListedColormap(list(colors.values())))
    axes[1].set(yticks=[], ylabel="Class")

    for label, values in frame.groupby("class", sort=True):
        axes[2].scatter(values["q_visual"], values["q_text"], s=10, alpha=0.55, label=label, color=colors[str(label)])
    axes[2].axvline(threshold, color="black", linestyle="--", linewidth=0.8)
    axes[2].axhline(threshold, color="black", linestyle="--", linewidth=0.8)
    axes[2].set(xlabel="q visual", ylabel="q text")
    axes[2].legend(ncol=5)
    axes[2].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
