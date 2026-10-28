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

"""Four-panel measured throughput figures; incomplete sweeps are rejected."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from prepare_throughput_2026_10 import ROOT
from report_cumulative_importance_2026_10 import DATASETS, MODELS


BATCHES = (1, 2, 4, 8, 16, 32, 64)
RATES = tuple(range(5, 51, 5))
OUT = Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/structured_throughput_1000")
SAMPLES = 1000


def report(model, dataset):
    source = ROOT / "results" / f"{model}__{dataset}" / "natural"
    arrays, states, records = [], [], []
    for method in ("original", "uniform", "nonuniform"):
        rates = (0,) if method == "original" else RATES
        values = np.full((7, len(rates)), np.nan)
        status = np.full(values.shape, "", dtype=object)
        for j, rate in enumerate(rates):
            first_oom = None
            for i, batch in enumerate(BATCHES):
                path = source / f"{method}_{rate:02d}_b{batch:02d}.json"
                row = json.loads(path.read_text())
                state = row["status"]
                if state == "complete":
                    if first_oom is not None or len(row["measurements"]) != 3:
                        raise ValueError(f"Invalid completion sequence: {path}")
                    if any(m["requests"] != SAMPLES // batch * batch for m in row["measurements"]):
                        raise ValueError("Incomplete request count")
                    for m in row["measurements"]:
                        if not np.isclose(m["req_per_second"], m["requests"] / m["elapsed_gpu_service_seconds"]):
                            raise ValueError(f"Throughput arithmetic mismatch: {path}")
                    if row["config"]["workload_samples"] != SAMPLES:
                        raise ValueError(f"Wrong experiment: {path}")
                    values[i, j] = np.median([m["req_per_second"] for m in row["measurements"]])
                elif state == "oom" and first_oom is None:
                    first_oom = batch
                elif state != "skipped_after_oom" or first_oom is None:
                    raise ValueError(f"Unrecognized/inconsistent status: {path}")
                status[i, j] = state
                records.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        method=method,
                        target_percent=rate,
                        actual_percent=row["actual_rate"] * 100,
                        batch_size=batch,
                        status=state,
                        req_per_second=values[i, j],
                        parameters=row["parameters"]["parameters_after"],
                        aligned_widths=json.dumps(row["aligned_widths"]),
                        source=str(path),
                        min_req_s=min((m["req_per_second"] for m in row.get("measurements", [])), default=np.nan),
                        max_req_s=max((m["req_per_second"] for m in row.get("measurements", [])), default=np.nan),
                        mean_task_score=np.mean([m["task_score"] for m in row["measurements"]]) if state == "complete" else np.nan,
                        gpu=row["config"]["runtime"]["gpu"],
                        device=row["config"]["runtime"]["device_identity"],
                    )
                )
        arrays.append(values)
        states.append(status)
    valid = np.concatenate([a[np.isfinite(a)] for a in arrays])
    if not len(valid):
        raise ValueError("No complete measurements")
    dense, uniform, nonuniform = arrays
    common = np.isfinite(uniform) & np.isfinite(nonuniform)
    gain = np.full_like(uniform, np.nan)
    gain[common] = 100 * (nonuniform[common] / uniform[common] - 1)
    bound = max(5, float(np.max(np.abs(gain[common]))) if common.any() else 5)
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.2), gridspec_kw={"width_ratios": [1.1, 5, 5, 5]})
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.25, top=0.75, wspace=0.2)
    fig.suptitle("Inference throughput across pruning rates and batch sizes", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.89,
        f"{MODELS[model]} | {DATASETS[dataset]} | {SAMPLES:,} requests | MLP widths rounded up to multiples of 8",
        ha="center",
    )
    fig.text(0.5, 0.84, "A40 | Measured GPU service throughput; median of 3 runs", ha="center", color="#64748B")
    norm = Normalize(float(valid.min()) * 0.95, float(valid.max()) * 1.02)
    for k, (axis, values, title) in enumerate(
        zip(axes, [*arrays, gain], ("Original", "Uniform pruning", "Nonuniform pruning", "Nonuniform vs uniform"))
    ):
        cmap = plt.get_cmap("YlGnBu" if k < 3 else "RdBu").copy()
        cmap.set_bad("#F1F5F9")
        scale = norm if k < 3 else TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
        im = axis.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, norm=scale)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(range(values.shape[1]), ["0%"] if k == 0 else [f"{r}%" for r in RATES], fontsize=8)
        axis.set_yticks(range(7), [str(b) for b in BATCHES] if k == 0 else [])
        for i, j in np.ndindex(values.shape):
            v = values[i, j]
            if np.isfinite(v):
                label = f"{v:+.1f}%" if k == 3 else f"{v:.1f}"
                if k in (1, 2):
                    label += f"\n{v / dense[i, 0]:.2f}x" if np.isfinite(dense[i, 0]) else "\nN/A"
                color = "white" if (scale(v) > 0.6 if k < 3 else abs(v) > bound * 0.65) else "#172033"
            else:
                label = ("OOM" if states[k][i, j] == "oom" else "Skipped") if k < 3 else "N/A"
                color = "#64748B"
                axis.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor="#F1F5F9", edgecolor="#CBD5E1", hatch="///", linewidth=0))
            axis.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)
        axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.5)
        axis.tick_params(which="both", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if k:
            axis.set_xlabel("Target pruning rate", labelpad=9)
        if k == 2:
            fig.colorbar(im, ax=list(axes[:3]), location="bottom", fraction=0.06, pad=0.18, label="Throughput (req/s)")
        if k == 3:
            fig.colorbar(
                im, ax=axis, location="bottom", fraction=0.06, pad=0.18, label="100 × (nonuniform / uniform − 1) [%]"
            )
    axes[0].set_ylabel("Batch size")
    fig.text(
        0.035,
        0.10,
        "Speedup uses original at the SAME batch size. First three panels share a req/s scale within each figure. N/A = comparison unavailable.",
        fontsize=10,
        color="#64748B",
    )
    fig.text(
        0.035,
        0.05,
        "Right: blue = nonuniform faster; red = uniform faster. Actual aligned rates are in CSV. Natural EOS; max 32 new tokens; CPU preprocessing excluded.",
        fontsize=10,
        color="#64748B",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{model}__{dataset}.{suffix}", dpi=220, facecolor="white")
    plt.close(fig)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    ROOT, OUT, SAMPLES = args.root, args.output, args.samples
    records = [row for model in MODELS for dataset in DATASETS for row in report(model, dataset)]
    with (OUT / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
