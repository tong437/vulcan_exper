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

"""Equal-run-weight overall U/N score drops at matched response budgets."""

import hashlib
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "reports/advisor_2026_09/figures/nonuniform_budget"
    source = output / "measured_comparison.csv"
    frame = pd.read_csv(source)
    expected_models = {"llava_1_5_7b", "qwen3_vl_2b_instruct", "qwen3_5_2b", "gemma_4_e2b_it"}
    expected_datasets = {"mme", "mmbench", "textvqa", "chartqa"}
    epsilons = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    expected = {
        (m, d, e, method)
        for m in expected_models
        for d in expected_datasets
        for e in epsilons
        for method in ("U", "N")
    }
    keys = ["model", "dataset", "epsilon", "method"]
    assert len(frame) == 192 and not frame.duplicated(keys).any()
    assert set(frame[keys].itertuples(index=False, name=None)) == expected
    assert (frame.samples == 1000).all()
    assert np.allclose(frame.baseline_score_percent - frame.score_percent, frame.score_drop_pp)
    assert frame.groupby(["model", "dataset"]).baseline_score_percent.nunique().eq(1).all()
    # Independently reconcile each plotted run against its persisted execution summary.
    for row in frame.itertuples():
        execution = json.loads(Path(row.source).with_name("summary.json").read_text())
        condition = "N" if row.method == "N" else "A2"
        measured = next(c for c in execution["conditions"] if c["condition"] == condition)
        baseline = next(c for c in execution["conditions"] if c["condition"] == "baseline")
        assert np.isclose(measured["score"] * 100, row.score_percent, rtol=0, atol=1e-9)
        assert np.isclose(baseline["score"] * 100, row.baseline_score_percent, rtol=0, atol=1e-9)
    rows = []
    for method in ("U", "N"):
        for epsilon in epsilons:
            group = frame[(frame.method == method) & (frame.epsilon == epsilon)]
            rows.append(
                dict(
                    method=method,
                    epsilon=epsilon,
                    runs=len(group),
                    mean_drop_pp=group.score_drop_pp.mean(),
                    sd_drop_pp=group.score_drop_pp.std(ddof=1),
                    mean_pruning_percent=group.mean_pruning_percent.mean(),
                )
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "overall_uniform_vs_nonuniform.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.23, top=0.75)
    fig.suptitle("Overall pruning: uniform vs nonuniform", fontsize=23, fontweight="bold", y=0.965)
    fig.text(
        0.54, 0.885, "4 models × 4 datasets | 1,000 frozen samples per run", ha="center", color="#475569", fontsize=14
    )
    fig.text(
        0.54,
        0.827,
        "Equal-weighted mean score drop from each run's own baseline",
        ha="center",
        color="#64748B",
        fontsize=12,
    )
    x = np.r_[0, epsilons]
    for method, label, color, marker, offset in [
        ("U", "U: uniform across layers", "#397DB5", "o", 16),
        ("N", "N: nonuniform across layers", "#D58A49", "s", -38),
    ]:
        group = summary[summary.method == method].sort_values("epsilon")
        mean = np.r_[0, group.mean_drop_pp.to_numpy()]
        sd = np.r_[0, group.sd_drop_pp.to_numpy()]
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        ax.plot(x, mean, color=color, marker=marker, lw=2.7, ms=7, label=label)
        ax.annotate(
            f"{mean[-1]:.2f} pp",
            (x[-1], mean[-1]),
            xytext=(-12, offset),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=12,
        )
    ax.axhline(0, color="#8190A2", ls="--", lw=1.3)
    ax.set_xticks(x, ["0", "0.05", "0.10", "0.15", "0.20", "0.25", "0.30"])
    ax.set(
        xlabel="Mean response-score budget ε",
        ylabel="Score drop from baseline (percentage points)",
        xlim=(-0.007, 0.308),
    )
    ax.grid(color="#E4EAF0", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#E2E8F0", fontsize=11)
    fig.text(
        0.12,
        0.13,
        "Shaded bands: ±1 SD across 16 runs (ddof=1); descriptive spread, not confidence intervals.",
        color="#64748B",
        fontsize=10,
    )
    fig.text(
        0.12,
        0.085,
        "Positive drop = worse performance. Task metrics differ; each model–dataset run has equal weight.",
        color="#64748B",
        fontsize=10,
    )
    fig.text(
        0.12,
        0.04,
        "Same response budget, not the same removal count. The point at ε=0 denotes the unpruned baseline.",
        color="#64748B",
        fontsize=10,
    )
    stem = output / "overall_uniform_vs_nonuniform"
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=220, facecolor="white")
    plt.close(fig)
    stem.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                weighting="equal weight across 16 model-dataset runs",
                band="mean +/- sample SD, ddof=1",
                execution_summaries_checked=192,
                baseline_anchor="unpruned; not a new epsilon=0 experiment",
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
