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

"""Plot matched-count U/N comparisons with paired group-bootstrap intervals."""

import argparse
import hashlib
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from report_epsilon_sweep_2026_10 import _paired_group_ci
from report_nonuniform_budget_2026_10 import DATASETS, EPSILONS, MODELS, read


def plot(frame, title, metric, output, example=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.8))
    fig.subplots_adjust(left=0.08, right=0.96, top=0.73, bottom=0.32, wspace=0.28)
    fig.suptitle("Matched-count pruning: uniform vs nonuniform", fontsize=23, fontweight="bold", y=0.97)
    fig.text(0.52, 0.91, title, ha="center", fontsize=16, color="#475569")
    fig.text(
        0.52,
        0.855,
        "SIMULATED EXAMPLE — not experimental results"
        if example
        else "Measured results | 1000 paired samples | Equal removed count for every sample",
        ha="center",
        fontsize=12,
        color="#A94E49" if example else "#64748B",
    )
    x = frame.pruning_percent.to_numpy()
    for col, label, color, marker in [
        ("u_score", "U: uniform", "#397DB5", "o"),
        ("n_score", "N_match: nonuniform", "#D98A49", "s"),
    ]:
        axes[0].plot(x, frame[col], color=color, marker=marker, lw=2.5, label=label)
    axes[0].axhline(frame.baseline.iloc[0], ls="--", color="#8B99AB", label=f"Baseline: {frame.baseline.iloc[0]:.1f}%")
    axes[0].set_ylabel(metric + " (%)")
    axes[0].set_title("Task score at the same removal count")
    axes[0].legend(fontsize=10)
    gain = frame.gain_pp.to_numpy()
    axes[1].vlines(x, frame.ci_low, frame.ci_high, color="#475569", lw=1.5)
    for bound in (frame.ci_low, frame.ci_high):
        axes[1].plot(x, bound, linestyle="none", marker="_", ms=9, color="#475569")
    axes[1].plot(x, gain, marker="o", color="#D98A49", lw=2)
    axes[1].axhline(0, color="#8B99AB", ls="--")
    axes[1].set_ylabel("Score(N_match) − Score(U) (pp)")
    axes[1].set_title("Paired score difference with 95% intervals")
    for ax in axes:
        ax.set_xlabel("Mean actual pruning rate (%)")
        ax.set_xticks(x, [f"{v:.1f}%" for v in x], fontsize=9)
        ax.grid(color="#E4EAF0")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    table_ax = fig.add_axes((0.08, 0.105, 0.88, 0.11))
    table_ax.axis("off")
    table_ax.text(0, 1.18, "Actual mean response share removed (B)", color="#475569")
    table = table_ax.table(
        cellText=[["U", *[f"{v:.4f}%" for v in frame.u_budget]], ["N_match", *[f"{v:.4f}%" for v in frame.n_budget]]],
        colLabels=["Method", *[f"ε={v:.2f}" for v in frame.epsilon]],
        bbox=(0, 0, 1, 1),
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#DDE5EC")
    fig.text(
        0.08,
        0.035,
        "Score axis is zoomed. Positive difference favors N_match. "
        + (
            "Illustrative intervals only."
            if example
            else "Intervals: paired group-bootstrap; not multiplicity-adjusted."
        ),
        fontsize=10,
        color="#64748B",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix("." + ext), dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1] / "artifacts/neuron_typing_generalization"
    output = project / "reports/advisor_2026_09/figures/matched_count"
    if args.example:
        frame = pd.DataFrame(
            dict(
                epsilon=EPSILONS,
                pruning_percent=[11, 20, 28, 35, 42, 48],
                baseline=[80] * 6,
                u_score=[79.5, 79, 77, 75, 72, 68],
                n_score=[79.4, 79.2, 77.8, 76, 72.6, 67.5],
                gain_pp=[-0.1, 0.2, 0.8, 1, 0.6, -0.5],
                ci_low=[-0.5, -0.3, 0.1, 0.2, -0.5, -1.7],
                ci_high=[0.3, 0.7, 1.5, 1.8, 1.7, 0.7],
                u_budget=[4.99, 9.99, 14.99, 19.99, 24.99, 29.99],
                n_budget=[4.3, 8.7, 13.2, 18.1, 22.9, 27.8],
            )
        )
        plot(
            frame,
            "Qwen3-VL-2B | MMBench | Illustrative layout",
            "Accuracy",
            output / "matched_count_SIMULATED_example",
            example=True,
        )
        return
    missing = []
    for m in MODELS:
        for d in DATASETS:
            for e in EPSILONS:
                p = root / "matched_count_2026_10" / f"{m}__{d}__eps_{e:.2f}".replace(".", "p") / "summary.json"
                if not p.exists():
                    missing.append(str(p))
    if missing:
        raise SystemExit(
            f"Incomplete matched-count experiment: {len(missing)}/96 conditions missing. No measured report generated."
        )
    records = []
    for model, label in MODELS.items():
        for ds, (ds_label, metric) in DATASETS.items():
            group = []
            for eps in EPSILONS:
                target = root / "matched_count_2026_10" / f"{model}__{ds}__eps_{eps:.2f}".replace(".", "p")
                config, n = read(target)
                uniform_path = Path(config["uniform_source"])
                assert (
                    hashlib.sha256((uniform_path / "predictions.jsonl").read_bytes()).hexdigest()
                    == config["uniform_predictions_sha256"]
                )
                uc, u = read(uniform_path)
                for key in ("model", "dataset", "epsilon", "source_run_id", "input_policy", "max_new_tokens"):
                    assert config[key] == uc[key]
                a, b = n["N_match"], u["A2"]
                for key in ("sample_id", "group_id", "selected_neurons", "actual_ratio"):
                    assert a[key].equals(b[key]), key
                for key in ("sample_id", "prediction", "score"):
                    assert n["baseline"][key].equals(u["baseline"][key]), key
                assert (a.actual_score_share <= b.actual_score_share + 1e-10).all()
                gain = (a.score - b.score).to_numpy() * 100
                low, high = _paired_group_ci(gain, a.group_id.to_numpy(), 20261013, 5000)
                row = dict(
                    model=model,
                    dataset=ds,
                    epsilon=eps,
                    baseline=n["baseline"].score.mean() * 100,
                    pruning_percent=a.actual_ratio.mean() * 100,
                    u_score=b.score.mean() * 100,
                    n_score=a.score.mean() * 100,
                    gain_pp=gain.mean(),
                    ci_low=low,
                    ci_high=high,
                    u_budget=b.actual_score_share.mean() * 100,
                    n_budget=a.actual_score_share.mean() * 100,
                    source=str(target),
                    uniform_source=str(uniform_path),
                    bootstrap_seed=20261013,
                    bootstrap_repeats=5000,
                )
                group.append(row)
                records.append(row)
            plot(pd.DataFrame(group), f"{label} | {ds_label}", metric, output / f"{model}__{ds}")
    pd.DataFrame(records).to_csv(output / "measured_comparison.csv", index=False)
    print("Wrote 16 measured comparisons and 96-row summary.")


if __name__ == "__main__":
    main()
