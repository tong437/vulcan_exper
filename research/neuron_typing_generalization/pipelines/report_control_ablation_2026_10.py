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

"""Validate and summarize the completed 16-run W/C ablation matrix."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {"mme": "MME", "mmbench": "MMBench", "textvqa": "TextVQA", "chartqa": "ChartQA"}
CONDITIONS = ("baseline", "W_random", "C_random", "W_max", "C_max", "W_min", "C_min", "global_random")
REPEATS = {condition: (10 if "random" in condition else 1) for condition in CONDITIONS}


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    return plt


def _load_run(root: Path, model: str, dataset: str) -> tuple[list[dict], dict]:
    directory = root / f"{model}__{dataset}"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["expected_samples"] != 1000 or summary["removal_fraction"] != 0.01:
        raise ValueError(f"Wrong sample count or deletion fraction: {directory}")
    if summary["random_repeats"] != 10:
        raise ValueError(f"Wrong random-repeat count: {directory}")
    rows = []
    seen = set()
    with (directory / "predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["sample_index"], row["condition"], row["repeat"])
            if key in seen:
                raise ValueError(f"Duplicate prediction key in {directory}: {key}")
            seen.add(key)
            if row["condition"] not in CONDITIONS or not 0 <= row["sample_index"] < 1000:
                raise ValueError(f"Unexpected condition or sample index in {directory}: {key}")
            if not 0 <= row["repeat"] < REPEATS[row["condition"]]:
                raise ValueError(f"Unexpected repeat in {directory}: {key}")
            if not 0 <= row["score"] <= 1:
                raise ValueError(f"Score outside [0, 1] in {directory}: {key}")
            if not 0 <= row["selected_neurons"] <= row["target_neurons"]:
                raise ValueError(f"Invalid deletion count in {directory}: {key}")
            rows.append(row)
    if len(rows) != 35000:
        raise ValueError(f"Expected 35,000 predictions, found {len(rows)}: {directory}")
    counts = Counter((row["condition"], row["repeat"]) for row in rows)
    if len(counts) != 35 or any(count != 1000 for count in counts.values()):
        raise ValueError(f"Incomplete condition/repeat runs in {directory}")
    sample_ids = defaultdict(set)
    for row in rows:
        sample_ids[row["sample_index"]].add(row["sample_id"])
    if len(sample_ids) != 1000 or any(len(ids) != 1 for ids in sample_ids.values()):
        raise ValueError(f"Sample IDs disagree across conditions in {directory}")
    return rows, summary


def _plot_overview(frame: pd.DataFrame, output_dir: Path) -> None:
    plt = _load_pyplot()
    figure, axes = plt.subplots(4, 4, figsize=(17.5, 12.5), sharey=False, facecolor="white")
    colors = {"W": "#397DB5", "C": "#8B6FB1"}
    methods = ("random", "max", "min")
    for model_index, model in enumerate(MODELS):
        for dataset_index, dataset in enumerate(DATASETS):
            axis = axes[model_index, dataset_index]
            data = frame[(frame.model == model) & (frame.dataset == dataset)].set_index("condition")
            for state_index, state in enumerate(("W", "C")):
                values = [float(data.loc[f"{state}_{method}", "drop_pp"]) for method in methods]
                errors = [
                    float(data.loc[f"{state}_{method}", "sd_pp"]) if method == "random" else 0 for method in methods
                ]
                axis.bar(
                    np.arange(3) + (state_index - 0.5) * 0.36,
                    values,
                    width=0.34,
                    yerr=errors,
                    color=colors[state],
                    capsize=2,
                    label=state,
                )
            reference = float(data.loc["global_random", "drop_pp"])
            axis.axhline(reference, color="#64748B", linestyle="--", linewidth=1)
            axis.axhline(0, color="#334155", linewidth=0.8)
            axis.set_xticks(np.arange(3), ["Random", "Max", "Min"])
            axis.set_title(f"{MODELS[model]} · {DATASETS[dataset]}", fontsize=10)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
            if dataset_index == 0:
                axis.set_ylabel("Score drop (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.975), frameon=False)
    figure.suptitle("W/C neuron ablation · 1% target per layer", fontsize=18, y=0.998)
    figure.text(0.5, 0.925, "Dashed line: global random. Error bars: SD over 10 random repeats.", ha="center")
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(output_dir / "wc_ablation_overview.png", dpi=200, facecolor="white")
    figure.savefig(output_dir / "wc_ablation_overview.pdf", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    condition_rows = []
    comparison_rows = []
    for model in MODELS:
        for dataset in DATASETS:
            rows, summary = _load_run(args.input_root, model, dataset)
            by_run = defaultdict(list)
            by_condition = defaultdict(list)
            for row in rows:
                by_run[(row["condition"], row["repeat"])].append(row["score"])
                by_condition[row["condition"]].append(row)
            means = {
                condition: np.array([np.mean(by_run[(condition, repeat)]) for repeat in range(REPEATS[condition])])
                for condition in CONDITIONS
            }
            baseline = float(means["baseline"][0])
            if not np.isclose(baseline, summary["baseline_score"], atol=1e-12):
                raise ValueError(f"Baseline differs from saved summary: {model} x {dataset}")
            condition_lookup = {item["condition"]: item for item in summary["conditions"]}
            for condition in CONDITIONS:
                scores = means[condition]
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                saved = condition_lookup[condition]
                if not np.isclose(mean_score, saved["score_mean"], atol=1e-12) or not np.isclose(
                    std_score, saved["score_std"], atol=1e-12
                ):
                    raise ValueError(f"Saved summary differs from predictions: {model} x {dataset} x {condition}")
                subset = by_condition[condition]
                condition_rows.append(
                    {
                        "model": model,
                        "dataset": dataset,
                        "condition": condition,
                        "baseline_percent": baseline * 100,
                        "score_percent": mean_score * 100,
                        "drop_pp": (baseline - mean_score) * 100,
                        "sd_pp": std_score * 100,
                        "repeats": len(scores),
                        "selected_mean": float(np.mean([row["selected_neurons"] for row in subset])),
                        "selected_min": min(row["selected_neurons"] for row in subset),
                        "selected_max": max(row["selected_neurons"] for row in subset),
                        "target_neurons": subset[0]["target_neurons"],
                        "shortfall_prediction_fraction": float(
                            np.mean([row["shortfall_layers"] > 0 for row in subset])
                        ),
                        "empty_prediction_fraction": float(np.mean([not row["prediction"] for row in subset])),
                    }
                )
            comparison_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "C_minus_W_random_drop_pp": 100 * float(np.mean(means["W_random"] - means["C_random"])),
                    "C_minus_W_random_sd_pp": 100 * float(np.std(means["W_random"] - means["C_random"], ddof=1)),
                    "C_minus_W_max_drop_pp": 100 * float(means["W_max"][0] - means["C_max"][0]),
                    "C_minus_W_min_drop_pp": 100 * float(means["W_min"][0] - means["C_min"][0]),
                }
            )
    conditions = pd.DataFrame(condition_rows)
    comparisons = pd.DataFrame(comparison_rows)
    conditions.to_csv(args.output_dir / "wc_ablation_conditions.csv", index=False)
    comparisons.to_csv(args.output_dir / "wc_ablation_comparisons.csv", index=False)
    _plot_overview(conditions, args.output_dir)
    lines = [
        "# W/C 1% neuron ablation: 16 completed runs",
        "",
        "Each model–dataset run contains 1,000 frozen examples and 35 generation conditions (35,000 predictions).",
        "Positive score drop means performance worsened relative to the same run's baseline. Random values show mean ± SD across 10 complete repeats; Max and Min use one deterministic run.",
        "The target is 1% per decoder layer. A state pool smaller than the target is deleted in full, without drawing from another class.",
        "",
        "| Model | Dataset | Baseline | W random drop | C random drop | W Max drop | C Max drop | W Min drop | C Min drop | Global random drop | C actual/target |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for dataset in DATASETS:
            data = conditions[(conditions.model == model) & (conditions.dataset == dataset)].set_index("condition")
            random_cell = lambda condition: (
                f"{data.loc[condition, 'drop_pp']:+.2f} ± {data.loc[condition, 'sd_pp']:.2f}"
            )
            lines.append(
                f"| {MODELS[model]} | {DATASETS[dataset]} | {data.loc['baseline', 'baseline_percent']:.2f}% | "
                f"{random_cell('W_random')} | {random_cell('C_random')} | "
                f"{data.loc['W_max', 'drop_pp']:+.2f} | {data.loc['C_max', 'drop_pp']:+.2f} | "
                f"{data.loc['W_min', 'drop_pp']:+.2f} | {data.loc['C_min', 'drop_pp']:+.2f} | "
                f"{random_cell('global_random')} | "
                f"{data.loc['C_random', 'selected_mean']:.1f}/{int(data.loc['C_random', 'target_neurons'])} |"
            )
    lines.extend(
        [
            "",
            "![W/C ablation overview](wc_ablation_overview.png)",
            "",
            "## Interpretation limits",
            "",
            "- Qwen3-VL and LLaVA have one C-pool shortfall layer in every sample. Their C intervention deletes fewer neurons than W or global random, so those contrasts also differ in total deletion dose.",
            "- C is defined from non-visual, non-question-text, non-padding input positions, including native template and special tokens. These interventions do not isolate a particular token or prove which position caused a score change.",
            "- The Qwen3-VL C conditions often produce empty output or repeat template text; the score collapse therefore reflects generation failure as well as answer correctness.",
            "- Random SD describes variation across the ten selection runs. It is not a confidence interval over newly sampled questions.",
            "",
            "Complete condition statistics: [CSV](wc_ablation_conditions.csv). W/C contrasts: [CSV](wc_ablation_comparisons.csv).",
            "",
        ]
    )
    (args.output_dir / "wc_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": 16,
                "predictions": 560000,
                "conditions": len(conditions),
                "maximum_absolute_drop_pp": float(conditions.drop_pp.abs().max()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
