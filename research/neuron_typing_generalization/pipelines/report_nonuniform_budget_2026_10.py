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

"""Audit paired U/N predictions and export 16 measured comparison figures."""

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from plot_nonuniform_budget_example_2026_10 import plot_comparison


MODELS = {
    "llava_1_5_7b": "LLaVA-1.5-7B",
    "qwen3_vl_2b_instruct": "Qwen3-VL-2B",
    "qwen3_5_2b": "Qwen3.5-2B",
    "gemma_4_e2b_it": "Gemma-4-E2B-it",
}
DATASETS = {
    "mme": ("MME", "Accuracy"),
    "mmbench": ("MMBench", "Accuracy"),
    "textvqa": ("TextVQA", "Soft accuracy"),
    "chartqa": ("ChartQA", "Relaxed accuracy"),
}
EPSILONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def read(directory):
    config = json.loads((directory / "run_config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    rows = pd.read_json(directory / "predictions.jsonl", lines=True)
    assert summary["expected_samples"] == 1000
    assert not rows.duplicated(["sample_index", "condition"]).any()
    groups = {c: g.sort_values("sample_index").reset_index(drop=True) for c, g in rows.groupby("condition")}
    for c, g in groups.items():
        assert len(g) == 1000 and g.sample_index.tolist() == list(range(1000))
        recorded = next(v for v in summary["conditions"] if v["condition"] == c)
        for field, stat in [
            ("score", "score"),
            ("actual_ratio", "mean_actual_ratio"),
            ("actual_score_share", "mean_actual_score_share"),
        ]:
            assert np.isclose(g[field].mean(), recorded[stat], atol=1e-10, rtol=0)
    return config, groups


def main():
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1] / "artifacts/neuron_typing_generalization"
    output = project / "reports/advisor_2026_09/figures/nonuniform_budget"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    figures = []
    for model, label in MODELS.items():
        for ds, (ds_label, metric) in DATASETS.items():
            scores, ratios, budgets = [[], []], [[], []], [[], []]
            shared_baseline = None
            for eps in EPSILONS:
                tag = f"{model}__{ds}__eps_{eps:.2f}".replace(".", "p")
                n_dir = root / "nonuniform_budget_2026_10" / tag
                u_dir = (
                    root / "adaptive_pruning_a1_a2_2026_10" / f"{model}__{ds}"
                    if eps == 0.05
                    else root / "adaptive_pruning_epsilon_sweep_2026_10" / tag
                )
                nc, n = read(n_dir)
                uc, u = read(u_dir)
                for key in [
                    "model",
                    "dataset",
                    "study",
                    "statistics_root",
                    "baseline_source",
                    "source_run_id",
                    "input_policy",
                    "max_new_tokens",
                    "epsilon",
                ]:
                    assert nc[key] == uc[key], (tag, key)
                assert nc["model"] == model and nc["dataset"] == ds and nc["epsilon"] == eps
                for key in ["sample_id", "group_id", "prediction", "score"]:
                    assert n["baseline"][key].equals(u["baseline"][key]), (tag, key)
                for key in ["sample_id", "group_id"]:
                    assert n["N"][key].equals(u["A2"][key])
                    assert n["N"][key].equals(n["baseline"][key])
                assert (n["N"].selected_neurons >= u["A2"].selected_neurons).all()
                if shared_baseline is None:
                    shared_baseline = n["baseline"]
                else:
                    assert shared_baseline.equals(n["baseline"])
                baseline = float(n["baseline"].score.mean() * 100)
                for i, (method, g, directory) in enumerate([("U", u["A2"], u_dir), ("N", n["N"], n_dir)]):
                    assert g.actual_score_share.between(0, eps + 1e-10).all()
                    assert g.actual_ratio.between(0, 1).all()
                    score, ratio, budget = [
                        float(g[k].mean() * 100) for k in ["score", "actual_ratio", "actual_score_share"]
                    ]
                    scores[i].append(score)
                    ratios[i].append(ratio)
                    budgets[i].append(budget)
                    records.append(
                        dict(
                            model=model,
                            dataset=ds,
                            epsilon=eps,
                            method=method,
                            baseline_score_percent=baseline,
                            score_percent=score,
                            score_drop_pp=baseline - score,
                            mean_pruning_percent=ratio,
                            mean_response_share_percent=budget,
                            samples=len(g),
                            source=str(directory / "predictions.jsonl"),
                        )
                    )
            stem = output / f"{model}__{ds}"
            plot_comparison(
                scores,
                ratios,
                budgets,
                baseline_score=baseline,
                subtitle=f"{label} | {ds_label} | Sample-adaptive selection",
                metric=metric,
                output=stem,
            )
            figures.append(stem)
    pd.DataFrame(records).to_csv(output / "measured_comparison.csv", index=False)
    with ZipFile(output / "nonuniform_budget_16_figures.zip", "w") as archive:
        for stem in figures:
            for extension in ("png", "pdf"):
                p = stem.with_suffix(f".{extension}")
                archive.write(p, p.name)
        archive.write(output / "measured_comparison.csv", "measured_comparison.csv")
    print("Verified 96 paired conditions; exported 16 PNGs + 16 PDFs and measured CSV.")


if __name__ == "__main__":
    main()
