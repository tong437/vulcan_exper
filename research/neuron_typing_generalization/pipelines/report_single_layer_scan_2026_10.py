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

"""Audit and plot observed single-layer calibration interventions."""

import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import numpy as np
import pandas as pd
from plot_layer_sensitivity_detail import plot_detail
from report_epsilon_sweep_2026_10 import _paired_group_ci
from report_nonuniform_budget_2026_10 import DATASETS, MODELS


def main():
    project = Path(__file__).resolve().parents[1]
    root = project.parents[1] / "artifacts/neuron_typing_generalization/single_layer_scan_2026_10"
    output = project / "reports/advisor_2026_09/figures/single_layer_scan"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for model in ("qwen3_vl_2b_instruct", "llava_1_5_7b"):
        for ds, (ds_label, metric) in DATASETS.items():
            directory = root / f"{model}__{ds}"
            cfg = json.loads((directory / "run_config.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            rows = pd.read_json(directory / "predictions.jsonl", lines=True)
            indices = cfg["split_manifest"]["calibration_indices"]
            assert len(indices) == summary["expected_samples"]
            assert not set(indices) & set(cfg["split_manifest"]["heldout_indices"])
            assert not rows.duplicated(["sample_index", "condition"]).any()
            assert len(rows) == len(indices) * (1 + len(cfg["scan_layers"]) * len(cfg["scan_rates"]))
            baseline = rows[rows.condition == "baseline"].sort_values("sample_index").reset_index(drop=True)
            assert baseline.sample_index.tolist() == indices
            source = pd.read_json(Path(cfg["baseline_source"]) / "predictions.jsonl", lines=True)
            source = (
                source[(source.condition == "baseline") & source.sample_index.isin(indices)]
                .sort_values("sample_index")
                .reset_index(drop=True)
            )
            for key in ("sample_id", "prediction", "score"):
                assert baseline[key].equals(source[key]), key
            summaries = {v["condition"]: v for v in summary["conditions"]}
            group = []
            for condition, data in rows[rows.condition != "baseline"].groupby("condition"):
                data = data.sort_values("sample_index").reset_index(drop=True)
                assert data.sample_index.tolist() == indices
                for key in ("sample_id", "group_id"):
                    assert data[key].equals(baseline[key])
                layer = int(data.target_layer.iloc[0])
                rate = float(data.target_layer_rate.iloc[0])
                assert layer in cfg["scan_layers"] and any(np.isclose(rate, r) for r in cfg["scan_rates"])
                rate = next(r for r in cfg["scan_rates"] if np.isclose(rate, r))
                width = cfg["d_ff_by_layer"][layer]
                count = int(np.floor(rate * width + 1e-9))
                for counts in data.selected_per_layer:
                    assert counts[layer] == count and sum(counts) == count
                assert np.allclose(data.actual_layer_rate, count / width)
                assert np.allclose(data.layer_response_share, data.actual_score_share * len(cfg["scan_layers"]))
                assert data.layer_response_share.between(0, 1).all()
                assert np.isclose(data.score.mean(), summaries[condition]["score"])
                drop = (baseline.score - data.score).to_numpy() * 100
                low, high = _paired_group_ci(drop, baseline.group_id.to_numpy(), 20261014, 2000)
                record = dict(
                    model=model,
                    dataset=ds,
                    layer=layer,
                    rate=rate,
                    samples=len(indices),
                    baseline_percent=baseline.score.mean() * 100,
                    score_percent=data.score.mean() * 100,
                    drop_pp=drop.mean(),
                    ci_low=low,
                    ci_high=high,
                    response_percent=data.layer_response_share.mean() * 100,
                    source=str(directory),
                    bootstrap_repeats=2000,
                    bootstrap_seed=20261014,
                )
                records.append(record)
                group.append(record)
            frame = pd.DataFrame(group)
            plot_detail(
                frame,
                f"{MODELS[model]} | {ds_label} | Calibration N={len(indices)} | Baseline: {baseline.score.mean() * 100:.1f}%",
                output / f"{model}__{ds}",
            )
            print(model, ds, "validated", len(frame), "layer/rate conditions", flush=True)
    pd.DataFrame(records).to_csv(output / "measured_layer_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
