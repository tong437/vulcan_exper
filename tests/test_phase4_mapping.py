# Copyright 2026 the LlamaFactory team.
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

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from fit_phase4_activation_mapping import add_phase4_ranking_columns, fit_mapping  # noqa: E402
from phase4_mapping_utils import (  # noqa: E402
    fit_linear_mapper,
    fit_randomized_projector,
    image_disjoint_split,
    masked_pool,
    regression_metrics,
    split_local_derangement,
    verify_image_disjoint_split,
)
from run_phase44a_dose_response import (  # noqa: E402
    _control_distribution,
    build_control_conditions,
    build_qstratified_control_table,
    global_random_condition,
    mapping_condition,
    q_condition,
    qstrat_column,
    qstrat_condition,
    summarize_results,
)


def test_image_split_and_shuffle_are_deterministic_disjoint_and_split_local():
    image_ids = [f"image_{index // 2}.jpg" for index in range(24)]
    first = image_disjoint_split(image_ids, train_ratio=0.6, validation_ratio=0.2, seed=7)
    second = image_disjoint_split(image_ids, train_ratio=0.6, validation_ratio=0.2, seed=7)
    assert first == second
    verification = verify_image_disjoint_split(image_ids, first)
    assert verification["is_image_disjoint"]
    assert sum(verification["row_counts"].values()) == len(image_ids)

    mapping = split_local_derangement(image_ids, first, seed=11)
    split_lookup = dict(zip(image_ids, first))
    assert all(source != target for source, target in mapping.items())
    assert all(split_lookup[source] == split_lookup[target] for source, target in mapping.items())


def test_masked_pool_supports_signed_mean_and_max_abs():
    values = torch.tensor(
        [
            [[1.0, -5.0], [3.0, 2.0], [100.0, 100.0]],
            [[-2.0, 1.0], [4.0, -7.0], [8.0, 9.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, True, True]])
    mean_max = masked_pool(values, mask, "mean_max_abs")
    assert mean_max.shape == (2, 4)
    assert mean_max[0].tolist() == pytest.approx([2.0, -1.5, 3.0, -5.0])
    assert mean_max[1].tolist() == pytest.approx([6.0, 1.0, 8.0, 9.0])


def test_ridge_mapper_recovers_a_low_rank_linear_mapping():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(120, 12))
    weights = rng.normal(size=(12, 8))
    targets = features @ weights + rng.normal(scale=0.01, size=(120, 8))
    projector = fit_randomized_projector(features[:80], rank=12, seed=4)
    mapper = fit_linear_mapper(
        projector.transform(features[:80]),
        targets[:80],
        alpha=0.01,
        seed=5,
    )
    metrics = regression_metrics(targets[80:], mapper.predict(projector.transform(features[80:])))
    assert metrics["variance_weighted_r2"] > 0.99
    assert metrics["mean_neuron_correlation"] > 0.99


def test_phase4_ranking_keeps_raw_dead_neuron_nan_but_produces_finite_scores():
    table = pd.DataFrame(
        {
            "layer": [0, 0, 0],
            "q_multimodal": [0.8, np.nan, 0.2],
            "mapping_r2": [0.1, 0.2, -0.3],
            "image_induced_mean_abs_delta": [1.0, 2.0, 3.0],
        }
    )
    ranked = add_phase4_ranking_columns(table, "layer")

    assert np.isnan(ranked.loc[1, "q_multimodal"])
    assert ranked.loc[1, "q_multimodal_rank_value"] == 0.0
    assert np.isfinite(ranked["combined_protection"]).all()


def test_qstratified_controls_match_mapping_q_histogram_and_exact_counts():
    rows = []
    width = 100
    for layer in range(2):
        for neuron in range(width):
            rows.append(
                {
                    "layer": layer,
                    "neuron_idx": neuron,
                    "q_visual": 0.0,
                    "q_text": 0.0,
                    "q_multimodal": neuron / width,
                    "q_multimodal_rank_value": neuron / width,
                    "q_unknown": 0.0,
                    "r_visual": 0.0,
                    "r_text": 0.0,
                    "r_multimodal": 0.0,
                    "r_unknown": 0.0,
                    "mapping_signal": float((neuron * 17 + layer * 3) % width),
                }
            )
    table = pd.DataFrame(rows)
    first, metadata = build_qstratified_control_table(
        table,
        ratios=[0.1],
        control_seeds=[1, 2],
        q_bins=5,
        base_seed=7,
    )
    second, _ = build_qstratified_control_table(
        table,
        ratios=[0.1],
        control_seeds=[1, 2],
        q_bins=5,
        base_seed=7,
    )

    target_counts = metadata["controls"]["p1000"]["per_layer_mapping_q_bin_counts"]
    for seed in (1, 2):
        column = qstrat_column(0.1, seed)
        assert first[column].equals(second[column])
        for layer, group in first.groupby("layer"):
            assert int(group[column].sum()) == 10
            q_bins = np.arange(width) * 5 // width
            selected_bins = q_bins[group[column].to_numpy()]
            assert np.bincount(selected_bins, minlength=5).tolist() == target_counts[str(layer)]


def test_control_distribution_uses_a_finite_sample_empirical_p():
    metrics = {
        "none": {"accuracy": 0.9, "yes_ratio": 0.5},
        "mapping": {"accuracy": 0.5, "yes_ratio": 1.0, "delta_accuracy": -0.4},
        "r1": {"accuracy": 0.8, "yes_ratio": 0.6, "delta_accuracy": -0.1},
        "r2": {"accuracy": 0.7, "yes_ratio": 0.7, "delta_accuracy": -0.2},
        "r3": {"accuracy": 0.6, "yes_ratio": 0.96, "delta_accuracy": -0.3},
    }
    result = _control_distribution(metrics, ["r1", "r2", "r3"], mapping_delta=-0.4)

    assert result["causal_enrichment"] == pytest.approx(0.2)
    assert result["empirical_p_mapping_more_damaging"] == 0.25
    assert result["yes_collapse_frequency"] == pytest.approx(1 / 3)


def test_control_condition_list_includes_every_matched_reference():
    conditions = build_control_conditions([0.01, 0.05], [1, 2])

    for ratio in (0.01, 0.05):
        assert q_condition(ratio) in conditions
        assert mapping_condition(ratio) in conditions
        assert global_random_condition(ratio, 1) in conditions
        assert qstrat_condition(ratio, 2) in conditions


def test_phase44a_summary_reports_monotonicity_and_matched_control_enrichment(tmp_path):
    ratios = [0.01, 0.05]
    seeds = [1, 2]
    main_metrics = {"none": {"accuracy": 0.9, "f1": 0.9, "yes_ratio": 0.5}}
    control_metrics = dict(main_metrics)
    for ratio, mapping_delta in zip(ratios, (-0.05, -0.30)):
        main_metrics[q_condition(ratio)] = {
            "accuracy": 0.9,
            "f1": 0.9,
            "yes_ratio": 0.5,
            "delta_accuracy": 0.0,
            "delta_f1": 0.0,
        }
        mapping_row = {
            "accuracy": 0.9 + mapping_delta,
            "f1": 0.9 + mapping_delta,
            "yes_ratio": 0.5 - mapping_delta,
            "delta_accuracy": mapping_delta,
            "delta_f1": mapping_delta,
        }
        main_metrics[mapping_condition(ratio)] = mapping_row
        control_metrics[mapping_condition(ratio)] = mapping_row
        for seed in seeds:
            for condition in (
                global_random_condition(ratio, seed),
                qstrat_condition(ratio, seed),
            ):
                control_metrics[condition] = {
                    "accuracy": 0.89,
                    "f1": 0.89,
                    "yes_ratio": 0.51,
                    "delta_accuracy": -0.01,
                    "delta_f1": -0.01,
                }
    main_file = tmp_path / "main.json"
    control_file = tmp_path / "controls.json"
    main_file.write_text(json.dumps({"metrics": main_metrics}), encoding="utf-8")
    control_file.write_text(json.dumps({"metrics": control_metrics}), encoding="utf-8")

    result = summarize_results(
        main_files={"random": main_file},
        control_file=control_file,
        ratios=ratios,
        control_seeds=seeds,
        max_accuracy_drop=0.01,
        max_yes_ratio_shift=0.05,
        min_dose_spearman=0.8,
        min_enriched_ratios=2,
        max_empirical_p=1 / 3,
    )

    assert result["gates"]["monotonic_dose_response"]["passed"]
    assert result["gates"]["mapping_causal_enrichment"]["passed"]
    assert result["main_dose_curve"]["random"]["first_unsafe_ratio"] == 0.01


def _score_table(num_layers: int, width: int) -> pd.DataFrame:
    rows = []
    for layer in range(num_layers):
        for neuron in range(width):
            rows.append(
                {
                    "layer": layer,
                    "neuron_idx": neuron,
                    "q_visual": 0.1,
                    "q_text": 0.2,
                    "q_multimodal": neuron / width,
                    "q_unknown": 0.0,
                    "r_visual": 0.1,
                    "r_text": 0.2,
                    "r_multimodal": neuron / width,
                    "r_unknown": 0.0,
                    "is_dead": False,
                }
            )
    return pd.DataFrame(rows)


def test_end_to_end_fit_writes_gated_augmented_scores(tmp_path):
    rng = np.random.default_rng(19)
    rows_count = 72
    layers = 2
    width = 6
    vision = rng.normal(size=(rows_count, 5)).astype(np.float32)
    question = rng.normal(size=(rows_count, 3)).astype(np.float32)
    weights = [rng.normal(size=(5, width)) for _ in range(layers)]
    delta = {
        str(layer): torch.tensor(vision @ weights[layer] + rng.normal(scale=0.01, size=(rows_count, width)))
        for layer in range(layers)
    }
    splits = ["train"] * 48 + ["validation"] * 12 + ["test"] * 12
    rows = [
        {
            "row_index": index,
            "source_index": index,
            "question_id": index,
            "image_id": f"image_{index}.jpg",
            "shuffled_image_id": f"other_{index}.jpg",
            "split": splits[index],
            "question": "Question?",
            "answer": "yes",
        }
        for index in range(rows_count)
    ]

    activation_dir = tmp_path / "activations"
    chunk_dir = activation_dir / "chunks"
    chunk_dir.mkdir(parents=True)
    torch.save(
        {
            "start_index": 0,
            "end_index": rows_count,
            "vision_features": torch.tensor(vision),
            "question_features": torch.tensor(question),
            "delta_activations": delta,
            "rows": rows,
        },
        chunk_dir / f"chunk_000000_{rows_count:06d}.pt",
    )
    (activation_dir / "collection_state.json").write_text(
        json.dumps(
            {
                "complete": True,
                "completed_rows": rows_count,
                "config": {"output_dir": str(activation_dir)},
            }
        ),
        encoding="utf-8",
    )
    score_file = tmp_path / "scores.parquet"
    _score_table(layers, width).to_parquet(score_file, index=False)
    output_dir = tmp_path / "fit"
    args = Namespace(
        activation_dir=str(activation_dir),
        score_file=str(score_file),
        output_dir=str(output_dir),
        feature_rank=8,
        target_rank=3,
        alphas="0.01,0.1",
        top_k=2,
        null_permutations=19,
        primary_model="reduced_rank",
        min_mapping_r2=0.1,
        min_incremental_r2=0.1,
        max_null_p=0.05,
        min_positive_layers=1,
        seed=23,
    )

    result = fit_mapping(args)

    assert result["gates"]["gate_a_mapping_exists"]["passed"]
    assert result["gates"]["gate_b_incremental_value"]["passed"]
    assert result["gates"]["phase4_group_causal_ablation_allowed"]
    augmented = pd.read_parquet(output_dir / "neuron_scores_with_mapping.parquet")
    assert {"mapping_signal", "combined_protection"}.issubset(augmented.columns)
    assert np.isfinite(augmented["mapping_signal"]).all()
