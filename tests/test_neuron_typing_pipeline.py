from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from build_pruning_baseline_scores import (  # noqa: E402
    add_pruning_baseline_scores,
    compute_group_weight_magnitude,
)
from dataset_guard import (  # noqa: E402
    assert_disjoint_manifests,
    build_dataset_manifest,
    slice_dataset,
)
from run_phase2_ablation import (  # noqa: E402
    build_type_mask,
    compute_relative_damage,
    get_layer_dims,
    infer_score_columns,
    paired_bootstrap_analysis,
    parse_ablation_spec,
    verify_mask_nesting,
    verify_matched_score_controls,
)
from score_neuron_types import compute_type_scores  # noqa: E402


class TinyDataset(list):
    def select(self, indices):
        return TinyDataset(self[index] for index in indices)


def test_dataset_slice_rejects_stale_short_cache():
    dataset = TinyDataset({"images": [f"img_{idx}.jpg"]} for idx in range(4))
    with pytest.raises(ValueError, match="stale"):
        slice_dataset(dataset, sample_offset=2, max_samples=3)


def test_manifest_rejects_corrupted_repeated_image():
    dataset = TinyDataset({"images": ["same.jpg"]} for _ in range(6))
    with pytest.raises(ValueError, match="likely corrupted"):
        build_dataset_manifest(
            dataset,
            list(range(6)),
            role="typing",
            dataset_name="dummy",
            tokenized_path=None,
            max_image_repeat=5,
        )


def test_manifest_isolation(tmp_path):
    current = {"role": "eval", "image_ids": ["a.jpg", "b.jpg"]}
    other = tmp_path / "typing.json"
    other.write_text(json.dumps({"role": "typing", "image_ids": ["c.jpg"]}))
    assert assert_disjoint_manifests(current, [other])["is_isolated"]
    other.write_text(json.dumps({"role": "typing", "image_ids": ["b.jpg"]}))
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint_manifests(current, [other])


def _score_table(neurons_per_layer: int = 10) -> pd.DataFrame:
    rows = []
    for layer in range(2):
        for neuron in range(neurons_per_layer):
            rows.append(
                {
                    "layer": layer,
                    "neuron_idx": neuron,
                    "q_visual": 0.0,
                    "q_text": 0.0,
                    "q_multimodal": float(neurons_per_layer - 1 - neuron) / neurons_per_layer,
                    "q_unknown": 0.0,
                    "r_visual": 0.0,
                    "r_text": 0.0,
                    "r_multimodal": float(neuron) / neurons_per_layer,
                    "r_unknown": float(neuron) / neurons_per_layer,
                    "weight_magnitude": float(neuron),
                    "activation_frequency": float(neurons_per_layer - 1 - neuron),
                }
            )
    return pd.DataFrame(rows)


def _mask(table, text):
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    return build_type_mask(
        table,
        parse_ablation_spec(text, 42),
        layer_col,
        neuron_col,
        score_cols,
        activation_col,
        get_layer_dims(table, layer_col, neuron_col),
        None,
        "per_layer",
        1.0,
        0.0,
    )


def test_masks_are_deterministic_nested_and_rank_band_is_difference():
    table = _score_table()
    small = _mask(table, "multimodal:0.2")
    large = _mask(table, "multimodal:0.5")
    repeated = _mask(table, "multimodal:0.5")
    band = _mask(table, "rank_band:multimodal:0.2:0.5")
    assert verify_mask_nesting({0.2: small, 0.5: large})["20% ⊂ 50%"]
    for layer in large:
        assert torch.equal(large[layer], repeated[layer])
        assert torch.equal(band[layer], large[layer] & ~small[layer])


def test_rank_window_selects_exact_deterministic_per_layer_slice():
    table = _score_table(neurons_per_layer=10)
    window = _mask(table, "rank_window:multimodal:2:4")
    repeated = _mask(table, "rank_window:multimodal:2:4")

    assert parse_ablation_spec("rank_window:multimodal:2:4", 42).result_name == ("rank_window:multimodal:2:4")
    for layer in window:
        assert window[layer].nonzero().flatten().tolist() == [2, 3, 4, 5]
        assert torch.equal(window[layer], repeated[layer])


def test_matched_random_uses_exact_rank_band_counts_and_partition():
    table = _score_table(neurons_per_layer=6)
    band = _mask(table, "rank_band:multimodal:0.2:0.5")
    matched_small = _mask(table, "matched_random:multimodal:0.2:seed7")
    matched_band = _mask(table, "matched_random:rank_band:multimodal:0.2:0.5:seed7")
    matched_large = _mask(table, "matched_random:multimodal:0.5:seed7")
    repeated = _mask(table, "matched_random:rank_band:multimodal:0.2:0.5:seed7")
    ratio_random = _mask(table, "random:0.3:seed7")

    assert (
        parse_ablation_spec("matched_random:rank_band:multimodal:0.2:0.5:seed7", 42).result_name
        == "matched_random:rank_band:multimodal:0.2:0.5:seed7"
    )
    for layer in band:
        assert int(band[layer].sum()) == 1
        assert int(matched_band[layer].sum()) == int(band[layer].sum())
        assert int(ratio_random[layer].sum()) == 2
        assert torch.equal(matched_band[layer], repeated[layer])
        assert not bool((matched_small[layer] & matched_band[layer]).any())
        assert torch.equal(matched_small[layer] | matched_band[layer], matched_large[layer])


def test_matched_score_uses_exact_reference_budget():
    table = _score_table(neurons_per_layer=6)
    reference_spec = parse_ablation_spec("rank_band:multimodal:0.2:0.5", 42)
    magnitude_spec = parse_ablation_spec("matched_score:weight_magnitude:lowest:rank_band:multimodal:0.2:0.5", 42)
    activation_spec = parse_ablation_spec("matched_score:activation_frequency:lowest:rank_band:multimodal:0.2:0.5", 42)
    specs = [reference_spec, magnitude_spec, activation_spec]
    masks = {spec.result_name: _mask(table, spec.result_name) for spec in specs}

    verification = verify_matched_score_controls(specs, masks)
    assert magnitude_spec.result_name == ("matched_score:weight_magnitude:lowest:rank_band:multimodal:0.2:0.5")
    assert all(row["counts_match"] for row in verification.values())
    for layer in masks[reference_spec.result_name]:
        assert masks[magnitude_spec.result_name][layer].nonzero().flatten().tolist() == [0]
        assert masks[activation_spec.result_name][layer].nonzero().flatten().tolist() == [5]


def test_pruning_baseline_scores_use_group_l2_and_threshold_frequency():
    gate = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    up = torch.tensor([[0.0, 0.0], [0.0, 12.0]])
    down = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    magnitude = compute_group_weight_magnitude(gate, up, down)
    assert magnitude.tolist() == pytest.approx([5.0, 13.0])

    table = pd.DataFrame(
        [
            {"layer": 0, "neuron_idx": 0, "r_unknown": 0.25},
            {"layer": 0, "neuron_idx": 1, "r_unknown": 0.75},
        ]
    )
    result = add_pruning_baseline_scores(table, {0: magnitude})
    assert result["weight_magnitude"].tolist() == pytest.approx([5.0, 13.0])
    assert result["activation_frequency"].tolist() == pytest.approx([0.75, 0.25])


def test_paired_bootstrap_uses_token_weighted_delta():
    result = paired_bootstrap_analysis(
        [10.0, 9.0],
        [10, 1],
        [20.0, 10.0],
        [10, 1],
        num_bootstrap=200,
        seed=1,
    )
    assert result["paired_delta_nll"] == pytest.approx(1.0)
    assert result["damaged_frac"] == 1.0


def test_relative_damage_reports_smoothed_empirical_p():
    result = compute_relative_damage(1.0, [0.0, 0.5])
    assert result["median_random_delta"] == pytest.approx(0.25)
    assert result["empirical_p_more_damaging"] == pytest.approx(1 / 3)


def test_scoring_uses_configured_threshold_and_r_tie_break():
    scores = {
        "layer_0": {
            "q_visual": [0.5],
            "q_text": [0.0],
            "q_multimodal": [0.5],
            "q_unknown": [0.0],
            "r_visual": [0.1],
            "r_text": [0.0],
            "r_multimodal": [0.8],
            "r_unknown": [0.1],
            "dead_mask": [False],
        }
    }
    frame = compute_type_scores(scores, high_conf_threshold=0.6)
    assert frame.loc[0, "dominant_type"] == "multimodal"
    assert bool(frame.loc[0, "dominant_tie"])
    assert frame.loc[0, "confidence_category"] == "mixed_low_confidence"
