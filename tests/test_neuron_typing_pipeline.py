from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from dataset_guard import (  # noqa: E402
    assert_disjoint_manifests,
    build_dataset_manifest,
    slice_dataset,
)
from run_phase2_ablation import (  # noqa: E402
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    paired_bootstrap_analysis,
    parse_ablation_spec,
    verify_mask_nesting,
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
            dataset, list(range(6)), role="typing", dataset_name="dummy",
            tokenized_path=None, max_image_repeat=5,
        )


def test_manifest_isolation(tmp_path):
    current = {"role": "eval", "image_ids": ["a.jpg", "b.jpg"]}
    other = tmp_path / "typing.json"
    other.write_text(json.dumps({"role": "typing", "image_ids": ["c.jpg"]}))
    assert assert_disjoint_manifests(current, [other])["is_isolated"]
    other.write_text(json.dumps({"role": "typing", "image_ids": ["b.jpg"]}))
    with pytest.raises(ValueError, match="overlaps"):
        assert_disjoint_manifests(current, [other])


def _score_table() -> pd.DataFrame:
    rows = []
    for layer in range(2):
        for neuron in range(10):
            rows.append({
                "layer": layer,
                "neuron_idx": neuron,
                "q_visual": 0.0,
                "q_text": 0.0,
                "q_multimodal": float(9 - neuron) / 10,
                "q_unknown": 0.0,
                "r_visual": 0.0,
                "r_text": 0.0,
                "r_multimodal": float(neuron) / 10,
                "r_unknown": float(neuron) / 10,
            })
    return pd.DataFrame(rows)


def _mask(table, text):
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    return build_type_mask(
        table, parse_ablation_spec(text, 42), layer_col, neuron_col, score_cols,
        activation_col, get_layer_dims(table, layer_col, neuron_col), None,
        "per_layer", 1.0, 0.0,
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


def test_paired_bootstrap_uses_token_weighted_delta():
    result = paired_bootstrap_analysis(
        [10.0, 9.0], [10, 1], [20.0, 10.0], [10, 1], num_bootstrap=200, seed=1,
    )
    assert result["paired_delta_nll"] == pytest.approx(1.0)
    assert result["damaged_frac"] == 1.0


def test_scoring_uses_configured_threshold_and_r_tie_break():
    scores = {
        "layer_0": {
            "q_visual": [0.5], "q_text": [0.0], "q_multimodal": [0.5], "q_unknown": [0.0],
            "r_visual": [0.1], "r_text": [0.0], "r_multimodal": [0.8], "r_unknown": [0.1],
            "dead_mask": [False],
        }
    }
    frame = compute_type_scores(scores, high_conf_threshold=0.6)
    assert frame.loc[0, "dominant_type"] == "multimodal"
    assert bool(frame.loc[0, "dominant_tie"])
    assert frame.loc[0, "confidence_category"] == "mixed_low_confidence"
