from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from run_phase26_completeness import (  # noqa: E402
    QBAND_CONDITION,
    _control_comparison,
    build_candidate_registry,
    build_experiment_plan,
    parse_named_files,
    summarize_screen,
)


def _score_table(width: int = 20) -> pd.DataFrame:
    rows = []
    for layer in range(2):
        for neuron in range(width):
            rows.append(
                {
                    "layer": layer,
                    "neuron_idx": neuron,
                    "q_visual": (width - neuron) / width,
                    "q_text": neuron / width,
                    "q_multimodal": (width - neuron) / width,
                    "q_unknown": neuron % 3 / 3,
                    "r_visual": 0.0,
                    "r_text": 0.0,
                    "r_multimodal": neuron / width,
                    "r_unknown": 1.0 - neuron / width,
                }
            )
    return pd.DataFrame(rows)


def test_candidate_registry_uses_exact_budget_and_explicit_q_unknown():
    candidates, budget = build_candidate_registry(
        intermediate_size=3584,
        tail_starts=[0.20, 0.25, 0.85],
        window_ratio=0.15,
    )
    assert budget == 537
    assert candidates["q_multimodal_p2000_p3500"]["condition"] == "rank_window:multimodal:717:537"
    assert candidates["q_multimodal_p8500_p10000"]["rank_start"] + budget == 3584
    assert candidates["q_unknown_top15"]["condition"].startswith("matched_score:q_unknown:highest:")
    assert candidates["q_multimodal_p05_p20"]["condition"] == QBAND_CONDITION


def test_experiment_plan_verifies_same_count_for_every_candidate(tmp_path):
    score_file = tmp_path / "scores.parquet"
    _score_table().to_parquet(score_file, index=False)
    args = SimpleNamespace(
        score_file=str(score_file),
        tail_starts="0.20,0.25,0.85",
        window_ratio=0.15,
        seed=7,
        max_caption_delta_nll=0.05,
        max_text_delta_nll=0.05,
        max_pope_accuracy_drop=0.01,
        max_yes_ratio_shift=0.05,
    )
    plan = build_experiment_plan(args)
    assert plan["per_layer_budget"] == 3
    assert plan["actual_budget_ratio"] == pytest.approx(0.15)
    for summary in plan["mask_summaries"].values():
        assert set(summary["per_layer_selected"].values()) == {3}


def test_parse_named_files_requires_all_three_pope_splits():
    with pytest.raises(ValueError, match="missing"):
        parse_named_files(["random=r.json", "popular=p.json"])
    result = parse_named_files(["random=r.json", "popular=p.json", "adversarial=a.json"])
    assert result["adversarial"] == "a.json"


def test_control_comparison_uses_correct_safety_direction():
    nll = _control_comparison(-0.2, [-0.1, 0.0, 0.1], lower_is_safer=True)
    accuracy = _control_comparison(-0.01, [-0.05, -0.04, -0.03], lower_is_safer=False)
    assert nll["candidate_safer_than_control_median"]
    assert nll["empirical_p_candidate_safer"] == pytest.approx(0.25)
    assert accuracy["candidate_safer_than_control_median"]
    assert accuracy["empirical_p_candidate_safer"] == pytest.approx(0.25)


def test_screen_summary_requires_caption_text_pope_and_yes_ratio(tmp_path):
    condition = QBAND_CONDITION
    plan = {
        "plan_sha256": "test",
        "candidates": {
            "safe": {"condition": condition, "family": "q_multimodal"},
            "unsafe_text": {"condition": "multimodal:0.15", "family": "q_multimodal"},
        },
    }
    screen_dir = tmp_path / "screen"
    screen_dir.mkdir()
    caption = {
        "metrics": {
            "none": {"nll": 1.0},
            condition: {"nll": 1.01, "delta_nll": 0.01},
            "multimodal:0.15": {"nll": 1.01, "delta_nll": 0.01},
        }
    }
    text_only = {
        "metrics": {
            "none": {"nll": 2.0},
            condition: {"nll": 2.01, "delta_nll": 0.01},
            "multimodal:0.15": {"nll": 2.2, "delta_nll": 0.2},
        }
    }
    pope = {
        "metrics": {
            "none": {"accuracy": 0.9, "yes_ratio": 0.5},
            condition: {"accuracy": 0.895, "delta_accuracy": -0.005, "yes_ratio": 0.51},
            "multimodal:0.15": {"accuracy": 0.9, "delta_accuracy": 0.0, "yes_ratio": 0.5},
        }
    }
    import json

    (screen_dir / "caption.json").write_text(json.dumps(caption))
    (screen_dir / "text_only.json").write_text(json.dumps(text_only))
    (screen_dir / "pope_random.json").write_text(json.dumps(pope))
    args = SimpleNamespace(
        max_caption_delta_nll=0.05,
        max_text_delta_nll=0.05,
        max_pope_accuracy_drop=0.01,
        max_yes_ratio_shift=0.05,
        max_finalists=3,
    )
    result = summarize_screen(plan, tmp_path, args)
    assert result["finalists"] == ["safe"]
    assert not result["candidates"]["unsafe_text"]["gates"]["text_only"]
