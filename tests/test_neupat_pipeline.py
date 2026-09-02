from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "scripts" / "vulcan" / "neuron_typing"
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SRC_DIR))

from analyze_neupat_overlap import build_overlap_outputs  # noqa: E402
from analyze_neupat_stability import build_stability_report  # noqa: E402
from build_neupat_joint_candidate import build_joint_candidate  # noqa: E402
from compare_neupat_replications import compare_tables  # noqa: E402
from prepare_neupat_text_probe import (  # noqa: E402
    SourceSpec,
    convert_halueval,
    normalize_prompt,
    select_source_examples,
)
from run_neupat_causality import add_yes_ratio_deltas, summarize_metric  # noqa: E402
from run_phase2_ablation import (  # noqa: E402
    build_type_mask,
    get_layer_dims,
    infer_score_columns,
    parse_ablation_spec,
    summarize_cutoffs,
    verify_matched_random_controls,
)

from llamafactory.train.vulcan.neupat import (  # noqa: E402
    NeuPATController,
    allocate_neupat_roles,
    select_importance_mass,
    validate_neupat_artifact,
)


class TinyMLP(torch.nn.Module):
    def __init__(self, hidden_size: int = 3, intermediate_size: int = 4):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([TinyLayer()]))
        self.add_module("decoder_layer", self.model.layers[0])


def _artifact() -> dict:
    return {
        "artifact_version": 1,
        "method": "neupat",
        "layers": {
            "0": {
                "intermediate_size": 4,
                "language": [0],
                "multimodal": [1],
                "shared": [2],
                "reserve": [3],
            }
        },
    }


def test_importance_mass_and_roles_are_deterministic_exhaustive():
    scores = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert select_importance_mass(scores, 0.7).tolist() == [0, 1]
    roles = allocate_neupat_roles(
        torch.tensor([5.0, 1.0, 4.0, 0.0]),
        torch.tensor([1.0, 5.0, 4.0, 0.0]),
        tau_text=0.6,
        tau_vision=0.6,
    )
    assert roles["language"].tolist() == [0]
    assert roles["multimodal"].tolist() == [1]
    assert roles["shared"].tolist() == [2]
    assert roles["reserve"].tolist() == [3]


def test_artifact_validation_rejects_overlap_and_missing_roles():
    assert validate_neupat_artifact(_artifact(), expected_layer_dims={0: 4})[0]["shared"] == [2]
    invalid = _artifact()
    invalid["layers"]["0"]["shared"] = [0, 2]
    with pytest.raises(ValueError, match="overlap"):
        validate_neupat_artifact(invalid)


def test_controller_masks_language_gradients_and_regularizes_shared(tmp_path):
    artifact_path = tmp_path / "roles.json"
    artifact_path.write_text(json.dumps(_artifact()))
    model = TinyModel()
    controller = NeuPATController(model, artifact_path, lambda_in=0.1, lambda_out=0.1)
    mlp = model.model.layers[0].mlp
    loss = mlp.gate_proj.weight.sum() + mlp.up_proj.weight.sum() + mlp.down_proj.weight.sum()
    loss.backward()
    assert torch.count_nonzero(mlp.gate_proj.weight.grad[0]) == 0
    assert torch.count_nonzero(mlp.up_proj.weight.grad[0]) == 0
    assert torch.count_nonzero(mlp.down_proj.weight.grad[:, 0]) == 0
    assert torch.count_nonzero(mlp.gate_proj.weight.grad[1:]) > 0
    assert torch.count_nonzero(mlp.down_proj.weight.grad[:, 1:]) > 0
    initial, _ = controller.regularization()
    assert initial.item() >= 0.0
    assert initial.item() == pytest.approx(0.0, abs=1e-7)
    with torch.no_grad():
        mlp.gate_proj.weight[2].add_(0.1)
        mlp.down_proj.weight[0, 2].add_(0.2)
    changed, log = controller.regularization()
    assert changed.item() > 0
    assert log["neupat_input_raw"] > 0
    assert log["neupat_output_raw"] > 0
    controller.remove_hooks()


def _q_table(width: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": 0,
                "neuron_idx": neuron,
                "q_visual": neuron / width,
                "q_text": (width - neuron) / width,
                "q_multimodal": (width - neuron) / width,
                "q_unknown": neuron / width,
                "r_visual": neuron / width,
                "r_text": (width - neuron) / width,
                "r_multimodal": (width - neuron) / width,
                "r_unknown": neuron / width,
            }
            for neuron in range(width)
        ]
    )


def _neupat_table(width: int = 10) -> pd.DataFrame:
    roles = ["language", "multimodal", "shared", "reserve"]
    rows = []
    for neuron in range(width):
        role = roles[neuron % len(roles)]
        rows.append(
            {
                "layer": 0,
                "neuron_idx": neuron,
                "neupat_text_importance": float(width - neuron),
                "neupat_vision_importance": float(neuron + 1),
                "neupat_overall_importance": float(width),
                "neupat_role": role,
                **{f"neupat_{name}": role == name for name in roles},
            }
        )
    return pd.DataFrame(rows)


def test_overlap_analysis_builds_protection_and_joint_eligibility():
    mapping = pd.DataFrame(
        [{"layer": 0, "neuron_idx": neuron, "mapping_signal": float(neuron)} for neuron in range(10)]
    )
    combined, report = build_overlap_outputs(
        _neupat_table(),
        _q_table(),
        mapping_scores=mapping,
        mapping_score_column="mapping_signal",
        mapping_protect_ratio=0.1,
        qband_start=0.1,
        qband_end=0.4,
    )
    assert int(combined["mapping_protect"].sum()) == 1
    assert bool((combined["joint_protect"] >= combined["neupat_role_protect"]).all())
    assert report["global"]["qband_delete_candidates"] == 3


def test_tau_stability_is_exact_at_reference_and_exhaustive():
    table = _neupat_table()
    table["neupat_vision_importance"] = table["neupat_vision_importance"].astype(float)
    report = build_stability_report(table, [0.5, 0.8], 0.8)
    reference = report["comparisons"]["0.8"]
    assert reference["exact_role_agreement"] == pytest.approx(1.0)
    assert all(values["jaccard"] == pytest.approx(1.0) for values in reference["role_overlap"].values())
    assert sum(report["comparisons"]["0.5"]["role_counts"].values()) == len(table)


def test_replication_comparison_reports_role_and_protection_overlap():
    reference = _neupat_table()
    comparison = reference.copy()
    comparison.loc[0, ["neupat_language", "neupat_multimodal"]] = [False, True]
    comparison.loc[0, "neupat_role"] = "multimodal"
    report = compare_tables(reference, comparison)
    assert report["exact_role_agreement"] == pytest.approx(0.9)
    assert report["language_protection_overlap"]["reference_count"] == 5
    assert report["language_protection_overlap"]["intersection"] == 4


def test_boolean_role_masks_and_matched_random_controls_have_exact_counts():
    table = _q_table().merge(_neupat_table(), on=["layer", "neuron_idx"])
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    role_spec = parse_ablation_spec("mask:neupat_language", 42)
    random_spec = parse_ablation_spec("matched_random:mask:neupat_language:seed7", 42)
    masks = {}
    for spec in (role_spec, random_spec):
        masks[spec.result_name] = build_type_mask(
            table,
            spec,
            layer_col,
            neuron_col,
            score_cols,
            activation_col,
            layer_dims,
            None,
            "per_layer",
            1.0,
            0.0,
        )
    assert masks[role_spec.result_name][0].nonzero().flatten().tolist() == [0, 4, 8]
    count_verification, _ = verify_matched_random_controls([role_spec, random_spec], masks)
    assert count_verification[random_spec.result_name]["counts_match"]


def test_boolean_role_mask_has_no_ranked_cutoff_summary():
    table = _q_table().merge(_neupat_table(), on=["layer", "neuron_idx"])
    layer_col, neuron_col, score_cols, activation_col = infer_score_columns(table, None)
    layer_dims = get_layer_dims(table, layer_col, neuron_col)
    spec = parse_ablation_spec("mask:neupat_language", 42)
    masks = build_type_mask(
        table,
        spec,
        layer_col,
        neuron_col,
        score_cols,
        activation_col,
        layer_dims,
        None,
        "per_layer",
        1.0,
        0.0,
    )
    assert summarize_cutoffs(table, spec, masks, layer_col, neuron_col, score_cols, activation_col, 1.0, 0.0) == {}


def test_joint_candidate_never_selects_protected_and_backfills_exact_budget():
    table = _q_table()
    table["joint_protect"] = False
    table.loc[table["neuron_idx"] == 2, "joint_protect"] = True
    masks, metadata = build_joint_candidate(
        table,
        band_start=0.1,
        band_end=0.4,
        budget_per_layer=3,
        allow_backfill=True,
    )
    selected = masks[0].nonzero().flatten().tolist()
    assert 2 not in selected
    assert len(selected) == 3
    assert metadata["per_layer"]["0"]["backfilled"] == 1
    assert not metadata["structural_pruning_allowed"]


def test_joint_candidate_uses_phase2_secondary_tie_break():
    table = _q_table()
    table["q_multimodal"] = 1.0
    table["r_multimodal"] = table["neuron_idx"].astype(float)
    table["joint_protect"] = table["neuron_idx"] == 7
    table["qband_delete_candidate"] = table["neuron_idx"].isin([8, 7, 6])
    masks, _ = build_joint_candidate(
        table,
        band_start=0.1,
        band_end=0.4,
        budget_per_layer=None,
        allow_backfill=False,
    )
    assert masks[0].nonzero().flatten().tolist() == [6, 8]


def test_causality_summary_compares_exact_count_controls_and_derives_yes_shift():
    payload = {
        "metrics": {
            "none": {"yes_ratio": 0.4},
            "mask:neupat_language": {"delta_nll": 0.3, "yes_ratio": 0.5},
            "matched_random:mask:neupat_language:seed1": {"delta_nll": 0.1, "yes_ratio": 0.35},
            "matched_random:mask:neupat_language:seed2": {"delta_nll": 0.2, "yes_ratio": 0.45},
        }
    }
    add_yes_ratio_deltas(payload)
    nll = summarize_metric(payload, "language", "delta_nll", control_seed_count=2)
    yes_ratio = summarize_metric(payload, "language", "delta_yes_ratio", control_seed_count=2)
    assert nll["random_mean"] == pytest.approx(0.15)
    assert nll["excess_vs_random"] == pytest.approx(0.15)
    assert yes_ratio["observed"] == pytest.approx(0.1)
    assert yes_ratio["random_mean"] == pytest.approx(0.0)


def test_probe_source_selection_filters_hallucinations_and_deduplicates():
    source = SourceSpec("halueval", "repo", "general", "data", convert_halueval)
    raw_rows = [
        {
            "row_idx": 0,
            "row": {
                "user_query": "Same prompt",
                "chatgpt_response": "Correct answer",
                "hallucination": "no",
            },
        },
        {
            "row_idx": 1,
            "row": {
                "user_query": "Same   prompt",
                "chatgpt_response": "Duplicate answer",
                "hallucination": "no",
            },
        },
        {
            "row_idx": 2,
            "row": {
                "user_query": "Bad response",
                "chatgpt_response": "Hallucinated answer",
                "hallucination": "yes",
            },
        },
        {
            "row_idx": 3,
            "row": {
                "user_query": "Another prompt",
                "chatgpt_response": "Another correct answer",
                "hallucination": "no",
            },
        },
    ]
    selected, counters = select_source_examples(
        source,
        raw_rows,
        target=2,
        seed=2026,
        max_chars=1000,
        global_prompt_hashes=set(),
    )
    assert len(selected) == 2
    assert len({row["prompt_sha256"] for row in selected}) == 2
    assert convert_halueval(raw_rows[2]["row"]) is None
    assert normalize_prompt({"instruction": "Same prompt", "input": ""}) == normalize_prompt(
        {"instruction": "Same   prompt", "input": ""}
    )
    assert set(counters) == {"converter_rejected", "empty_rejected", "length_rejected", "duplicate_rejected"}
