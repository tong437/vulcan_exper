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
from evaluate_neupat_forgetting_stress import language_command as stress_language_command  # noqa: E402
from evaluate_neupat_forgetting_stress import vqa_command as stress_vqa_command  # noqa: E402
from evaluate_neupat_sft_language import paired_weighted_bootstrap  # noqa: E402
from prepare_neupat_stress_splits import build_splits  # noqa: E402
from prepare_neupat_text_probe import (  # noqa: E402
    SourceSpec,
    convert_halueval,
    normalize_prompt,
    select_source_examples,
)
from run_neupat_causality import (  # noqa: E402
    add_yes_ratio_deltas,
    build_conditions,
    paired_excess_nll_bootstrap,
    summarize_metric,
)
from run_neupat_forgetting_stress import training_command as stress_training_command  # noqa: E402
from run_neupat_sft_matrix import latest_resume_checkpoint  # noqa: E402
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


def test_protection_set_is_the_default_causal_hypothesis():
    assert build_conditions(2) == [
        "mask:neupat_role_protect",
        "matched_random:mask:neupat_role_protect:seed1",
        "matched_random:mask:neupat_role_protect:seed2",
    ]


def test_paired_excess_bootstrap_uses_aligned_token_weighted_examples():
    def rows(deltas):
        return [
            {"source_index": index, "delta_nll": delta, "token_count": weight}
            for index, (delta, weight) in enumerate(zip(deltas, [1, 3]))
        ]

    payload = {
        "metrics": {
            "mask:neupat_role_protect": {"per_example": rows([3.0, 5.0])},
            "matched_random:mask:neupat_role_protect:seed1": {"per_example": rows([1.0, 1.0])},
            "matched_random:mask:neupat_role_protect:seed2": {"per_example": rows([1.0, 3.0])},
        }
    }
    result = paired_excess_nll_bootstrap(
        payload,
        "language_protection",
        control_seed_count=2,
        bootstrap_samples=100,
        bootstrap_seed=7,
    )
    assert result["available"]
    assert result["excess_nll"] == pytest.approx(2.75)
    assert result["ci_low"] > 0


def test_post_sft_language_bootstrap_is_paired_and_token_weighted():
    base = [
        {"source_index": 0, "nll_sum": 1.0, "token_count": 1},
        {"source_index": 1, "nll_sum": 3.0, "token_count": 3},
    ]
    tuned = [
        {"source_index": 0, "nll_sum": 2.0, "token_count": 1},
        {"source_index": 1, "nll_sum": 9.0, "token_count": 3},
    ]
    result = paired_weighted_bootstrap(tuned, base, samples=100, seed=7)
    assert result["delta_nll"] == pytest.approx(1.75)
    assert result["ci_low"] > 0

    misaligned = [dict(tuned[1]), dict(tuned[0])]
    with pytest.raises(ValueError, match="not aligned"):
        paired_weighted_bootstrap(misaligned, base, samples=100, seed=7)


def test_matrix_resume_uses_highest_complete_checkpoint(tmp_path):
    for step in (50, 100, 150):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").touch()
        (checkpoint / f"global_step{step}").mkdir()
    (tmp_path / "checkpoint-150" / "adapter_model.safetensors").unlink()
    assert latest_resume_checkpoint(tmp_path) == tmp_path / "checkpoint-100"


def test_matrix_resume_accepts_standard_trainer_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint-50"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 50}), encoding="utf-8")
    (checkpoint / "model.safetensors").touch()
    (checkpoint / "optimizer.pt").touch()
    (checkpoint / "scheduler.pt").touch()
    assert latest_resume_checkpoint(tmp_path) == checkpoint


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


def test_stress_split_is_disjoint_by_image_content_and_canonicalized(tmp_path):
    image_root = tmp_path / "dataset"
    image_dir = image_root / "images"
    image_dir.mkdir(parents=True)
    contents = (b"same-a", b"same-a", b"b", b"c", b"same-d", b"same-d")
    rows = []
    for index, content in enumerate(contents):
        image_path = image_dir / f"{index}.png"
        image_path.write_bytes(content)
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": "<image>Question?"},
                    {"role": "assistant", "content": "yes" if index % 2 else "no"},
                ],
                "images": [f"images/{index}.png"],
            }
        )
    input_file = image_root / "input.jsonl"
    input_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    args = SimpleNamespace(
        input_file=str(input_file),
        image_root=str(image_root),
        train_output=str(image_root / "stress_train.jsonl"),
        dev_output=str(image_root / "stress_dev.jsonl"),
        manifest_file=str(image_root / "manifest.json"),
        dataset_info_file=str(image_root / "dataset_info.json"),
        legacy_test_file=str(image_root / "missing.jsonl"),
        dev_fraction=0.5,
        seed=17,
        balance_trials=32,
        force=False,
    )
    manifest = build_splits(args)
    assert manifest["train_dev_image_hash_overlap"] == 0
    assert manifest["train"]["unique_image_hashes"] == 2
    assert manifest["dev"]["unique_image_hashes"] == 2
    assert manifest["train"]["rows"] + manifest["dev"]["rows"] == len(rows)
    dataset_info = json.loads(Path(args.dataset_info_file).read_text())
    assert dataset_info["vqa_rad_stress_train"]["file_name"] == "stress_train.jsonl"
    assert dataset_info["vqa_rad_stress_dev"]["file_name"] == "stress_dev.jsonl"
    for path in (Path(args.train_output), Path(args.dev_output)):
        split_rows = [json.loads(line) for line in path.read_text().splitlines()]
        references_by_content = {}
        for row in split_rows:
            content = (image_root / row["images"][0]).read_bytes()
            references_by_content.setdefault(content, set()).add(row["images"][0])
        assert all(len(references) == 1 for references in references_by_content.values())


def test_stress_commands_are_candidate_scoped_and_do_not_open_lockbox(tmp_path):
    candidate = {"learning_rate": 2e-5, "num_train_epochs": 3.0}
    train = stress_training_command(Path("config.yaml"), tmp_path / "candidate", candidate, seed=11)
    assert "learning_rate=2e-05" in train
    assert "num_train_epochs=3.0" in train
    assert "seed=11" in train

    args = SimpleNamespace(
        language_config="language_dev.yaml",
        score_file="scores.parquet",
        max_language_samples=500,
        bootstrap_samples=1000,
        bootstrap_seed=11,
        training_config="train.yaml",
    )
    language = stress_language_command(args, "model", tmp_path / "language.json")
    vqa = stress_vqa_command(args, "model", tmp_path / "vqa.json")
    assert "language_dev.yaml" in language
    assert "datasets/vqa_rad/stress_dev.jsonl" in vqa
    assert "--allow_excessive_image_repeats" in vqa
    assert all("lockbox" not in item for item in (*language, *vqa))
