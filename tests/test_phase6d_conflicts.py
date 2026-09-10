# Copyright 2025 the LlamaFactory team.
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

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_phase6d_conflicts import analyze  # noqa: E402
from phase6d_conflicts import ConflictContrast, build_localization_variants, one_layer  # noqa: E402
from phase6d_delta_debug import (  # noqa: E402
    choose_reduction,
    delta_debug_proposals,
    deterministic_partitions,
)


def _contrast() -> ConflictContrast:
    return ConflictContrast("a_b__base_a__target_a", "a_b", "a", "b", "a", "b", "a")


def _variants(random_seeds: int = 2):
    winners = {
        "a": {0: {0, 1, 2}, 1: {0, 1, 2}},
        "b": {0: {0, 3, 4}, 1: {0, 3}},
    }
    dims = {0: 8, 1: 8}
    return build_localization_variants(
        winners,
        dims,
        random_seeds=random_seeds,
        seed=41,
        contrasts=(_contrast(),),
    )


def test_one_layer_preserves_only_selected_layer() -> None:
    values = {0: {1, 2}, 1: {3}, 2: {4, 5}}
    assert one_layer(values, 1) == {0: set(), 1: {3}, 2: set()}


def test_localization_variants_have_exact_endpoints_and_treatments() -> None:
    variants = _variants()
    assert len(variants) == 2 + 2 * (2 + 2 * 2)
    by_id = {variant["variant_id"]: variant for variant in variants}
    prefix = "localize__a_b__base_a__target_a"
    assert by_id["endpoint__a_b__base_a__target_a__safe_base"]["kept"] == {
        0: {0, 1, 2},
        1: {0, 1, 2},
    }
    assert by_id["endpoint__a_b__base_a__target_a__failed_union"]["kept"] == {
        0: {0, 1, 2, 3, 4},
        1: {0, 1, 2, 3},
    }
    assert by_id[f"{prefix}__layer_00__add__treatment"]["kept"] == {
        0: {0, 1, 2, 3, 4},
        1: {0, 1, 2},
    }
    assert by_id[f"{prefix}__layer_00__loo__treatment"]["kept"] == {
        0: {0, 1, 2},
        1: {0, 1, 2, 3},
    }


def test_random_controls_match_layer_and_count_without_donor_contamination() -> None:
    variants = _variants()
    base = {0: {0, 1, 2}, 1: {0, 1, 2}}
    union = {0: {0, 1, 2, 3, 4}, 1: {0, 1, 2, 3}}
    prefix = "localize__a_b__base_a__target_a"
    for layer, count in ((0, 2), (1, 1)):
        for replicate in range(2):
            add = next(
                variant
                for variant in variants
                if variant["variant_id"] == f"{prefix}__layer_{layer:02d}__add__random_r{replicate:02d}"
            )["kept"]
            loo = next(
                variant
                for variant in variants
                if variant["variant_id"] == f"{prefix}__layer_{layer:02d}__loo__random_r{replicate:02d}"
            )["kept"]
            assert len(add[layer] - base[layer]) == count
            assert not (add[layer] - base[layer]) & union[layer]
            assert len(union[layer] - loo[layer]) == count
            assert union[layer] - loo[layer] <= base[layer]
            other_layer = 1 - layer
            assert add[other_layer] == base[other_layer]
            assert loo[other_layer] == union[other_layer]


def test_delta_debug_proposes_chunks_and_complements_deterministically() -> None:
    identities = set(range(11))
    first = deterministic_partitions(identities, 3, seed=7)
    second = deterministic_partitions(identities, 3, seed=7)
    assert first == second
    assert set().union(*first) == identities
    assert max(map(len, first)) - min(map(len, first)) <= 1
    proposals = delta_debug_proposals(identities, 3, seed=7)
    assert {proposal.kind for proposal in proposals} == {"chunk", "complement"}
    assert all(0 < len(proposal.identities) < len(identities) for proposal in proposals)
    assert len({proposal.identities for proposal in proposals}) == len(proposals)


def test_choose_reduction_prefers_size_then_margin_then_hash() -> None:
    rows = [
        {"identities": {1, 2, 3}, "joint_nll_margin": 10.0, "subset_sha256": "a"},
        {"identities": {1, 2}, "joint_nll_margin": 1.0, "subset_sha256": "b"},
        {"identities": {3, 4}, "joint_nll_margin": 2.0, "subset_sha256": "c"},
        {"identities": {5, 6}, "joint_nll_margin": 2.0, "subset_sha256": "a"},
    ]
    assert choose_reduction(rows) == rows[3]


def test_analysis_shortlists_only_bidirectional_exact_layers(tmp_path: Path) -> None:
    contrast = {
        "contrast_id": "a_b__base_a__target_a",
        "edge_id": "a_b",
        "left": "a",
        "right": "b",
        "base": "a",
        "donor": "b",
        "target": "a",
    }
    specs = [
        ("base", "endpoint", {"endpoint": "safe_base", "layer": None}, True, 1.0),
        ("union", "endpoint", {"endpoint": "failed_union", "layer": None}, False, 2.0),
        (
            "l0_add",
            "layer_localization",
            {"endpoint": None, "layer": 0, "intervention": "add_one_layer", "control": False, "increment_count": 2},
            False,
            2.5,
        ),
        (
            "l0_loo",
            "layer_localization",
            {
                "endpoint": None,
                "layer": 0,
                "intervention": "leave_one_layer_out",
                "control": False,
                "increment_count": 2,
            },
            True,
            0.8,
        ),
        (
            "l0_add_r",
            "layer_localization",
            {"endpoint": None, "layer": 0, "intervention": "add_one_layer", "control": True, "increment_count": 2},
            True,
            1.1,
        ),
        (
            "l0_loo_r",
            "layer_localization",
            {
                "endpoint": None,
                "layer": 0,
                "intervention": "leave_one_layer_out",
                "control": True,
                "increment_count": 2,
            },
            False,
            2.1,
        ),
        (
            "l1_add",
            "layer_localization",
            {"endpoint": None, "layer": 1, "intervention": "add_one_layer", "control": False, "increment_count": 1},
            True,
            1.2,
        ),
        (
            "l1_loo",
            "layer_localization",
            {
                "endpoint": None,
                "layer": 1,
                "intervention": "leave_one_layer_out",
                "control": False,
                "increment_count": 1,
            },
            True,
            1.4,
        ),
        (
            "l1_add_r",
            "layer_localization",
            {"endpoint": None, "layer": 1, "intervention": "add_one_layer", "control": True, "increment_count": 1},
            True,
            1.1,
        ),
        (
            "l1_loo_r",
            "layer_localization",
            {
                "endpoint": None,
                "layer": 1,
                "intervention": "leave_one_layer_out",
                "control": True,
                "increment_count": 1,
            },
            False,
            2.1,
        ),
    ]
    rows = []
    variants = []
    for variant_id, family, metadata, passed, nll in specs:
        full_metadata = {"contrast": contrast, **metadata}
        rows.append(
            {
                "evaluation_id": variant_id,
                "variant_id": variant_id,
                "family": family,
                "target_sample_id": "a",
                "kept_sha256": variant_id,
                "variant_metadata": full_metadata,
                "automatic_semantic_pass": passed,
                "gold_proxy": {"mean_nll": nll},
                "generation": {
                    "final_caption": variant_id,
                    "semantic": {"automatic_pass": passed},
                },
            }
        )
        variants.append(
            {
                "variant_id": variant_id,
                "target": "a",
                "kept_sha256": variant_id,
            }
        )
    evaluations_path = tmp_path / "evaluations.jsonl"
    evaluations_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "complete": True,
                "expected_evaluations": len(rows),
                "evaluations_file": str(evaluations_path),
                "variants": variants,
                "contrasts": [contrast],
            }
        ),
        encoding="utf-8",
    )
    output = analyze(result_path)
    assert output["integrity"]["all_endpoints_reproduced"]
    assert output["summary"]["bidirectional_exact_slots"] == 1
    assert output["summary"]["identity_specific_slots"] == 1
    assert output["identity_refinement_shortlist"][0]["layer"] == 0
