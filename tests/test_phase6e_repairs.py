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

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPTS))

from phase6e_repairs import FrozenCandidate, RepairSpec, assemble_conflicts, build_pair_variants  # noqa: E402
from run_phase6d_identity_refinement import identity_sha256  # noqa: E402


def toy_spec() -> RepairSpec:
    identities = {4, 5}
    return RepairSpec(
        edge_id="left_right",
        left="left",
        right="right",
        safe_base="right",
        donor="left",
        candidates=(FrozenCandidate("candidate", 1, 2, identity_sha256(identities)),),
    )


def toy_payload() -> dict:
    identities = [4, 5]
    return {
        "candidate_id": "candidate",
        "layer": 1,
        "final_size": 2,
        "final_identities": identities,
        "final_identity_sha256": identity_sha256(set(identities)),
        "contrast": {"edge_id": "left_right", "base": "right", "donor": "left"},
    }


def test_build_pair_variants_uses_same_donor_increment() -> None:
    dims = {0: 8, 1: 8}
    winners = {
        "right": {0: {0, 1, 2}, 1: {0, 1, 2, 3}},
        "left": {0: {0, 1, 2, 6}, 1: {0, 1, 2, 3, 4, 5, 6, 7}},
    }
    spec = toy_spec()
    conflicts = assemble_conflicts(spec, {"candidate": toy_payload()}, dims)
    variants = build_pair_variants(spec, winners, conflicts, dims, random_seeds=3, seed=7)

    assert [variant["condition"] for variant in variants[:3]] == ["safe_base", "failed_union", "repaired_union"]
    assert variants[2]["kept"][1] == {0, 1, 2, 3, 6, 7}
    for control in variants[3:]:
        removed = variants[1]["kept"][1] - control["kept"][1]
        assert len(removed) == 2
        assert removed <= {4, 5, 6, 7}
        assert variants[1]["kept"][0] == control["kept"][0]


def test_candidate_provenance_drift_is_rejected() -> None:
    payload = toy_payload()
    payload["contrast"]["donor"] = "wrong"
    with pytest.raises(ValueError, match="donor"):
        assemble_conflicts(toy_spec(), {"candidate": payload}, {0: 8, 1: 8})


def test_conflicts_must_lie_in_donor_increment() -> None:
    dims = {0: 8, 1: 8}
    winners = {
        "right": {0: {0, 1}, 1: {0, 1, 4}},
        "left": {0: {0, 1}, 1: {0, 1, 5}},
    }
    spec = toy_spec()
    conflicts = assemble_conflicts(spec, {"candidate": toy_payload()}, dims)
    with pytest.raises(ValueError, match="donor increment"):
        build_pair_variants(spec, winners, conflicts, dims, random_seeds=1, seed=7)
