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

"""Frozen repaired-union constructors for Phase 6E."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from phase6c_subnets import (
    LayerSets,
    derive_seed,
    deterministic_sample,
    kept_sha256,
    layer_counts,
    layerwise_difference,
    layerwise_union,
    summarize_kept,
    validate_kept,
)
from run_phase6d_identity_refinement import identity_sha256


@dataclass(frozen=True)
class FrozenCandidate:
    candidate_id: str
    layer: int
    size: int
    sha256: str


@dataclass(frozen=True)
class RepairSpec:
    edge_id: str
    left: str
    right: str
    safe_base: str
    donor: str
    candidates: tuple[FrozenCandidate, ...]

    @property
    def targets(self) -> tuple[str, str]:
        return self.left, self.right


REPAIR_SPECS = (
    RepairSpec(
        edge_id="bicycle_toilet",
        left="bicycle_clock",
        right="toilet_spatial",
        safe_base="toilet_spatial",
        donor="bicycle_clock",
        candidates=(
            FrozenCandidate(
                candidate_id="bicycle_toilet__base_toilet__target_toilet__layer_20",
                layer=20,
                size=135,
                sha256="2c1ae5525bc111399c2e441a9b9fc03f93b68a0e7edad770d8824b2911f194cc",
            ),
        ),
    ),
    RepairSpec(
        edge_id="bicycle_violin",
        left="bicycle_clock",
        right="violin_kitchen",
        safe_base="bicycle_clock",
        donor="violin_kitchen",
        candidates=(
            FrozenCandidate(
                candidate_id="bicycle_violin__base_bicycle__target_bicycle__layer_20",
                layer=20,
                size=337,
                sha256="a2a688136bb1cc8721ba7ad3b8d2709da50640910fd0e67223940bfca5b5b800",
            ),
            FrozenCandidate(
                candidate_id="bicycle_violin__base_bicycle__target_bicycle__layer_00",
                layer=0,
                size=325,
                sha256="fb4541d193a2340e7a1875099f3ba813406011e776ad46b7975c8bb41e97cd21",
            ),
        ),
    ),
)


def validate_candidate_payload(candidate: FrozenCandidate, payload: dict[str, Any], spec: RepairSpec) -> set[int]:
    identities = set(payload["final_identities"])
    contrast = payload["contrast"]
    checks = {
        "candidate_id": payload["candidate_id"] == candidate.candidate_id,
        "layer": int(payload["layer"]) == candidate.layer,
        "size": len(identities) == payload["final_size"] == candidate.size,
        "sha256": identity_sha256(identities) == payload["final_identity_sha256"] == candidate.sha256,
        "edge": contrast["edge_id"] == spec.edge_id,
        "base": contrast["base"] == spec.safe_base,
        "donor": contrast["donor"] == spec.donor,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Frozen candidate {candidate.candidate_id} failed checks: {failed}.")
    return identities


def assemble_conflicts(
    spec: RepairSpec,
    payloads: dict[str, dict[str, Any]],
    layer_dims: dict[int, int],
) -> LayerSets:
    conflicts = {layer: set() for layer in layer_dims}
    for candidate in spec.candidates:
        if candidate.candidate_id not in payloads:
            raise ValueError(f"Missing frozen candidate {candidate.candidate_id}.")
        identities = validate_candidate_payload(candidate, payloads[candidate.candidate_id], spec)
        if conflicts[candidate.layer] & identities:
            raise ValueError(f"Frozen candidate {candidate.candidate_id} overlaps another candidate in its layer.")
        conflicts[candidate.layer].update(identities)
    validate_kept(conflicts, layer_dims, allow_empty_layer=True)
    return conflicts


def build_pair_variants(
    spec: RepairSpec,
    winners: dict[str, LayerSets],
    conflicts: LayerSets,
    layer_dims: dict[int, int],
    *,
    random_seeds: int,
    seed: int,
) -> list[dict[str, Any]]:
    if random_seeds < 1:
        raise ValueError("random_seeds must be positive.")
    safe_base = winners[spec.safe_base]
    donor = winners[spec.donor]
    union = layerwise_union(safe_base, donor)
    donor_increment = layerwise_difference(donor, safe_base)
    if any(not conflicts[layer] <= donor_increment[layer] for layer in layer_dims):
        raise ValueError(f"Repair identities for {spec.edge_id} are not contained in the frozen donor increment.")
    repaired = layerwise_difference(union, conflicts)
    counts = layer_counts(conflicts)

    def variant(condition: str, kept: LayerSets, **metadata: Any) -> dict[str, Any]:
        validate_kept(kept, layer_dims)
        return {
            "variant_id": f"{spec.edge_id}__{condition}",
            "edge_id": spec.edge_id,
            "targets": list(spec.targets),
            "condition": condition,
            "kept": kept,
            "kept_sha256": kept_sha256(kept),
            "mask_summary": summarize_kept(kept, layer_dims),
            "metadata": {
                "safe_base": spec.safe_base,
                "donor": spec.donor,
                "repair_counts_by_layer": {str(layer): count for layer, count in counts.items() if count},
                **metadata,
            },
        }

    variants = [
        variant("safe_base", safe_base),
        variant("failed_union", union),
        variant(
            "repaired_union",
            repaired,
            conflict_candidate_ids=[candidate.candidate_id for candidate in spec.candidates],
            conflict_identity_sha256_by_layer={
                str(layer): identity_sha256(identities) for layer, identities in conflicts.items() if identities
            },
        ),
    ]
    for replicate in range(random_seeds):
        control_seed = derive_seed(seed, spec.edge_id, "random_repair", replicate)
        random_removed = deterministic_sample(donor_increment, counts, seed=control_seed)
        overlap = {
            str(layer): len(random_removed[layer] & conflicts[layer])
            for layer in layer_dims
            if counts[layer]
        }
        variants.append(
            variant(
                f"random_repair_r{replicate:02d}",
                layerwise_difference(union, random_removed),
                control=True,
                replicate=replicate,
                control_seed=control_seed,
                control_pool="same_donor_increment_including_frozen_conflicts",
                overlap_with_conflicts_by_layer=overlap,
                removed_identity_sha256_by_layer={
                    str(layer): identity_sha256(identities)
                    for layer, identities in random_removed.items()
                    if identities
                },
            )
        )
    return variants
