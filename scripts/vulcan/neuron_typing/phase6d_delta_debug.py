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

"""Deterministic, non-monotonic subset proposals for Phase 6D identity refinement."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from phase6c_subnets import derive_seed


@dataclass(frozen=True)
class SubsetProposal:
    """A deterministic chunk or complement proposed by delta debugging."""

    kind: str
    partition_index: int
    identities: frozenset[int]

    @property
    def sha256(self) -> str:
        payload = ",".join(str(identity) for identity in sorted(self.identities)).encode()
        return hashlib.sha256(payload).hexdigest()


def deterministic_partitions(
    identities: set[int] | frozenset[int], granularity: int, *, seed: int
) -> list[frozenset[int]]:
    """Partition identities reproducibly without treating numeric adjacency as structure."""
    if not identities:
        raise ValueError("Cannot partition an empty identity set.")
    if not 2 <= granularity <= len(identities):
        raise ValueError(f"granularity must lie in [2, {len(identities)}].")
    ordered = sorted(identities)
    random.Random(derive_seed(seed, len(ordered), granularity)).shuffle(ordered)
    partitions = [frozenset(ordered[index::granularity]) for index in range(granularity)]
    if any(not partition for partition in partitions) or set().union(*partitions) != set(identities):
        raise RuntimeError("Deterministic partition construction failed.")
    return partitions


def delta_debug_proposals(
    identities: set[int] | frozenset[int], granularity: int, *, seed: int
) -> list[SubsetProposal]:
    """Return all unique proper chunks and complements at one granularity."""
    full = frozenset(identities)
    partitions = deterministic_partitions(full, granularity, seed=seed)
    proposals = []
    seen: set[frozenset[int]] = set()
    for kind in ("chunk", "complement"):
        for index, partition in enumerate(partitions):
            subset = partition if kind == "chunk" else full - partition
            if not subset or subset == full or subset in seen:
                continue
            seen.add(subset)
            proposals.append(SubsetProposal(kind, index, subset))
    return proposals


def choose_reduction(passing: list[dict[str, object]]) -> dict[str, object] | None:
    """Choose a passing reduction using the frozen size/margin/hash tie-break."""
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (
            len(row["identities"]),
            -float(row["joint_nll_margin"]),
            str(row["subset_sha256"]),
        ),
    )
