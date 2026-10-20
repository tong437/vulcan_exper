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

"""Deterministic sampling helpers for the September 2026 advisor study."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping


def stable_digest(*parts: object) -> str:
    """Return an unambiguous SHA-256 digest for a sequence of values."""
    payload = b"".join(len(str(part).encode()).to_bytes(8, "big") + str(part).encode() for part in parts)
    return hashlib.sha256(payload).hexdigest()


def largest_remainder_quotas(capacities: Mapping[str, int], total: int) -> dict[str, int]:
    """Allocate an integer proportional quota with deterministic tie-breaking.

    The implementation also handles capacity constraints, although ordinary proportional allocation cannot exceed a
    stratum's capacity when ``total <= sum(capacities.values())``.
    """
    clean = {str(key): int(value) for key, value in capacities.items() if int(value) > 0}
    capacity_total = sum(clean.values())
    if total < 0 or total > capacity_total:
        raise ValueError(f"Requested {total} units from capacity {capacity_total}.")
    if total == 0:
        return dict.fromkeys(clean, 0)

    ideals = {key: total * value / capacity_total for key, value in clean.items()}
    quotas = {key: min(value, math.floor(ideals[key])) for key, value in clean.items()}
    remaining = total - sum(quotas.values())
    order = sorted(clean, key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] < clean[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("Unable to distribute the requested quota within stratum capacities.")
    return quotas


def deterministic_take(unit_ids: list[str], count: int, *, seed: int, namespace: str) -> list[str]:
    """Select units without replacement by a stable hash rank."""
    if count < 0 or count > len(unit_ids):
        raise ValueError(f"Cannot select {count} unique units from {len(unit_ids)} candidates.")
    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Unit IDs must be unique.")
    return sorted(unit_ids, key=lambda unit_id: (stable_digest(seed, namespace, unit_id), unit_id))[:count]
