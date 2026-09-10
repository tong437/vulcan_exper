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


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase6c_subnets import (  # noqa: E402
    complement_pool,
    deletion_masks,
    deterministic_sample,
    dose_counts,
    frequency_core,
    kept_from_deleted,
    layer_counts,
    layerwise_difference,
    layerwise_union,
)


def _winners():
    return {
        "a": {0: {0, 1, 2}, 1: {0, 2}},
        "b": {0: {1, 2, 3}, 1: {0, 1}},
        "c": {0: {1, 2, 4}, 1: {0, 3}},
    }


def test_frequency_cores_and_shell_reconstruction() -> None:
    winners = _winners()
    c3 = frequency_core(winners, 3)
    c2 = frequency_core(winners, 2)
    assert c3 == {0: {1, 2}, 1: {0}}
    assert c2 == c3
    for kept in winners.values():
        shell = layerwise_difference(kept, c3)
        assert layerwise_union(c3, shell) == kept


def test_exact_layerwise_sampling_is_stable() -> None:
    pool = {0: set(range(20)), 1: set(range(10))}
    counts = {0: 5, 1: 3}
    first = deterministic_sample(pool, counts, seed=17)
    second = deterministic_sample(pool, counts, seed=17)
    other = deterministic_sample(pool, counts, seed=18)
    assert first == second
    assert first != other
    assert layer_counts(first) == counts


def test_deleted_roundtrip_and_masks() -> None:
    dims = {0: 5, 1: 4}
    deleted = {"0": [3, 4], "1": [1, 3]}
    kept = kept_from_deleted(deleted, dims)
    masks = deletion_masks(kept, dims)
    assert kept == {0: {0, 1, 2}, 1: {0, 2}}
    assert masks[0].tolist() == [False, False, False, True, True]
    assert masks[1].tolist() == [False, True, False, True]
    assert complement_pool(kept, dims) == {0: {3, 4}, 1: {1, 3}}


def test_dose_counts_preserve_each_layer() -> None:
    values = {0: set(range(9)), 1: set(range(3))}
    assert dose_counts(values, 0.1) == {0: 1, 1: 1}
    assert dose_counts(values, 0.5) == {0: 4, 1: 2}
    assert dose_counts(values, 1.0) == {0: 9, 1: 3}
