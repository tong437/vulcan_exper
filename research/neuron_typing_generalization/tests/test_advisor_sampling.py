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

import pytest
from neuron_typing_generalization.advisor_sampling import (
    deterministic_take,
    largest_remainder_quotas,
    stable_digest,
)


def test_largest_remainder_quotas_is_exact_and_deterministic() -> None:
    assert largest_remainder_quotas({"b": 1, "a": 1, "c": 2}, 3) == {"b": 1, "a": 1, "c": 1}
    assert largest_remainder_quotas({"human": 625, "augmented": 184}, 500) == {
        "human": 386,
        "augmented": 114,
    }


def test_largest_remainder_rejects_excess_request() -> None:
    with pytest.raises(ValueError, match="capacity"):
        largest_remainder_quotas({"a": 2}, 3)


def test_deterministic_take_is_order_independent() -> None:
    forward = deterministic_take(["a", "b", "c", "d"], 2, seed=42, namespace="test")
    reverse = deterministic_take(["d", "c", "b", "a"], 2, seed=42, namespace="test")
    assert forward == reverse
    assert len(forward) == len(set(forward)) == 2


def test_stable_digest_separates_part_boundaries() -> None:
    assert stable_digest("ab", "c") != stable_digest("a", "bc")
