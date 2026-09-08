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
import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "vulcan" / "neuron_typing"
sys.path.insert(0, str(SCRIPT_DIR))

from scan_phase5_core_extension_physical import (  # noqa: E402
    extend_core_mask,
    parse_positive_ints,
    ranked_extension_neurons,
    refresh_summary,
)


def test_ranked_extension_neurons_excludes_core_and_is_stable():
    scores = torch.tensor([2.0, 1.0, 1.0, 0.5, 3.0])
    base_mask = torch.tensor([False, True, False, True, False])
    assert ranked_extension_neurons(scores, base_mask, [1, 2, 3]) == {1: 2, 2: 0, 3: 4}


def test_extend_core_mask_preserves_core_and_adds_exactly_one():
    base = {0: torch.tensor([True, False]), 1: torch.tensor([False, False])}
    extended = extend_core_mask(base, 1, 0)
    assert extended[0].tolist() == [True, False]
    assert extended[1].tolist() == [True, False]
    assert base[1].tolist() == [False, False]
    with pytest.raises(ValueError, match="already deleted"):
        extend_core_mask(base, 0, 0)


def test_positive_int_parser_deduplicates_and_rejects_zero():
    assert parse_positive_ints("1,2,1") == [1, 2]
    with pytest.raises(ValueError, match="positive"):
        parse_positive_ints("0,1")


def test_refresh_summary_prefers_lowest_kl_strict_candidate():
    result = {
        "candidates": {
            "failed": {
                "added_layer": 0,
                "added_neuron": 1,
                "candidate_rank": 1,
                "generation": {"exact_match": False, "common_prefix_tokens": 400},
                "cached_fidelity": None,
                "strict_feasible": False,
            },
            "strict": {
                "added_layer": 2,
                "added_neuron": 3,
                "candidate_rank": 2,
                "generation": {"exact_match": True, "common_prefix_tokens": 512},
                "cached_fidelity": {"mean_kl": 0.0008},
                "strict_feasible": True,
            },
        }
    }
    refresh_summary(result)
    assert result["strict_candidates"] == ["strict"]
    assert result["best_strict_candidate"]["candidate"] == "strict"
    assert result["best_prefix"]["candidate"] == "strict"
