# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small arithmetic tests for sample-overlap diagnostics."""

import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from test_sample_pruning_variability_2026_10 import POPCOUNT, overlap_metrics, sample_pairs


def test_overlap_and_layer_null():
    result = overlap_metrics(3, [2, 2], [2, 2], np.array([4, 4]))
    assert result["jaccard"] == 3 / 5
    assert result["layer_random_reference"] == 2 / 6
    assert result["adjusted_overlap"] == 0.5


def test_identical_masks():
    result = overlap_metrics(4, [2, 2], [2, 2], np.array([4, 4]))
    assert result["jaccard"] == 1
    assert result["adjusted_overlap"] == 1


def test_packed_intersection_and_padding():
    a = np.array([True, False, True, True, False])
    b = np.array([True, True, False, True, False])
    assert POPCOUNT[np.packbits(a) & np.packbits(b)].sum() == (a & b).sum() == 2


def test_cross_group_unique_reproducible_pairs():
    groups = ["a", "a", "b", "c", "d"]
    pairs = sample_pairs(groups, 8)
    assert pairs == sample_pairs(groups, 8)
    assert len(set(pairs)) == 8
    assert all(i < j and groups[i] != groups[j] for i, j in pairs)
