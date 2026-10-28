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

"""Exact global cumulative costs and edge cases."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PIPELINES = Path(__file__).resolve().parents[1] / "pipelines"
sys.path.insert(0, str(PIPELINES))
spec = importlib.util.spec_from_file_location(
    "global_report", PIPELINES / "report_global_cumulative_importance_2026_10.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_variable_width_and_padding():
    q = np.zeros((2, 3, 3))
    q[0, 0] = [1, 3, np.nan]
    q[1, 0] = [0, 1, 1]
    # Layer masses are each 1/2 even with unequal widths; padded NaN is ignored.
    c = report.global_cumulative(q, [2, 3])
    np.testing.assert_allclose(c, [0, 0, 0.125, 0.375, 0.625, 1])
    assert report.budget_count(c, 0) == 1
    assert report.budget_count(c, 0.375) == 3
    assert report.budget_count(c, 0.374) == 2


def test_uniform_cost_diagonal():
    c = report.global_cumulative(np.ones((2, 3, 4)), [4, 4])
    np.testing.assert_allclose(c, np.arange(9) / 8)


@pytest.mark.parametrize("bad", [-1, np.nan, np.inf])
def test_invalid_scores(bad):
    q = np.ones((1, 3, 2))
    q[0, 0, 0] = bad
    with pytest.raises(ValueError):
        report.global_cumulative(q, [2])


def test_zero_layer_rejected():
    with pytest.raises(ValueError):
        report.global_cumulative(np.zeros((1, 3, 2)), [2])


def test_global_beats_uniform_at_same_count():
    q = np.zeros((2, 3, 2))
    q[:, 0] = [[1, 99], [40, 60]]
    c = report.global_cumulative(q, [2, 2])
    # For two removals global prefix is optimal among all subsets.
    import itertools

    costs = [0.005, 0.495, 0.2, 0.3]
    assert c[2] == min(sum(pair) for pair in itertools.combinations(costs, 2))
