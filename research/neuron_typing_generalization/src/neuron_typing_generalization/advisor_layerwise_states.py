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

"""Layerwise V/T/M/W response-state distributions for the advisor study."""

from __future__ import annotations

import numpy as np
import pandas as pd


STATE_ORDER = ("V", "T", "M", "W")
SHARE_COLUMNS = tuple(f"share_{state}" for state in STATE_ORDER)


def layerwise_state_distribution(profile: pd.DataFrame) -> pd.DataFrame:
    """Average per-neuron state frequencies within each decoder layer.

    The estimator is ``p_c^(l) = mean_j(N_c^(l,j) / N)``. The current
    advisor study requires one common positive sample count for every neuron
    in a run and never pads layers to a common width.
    """
    required = {"layer", "neuron", "valid_samples", *(f"count_{state}" for state in STATE_ORDER)}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Missing layerwise state columns: {sorted(missing)}")

    numeric = profile[list(required)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if numeric.isna().any().any() or not np.isfinite(values).all():
        raise ValueError("Layer, neuron, sample, and state-count values must be finite numbers.")
    if (numeric < 0).any().any():
        raise ValueError("Layer, neuron, sample, and state-count values must be nonnegative.")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("Layer, neuron, sample, and state-count values must be integers.")
    if profile.duplicated(["layer", "neuron"]).any():
        raise ValueError("Each (layer, neuron) index must be unique within a run.")

    sample_counts = numeric["valid_samples"].unique()
    if len(sample_counts) != 1 or sample_counts[0] <= 0:
        raise ValueError("All neurons in a run must have the same positive valid-sample count.")
    count_sum = sum(numeric[f"count_{state}"].to_numpy(dtype=np.float64) for state in STATE_ORDER)
    if not np.array_equal(count_sum, numeric["valid_samples"].to_numpy(dtype=np.float64)):
        raise ValueError("The four state counts must sum to valid_samples for every neuron.")

    working = pd.DataFrame(
        {
            "layer": numeric["layer"].astype(np.int64),
            **{
                f"share_{state}": numeric[f"count_{state}"] / numeric["valid_samples"]
                for state in STATE_ORDER
            },
        }
    )
    grouped = working.groupby("layer", sort=True, observed=True)
    result = grouped[list(SHARE_COLUMNS)].mean()
    result.insert(0, "neuron_count", grouped.size().astype(np.int64))
    result.insert(1, "sample_count", int(sample_counts[0]))
    result = result.reset_index()

    if "d_ff" in profile.columns:
        d_ff = pd.to_numeric(profile["d_ff"], errors="coerce")
        if d_ff.isna().any() or not np.isfinite(d_ff.to_numpy(dtype=np.float64)).all():
            raise ValueError("d_ff values must be finite.")
        if (d_ff <= 0).any() or not np.equal(d_ff, np.floor(d_ff)).all():
            raise ValueError("d_ff values must be positive integers.")
        layer_width = pd.DataFrame({"layer": numeric["layer"].astype(np.int64), "d_ff": d_ff.astype(np.int64)})
        unique_widths = layer_width.groupby("layer")["d_ff"].nunique()
        if (unique_widths != 1).any():
            raise ValueError("Each layer must have exactly one d_ff value.")
        widths = layer_width.groupby("layer")["d_ff"].first()
        actual = result.set_index("layer")["neuron_count"]
        if not actual.equals(widths.rename("neuron_count")):
            raise ValueError("Each layer must contain exactly d_ff real neurons; padding or omissions are forbidden.")
        result.insert(2, "d_ff", result["layer"].map(widths).astype(np.int64))

    row_sums = result[list(SHARE_COLUMNS)].sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-12, atol=1e-12):
        raise ValueError("Layerwise V/T/M/W shares must sum to one.")
    result["share_sum"] = row_sums
    result["share_sum_residual"] = row_sums - 1.0
    return result
