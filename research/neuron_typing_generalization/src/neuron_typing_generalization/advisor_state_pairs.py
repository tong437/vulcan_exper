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

"""State-pair decomposition of the advisor's per-neuron change rate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STATE_PAIRS = (("V", "T"), ("V", "M"), ("V", "W"), ("T", "M"), ("T", "W"), ("M", "W"))
PAIR_LABELS = tuple(f"{left}{right}" for left, right in STATE_PAIRS)
PAIR_COLUMNS = tuple(f"R_{label}" for label in PAIR_LABELS)


def compute_state_pair_contributions(profile: pd.DataFrame) -> pd.DataFrame:
    """Compute six absolute per-neuron state-pair probabilities.

    For a neuron with ``n`` valid samples, the contribution of the unordered
    pair ``ab`` is ``2 * n_a * n_b / (n * (n - 1))``. Neurons with fewer than
    two valid samples receive NaN because their distinct-sample rate is
    undefined.
    """
    required = {"valid_samples", *(f"count_{state}" for state in "VTMW")}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Missing state-count columns: {sorted(missing)}")

    numeric = profile[list(required)].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise ValueError("State counts and valid-sample counts must be finite numbers.")
    if (numeric < 0).any().any():
        raise ValueError("State counts and valid-sample counts must be nonnegative.")
    if not np.equal(numeric.to_numpy(dtype=np.float64), np.floor(numeric.to_numpy(dtype=np.float64))).all():
        raise ValueError("State counts and valid-sample counts must be integers.")

    valid = numeric["valid_samples"].to_numpy(dtype=np.float64)
    count_sum = sum(numeric[f"count_{state}"].to_numpy(dtype=np.float64) for state in "VTMW")
    if not np.array_equal(count_sum, valid):
        raise ValueError("The four state counts must sum to valid_samples for every neuron.")

    denominator = valid * (valid - 1.0)
    defined = denominator > 0
    output = pd.DataFrame(index=profile.index)
    for (left, right), column in zip(STATE_PAIRS, PAIR_COLUMNS):
        numerator = (
            2.0
            * numeric[f"count_{left}"].to_numpy(dtype=np.float64)
            * numeric[f"count_{right}"].to_numpy(dtype=np.float64)
        )
        output[column] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=defined,
        )

    output["R_sum"] = output[list(PAIR_COLUMNS)].sum(axis=1, min_count=len(PAIR_COLUMNS))
    if "change_rate_distinct_pairs" in profile.columns:
        stored = pd.to_numeric(profile["change_rate_distinct_pairs"], errors="coerce").to_numpy(dtype=np.float64)
        comparable = defined & np.isfinite(stored)
        if np.any(defined & ~np.isfinite(stored)):
            raise ValueError("A defined neuron is missing its stored distinct-pair change rate.")
        residual = np.full_like(stored, np.nan, dtype=np.float64)
        residual[comparable] = output.loc[comparable, "R_sum"].to_numpy() - stored[comparable]
        output["identity_residual"] = residual
        if comparable.any() and not np.allclose(residual[comparable], 0.0, rtol=1e-10, atol=1e-12):
            raise ValueError("The six state-pair contributions do not reproduce the stored neuron change rate.")
    return output


def summarize_state_pair_contributions(profile: pd.DataFrame) -> dict[str, Any]:
    """Average absolute contributions, then normalize them into composition shares."""
    contributions = compute_state_pair_contributions(profile)
    defined = contributions["R_sum"].notna()
    if not defined.any():
        raise ValueError("No neurons have at least two valid samples.")

    means = contributions.loc[defined, list(PAIR_COLUMNS)].mean(axis=0)
    overall = float(means.sum())
    if not np.isfinite(overall) or overall <= 0.0:
        raise ValueError("State-pair composition is undefined when the overall change rate is zero.")
    if overall > 1.0 + 1e-12:
        raise ValueError("The overall change rate cannot exceed one.")
    shares = means / overall

    if "change_rate_distinct_pairs" in profile.columns:
        stored_mean = float(pd.to_numeric(profile.loc[defined, "change_rate_distinct_pairs"]).mean())
        if not np.isclose(overall, stored_mean, rtol=1e-10, atol=1e-12):
            raise ValueError("Mean state-pair contributions do not reproduce the reported overall change rate.")
    else:
        stored_mean = overall

    residual = contributions.loc[defined, "identity_residual"] if "identity_residual" in contributions else None
    return {
        "included_neurons": int(defined.sum()),
        "excluded_neurons": int((~defined).sum()),
        "overall_change_rate": overall,
        "stored_overall_change_rate": stored_mean,
        "absolute_contributions": {label: float(means[f"R_{label}"]) for label in PAIR_LABELS},
        "composition_shares": {label: float(shares[f"R_{label}"]) for label in PAIR_LABELS},
        "maximum_neuron_identity_residual": (
            float(residual.abs().max()) if residual is not None and not residual.empty else None
        ),
    }
