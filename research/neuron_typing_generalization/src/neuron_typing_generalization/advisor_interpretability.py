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

"""Entropy and effective-category summaries for the advisor's 4 x 4 study."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


EFFECTIVE_BIN_WIDTH = 0.05
EFFECTIVE_BINS = np.linspace(1.0, 4.0, 61)
REFERENCE_BANDS = (
    ("near_1", 1.0, 1.5),
    ("near_2", 1.5, 2.5),
    ("near_3", 2.5, 3.5),
    ("near_4", 3.5, 4.0 + 1e-12),
)


def summarize_effective_categories(profile: pd.DataFrame) -> tuple[dict[str, Any], np.ndarray]:
    """Validate and summarize per-neuron entropy/effective-category values.

    The stored entropy is the unnormalized Shannon entropy in nats. The
    normalized entropy is ``entropy / log(4)`` and the Hill number is therefore
    ``exp(entropy) == 4 ** normalized_entropy``.
    """
    required = {"valid_samples", "entropy", "normalized_entropy", "effective_class_count"}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Missing effective-category columns: {sorted(missing)}")

    finite = profile.loc[
        profile[list(required)].notna().all(axis=1) & np.isfinite(profile[list(required)]).all(axis=1)
    ].copy()
    if finite.empty:
        raise ValueError("No finite per-neuron entropy values were found.")
    if (finite["valid_samples"] <= 0).any():
        raise ValueError("Effective-category summaries require at least one valid sample per neuron.")

    entropy = finite["entropy"].to_numpy(dtype=np.float64)
    normalized = finite["normalized_entropy"].to_numpy(dtype=np.float64)
    effective = finite["effective_class_count"].to_numpy(dtype=np.float64)
    if not np.allclose(normalized, entropy / np.log(4.0), rtol=1e-10, atol=1e-10):
        raise ValueError("Stored normalized entropy is inconsistent with entropy / log(4).")
    if not np.allclose(effective, np.exp(entropy), rtol=1e-10, atol=1e-10):
        raise ValueError("Stored effective category count is inconsistent with exp(entropy).")
    if np.any(effective < 1.0 - 1e-10) or np.any(effective > 4.0 + 1e-10):
        raise ValueError("Effective category counts must lie in [1, 4].")

    effective = np.clip(effective, 1.0, 4.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    q10, q25, median, q75, q90 = np.quantile(effective, [0.10, 0.25, 0.50, 0.75, 0.90])
    histogram, _ = np.histogram(effective, bins=EFFECTIVE_BINS)
    percentages = histogram.astype(np.float64) / len(effective) * 100.0
    summary: dict[str, Any] = {
        "neuron_count": int(len(effective)),
        "normalized_entropy_mean": float(normalized.mean()),
        "normalized_entropy_median": float(np.median(normalized)),
        "effective_category_mean": float(effective.mean()),
        "effective_category_q10": float(q10),
        "effective_category_q25": float(q25),
        "effective_category_median": float(median),
        "effective_category_q75": float(q75),
        "effective_category_q90": float(q90),
    }
    for label, lower, upper in REFERENCE_BANDS:
        summary[f"{label}_proportion"] = float(np.mean((effective >= lower) & (effective < upper)))
    return summary, percentages


def smooth_histogram(percentages: np.ndarray, bandwidth: float = 0.12) -> np.ndarray:
    """Gaussian-smooth histogram percentages while retaining per-bin units."""
    values = np.asarray(percentages, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(EFFECTIVE_BINS) - 1:
        raise ValueError(f"Expected {len(EFFECTIVE_BINS) - 1} histogram bins.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    sigma_bins = bandwidth / EFFECTIVE_BIN_WIDTH
    radius = max(1, int(np.ceil(4 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_bins))
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="same")[radius:-radius]
