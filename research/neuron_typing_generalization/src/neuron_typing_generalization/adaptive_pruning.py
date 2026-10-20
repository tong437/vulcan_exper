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

"""Exact sample-level A1 knee and A2 score-budget FFN ablation plans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AdaptiveSelection:
    """A uniform-ratio, lowest-score selection for every real decoder layer."""

    method: str
    candidate_ratio: float
    candidate_score_share: float
    actual_ratio: float
    actual_score_share: float
    selected: dict[int, NDArray[np.int64]]
    selected_per_layer: tuple[int, ...]


def _sample_curves(
    q_three: NDArray[np.floating], widths: Sequence[int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.int64]], list[NDArray[np.float64]]]:
    """Evaluate the layer-equal mean curve at every real layer's neuron-count knot."""
    q_three = np.asarray(q_three)
    widths = tuple(int(width) for width in widths)
    if q_three.ndim != 3 or q_three.shape[1] != 3 or q_three.shape[0] != len(widths):
        raise ValueError("q must have shape [layers, 3, maximum_d_ff].")
    if not widths or any(width < 1 or width > q_three.shape[2] for width in widths):
        raise ValueError("Real decoder layer widths are invalid.")
    u = np.unique(np.concatenate([np.arange(width + 1, dtype=np.float64) / width for width in set(widths)]))
    mean = np.zeros(len(u), dtype=np.float64)
    orders: list[NDArray[np.int64]] = []
    cumulatives: list[NDArray[np.float64]] = []
    for layer, width in enumerate(widths):
        scores = q_three[layer, :, :width].astype(np.float64).sum(axis=0)
        if not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError(f"Layer {layer} has invalid nonnegative scores.")
        total = scores.sum()
        if total <= 0:
            raise ValueError(f"Layer {layer} has zero total score.")
        order = np.lexsort((np.arange(width), scores)).astype(np.int64)
        cumulative = np.r_[0.0, np.cumsum(scores[order], dtype=np.float64) / total]
        cumulative[-1] = 1.0
        orders.append(order)
        cumulatives.append(cumulative)
        if len(u) == width + 1 and np.array_equal(u, np.arange(width + 1) / width):
            mean += cumulative
        else:
            mean += np.interp(u, np.arange(width + 1) / width, cumulative)
    mean /= len(widths)
    mean[0], mean[-1] = 0.0, 1.0
    if np.any(np.diff(mean) < -1e-10) or np.any(mean > u + 1e-10):
        raise ValueError("Sample cumulative importance curve violates sorting invariants.")
    return u, mean, orders, cumulatives


def adaptive_selections(
    q_three: NDArray[np.floating], widths: Sequence[int], *, epsilon: float = 0.05
) -> dict[str, AdaptiveSelection]:
    """Choose A1/A2 continuous rates, then floor the same rate in every layer.

    A1 is exact for the piecewise-linear mean curve because its maximum
    vertical gap occurs at a knot. A2 linearly interpolates the rightmost
    crossing of the monotone mean curve with epsilon. The budget is on the
    equal-layer mean, not on each layer separately.
    """
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be finite and lie in [0, 1).")
    widths = tuple(int(width) for width in widths)
    u, mean, orders, cumulatives = _sample_curves(q_three, widths)
    gap = u - mean
    knee = int(np.flatnonzero(np.isclose(gap, gap.max(), atol=1e-12, rtol=1e-10))[0])
    a1_rate = float(u[knee])
    right = int(np.searchsorted(mean, epsilon, side="right"))
    if right == 0:
        a2_rate = 0.0
    elif right == len(mean):
        a2_rate = 1.0
    else:
        left = right - 1
        if np.isclose(mean[left], epsilon, atol=1e-14, rtol=0):
            a2_rate = float(u[left])
        else:
            a2_rate = float(u[left] + (epsilon - mean[left]) * (u[right] - u[left]) / (mean[right] - mean[left]))
    result = {}
    for method, rate in (("A1", a1_rate), ("A2", a2_rate)):
        counts = tuple(min(width, int(np.floor(rate * width + 1e-9))) for width in widths)
        selected = {layer: np.sort(order[: counts[layer]]) for layer, order in enumerate(orders)}
        actual_share = float(np.mean([curve[count] for curve, count in zip(cumulatives, counts)]))
        if method == "A2" and actual_share > epsilon + 1e-10:
            raise ValueError("Discrete A2 selection exceeds its mean score budget.")
        result[method] = AdaptiveSelection(
            method=method,
            candidate_ratio=rate,
            candidate_score_share=float(np.interp(rate, u, mean)),
            actual_ratio=sum(counts) / sum(widths),
            actual_score_share=actual_share,
            selected=selected,
            selected_per_layer=counts,
        )
    return result


def nonuniform_selection(
    q_three: NDArray[np.floating], widths: Sequence[int], *, epsilon: float, target_count: int | None = None
) -> AdaptiveSelection:
    """Select a maximum budgeted count, or minimize cost at a frozen U count.

    N/N_match have no continuous uniform candidate: candidate fields equal realized
    discrete totals. Ties are ordered by layer, then neuron index.
    """
    if not np.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("epsilon must be finite and lie in [0, 1).")
    widths = tuple(int(width) for width in widths)
    uniform = adaptive_selections(q_three, widths, epsilon=epsilon)["A2"]
    if target_count is not None:
        if isinstance(target_count, bool) or not isinstance(target_count, (int, np.integer)):
            raise ValueError("target_count must be an integer.")
        if target_count != sum(uniform.selected_per_layer):
            raise ValueError("Frozen uniform count disagrees with recomputed uniform selection.")
    scores = [np.asarray(q_three[l, :, :width], dtype=np.float64).sum(axis=0) for l, width in enumerate(widths)]
    costs = np.concatenate([s / s.sum() / len(widths) for s in scores])
    order = np.argsort(costs, kind="stable")
    cumulative = np.cumsum(costs[order], dtype=np.float64)
    count = int(np.searchsorted(cumulative, epsilon, side="right"))
    # Same tolerance as the existing budget audit, only for roundoff at exact boundaries.
    while count < len(order) and cumulative[count] <= epsilon + 1e-12:
        count += 1
    if target_count is not None:
        count = int(target_count)
    offsets = np.r_[0, np.cumsum(widths)]
    chosen = np.sort(order[:count])
    selected = {
        layer: (chosen[(chosen >= offsets[layer]) & (chosen < offsets[layer + 1])] - offsets[layer])
        for layer in range(len(widths))
    }
    counts = tuple(len(selected[layer]) for layer in range(len(widths)))
    share = float(np.mean([s[selected[layer]].sum() / s.sum() for layer, s in enumerate(scores)]))
    if share > epsilon + 1e-10 or count < sum(uniform.selected_per_layer):
        raise ValueError("Nonuniform selection failed budget or uniform-count dominance audit.")
    if target_count is not None and share > uniform.actual_score_share + 1e-10:
        raise ValueError("Matched-count response cost exceeds uniform selection cost.")
    ratio = count / sum(widths)
    return AdaptiveSelection(
        "N_match" if target_count is not None else "N", ratio, share, ratio, share, selected, counts
    )


def single_layer_selections(q_three, widths, layers, rates):
    """Select lowest scores in exactly one layer; return independent masks."""
    _, _, orders, curves = _sample_curves(q_three, widths)
    result = {}
    for layer in layers:
        if layer not in range(len(widths)):
            raise ValueError("Invalid target layer.")
        for rate in rates:
            if not np.isfinite(rate) or not 0 <= rate < 1:
                raise ValueError("Invalid layer removal rate.")
            count = int(np.floor(rate * widths[layer] + 1e-9))
            selected = {
                l: np.sort(order[:count]) if l == layer else np.empty(0, dtype=np.int64)
                for l, order in enumerate(orders)
            }
            counts = tuple(count if l == layer else 0 for l in range(len(widths)))
            name = f"L{layer:02d}_r{rate:.2f}"
            share = float(curves[layer][count]) / len(widths)
            result[name] = AdaptiveSelection(
                name, count / sum(widths), share, count / sum(widths), share, selected, counts
            )
    return result
