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

"""Synthetic CPU scale checks using the real Qwen3.5-2B decoder dimensions."""

from __future__ import annotations

import resource
import time
from typing import Any

import numpy as np

from .analysis import change_rates_from_counts, maximum_change_pair, neuron_stability


def run_scale_benchmark(
    sample_count: int = 128,
    layer_count: int = 24,
    d_ff: int = 6144,
    chunk_size: int = 32,
    seed: int = 42,
) -> dict[str, Any]:
    """Exercise count, stability, and chunked pair algorithms at real neuron dimensions."""
    if min(sample_count, layer_count, d_ff, chunk_size) <= 0:
        raise ValueError("All benchmark dimensions must be positive.")
    if sample_count < 2:
        raise ValueError("At least two samples are required for pair search.")
    generator = np.random.default_rng(seed)
    classes = generator.integers(0, 4, size=(sample_count, layer_count, d_ff), dtype=np.int8)
    invalid_mask = generator.random(classes.shape) < 0.001
    classes[invalid_mask] = -1
    started = time.perf_counter()
    rates = change_rates_from_counts(classes)
    after_rates = time.perf_counter()
    stability = neuron_stability(classes)
    after_stability = time.perf_counter()
    maximum = maximum_change_pair(classes, chunk_size=chunk_size)
    after_maximum = time.perf_counter()
    repeat = maximum_change_pair(classes, chunk_size=max(1, chunk_size // 2))
    if maximum != repeat:
        raise AssertionError("Maximum-pair result changed with chunk size.")
    peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "status": "passed",
        "seed": seed,
        "shape": {"samples": sample_count, "layers": layer_count, "d_ff": d_ff},
        "class_tensor_bytes": int(classes.nbytes),
        "invalid_observations": int(invalid_mask.sum()),
        "change_rate_including_self": rates.including_self,
        "changed_ordered_comparisons": rates.including_self_changed,
        "valid_ordered_comparisons_including_self": rates.including_self_comparisons,
        "change_rate_distinct_pairs": rates.distinct_pairs,
        "valid_distinct_ordered_comparisons": rates.distinct_pair_comparisons,
        "maximum_pair": {
            "left": maximum.left,
            "right": maximum.right,
            "rate": maximum.rate,
            "valid_neurons": maximum.valid_neurons,
        },
        "stability_shape": list(stability.counts.shape),
        "chunk_sizes_verified": [chunk_size, max(1, chunk_size // 2)],
        "elapsed_seconds": {
            "change_rates": after_rates - started,
            "stability": after_stability - after_rates,
            "maximum_pair_first": after_maximum - after_stability,
            "maximum_pair_repeat": time.perf_counter() - after_maximum,
            "total": time.perf_counter() - started,
        },
        "peak_process_rss_kib": peak_rss_kib,
    }
