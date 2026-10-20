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

"""Auditable Hellinger/average-linkage pilot clustering for neuron state profiles."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


PROBABILITY_COLUMNS = ("proportion_V", "proportion_T", "proportion_M", "proportion_W")


def hellinger_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the Hellinger distance between two categorical distributions."""
    left_probability = _validate_probability_matrix(np.asarray(left, dtype=np.float64)[None, :])[0]
    right_probability = _validate_probability_matrix(np.asarray(right, dtype=np.float64)[None, :])[0]
    return float(np.linalg.norm(np.sqrt(left_probability) - np.sqrt(right_probability)) / np.sqrt(2.0))


def condensed_hellinger(probabilities: np.ndarray) -> np.ndarray:
    """Return the SciPy condensed Hellinger distance vector."""
    values = _validate_probability_matrix(probabilities)
    return pdist(np.sqrt(values), metric="euclidean") / np.sqrt(2.0)


def _validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("probabilities must have shape [neurons, 4].")
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values.")
    if np.any(values < -1e-12) or np.any(values > 1.0 + 1e-12):
        raise ValueError("probabilities must lie in [0, 1].")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("each probability vector must sum to one.")
    return np.clip(values, 0.0, 1.0)


def _largest_remainder_quotas(layer_sizes: pd.Series, target: int) -> dict[int, int]:
    if target <= 0 or target > int(layer_sizes.sum()):
        raise ValueError("target must be positive and no larger than the population.")
    ideal = layer_sizes.astype(np.float64) * target / int(layer_sizes.sum())
    quotas = np.floor(ideal).astype(int)
    remaining = target - int(quotas.sum())
    order = sorted(layer_sizes.index, key=lambda layer: (-(ideal[layer] - quotas[layer]), int(layer)))
    for layer in order[:remaining]:
        quotas[layer] += 1
    if int(quotas.sum()) != target or (quotas > layer_sizes).any():
        raise AssertionError("invalid layer-stratified quota allocation.")
    return {int(layer): int(quota) for layer, quota in quotas.items()}


def stratified_neuron_sample(profile: pd.DataFrame, maximum_neurons: int, seed: int) -> pd.DataFrame:
    """Select a deterministic neuron sample proportional to the layer widths."""
    required = {"layer", "neuron", *PROBABILITY_COLUMNS}
    missing = required.difference(profile.columns)
    if missing:
        raise ValueError(f"Missing profile columns: {sorted(missing)}")
    if profile.duplicated(["layer", "neuron"]).any():
        raise ValueError("Neuron identifiers must be unique within a run.")
    target = min(int(maximum_neurons), len(profile))
    if target < 2:
        raise ValueError("At least two neurons are required for clustering.")
    layer_sizes = profile.groupby("layer", sort=True).size()
    quotas = _largest_remainder_quotas(layer_sizes, target)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for layer, values in profile.groupby("layer", sort=True):
        indices = values.index.to_numpy(dtype=np.int64)
        chosen = rng.choice(indices, size=quotas[int(layer)], replace=False)
        selected.append(np.sort(chosen))
    result = profile.loc[np.concatenate(selected)].copy()
    return result.sort_values(["layer", "neuron"], kind="stable").reset_index(drop=True)


def stable_run_seed(seed: int, run_id: str) -> int:
    """Derive a stable run-specific NumPy seed from the global seed and run ID."""
    digest = hashlib.sha256(f"{seed}:{run_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def average_linkage_clusters(
    probabilities: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Cluster probability vectors using exact average linkage on Hellinger distance."""
    values = _validate_probability_matrix(probabilities)
    if not 0 < threshold <= 1:
        raise ValueError("threshold must lie in (0, 1].")
    condensed = condensed_hellinger(values)
    linkage_matrix = linkage(condensed, method="average", optimal_ordering=False)
    raw_labels = fcluster(linkage_matrix, t=threshold, criterion="distance").astype(np.int32)

    descriptors = []
    for raw_label in np.unique(raw_labels):
        member_indices = np.flatnonzero(raw_labels == raw_label)
        centroid = values[member_indices].mean(axis=0)
        descriptors.append((int(raw_label), len(member_indices), tuple(centroid.tolist()), int(member_indices.min())))
    descriptors.sort(key=lambda item: (-item[1], item[2], item[3]))
    canonical = {raw_label: cluster_id for cluster_id, (raw_label, _, _, _) in enumerate(descriptors, start=1)}
    labels = np.asarray([canonical[int(raw_label)] for raw_label in raw_labels], dtype=np.int32)

    cluster_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(np.unique(labels)):
        member_indices = np.flatnonzero(labels == cluster_id)
        members = values[member_indices]
        centroid = members.mean(axis=0)
        centroid_distance = np.linalg.norm(np.sqrt(members) - np.sqrt(centroid), axis=1) / np.sqrt(2.0)
        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "neuron_count": int(len(member_indices)),
                "neuron_proportion": float(len(member_indices) / len(values)),
                **{f"centroid_{label}": float(value) for label, value in zip(("V", "T", "M", "W"), centroid)},
                "mean_hellinger_to_centroid": float(centroid_distance.mean()),
                "p90_hellinger_to_centroid": float(np.quantile(centroid_distance, 0.90)),
                "max_hellinger_to_centroid": float(centroid_distance.max()),
            }
        )
    return labels, linkage_matrix, cluster_rows


def condensed_distance_bytes(neuron_count: int, dtype_bytes: int = 8) -> int:
    """Return storage required for a condensed pairwise distance vector."""
    return int(neuron_count * (neuron_count - 1) // 2 * dtype_bytes)
