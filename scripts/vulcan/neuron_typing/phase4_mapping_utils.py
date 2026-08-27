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

"""Shared, model-independent utilities for Phase-4 activation mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


SPLIT_NAMES = ("train", "validation", "test")


def image_disjoint_split(
    image_ids: list[str],
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> list[str]:
    """Assign rows to deterministic image-disjoint train/validation/test splits."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one.")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be strictly between zero and one.")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be smaller than one.")

    unique_images = np.asarray(sorted(set(image_ids)), dtype=object)
    if len(unique_images) < 6:
        raise ValueError("Phase-4 mapping requires at least six unique images.")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_images)

    train_count = max(2, int(round(len(unique_images) * train_ratio)))
    validation_count = max(2, int(round(len(unique_images) * validation_ratio)))
    if train_count + validation_count > len(unique_images) - 2:
        overflow = train_count + validation_count - (len(unique_images) - 2)
        reducible_train = max(0, train_count - 2)
        train_reduction = min(overflow, reducible_train)
        train_count -= train_reduction
        overflow -= train_reduction
        validation_count -= overflow
    if min(train_count, validation_count, len(unique_images) - train_count - validation_count) < 2:
        raise ValueError("Every Phase-4 split must contain at least two images for split-local shuffling.")

    image_to_split = {
        image_id: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, image_id in enumerate(unique_images.tolist())
    }
    return [image_to_split[image_id] for image_id in image_ids]


def verify_image_disjoint_split(image_ids: list[str], splits: list[str]) -> dict[str, Any]:
    """Validate row alignment, split names, and image disjointness."""
    if len(image_ids) != len(splits):
        raise ValueError("image_ids and splits have different lengths.")
    unknown = sorted(set(splits) - set(SPLIT_NAMES))
    if unknown:
        raise ValueError(f"Unknown split names: {unknown}")

    image_sets = {
        split: {image_id for image_id, row_split in zip(image_ids, splits) if row_split == split}
        for split in SPLIT_NAMES
    }
    overlaps = {
        f"{left}_vs_{right}": sorted(image_sets[left] & image_sets[right])
        for left_index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[left_index + 1 :]
    }
    if any(overlaps.values()):
        raise ValueError(f"Image-disjoint split validation failed: {overlaps}")
    if any(len(image_sets[split]) < 2 for split in SPLIT_NAMES):
        raise ValueError("Every split must contain at least two unique images.")
    return {
        "is_image_disjoint": True,
        "row_counts": {split: splits.count(split) for split in SPLIT_NAMES},
        "image_counts": {split: len(image_sets[split]) for split in SPLIT_NAMES},
        "overlaps": overlaps,
    }


def split_local_derangement(image_ids: list[str], splits: list[str], *, seed: int) -> dict[str, str]:
    """Build a deterministic one-to-one image derangement independently inside every split."""
    verification = verify_image_disjoint_split(image_ids, splits)
    del verification
    mapping: dict[str, str] = {}
    for split_index, split in enumerate(SPLIT_NAMES):
        split_images = sorted({image_id for image_id, row_split in zip(image_ids, splits) if row_split == split})
        rng = np.random.default_rng(seed + split_index)
        shift = int(rng.integers(1, len(split_images)))
        shifted = split_images[shift:] + split_images[:shift]
        mapping.update(dict(zip(split_images, shifted)))

    if set(mapping) != set(image_ids):
        raise RuntimeError("Split-local image mapping does not cover every image.")
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("Split-local image derangement contains a fixed point.")
    split_lookup = dict(zip(image_ids, splits))
    if any(split_lookup[source] != split_lookup[target] for source, target in mapping.items()):
        raise RuntimeError("Split-local image derangement crossed a data split.")
    return mapping


def masked_pool(values: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool a `[batch, sequence, width]` tensor over a boolean token mask."""
    if values.ndim != 3 or mask.ndim != 2 or values.shape[:2] != mask.shape:
        raise ValueError(f"Cannot pool values {tuple(values.shape)} with mask {tuple(mask.shape)}.")
    counts = mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("At least one sample has no tokens for the requested pooling operation.")

    expanded = mask.unsqueeze(-1)
    mean = (values.float() * expanded).sum(dim=1) / counts.unsqueeze(1)
    if mode == "mean":
        return mean
    if mode not in {"max_abs", "mean_max_abs"}:
        raise ValueError(f"Unknown pooling mode: {mode}")

    absolute = values.float().abs().masked_fill(~expanded, float("-inf"))
    indices = absolute.argmax(dim=1, keepdim=True)
    max_abs = values.float().gather(1, indices).squeeze(1)
    return max_abs if mode == "max_abs" else torch.cat((mean, max_abs), dim=1)


@dataclass
class RandomizedProjector:
    """Training-only standardization followed by a fixed linear projection."""

    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(values, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.components.T


def fit_randomized_projector(
    values: np.ndarray,
    *,
    rank: int,
    seed: int,
    oversamples: int = 8,
    power_iterations: int = 1,
) -> RandomizedProjector:
    """Fit a compact randomized PCA-style feature projector without sklearn."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Projector input must be a 2-D matrix with at least two rows.")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (values - mean) / scale
    max_rank = min(standardized.shape[0] - 1, standardized.shape[1])
    effective_rank = min(rank, max_rank)
    if effective_rank < 1:
        raise ValueError("The requested projection has zero effective rank.")

    sketch_rank = min(max_rank, effective_rank + oversamples)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((standardized.shape[1], sketch_rank))
    sample_basis, _ = np.linalg.qr(standardized @ omega, mode="reduced")
    for _ in range(power_iterations):
        feature_basis, _ = np.linalg.qr(standardized.T @ sample_basis, mode="reduced")
        sample_basis, _ = np.linalg.qr(standardized @ feature_basis, mode="reduced")
    small_matrix = sample_basis.T @ standardized
    _, _, right_vectors = np.linalg.svd(small_matrix, full_matrices=False)
    components = right_vectors[:effective_rank]
    return RandomizedProjector(mean=mean, scale=scale, components=components)


@dataclass
class LinearActivationMapper:
    """A ridge mapper with an optional reduced-rank target basis."""

    target_mean: np.ndarray
    coefficients: np.ndarray
    target_components: np.ndarray | None = None

    def predict(self, projected_features: np.ndarray) -> np.ndarray:
        prediction = np.asarray(projected_features, dtype=np.float64) @ self.coefficients
        if self.target_components is not None:
            prediction = prediction @ self.target_components
        return prediction + self.target_mean


@dataclass
class LinearMapperWorkspace:
    """Alpha-independent sufficient statistics for repeated ridge fits."""

    target_mean: np.ndarray
    gram: np.ndarray
    cross_product: np.ndarray
    target_components: np.ndarray | None

    def fit(self, alpha: float) -> LinearActivationMapper:
        regularized = self.gram + float(alpha) * np.eye(self.gram.shape[0], dtype=np.float64)
        coefficients = np.linalg.solve(regularized, self.cross_product)
        return LinearActivationMapper(
            target_mean=self.target_mean,
            coefficients=coefficients,
            target_components=self.target_components,
        )


def fit_target_components(targets: np.ndarray, rank: int, seed: int) -> np.ndarray:
    """Fit a reduced-rank output basis on training targets only."""
    centered = targets - targets.mean(axis=0)
    max_rank = min(centered.shape[0] - 1, centered.shape[1])
    effective_rank = min(rank, max_rank)
    rng = np.random.default_rng(seed)
    sketch_rank = min(max_rank, effective_rank + 8)
    omega = rng.standard_normal((centered.shape[1], sketch_rank))
    sample_basis, _ = np.linalg.qr(centered @ omega, mode="reduced")
    small_matrix = sample_basis.T @ centered
    _, _, right_vectors = np.linalg.svd(small_matrix, full_matrices=False)
    return right_vectors[:effective_rank]


def prepare_linear_mapper(
    projected_features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    target_rank: int | None = None,
    target_components: np.ndarray | None = None,
) -> LinearMapperWorkspace:
    """Cache alpha-independent sufficient statistics for ridge fitting."""
    features = np.asarray(projected_features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError("Features and targets must be aligned 2-D matrices.")
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    regression_targets = centered_targets
    if target_components is not None and target_rank is not None:
        raise ValueError("Pass target_rank or precomputed target_components, not both.")
    if target_components is None and target_rank is not None and target_rank < min(centered_targets.shape):
        target_components = fit_target_components(centered_targets, target_rank, seed)
    if target_components is not None:
        regression_targets = centered_targets @ target_components.T

    gram = features.T @ features
    return LinearMapperWorkspace(
        target_mean=target_mean,
        gram=gram,
        cross_product=features.T @ regression_targets,
        target_components=target_components,
    )


def fit_linear_mapper(
    projected_features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
    seed: int,
    target_rank: int | None = None,
    target_components: np.ndarray | None = None,
) -> LinearActivationMapper:
    """Fit full-output ridge or reduced-rank ridge."""
    workspace = prepare_linear_mapper(
        projected_features,
        targets,
        seed=seed,
        target_rank=target_rank,
        target_components=target_components,
    )
    return workspace.fit(alpha)


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Compute variance-weighted and per-neuron R2/correlation."""
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("Targets and predictions must be aligned 2-D matrices.")
    residual_sum = np.square(targets - predictions).sum(axis=0)
    centered = targets - targets.mean(axis=0)
    total_sum = np.square(centered).sum(axis=0)
    valid = total_sum > 1e-12
    per_target_r2 = np.full(targets.shape[1], np.nan, dtype=np.float64)
    per_target_r2[valid] = 1.0 - residual_sum[valid] / total_sum[valid]
    variance_weighted_r2 = 1.0 - residual_sum[valid].sum() / total_sum[valid].sum() if valid.any() else float("nan")

    prediction_centered = predictions - predictions.mean(axis=0)
    denominator = np.sqrt(np.square(centered).sum(axis=0) * np.square(prediction_centered).sum(axis=0))
    correlation = np.full(targets.shape[1], np.nan, dtype=np.float64)
    correlation[denominator > 1e-12] = (
        (centered * prediction_centered).sum(axis=0)[denominator > 1e-12] / denominator[denominator > 1e-12]
    )
    return {
        "variance_weighted_r2": float(variance_weighted_r2),
        "mean_neuron_r2": float(np.nanmean(per_target_r2)),
        "median_neuron_r2": float(np.nanmedian(per_target_r2)),
        "mean_neuron_correlation": float(np.nanmean(correlation)),
        "per_neuron_r2": per_target_r2,
        "per_neuron_correlation": correlation,
    }


def topk_trigger_metrics(targets: np.ndarray, predictions: np.ndarray, top_k: int) -> dict[str, float]:
    """Measure per-sample recovery of the largest absolute activation changes."""
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    if targets.shape != predictions.shape or targets.ndim != 2:
        raise ValueError("Targets and predictions must be aligned 2-D matrices.")
    effective_k = min(top_k, targets.shape[1])
    target_top = np.argpartition(np.abs(targets), -effective_k, axis=1)[:, -effective_k:]
    predicted_top = np.argpartition(np.abs(predictions), -effective_k, axis=1)[:, -effective_k:]
    overlaps = np.asarray(
        [len(set(target_row.tolist()) & set(prediction_row.tolist())) for target_row, prediction_row in zip(target_top, predicted_top)]
    )
    precision_recall = overlaps / effective_k
    return {
        "top_k": int(effective_k),
        "precision_at_k": float(precision_recall.mean()),
        "recall_at_k": float(precision_recall.mean()),
        "random_expected_precision_at_k": float(effective_k / targets.shape[1]),
    }
