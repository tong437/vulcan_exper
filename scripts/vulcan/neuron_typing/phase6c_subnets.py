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

"""Pure set constructors for Phase 6C core, shell, and union interventions."""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Iterable
from typing import TypeAlias

import torch
from phase3_structural_utils import canonical_json_sha256


LayerSets: TypeAlias = dict[int, set[int]]


def validate_kept(kept: LayerSets, layer_dims: dict[int, int], *, allow_empty_layer: bool = False) -> None:
    if set(kept) != set(layer_dims):
        raise ValueError(f"Kept layers do not match layer dims: {sorted(kept)} != {sorted(layer_dims)}.")
    for layer, width in layer_dims.items():
        identities = kept[layer]
        if not allow_empty_layer and not identities:
            raise ValueError(f"Layer {layer} keeps no neurons.")
        if any(neuron < 0 or neuron >= width for neuron in identities):
            raise ValueError(f"Layer {layer} contains an out-of-range neuron identity.")


def kept_from_deleted(deleted: dict[str | int, list[int]], layer_dims: dict[int, int]) -> LayerSets:
    normalized = {int(layer): set(neurons) for layer, neurons in deleted.items()}
    if set(normalized) != set(layer_dims):
        raise ValueError("Deletion-mask layers do not match layer dims.")
    if any(
        len(neurons) != len(deleted[str(layer)] if str(layer) in deleted else deleted[layer])
        for layer, neurons in normalized.items()
    ):
        raise ValueError("Deletion masks contain duplicate neuron identities.")
    kept = {layer: set(range(width)) - normalized[layer] for layer, width in layer_dims.items()}
    validate_kept(kept, layer_dims)
    return kept


def kept_frequency(winners: dict[str, LayerSets]) -> dict[int, Counter[int]]:
    if not winners:
        raise ValueError("At least one winner is required.")
    layers = set(next(iter(winners.values())))
    if any(set(kept) != layers for kept in winners.values()):
        raise ValueError("Winner masks have inconsistent layers.")
    return {layer: Counter(neuron for kept in winners.values() for neuron in kept[layer]) for layer in sorted(layers)}


def frequency_core(winners: dict[str, LayerSets], minimum_frequency: int) -> LayerSets:
    if not 1 <= minimum_frequency <= len(winners):
        raise ValueError(f"minimum_frequency must lie in [1, {len(winners)}].")
    return {
        layer: {neuron for neuron, count in counts.items() if count >= minimum_frequency}
        for layer, counts in kept_frequency(winners).items()
    }


def layerwise_union(*values: LayerSets) -> LayerSets:
    if not values:
        raise ValueError("At least one layer set is required.")
    layers = set(values[0])
    if any(set(value) != layers for value in values):
        raise ValueError("Layer sets do not share the same layers.")
    return {layer: set().union(*(value[layer] for value in values)) for layer in sorted(layers)}


def layerwise_difference(left: LayerSets, right: LayerSets) -> LayerSets:
    if set(left) != set(right):
        raise ValueError("Layer sets do not share the same layers.")
    return {layer: left[layer] - right[layer] for layer in sorted(left)}


def layer_counts(values: LayerSets) -> dict[int, int]:
    return {layer: len(neurons) for layer, neurons in sorted(values.items())}


def derive_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def deterministic_sample(pool: LayerSets, counts: dict[int, int], *, seed: int) -> LayerSets:
    if set(pool) != set(counts):
        raise ValueError("Sampling pool and requested counts do not share the same layers.")
    result = {}
    for layer in sorted(pool):
        requested = counts[layer]
        candidates = sorted(pool[layer])
        if not 0 <= requested <= len(candidates):
            raise ValueError(f"Cannot sample {requested} of {len(candidates)} identities from layer {layer}.")
        rng = random.Random(derive_seed(seed, layer))
        result[layer] = set(rng.sample(candidates, requested))
    return result


def dose_counts(values: LayerSets, dose: float) -> dict[int, int]:
    if not 0 < dose <= 1:
        raise ValueError("Dose must lie in (0, 1].")
    return {
        layer: len(neurons) if dose == 1 else max(1, round(len(neurons) * dose))
        for layer, neurons in sorted(values.items())
    }


def deletion_masks(kept: LayerSets, layer_dims: dict[int, int]) -> dict[int, torch.Tensor]:
    validate_kept(kept, layer_dims)
    masks = {layer: torch.ones(width, dtype=torch.bool) for layer, width in layer_dims.items()}
    for layer, identities in kept.items():
        masks[layer][torch.tensor(sorted(identities), dtype=torch.long)] = False
    return masks


def canonical_kept_ids(kept: LayerSets) -> dict[str, list[int]]:
    return {str(layer): sorted(neurons) for layer, neurons in sorted(kept.items())}


def kept_sha256(kept: LayerSets) -> str:
    return canonical_json_sha256(canonical_kept_ids(kept))


def summarize_kept(kept: LayerSets, layer_dims: dict[int, int]) -> dict[str, object]:
    validate_kept(kept, layer_dims)
    total = sum(layer_dims.values())
    retained = sum(len(neurons) for neurons in kept.values())
    return {
        "total_neurons": total,
        "kept_neurons": retained,
        "deleted_neurons": total - retained,
        "keep_ratio": retained / total,
        "kept_by_layer": {str(layer): len(kept[layer]) for layer in sorted(kept)},
    }


def complement_pool(excluded: LayerSets, layer_dims: dict[int, int]) -> LayerSets:
    validate_kept(excluded, layer_dims, allow_empty_layer=True)
    return {layer: set(range(width)) - excluded[layer] for layer, width in layer_dims.items()}


def subtract_many(base: LayerSets, removed: Iterable[LayerSets]) -> LayerSets:
    result = {layer: set(neurons) for layer, neurons in base.items()}
    for value in removed:
        result = layerwise_difference(result, value)
    return result
