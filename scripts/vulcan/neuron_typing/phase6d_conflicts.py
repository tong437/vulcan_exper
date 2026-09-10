# Copyright 2025 the LlamaFactory team.
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

"""Pure constructors for Phase 6D negative-interference localization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from phase6c_subnets import (
    LayerSets,
    complement_pool,
    derive_seed,
    deterministic_sample,
    kept_sha256,
    layerwise_difference,
    layerwise_union,
    summarize_kept,
    validate_kept,
)


@dataclass(frozen=True)
class ConflictContrast:
    """One safe-base to failed-union target-level causal contrast."""

    contrast_id: str
    edge_id: str
    left: str
    right: str
    base: str
    donor: str
    target: str

    @property
    def union_variant_id(self) -> str:
        return f"union__{self.left}__{self.right}"


# Frozen from Phase 6C before inspecting any Phase 6D layer intervention.
CONFLICT_CONTRASTS = (
    ConflictContrast(
        "bicycle_violin__base_bicycle__target_bicycle",
        "bicycle_violin",
        "bicycle_clock",
        "violin_kitchen",
        "bicycle_clock",
        "violin_kitchen",
        "bicycle_clock",
    ),
    ConflictContrast(
        "bicycle_violin__base_bicycle__target_violin",
        "bicycle_violin",
        "bicycle_clock",
        "violin_kitchen",
        "bicycle_clock",
        "violin_kitchen",
        "violin_kitchen",
    ),
    ConflictContrast(
        "bicycle_violin__base_violin__target_bicycle",
        "bicycle_violin",
        "bicycle_clock",
        "violin_kitchen",
        "violin_kitchen",
        "bicycle_clock",
        "bicycle_clock",
    ),
    ConflictContrast(
        "bicycle_violin__base_violin__target_violin",
        "bicycle_violin",
        "bicycle_clock",
        "violin_kitchen",
        "violin_kitchen",
        "bicycle_clock",
        "violin_kitchen",
    ),
    ConflictContrast(
        "bicycle_toilet__base_toilet__target_toilet",
        "bicycle_toilet",
        "bicycle_clock",
        "toilet_spatial",
        "toilet_spatial",
        "bicycle_clock",
        "toilet_spatial",
    ),
    ConflictContrast(
        "violin_people__base_people__target_people",
        "violin_people",
        "violin_kitchen",
        "people_pizza",
        "people_pizza",
        "violin_kitchen",
        "people_pizza",
    ),
)


def one_layer(values: LayerSets, selected_layer: int) -> LayerSets:
    """Keep values from exactly one layer and empty sets elsewhere."""
    if selected_layer not in values:
        raise ValueError(f"Unknown layer {selected_layer}.")
    return {layer: set(neurons) if layer == selected_layer else set() for layer, neurons in values.items()}


def build_localization_variants(
    winners: dict[str, LayerSets],
    layer_dims: dict[int, int],
    *,
    random_seeds: int,
    seed: int,
    contrasts: tuple[ConflictContrast, ...] = CONFLICT_CONTRASTS,
) -> list[dict[str, Any]]:
    """Build endpoint, layer treatment, and exact-count identity-control masks."""
    if random_seeds < 1:
        raise ValueError("random_seeds must be positive.")
    if any(contrast.base not in winners or contrast.donor not in winners for contrast in contrasts):
        raise ValueError("A frozen conflict contrast refers to an unknown winner.")
    variants: list[dict[str, Any]] = []

    def add(
        family: str,
        variant_id: str,
        kept: LayerSets,
        contrast: ConflictContrast,
        metadata: dict[str, Any],
    ) -> None:
        validate_kept(kept, layer_dims)
        variants.append(
            {
                "family": family,
                "variant_id": variant_id,
                "target": contrast.target,
                "kept": kept,
                "kept_sha256": kept_sha256(kept),
                "mask_summary": summarize_kept(kept, layer_dims),
                "metadata": {"contrast": asdict(contrast), **metadata},
            }
        )

    for contrast in contrasts:
        base = winners[contrast.base]
        donor = winners[contrast.donor]
        donor_increment = layerwise_difference(donor, base)
        union = layerwise_union(base, donor)
        outside_union = complement_pool(union, layer_dims)
        if any(not donor_increment[layer] for layer in layer_dims):
            raise ValueError(f"Contrast {contrast.contrast_id} has an empty donor increment layer.")

        add(
            "endpoint",
            f"endpoint__{contrast.contrast_id}__safe_base",
            base,
            contrast,
            {"endpoint": "safe_base", "intervention": None, "layer": None, "replicate": None},
        )
        add(
            "endpoint",
            f"endpoint__{contrast.contrast_id}__failed_union",
            union,
            contrast,
            {"endpoint": "failed_union", "intervention": None, "layer": None, "replicate": None},
        )

        for layer in sorted(layer_dims):
            layer_increment = one_layer(donor_increment, layer)
            increment_count = len(layer_increment[layer])
            common_metadata = {
                "endpoint": None,
                "layer": layer,
                "increment_count": increment_count,
                "donor_increment_sha256": kept_sha256(layer_increment),
            }
            add(
                "layer_localization",
                f"localize__{contrast.contrast_id}__layer_{layer:02d}__add__treatment",
                layerwise_union(base, layer_increment),
                contrast,
                {**common_metadata, "intervention": "add_one_layer", "control": False, "replicate": None},
            )
            add(
                "layer_localization",
                f"localize__{contrast.contrast_id}__layer_{layer:02d}__loo__treatment",
                layerwise_difference(union, layer_increment),
                contrast,
                {**common_metadata, "intervention": "leave_one_layer_out", "control": False, "replicate": None},
            )
            counts = {
                candidate_layer: increment_count if candidate_layer == layer else 0 for candidate_layer in layer_dims
            }
            for replicate in range(random_seeds):
                control_seed = derive_seed(seed, contrast.contrast_id, layer, replicate)
                random_addition = deterministic_sample(outside_union, counts, seed=control_seed)
                random_removal = deterministic_sample(base, counts, seed=derive_seed(control_seed, "loo"))
                add(
                    "layer_localization",
                    f"localize__{contrast.contrast_id}__layer_{layer:02d}__add__random_r{replicate:02d}",
                    layerwise_union(base, random_addition),
                    contrast,
                    {
                        **common_metadata,
                        "intervention": "add_one_layer",
                        "control": True,
                        "replicate": replicate,
                        "control_pool": "outside_failed_union",
                    },
                )
                add(
                    "layer_localization",
                    f"localize__{contrast.contrast_id}__layer_{layer:02d}__loo__random_r{replicate:02d}",
                    layerwise_difference(union, random_removal),
                    contrast,
                    {
                        **common_metadata,
                        "intervention": "leave_one_layer_out",
                        "control": True,
                        "replicate": replicate,
                        "control_pool": "safe_base_resident",
                    },
                )

    variant_ids = [variant["variant_id"] for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise RuntimeError("Phase 6D generated duplicate variant IDs.")
    return variants
