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

"""Freeze a strict Phase-5 atomic physical candidate as a structural artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase3_structural_utils import (
    canonical_json_sha256,
    neuron_ids_to_masks,
    resolve_existing_path,
    sha256_file,
    theoretical_mlp_parameter_reduction,
)
from phase5_structural_utils import (
    build_partial_singleton_cluster_idx,
    target_layer_dims,
    validate_partial_singleton_cluster_idx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a structural artifact from a Phase-5 atomic frontier.")
    parser.add_argument("--atomic_frontier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate",
        default=None,
        help="Candidate name; default uses the frontier's best_strict_candidate.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def infer_hidden_size(static: dict[str, Any], layer_dims: dict[int, int]) -> int:
    """Infer the bias-free SwiGLU hidden width from the frozen parameter accounting."""
    removable = static.get("parameter_scope", {}).get("removable_by_layer", {})
    per_neuron_values = set()
    for layer, width in layer_dims.items():
        layer_parameters = int(removable.get(str(layer), 0))
        if layer_parameters <= 0 or layer_parameters % width:
            raise ValueError(f"Cannot infer per-neuron parameter count for layer {layer}.")
        per_neuron_values.add(layer_parameters // width)
    if len(per_neuron_values) != 1:
        raise ValueError(f"Atomic artifact requires a shared FFN parameterization, got {per_neuron_values}.")
    parameters_per_neuron = per_neuron_values.pop()
    if parameters_per_neuron % 3:
        raise ValueError(f"Expected three bias-free MLP projections per neuron, got {parameters_per_neuron}.")
    return parameters_per_neuron // 3


def build_atomic_artifact(
    atomic_path: str | Path, candidate_name: str | None = None
) -> tuple[dict[str, list[int]], list[Any], dict[str, Any]]:
    atomic_path = Path(atomic_path).resolve()
    atomic = load_json(atomic_path)
    if not atomic.get("complete"):
        raise ValueError("Phase-5 atomic physical frontier is incomplete.")

    selected = candidate_name or (atomic.get("best_strict_candidate") or {}).get("candidate")
    if selected not in atomic.get("candidates", {}):
        raise ValueError(f"Unknown or missing atomic candidate: {selected!r}.")
    candidate = atomic["candidates"][selected]
    fidelity = candidate.get("cached_fidelity")
    generation = candidate.get("generation", {})
    if not candidate.get("strict_feasible") or fidelity is None:
        raise ValueError(f"Structural export requires a strict atomic candidate, got {selected!r}.")
    if not generation.get("exact_match") or fidelity.get("token_agreement") != 1.0:
        raise ValueError(f"Atomic candidate {selected!r} does not preserve the frozen trajectory.")
    if not all(candidate.get("structure_checks", {}).values()):
        raise ValueError(f"Atomic candidate {selected!r} failed its physical structure checks.")

    static_path = resolve_existing_path(atomic["config"]["static_frontier"], relative_to=atomic_path)
    static = load_json(static_path)
    mask_path = resolve_existing_path(candidate["mask_file"], relative_to=atomic_path)
    neuron_ids = load_json(mask_path)
    if canonical_json_sha256(neuron_ids) != candidate["mask_sha256"]:
        raise ValueError("Atomic mask file does not match the candidate mask hash.")

    layer_dims = {int(layer): int(width) for layer, width in static["layer_widths"].items()}
    masks = neuron_ids_to_masks(neuron_ids, layer_dims)
    cluster_idx = build_partial_singleton_cluster_idx(masks)
    validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
    deletion_budget = int(candidate["total_deletion_count"])
    if validation["mask_summary"]["total_pruned"] != deletion_budget:
        raise ValueError("Atomic mask deletion count does not match the candidate record.")

    config_path = resolve_existing_path(static["config"]["config_path"], relative_to=static_path)
    kl_tolerance = float(static["config"]["kl_tolerance"])
    metadata = {
        "phase": "5E-atomic",
        "run": selected,
        "source_frontier_kind": "atomic_physical",
        "source_gate": "structural_strict",
        "source_strict_feasible": True,
        "source_behavioral_feasible": bool(candidate.get("behavior_preserved")),
        "source_physical_probe_eligible": True,
        # Keep the compatibility name consumed by the standard Phase-5 screeners.
        "learned_frontier": str(atomic_path),
        "learned_frontier_sha256": sha256_file(atomic_path),
        "atomic_frontier": str(atomic_path),
        "atomic_frontier_sha256": sha256_file(atomic_path),
        "static_frontier": str(static_path),
        "static_frontier_sha256": sha256_file(static_path),
        "source_mask_file": str(mask_path),
        "model_name_or_path": static["config"]["model_name_or_path"],
        "config_path": str(config_path),
        "mask_sha256": canonical_json_sha256(neuron_ids),
        "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
        "deletion_budget": deletion_budget,
        "deletions_by_layer": candidate["deletions_by_layer"],
        "deleted_neuron_ids_by_layer": candidate["deleted_neuron_ids_by_layer"],
        "layer_dims": layer_dims,
        "target_layer_dims": target_layer_dims(masks),
        "validation": validation,
        "theoretical_reduction": theoretical_mlp_parameter_reduction(masks, infer_hidden_size(static, layer_dims)),
        "kl_tolerance": kl_tolerance,
        "frozen_search_horizon": int(atomic["config"]["max_new_tokens"]),
        "source_fidelity": fidelity,
        "source_fidelity_path": "cached",
        "phase5b_teacher_fidelity": None,
        "phase5b_generation": generation,
    }
    return neuron_ids, cluster_idx, metadata


def main() -> None:
    args = parse_args()
    neuron_ids, cluster_idx, metadata = build_atomic_artifact(args.atomic_frontier, args.candidate)
    artifact_dir = Path(args.output_dir).resolve() / "artifact"
    write_json(artifact_dir / "mask.json", neuron_ids)
    write_json(artifact_dir / "cluster_idx.json", cluster_idx)
    write_json(artifact_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "candidate": metadata["run"],
                "deletion_budget": metadata["deletion_budget"],
                "output": str(artifact_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
