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

"""Freeze a strict Phase-5 core-extension candidate as a structural artifact."""

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
    parser = argparse.ArgumentParser(description="Build an artifact from a Phase-5 core-extension frontier.")
    parser.add_argument("--extension_frontier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate", default=None, help="Candidate name; default uses best_strict_candidate.")
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_core_extension_artifact(
    frontier_path: str | Path, candidate_name: str | None = None
) -> tuple[dict[str, list[int]], list[Any], dict[str, Any]]:
    frontier_path = Path(frontier_path).resolve()
    frontier = load_json(frontier_path)
    if not frontier.get("complete"):
        raise ValueError("Phase-5 core-extension frontier is incomplete.")
    selected = candidate_name or (frontier.get("best_strict_candidate") or {}).get("candidate")
    if selected not in frontier.get("candidates", {}):
        raise ValueError(f"Unknown or missing core-extension candidate: {selected!r}.")
    candidate = frontier["candidates"][selected]
    fidelity = candidate.get("cached_fidelity")
    generation = candidate.get("generation", {})
    if not candidate.get("strict_feasible") or fidelity is None:
        raise ValueError(f"Structural export requires a strict core-extension candidate, got {selected!r}.")
    if not generation.get("exact_match") or fidelity.get("token_agreement") != 1.0:
        raise ValueError(f"Core-extension candidate {selected!r} does not preserve the frozen trajectory.")
    if not all(candidate.get("structure_checks", {}).values()):
        raise ValueError(f"Core-extension candidate {selected!r} failed physical structure checks.")

    base_artifact_dir = Path(frontier["config"]["base_artifact_dir"]).resolve()
    base_metadata_path = base_artifact_dir / "metadata.json"
    base_metadata = load_json(base_metadata_path)
    if sha256_file(base_metadata_path) != frontier["config"]["base_metadata_sha256"]:
        raise ValueError("Base artifact metadata changed after the extension scan.")
    static_path = resolve_existing_path(frontier["config"]["static_frontier"], relative_to=frontier_path)
    mask_path = resolve_existing_path(candidate["mask_file"], relative_to=frontier_path)
    neuron_ids = load_json(mask_path)
    if canonical_json_sha256(neuron_ids) != candidate["mask_sha256"]:
        raise ValueError("Core-extension mask file does not match the candidate mask hash.")

    layer_dims = {int(layer): int(width) for layer, width in base_metadata["layer_dims"].items()}
    masks = neuron_ids_to_masks(neuron_ids, layer_dims)
    cluster_idx = build_partial_singleton_cluster_idx(masks)
    validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
    deletion_budget = int(candidate["total_deletion_count"])
    if validation["mask_summary"]["total_pruned"] != deletion_budget:
        raise ValueError("Core-extension mask deletion count does not match the candidate record.")
    deletions_by_layer = {
        str(layer): len(neuron_ids)
        for layer, neuron_ids in candidate["deleted_neuron_ids_by_layer"].items()
        if neuron_ids
    }
    if sum(deletions_by_layer.values()) != deletion_budget:
        raise ValueError("Core-extension per-layer deletion counts do not match the candidate deletion budget.")

    metadata = {
        **base_metadata,
        "phase": "5F-core-extension",
        "run": selected,
        "source_frontier_kind": "core_extension_physical",
        "source_gate": "structural_strict",
        "source_strict_feasible": True,
        "source_behavioral_feasible": bool(candidate.get("behavior_preserved")),
        "source_physical_probe_eligible": True,
        # Compatibility field consumed by the standard Phase-5 screeners.
        "learned_frontier": str(frontier_path),
        "learned_frontier_sha256": sha256_file(frontier_path),
        "extension_frontier": str(frontier_path),
        "extension_frontier_sha256": sha256_file(frontier_path),
        "base_artifact_dir": str(base_artifact_dir),
        "base_metadata_sha256": sha256_file(base_metadata_path),
        "static_frontier": str(static_path),
        "static_frontier_sha256": sha256_file(static_path),
        "source_mask_file": str(mask_path),
        "mask_sha256": canonical_json_sha256(neuron_ids),
        "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
        "deletion_budget": deletion_budget,
        "deletions_by_layer": deletions_by_layer,
        "added_layer": int(candidate["added_layer"]),
        "added_neuron": int(candidate["added_neuron"]),
        "candidate_rank": int(candidate["candidate_rank"]),
        "deleted_neuron_ids_by_layer": candidate["deleted_neuron_ids_by_layer"],
        "target_layer_dims": target_layer_dims(masks),
        "validation": validation,
        "theoretical_reduction": theoretical_mlp_parameter_reduction(
            masks, int(base_metadata["theoretical_reduction"]["hidden_size"])
        ),
        "kl_tolerance": float(frontier["config"]["kl_tolerance"]),
        "frozen_search_horizon": int(frontier["config"]["max_new_tokens"]),
        "source_fidelity": fidelity,
        "source_fidelity_path": "cached",
        "phase5b_teacher_fidelity": None,
        "phase5b_generation": generation,
    }
    return neuron_ids, cluster_idx, metadata


def main() -> None:
    args = parse_args()
    neuron_ids, cluster_idx, metadata = build_core_extension_artifact(args.extension_frontier, args.candidate)
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
