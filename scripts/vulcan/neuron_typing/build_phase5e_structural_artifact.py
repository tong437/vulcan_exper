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

"""Freeze a Phase 5E static or learned semantic-pass mask for physical screening."""

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
from phase5e_semantics import SEMANTIC_CONTRACT_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a physical Phase 5E singleton artifact.")
    parser.add_argument("--frontier", required=True, help="Phase 5E static_frontier.json or learned_frontier.json.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate", default=None, help="Condition/run name; default uses the frontier best entry.")
    return parser.parse_args()


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _select_candidate(frontier: dict[str, Any], candidate: str | None) -> tuple[str, dict[str, Any], str]:
    if "runs" in frontier:
        selected = candidate or (frontier.get("best_automatic_semantic_pass") or {}).get("run")
        collection = frontier["runs"]
        source = "learned"
    elif "conditions" in frontier:
        if candidate is None:
            best_rows = list((frontier.get("best_semantic_pass") or {}).values())
            selected = max(best_rows, key=lambda row: row["deletion_budget"])["condition"] if best_rows else None
        else:
            selected = candidate
        collection = frontier["conditions"]
        source = "static"
    else:
        raise ValueError("Unrecognized Phase 5E frontier schema.")
    if selected not in collection:
        raise ValueError(f"Unknown or missing Phase 5E candidate: {selected!r}.")
    return selected, collection[selected], source


def build_artifact(frontier_path: str | Path, candidate: str | None = None) -> tuple[dict, list, dict]:
    frontier_path = Path(frontier_path).resolve()
    frontier = _load_json(frontier_path)
    if not frontier.get("complete") or frontier.get("config", {}).get("trace_target") != "gold_caption":
        raise ValueError("A complete gold-caption Phase 5E frontier is required.")
    selected, row, source = _select_candidate(frontier, candidate)
    if not row.get("automatic_semantic_pass"):
        raise ValueError(f"Phase 5E-C requires an automatic semantic-pass candidate, got {selected!r}.")
    row_contract_version = row.get("generation", {}).get("semantic", {}).get("contract_version")
    if row_contract_version != SEMANTIC_CONTRACT_VERSION:
        raise ValueError(
            "Phase 5E candidate uses a stale semantic contract: "
            f"{row_contract_version!r} != {SEMANTIC_CONTRACT_VERSION!r}."
        )

    if source == "learned":
        static_path = resolve_existing_path(frontier["config"]["static_frontier"], relative_to=frontier_path)
        static = _load_json(static_path)
    else:
        static_path = frontier_path
        static = frontier
    mask_path = resolve_existing_path(row["mask_file"], relative_to=frontier_path)
    neuron_ids = _load_json(mask_path)
    if canonical_json_sha256(neuron_ids) != row["mask_hash"]:
        raise ValueError("Phase 5E candidate mask hash mismatch.")
    layer_dims = {int(layer): int(width) for layer, width in static["layer_widths"].items()}
    masks = neuron_ids_to_masks(neuron_ids, layer_dims)
    cluster_idx = build_partial_singleton_cluster_idx(masks)
    validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
    budget = int(row["deletion_budget"])
    if validation["mask_summary"]["total_pruned"] != budget:
        raise ValueError("Mask deletion count does not match the frontier candidate budget.")
    removed_parameters = int(row["parameter_summary"]["removed_parameters"])
    if removed_parameters % (3 * budget):
        raise ValueError("Cannot infer an integral gated-MLP hidden size from the parameter summary.")
    hidden_size = removed_parameters // (3 * budget)
    config_path = resolve_existing_path(static["config"]["config_path"], relative_to=static_path)
    metadata = {
        "phase": "5E-C",
        "candidate": selected,
        "source_search": source,
        "source_frontier": str(frontier_path),
        "source_frontier_sha256": sha256_file(frontier_path),
        "static_frontier": str(static_path),
        "static_frontier_sha256": sha256_file(static_path),
        "source_mask_file": str(mask_path),
        "model_name_or_path": static["config"]["model_name_or_path"],
        "config_path": str(config_path),
        "reference_caption": static["config"]["reference_caption"],
        "semantic_contract_version": row["generation"]["semantic"]["contract_version"],
        "mask_sha256": canonical_json_sha256(neuron_ids),
        "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
        "deletion_budget": budget,
        "layer_dims": layer_dims,
        "target_layer_dims": target_layer_dims(masks),
        "validation": validation,
        "theoretical_reduction": theoretical_mlp_parameter_reduction(masks, hidden_size),
        "source_gold_proxy": row["gold_proxy"],
        "source_generation": row["generation"],
        "source_automatic_semantic_pass": True,
        "source_human_confirmation": row.get("human_confirmation"),
    }
    return neuron_ids, cluster_idx, metadata


def main() -> None:
    args = parse_args()
    neuron_ids, cluster_idx, metadata = build_artifact(args.frontier, args.candidate)
    artifact_dir = Path(args.output_dir).resolve() / "artifact"
    _write_json(artifact_dir / "mask.json", neuron_ids)
    _write_json(artifact_dir / "cluster_idx.json", cluster_idx)
    _write_json(artifact_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "candidate": metadata["candidate"],
                "deletion_budget": metadata["deletion_budget"],
                "output": str(artifact_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
