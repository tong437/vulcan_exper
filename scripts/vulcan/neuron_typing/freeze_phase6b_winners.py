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

"""Freeze Phase 6B winner masks and structural singleton artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase3_structural_utils import canonical_json_sha256, neuron_ids_to_masks, resolve_existing_path  # noqa: E402
from phase5_structural_utils import (  # noqa: E402
    build_partial_singleton_cluster_idx,
    target_layer_dims,
    validate_partial_singleton_cluster_idx,
)
from phase6b_semantics import evaluate_contract_semantics  # noqa: E402
from run_phase5_single_sample_frontier import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze the selected Phase 6B single-sample masks.")
    parser.add_argument("--source_file", default="data/phase6b_single_samples/winner_sources.json")
    parser.add_argument("--sample_file", default="data/phase6b_single_samples/frozen_samples.json")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_row(result: dict[str, Any], sample_id: str, run_name: str) -> dict[str, Any]:
    if "samples" in result:
        return result["samples"][sample_id]["runs"][run_name]
    return result["runs"][run_name]


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_file).resolve()
    sample_path = Path(args.sample_file).resolve()
    source = load_json(source_path)
    sample_payload = load_json(sample_path)
    samples = {sample["sample_id"]: sample for sample in sample_payload["samples"]}
    if {row["sample_id"] for row in source["winners"]} != set(samples):
        raise ValueError("Winner sources must cover every frozen sample exactly once.")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    winners = []
    for selection in source["winners"]:
        sample_id = selection["sample_id"]
        sample = samples[sample_id]
        result_path = resolve_existing_path(selection["result_file"], relative_to=source_path)
        result = load_json(result_path)
        row = result_row(result, sample_id, selection["run"])
        stop = row["generation"]["stop"]
        semantic = evaluate_contract_semantics(
            row["generation"]["raw_text"],
            contract=sample["contract"],
            terminated_normally=stop["terminated_normally"],
            hit_token_limit=stop["hit_token_limit"],
        )
        if not semantic["automatic_pass"]:
            raise ValueError(f"Selected winner no longer passes the current strict contract: {sample_id}.")
        mask_path = resolve_existing_path(row["mask_file"], relative_to=result_path)
        neuron_ids = load_json(mask_path)
        if canonical_json_sha256(neuron_ids) != row["mask_hash"]:
            raise ValueError(f"Mask hash mismatch for {sample_id}.")
        kept_by_layer = row["mask_summary"]["kept_by_layer"]
        layer_dims = {int(layer): int(kept_by_layer[layer]) + len(neuron_ids[layer]) for layer in kept_by_layer}
        masks = neuron_ids_to_masks(neuron_ids, layer_dims)
        cluster_idx = build_partial_singleton_cluster_idx(masks)
        validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
        sample_dir = output_dir / sample_id
        write_json(sample_dir / "mask.json", neuron_ids)
        write_json(sample_dir / "cluster_idx.json", cluster_idx)
        metadata = {
            "phase": "6B-frozen-winner",
            "sample_id": sample_id,
            "semantic_stratum": sample["semantic_stratum"],
            "image": sample["image"],
            "gold_caption": sample["gold_caption"],
            "contract": sample["contract"],
            "source_result": str(result_path),
            "source_run": selection["run"],
            "source_mask": str(mask_path),
            "mask_sha256": row["mask_hash"],
            "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
            "deletion_budget": row["deletion_budget"],
            "kept_neurons": row["mask_summary"]["kept_neurons"],
            "keep_ratio": row["mask_summary"]["keep_ratio"],
            "layer_dims": layer_dims,
            "target_layer_dims": target_layer_dims(masks),
            "validation": validation,
            "source_parameter_summary": row["parameter_summary"],
            "source_gold_proxy": row["gold_proxy"],
            "source_generation": {**row["generation"], "semantic": semantic},
        }
        write_json(sample_dir / "metadata.json", metadata)
        winners.append(metadata)
    manifest = {
        "complete": True,
        "selection_rule": source["selection_rule"],
        "source_file": str(source_path),
        "sample_file": str(sample_path),
        "canonical_prompt": sample_payload["canonical_prompt"],
        "winners": winners,
    }
    write_json(output_dir / "frozen_winners.json", manifest)
    print(json.dumps({row["sample_id"]: row["deletion_budget"] for row in winners}, indent=2))


if __name__ == "__main__":
    main()
