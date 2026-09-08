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

"""Freeze a feasible Phase-5 learned mask as a structural pruning artifact."""

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
    parser = argparse.ArgumentParser(description="Build a Phase-5C structural singleton artifact.")
    parser.add_argument("--learned_frontier", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run", default=None, help="Feasible run name; default uses best_feasible.")
    parser.add_argument(
        "--allow_behavioral",
        action="store_true",
        help="Allow exact-generation, KL-feasible runs that fail only the strict full-forward agreement gate.",
    )
    parser.add_argument(
        "--allow_physical_probe",
        action="store_true",
        help="Allow cached top-1/exact-generation candidates above the KL limit for diagnostic physical screening.",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_artifact(
    learned_path: str | Path,
    run_name: str | None = None,
    *,
    allow_behavioral: bool = False,
    allow_physical_probe: bool = False,
) -> tuple[dict[str, Any], list[Any], dict]:
    learned_path = Path(learned_path).resolve()
    learned = load_json(learned_path)
    if not learned.get("complete"):
        raise ValueError("Phase-5B learned frontier is incomplete.")
    if run_name is not None:
        selected = run_name
    elif allow_behavioral:
        selected = (learned.get("best_behavioral_feasible") or {}).get("run") or (
            learned.get("best_feasible") or {}
        ).get("run")
    else:
        selected = (
            (learned.get("best_robust_feasible") or {}).get("run")
            or (learned.get("best_strict_feasible") or {}).get("run")
            or (learned.get("best_feasible") or {}).get("run")
        )
    if selected not in learned.get("runs", {}):
        raise ValueError(f"Unknown or missing Phase-5B run: {selected!r}.")
    run = learned["runs"][selected]
    fidelity = run.get("cached_fidelity") or run.get("teacher_fidelity")
    if fidelity is None:
        raise ValueError("Phase-5 run contains neither cached_fidelity nor teacher_fidelity.")
    strict_feasible = bool(run.get("strict_feasible", run.get("feasible", False)))
    behavioral_feasible = bool(
        run.get(
            "behavioral_feasible",
            run["generation"]["exact_match"] and fidelity["mean_kl"] <= learned["config"]["kl_tolerance"],
        )
    )
    physical_probe_eligible = bool(
        "cached_fidelity" in run
        and run["generation"]["exact_match"]
        and fidelity.get("token_agreement") == 1.0
        and fidelity.get("generated_token_agreement") == 1.0
    )
    if (
        not strict_feasible
        and not (allow_behavioral and behavioral_feasible)
        and not (allow_physical_probe and physical_probe_eligible)
    ):
        raise ValueError(f"Phase-5C requires a strict-feasible run, got {selected!r}.")

    static_path = resolve_existing_path(learned["config"]["static_frontier"], relative_to=learned_path)
    static = load_json(static_path)
    mask_path = resolve_existing_path(run["mask_file"], relative_to=learned_path)
    neuron_ids = load_json(mask_path)
    if canonical_json_sha256(neuron_ids) != run["mask_hash"]:
        raise ValueError("Learned mask file does not match the run mask hash.")

    layer_dims = {int(layer): int(width) for layer, width in static["layer_widths"].items()}
    masks = neuron_ids_to_masks(neuron_ids, layer_dims)
    cluster_idx = build_partial_singleton_cluster_idx(masks)
    validation = validate_partial_singleton_cluster_idx(cluster_idx, masks)
    if validation["mask_summary"]["total_pruned"] != run["deletion_budget"]:
        raise ValueError("Mask deletion count does not match the learned run budget.")

    model_path = static["config"]["model_name_or_path"]
    config_path = resolve_existing_path(static["config"]["config_path"], relative_to=static_path)
    hidden_size = run["parameter_summary"]["removed_parameters"] // (3 * run["deletion_budget"])
    metadata = {
        "phase": "5C",
        "run": selected,
        "source_gate": ("strict" if strict_feasible else "behavioral" if behavioral_feasible else "physical_probe"),
        "source_strict_feasible": strict_feasible,
        "source_behavioral_feasible": behavioral_feasible,
        "source_physical_probe_eligible": physical_probe_eligible,
        "learned_frontier": str(learned_path),
        "learned_frontier_sha256": sha256_file(learned_path),
        "static_frontier": str(static_path),
        "static_frontier_sha256": sha256_file(static_path),
        "source_mask_file": str(mask_path),
        "model_name_or_path": model_path,
        "config_path": str(config_path),
        "mask_sha256": canonical_json_sha256(neuron_ids),
        "cluster_idx_sha256": canonical_json_sha256(cluster_idx),
        "deletion_budget": run["deletion_budget"],
        "layer_dims": layer_dims,
        "target_layer_dims": target_layer_dims(masks),
        "validation": validation,
        "theoretical_reduction": theoretical_mlp_parameter_reduction(masks, hidden_size),
        "source_fidelity": fidelity,
        "source_fidelity_path": "cached" if "cached_fidelity" in run else "full_forward",
        "phase5b_teacher_fidelity": run.get("teacher_fidelity"),
        "phase5b_generation": run["generation"],
    }
    return neuron_ids, cluster_idx, metadata


def main() -> None:
    args = parse_args()
    neuron_ids, cluster_idx, metadata = build_artifact(
        args.learned_frontier,
        args.run,
        allow_behavioral=args.allow_behavioral,
        allow_physical_probe=args.allow_physical_probe,
    )
    output_dir = Path(args.output_dir).resolve()
    artifact_dir = output_dir / "artifact"
    write_json(artifact_dir / "mask.json", neuron_ids)
    write_json(artifact_dir / "cluster_idx.json", cluster_idx)
    write_json(artifact_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {"run": metadata["run"], "deletion_budget": metadata["deletion_budget"], "output": str(artifact_dir)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
