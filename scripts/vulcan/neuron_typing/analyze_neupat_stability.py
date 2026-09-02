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

"""Measure NeuPAT role sensitivity to the cumulative-importance threshold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.train.vulcan import allocate_neupat_roles  # noqa: E402


ROLES = ("language", "multimodal", "shared", "reserve")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NeuPAT role stability over tau values.")
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--tau", action="append", type=float, default=[])
    parser.add_argument("--reference_tau", type=float, default=0.8)
    return parser.parse_args()


def overlap(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | int]:
    intersection = int((left & right).sum().item())
    union = int((left | right).sum().item())
    return {
        "reference_count": int(left.sum().item()),
        "comparison_count": int(right.sum().item()),
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
        "reference_recall": intersection / int(left.sum().item()) if bool(left.any()) else 1.0,
    }


def allocate_table_roles(table: pd.DataFrame, tau: float) -> dict[str, torch.Tensor]:
    required = {"layer", "neuron_idx", "neupat_text_importance", "neupat_vision_importance"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Score table is missing required columns: {missing}.")
    masks = {role: torch.zeros(len(table), dtype=torch.bool) for role in ROLES}
    for _, group in table.groupby("layer", sort=True):
        ordered = group.sort_values("neuron_idx")
        expected = list(range(len(ordered)))
        if ordered["neuron_idx"].astype(int).tolist() != expected:
            raise ValueError("Every layer must contain contiguous neuron_idx values starting at zero.")
        allocation = allocate_neupat_roles(
            torch.tensor(ordered["neupat_text_importance"].to_numpy()),
            torch.tensor(ordered["neupat_vision_importance"].to_numpy()),
            tau_text=tau,
            tau_vision=tau,
        )
        positions = torch.tensor(ordered.index.to_numpy(), dtype=torch.long)
        for role in ROLES:
            masks[role][positions[allocation[role]]] = True
    return masks


def build_stability_report(table: pd.DataFrame, taus: list[float], reference_tau: float) -> dict[str, Any]:
    if not taus:
        raise ValueError("At least one tau value is required.")
    if reference_tau not in taus:
        taus = [*taus, reference_tau]
    taus = sorted(set(taus))
    if any(not 0.0 < tau <= 1.0 for tau in taus):
        raise ValueError("All tau values must be in (0, 1].")

    allocations = {tau: allocate_table_roles(table, tau) for tau in taus}
    reference = allocations[reference_tau]
    reference_labels = torch.empty(len(table), dtype=torch.int8)
    for role_index, role in enumerate(ROLES):
        reference_labels[reference[role]] = role_index

    comparisons: dict[str, Any] = {}
    for tau in taus:
        masks = allocations[tau]
        labels = torch.empty(len(table), dtype=torch.int8)
        for role_index, role in enumerate(ROLES):
            labels[masks[role]] = role_index
        protected_reference = reference["language"] | reference["shared"]
        protected_comparison = masks["language"] | masks["shared"]
        comparisons[str(tau)] = {
            "role_counts": {role: int(masks[role].sum().item()) for role in ROLES},
            "exact_role_agreement": float((labels == reference_labels).float().mean().item()),
            "role_overlap": {role: overlap(reference[role], masks[role]) for role in ROLES},
            "language_protection_overlap": overlap(protected_reference, protected_comparison),
        }
    return {
        "artifact_version": 1,
        "method": "neupat_tau_stability",
        "neurons": len(table),
        "layers": int(table["layer"].nunique()),
        "reference_tau": reference_tau,
        "taus": taus,
        "comparisons": comparisons,
        "limitation": "Threshold stability does not replace an independent-sample probe replication.",
    }


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(args.score_file)
    report = build_stability_report(table, args.tau or [0.7, 0.8, 0.9], args.reference_tau)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
