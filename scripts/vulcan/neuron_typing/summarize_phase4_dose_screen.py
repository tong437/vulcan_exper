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

"""Select the largest language-safe candidate from a Phase-4 C4 dose screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Phase-4 protected deletion-dose screen.")
    parser.add_argument("--metadata_file", required=True)
    parser.add_argument("--c4_result", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_delta_nll", type=float, default=0.05)
    parser.add_argument("--max_protected_minus_q_nll", type=float, default=0.02)
    return parser.parse_args()


def _delta(metric: dict[str, Any], baseline_nll: float) -> float:
    return float(metric.get("delta_nll", float(metric["nll"]) - baseline_nll))


def summarize_screen(
    metadata: dict[str, Any],
    c4_result: dict[str, Any],
    *,
    max_delta_nll: float,
    max_protected_minus_q_nll: float,
) -> dict[str, Any]:
    metrics = c4_result["metrics"]
    if "none" not in metrics:
        raise ValueError("C4 result is missing the unablated baseline.")
    baseline_nll = float(metrics["none"]["nll"])
    doses: dict[str, Any] = {}
    passing: list[tuple[float, str]] = []
    for slug, condition in metadata["conditions"].items():
        columns = condition["columns"]
        q_name = f"mask:{columns['q_only']}"
        protected_name = f"mask:{columns['combined']}"
        missing = [name for name in (q_name, protected_name) if name not in metrics]
        if missing:
            raise ValueError(f"C4 result is missing dose conditions: {missing}")
        q_delta = _delta(metrics[q_name], baseline_nll)
        protected_delta = _delta(metrics[protected_name], baseline_nll)
        difference = protected_delta - q_delta
        checks = {
            "absolute_c4_safety": protected_delta <= max_delta_nll,
            "noninferior_to_equal_budget_q_only": difference <= max_protected_minus_q_nll,
        }
        passed = all(checks.values())
        ratio = float(condition["deletion_ratio"])
        if passed:
            passing.append((ratio, slug))
        doses[slug] = {
            "deletion_ratio": ratio,
            "q_only_condition": q_name,
            "protected_condition": protected_name,
            "q_only_delta_nll": q_delta,
            "protected_delta_nll": protected_delta,
            "protected_minus_q_only_delta_nll": difference,
            "protected_paired_ci95": [
                metrics[protected_name].get("paired_ci_lo"),
                metrics[protected_name].get("paired_ci_hi"),
            ],
            "checks": checks,
            "passed": passed,
        }
    selected_slug = max(passing)[1] if passing else None
    selected = None
    if selected_slug is not None:
        selected = {"slug": selected_slug, **doses[selected_slug]}
        selected["mapping_control_condition"] = f"mask:{metadata['conditions'][selected_slug]['columns']['mapping']}"
    return {
        "complete": True,
        "screen_passed": selected is not None,
        "structural_checkpoint_allowed": False,
        "baseline_nll": baseline_nll,
        "thresholds": {
            "max_delta_nll": max_delta_nll,
            "max_protected_minus_q_nll": max_protected_minus_q_nll,
        },
        "selection_rule": "largest deletion ratio passing both C4 checks",
        "doses": doses,
        "selected": selected,
        "warning": "Screen selection only authorizes Caption/POPE/VQA-Med hook evaluation, not structural pruning.",
    }


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_file)
    c4_path = Path(args.c4_result)
    result = summarize_screen(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        json.loads(c4_path.read_text(encoding="utf-8")),
        max_delta_nll=args.max_delta_nll,
        max_protected_minus_q_nll=args.max_protected_minus_q_nll,
    )
    result["inputs"] = {
        "metadata_file": str(metadata_path.resolve()),
        "c4_result": str(c4_path.resolve()),
    }
    output = Path(args.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
