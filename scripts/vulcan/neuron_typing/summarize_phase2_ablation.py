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

"""Summarize Phase-2 ablation metrics relative to the no-ablation baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize run_phase2_ablation.py JSON metrics.")
    parser.add_argument("metrics_file", help="JSON output produced by run_phase2_ablation.py.")
    parser.add_argument("--baseline", default="none", help="Baseline metric key.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.metrics_file).open(encoding="utf-8") as f:
        payload = json.load(f)

    metrics = payload["metrics"]
    if args.baseline not in metrics:
        raise ValueError(f"Baseline {args.baseline!r} not found. Available keys: {list(metrics)}")

    baseline = metrics[args.baseline]
    header = f"{'ablation':36s} {'n':>7s} {'tokens':>10s} {'nll':>10s} {'Δnll':>10s} {'ppl':>10s} {'Δppl':>10s}"
    print(header)
    print("-" * len(header))
    for name, row in metrics.items():
        delta_nll = row["nll"] - baseline["nll"]
        delta_ppl = row["ppl"] - baseline["ppl"]
        print(
            f"{name:36s} "
            f"{row['num_examples']:7d} "
            f"{row['num_label_tokens']:10d} "
            f"{row['nll']:10.6f} "
            f"{delta_nll:+10.6f} "
            f"{row['ppl']:10.4f} "
            f"{delta_ppl:+10.4f}"
        )

    paired_rows = [(name, row) for name, row in metrics.items() if "paired_delta_nll" in row]
    if paired_rows:
        print("\nPaired per-example analysis")
        paired_header = (
            f"{'ablation':36s} {'paired Δnll':>12s} {'95% CI':>25s} "
            f"{'improved':>10s} {'damaged':>10s}"
        )
        print(paired_header)
        print("-" * len(paired_header))
        for name, row in paired_rows:
            ci = (
                f"[{row['paired_ci_lo']:+.6f}, {row['paired_ci_hi']:+.6f}]"
                if "paired_ci_lo" in row else "n/a"
            )
            print(
                f"{name:36s} {row['paired_delta_nll']:+12.6f} {ci:>25s} "
                f"{row['improved_frac']:10.2%} {row['damaged_frac']:10.2%}"
            )

    relative_damage = payload.get("relative_damage", {})
    if relative_damage:
        print("\nRelative damage versus matched-ratio random seeds")
        for name, row in relative_damage.items():
            print(
                f"{name:36s} relative_damage={row['relative_damage']:+.6f} "
                f"random_mean={row['mean_random_delta']:+.6f} z={row['z_score']:+.3f}"
            )


if __name__ == "__main__":
    main()
