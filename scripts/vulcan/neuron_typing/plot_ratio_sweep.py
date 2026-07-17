"""Plot Phase-2 ratio sweep curves: ΔNLL vs ablation ratio."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> dict[str, float]:
    with path.open() as f:
        data = json.load(f)
    baseline_nll = data["metrics"]["none"]["nll"]
    return {name: m["nll"] - baseline_nll for name, m in data["metrics"].items()}


def parse_ratio(name: str) -> float | None:
    parts = name.split(":")
    if parts[0] in ("none",):
        return 0.0
    try:
        return float(parts[1])
    except (IndexError, ValueError):
        return None


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: plot_ratio_sweep.py <ratio_sweep_v2.json> [layer_random_5_80.json]")
        sys.exit(1)

    sweep_path = Path(sys.argv[1])
    sweep = load_metrics(sweep_path)

    lr_extra = {}
    if len(sys.argv) >= 3:
        lr_extra = load_metrics(Path(sys.argv[2]))

    # Collect typed curves (deterministic, single value per ratio)
    typed_names = ["multimodal", "unknown", "unknown_safe"]
    typed_curves: dict[str, list[tuple[float, float]]] = {t: [(0.0, 0.0)] for t in typed_names}
    for name, dnll in sweep.items():
        if name == "none":
            continue
        ratio = parse_ratio(name)
        if ratio is None:
            continue
        for t in typed_names:
            if name.startswith(t + ":") or name == t:
                typed_curves[t].append((ratio, dnll))

    # Collect random multi-seed data
    random_points: dict[float, list[float]] = {0.0: [0.0]}
    for name, dnll in sweep.items():
        if "seed" in name and name.startswith("random:"):
            ratio = parse_ratio(name)
            if ratio is not None:
                random_points.setdefault(ratio, []).append(dnll)

    # Merge layer_random extra data (same as random with different seeds)
    for name, dnll in lr_extra.items():
        if name == "none":
            continue
        if "seed" in name and name.startswith("layer_random:"):
            ratio = parse_ratio(name)
            if ratio is not None:
                random_points.setdefault(ratio, []).append(dnll)

    # Compute random mean ± std
    random_ratios = sorted(random_points.keys())
    random_mean = [np.mean(random_points[r]) for r in random_ratios]
    random_std = [np.std(random_points[r]) if len(random_points[r]) > 1 else 0.0 for r in random_ratios]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Random band
    random_mean_arr = np.array(random_mean)
    random_std_arr = np.array(random_std)
    ax.fill_between(
        random_ratios,
        random_mean_arr - random_std_arr,
        random_mean_arr + random_std_arr,
        alpha=0.2, color="gray", label="random ± std"
    )
    ax.plot(random_ratios, random_mean_arr, "o-", color="gray", linewidth=2, markersize=6, label="random mean")

    # Typed curves
    colors = {"multimodal": "#e74c3c", "unknown": "#3498db", "unknown_safe": "#2ecc71"}
    markers = {"multimodal": "s", "unknown": "^", "unknown_safe": "D"}
    for t in typed_names:
        pts = sorted(typed_curves[t])
        x, y = zip(*pts)
        ax.plot(x, y, f"{markers[t]}-", color=colors[t], linewidth=2, markersize=7, label=t)

    # Reference lines
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Ablation Ratio", fontsize=13)
    ax.set_ylabel("ΔNLL (vs baseline)", fontsize=13)
    ax.set_title("Phase-2 Ratio Sweep: ΔNLL vs Ablation Ratio", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add annotation for regularization zone
    ax.annotate(
        "regularization zone\n(ΔNLL < 0)",
        xy=(0.15, -0.5), fontsize=10, color="gray", ha="center", style="italic"
    )

    plt.tight_layout()
    out_path = sweep_path.parent / "ratio_sweep_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
