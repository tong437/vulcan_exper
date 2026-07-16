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

"""Statistical tests for neuron typing: blocked permutation test and bootstrap CI."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical tests for neuron typing.")
    parser.add_argument("--input_dir", required=True, help="Directory with neuron type scores.")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/stats).")
    parser.add_argument("--n_permutations", type=int, default=10000, help="Number of permutations.")
    parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap samples.")
    parser.add_argument("--ci_level", type=float, default=0.95, help="Confidence interval level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


FA_LAYERS = {3, 7, 11, 15, 19, 23}

ATTENTION_BLOCKS = [
    (3, [0, 1, 2]),
    (7, [4, 5, 6]),
    (11, [8, 9, 10]),
    (15, [12, 13, 14]),
    (19, [16, 17, 18]),
    (23, [20, 21, 22]),
]


def load_neuron_scores(input_dir: str) -> pd.DataFrame:
    parquet_path = Path(input_dir) / "neuron_type_scores.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"neuron_type_scores.parquet not found in {input_dir}")
    return pd.read_parquet(parquet_path)


def compute_layer_type_ratios(df: pd.DataFrame, threshold: float) -> dict[int, dict[str, float]]:
    """Compute per-layer ratios of high-confidence neurons of each type."""
    ratios = {}
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        total = len(layer_df)
        if total == 0:
            continue

        ratios[int(layer)] = {
            "visual": (layer_df["q_visual"] >= threshold).sum() / total,
            "text": (layer_df["q_text"] >= threshold).sum() / total,
            "multimodal": (layer_df["q_multimodal"] >= threshold).sum() / total,
            "unknown": (layer_df["q_unknown"] >= threshold).sum() / total,
        }

    return ratios


def blocked_permutation_test(
    layer_ratios: dict[int, dict[str, float]],
    neuron_type: str,
    n_permutations: int,
    seed: int,
) -> dict:
    """Blocked permutation test comparing FA vs GDN layers.

    Each block consists of one FA layer and its adjacent GDN layers.
    The test swaps FA/GDN labels within each block.

    Returns:
        dict with observed_diff, p_value, and permutation distribution
    """
    rng = np.random.RandomState(seed)

    blocks = []
    for fa_layer, gdn_layers in ATTENTION_BLOCKS:
        if fa_layer not in layer_ratios:
            continue

        fa_ratio = layer_ratios[fa_layer][neuron_type]
        gdn_ratios = [layer_ratios[l][neuron_type] for l in gdn_layers if l in layer_ratios]
        if not gdn_ratios:
            continue

        blocks.append((fa_ratio, np.mean(gdn_ratios)))

    if len(blocks) < 2:
        return {"error": "Insufficient blocks for permutation test", "n_blocks": len(blocks)}

    fa_obs = [b[0] for b in blocks]
    gd_obs = [b[1] for b in blocks]
    observed_diff = np.mean(fa_obs) - np.mean(gd_obs)

    n_blocks = len(blocks)
    max_exact_perms = 2 ** n_blocks
    use_exact = n_permutations >= max_exact_perms

    if use_exact:
        perm_diffs = []
        for i in range(max_exact_perms):
            perm_fa, perm_gd = [], []
            for j, (fa_r, gd_r) in enumerate(blocks):
                if (i >> j) & 1:
                    perm_fa.append(fa_r)
                    perm_gd.append(gd_r)
                else:
                    perm_fa.append(gd_r)
                    perm_gd.append(fa_r)
            perm_diffs.append(np.mean(perm_fa) - np.mean(perm_gd))
        perm_diffs = np.array(perm_diffs)
    else:
        perm_diffs = np.zeros(n_permutations)
        for i in range(n_permutations):
            perm_fa, perm_gd = [], []
            for fa_r, gd_r in blocks:
                if rng.random() < 0.5:
                    perm_fa.append(fa_r)
                    perm_gd.append(gd_r)
                else:
                    perm_fa.append(gd_r)
                    perm_gd.append(fa_r)
            perm_diffs[i] = np.mean(perm_fa) - np.mean(perm_gd)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    return {
        "neuron_type": neuron_type,
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n_blocks": n_blocks,
        "n_permutations": len(perm_diffs),
        "max_exact_perms": max_exact_perms,
        "used_exact": use_exact,
        "p_value_floor": 1.0 / max_exact_perms if use_exact else None,
        "perm_diffs_mean": float(np.mean(perm_diffs)),
        "perm_diffs_std": float(np.std(perm_diffs)),
    }


def bootstrap_ci(
    layer_ratios: dict[int, dict[str, float]],
    neuron_type: str,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> dict:
    """Bootstrap confidence interval for FA-GDN difference.

    Resamples layers within FA and GDN groups independently.
    """
    rng = np.random.RandomState(seed)

    fa_layers = [l for l in layer_ratios if l in FA_LAYERS]
    gdn_layers = [l for l in layer_ratios if l not in FA_LAYERS]

    if len(fa_layers) < 2 or len(gdn_layers) < 2:
        return {"error": "Insufficient layers for bootstrap", "n_fa": len(fa_layers), "n_gdn": len(gdn_layers)}

    fa_ratios = np.array([layer_ratios[l][neuron_type] for l in fa_layers])
    gdn_ratios = np.array([layer_ratios[l][neuron_type] for l in gdn_layers])

    observed_diff = np.mean(fa_ratios) - np.mean(gdn_ratios)

    boot_diffs = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        fa_sample = rng.choice(fa_ratios, size=len(fa_ratios), replace=True)
        gdn_sample = rng.choice(gdn_ratios, size=len(gdn_ratios), replace=True)
        boot_diffs[i] = np.mean(fa_sample) - np.mean(gdn_sample)

    alpha = 1 - ci_level
    ci_lo = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_hi = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

    return {
        "neuron_type": neuron_type,
        "observed_diff": float(observed_diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "n_fa_layers": len(fa_layers),
        "n_gdn_layers": len(gdn_layers),
        "boot_mean": float(np.mean(boot_diffs)),
        "boot_std": float(np.std(boot_diffs)),
    }


def run_all_tests(
    df: pd.DataFrame,
    threshold: float,
    n_permutations: int,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> dict:
    """Run all statistical tests for all neuron types."""
    layer_ratios = compute_layer_type_ratios(df, threshold)

    results = {}
    for neuron_type in ["visual", "text", "multimodal", "unknown"]:
        print(f"\nTesting {neuron_type} neurons...")

        perm_result = blocked_permutation_test(
            layer_ratios, neuron_type, n_permutations, seed
        )
        boot_result = bootstrap_ci(
            layer_ratios, neuron_type, n_bootstrap, ci_level, seed
        )

        results[neuron_type] = {
            "permutation_test": perm_result,
            "bootstrap_ci": boot_result,
            "layer_ratios": {
                str(l): ratios[neuron_type]
                for l, ratios in layer_ratios.items()
            },
        }

        if "error" not in perm_result:
            print(f"  Permutation test: diff={perm_result['observed_diff']:+.4f}, p={perm_result['p_value']:.4f}")
        if "error" not in boot_result:
            print(f"  Bootstrap CI: [{boot_result['ci_lo']:+.4f}, {boot_result['ci_hi']:+.4f}]")

    return results


def save_results(output_dir: str, results: dict, config: dict):
    """Save statistical test results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "perm_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved perm_test_results.json")

    config_path = output_path / "test_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def print_summary(results: dict):
    """Print a human-readable summary of statistical tests."""
    print(f"\n{'='*60}")
    print("Statistical Test Summary")
    print(f"{'='*60}")
    print(f"\n{'Type':<15} {'Diff':>8} {'p-value':>10} {'95% CI':>20}")
    print("-" * 55)

    for neuron_type in ["visual", "text", "multimodal", "unknown"]:
        r = results[neuron_type]
        perm = r["permutation_test"]
        boot = r["bootstrap_ci"]

        if "error" in perm:
            print(f"{neuron_type:<15} {'ERROR':>8} {perm['error']}")
            continue

        diff = f"{perm['observed_diff']:+.4f}"
        pval = f"{perm['p_value']:.4f}"
        if "error" in boot:
            ci = "N/A"
        else:
            ci = f"[{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}]"

        print(f"{neuron_type:<15} {diff:>8} {pval:>10} {ci:>20}")

    print("\nNote: n_blocks=6, p-value resolution floor = 1/64 ≈ 0.016")


def main():
    args = parse_args()
    output_dir = args.output_dir or str(Path(args.input_dir) / "stats")

    print(f"Loading neuron type scores from {args.input_dir}")
    df = load_neuron_scores(args.input_dir)

    config = {
        "input_dir": args.input_dir,
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
        "ci_level": args.ci_level,
        "seed": args.seed,
        "threshold": 0.7,
    }

    results = run_all_tests(
        df,
        threshold=0.7,
        n_permutations=args.n_permutations,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        seed=args.seed,
    )

    save_results(output_dir, results, config)
    print_summary(results)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
