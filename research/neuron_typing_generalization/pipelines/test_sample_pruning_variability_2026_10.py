#!/usr/bin/env python3
# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exploratory sample variability; no new generation or advisor-plan changes."""

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from report_global_cumulative_importance_2026_10 import BASE, DATASETS, MODELS, PercentFormatter, plt, save_csv


OUT = Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/sample_pruning_variability_test")
EPS = (0.05, 0.15, 0.30)
COLORS = ("#397DB5", "#D58A49", "#599D8C")
POPCOUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def sample_pairs(groups, count=2000):
    """Uniform unique sample pairs conditional on distinct group IDs."""
    rng = np.random.default_rng(20261022)
    pairs = set()
    while len(pairs) < count:
        i, j = sorted(rng.choice(len(groups), 2, replace=False).tolist())
        if groups[i] != groups[j]:
            pairs.add((i, j))
    return sorted(pairs)


def digest_mask(mask, offsets):
    digest = hashlib.sha256()
    for layer in range(len(offsets) - 1):
        indices = np.flatnonzero(mask[offsets[layer] : offsets[layer + 1]])
        digest.update(f"{layer}:".encode())
        digest.update(np.asarray(indices, dtype="<i8").tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def overlap_metrics(intersection, counts_a, counts_b, widths):
    a, b = sum(counts_a), sum(counts_b)
    expected_global = a * b / sum(widths)
    expected_layer = np.sum(np.asarray(counts_a) * np.asarray(counts_b) / widths)
    union = a + b - intersection
    # Plug-in Jaccard references, NOT exact E[Jaccard].
    return dict(
        jaccard=intersection / union if union else 1.0,
        global_random_reference=expected_global / (a + b - expected_global) if a + b else 1.0,
        layer_random_reference=expected_layer / (a + b - expected_layer) if a + b else 1.0,
        adjusted_overlap=(intersection - expected_layer) / (min(a, b) - expected_layer)
        if min(a, b) > expected_layer
        else np.nan,
    )


def plot(name, total, layer_rates, pair_rows, output):
    model, dataset = name.split("__")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), gridspec_kw={"width_ratios": [1, 1.45, 1.1]})
    fig.subplots_adjust(left=0.055, right=0.98, top=0.76, bottom=0.25, wspace=0.28)
    fig.suptitle("Sample variability of nonuniform pruning", fontsize=23, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.90,
        f"{MODELS[model]} | {DATASETS[dataset]} | 1,000 samples | 2,000 cross-group pairs",
        ha="center",
        color="#475569",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.84,
        "Same response budget; real neuron identities preserved for overlap tests",
        ha="center",
        color="#64748B",
        fontsize=11,
    )
    ax = axes[0]
    boxes = ax.boxplot(total.T, positions=range(3), widths=0.45, whis=(10, 90), showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks(range(3), [f"{v:.2f}" for v in EPS])
    ax.set(title="Total pruning-rate distribution", xlabel="Response budget ε", ylabel="Actual global pruning rate")
    lo, hi = total.min(), total.max()
    ax.set_ylim(max(0, lo - 0.07), min(1, hi + 0.1))
    for e in range(3):
        p10, p90 = np.percentile(total[e], [10, 90])
        ax.text(
            e,
            np.max(total[e]) + 0.025,
            f"Mean {total[e].mean():.2%}\nP10–90: {p10:.2%}–{p90:.2%}",
            ha="center",
            fontsize=8,
        )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax = axes[1]
    for e, (epsilon, color) in enumerate(zip(EPS, COLORS)):
        mean = layer_rates[e].mean(axis=0)
        p10, p90 = np.percentile(layer_rates[e], [10, 90], axis=0)
        ax.plot(range(len(mean)), mean, color=color, lw=1.8, label=f"ε={epsilon:.2f}")
        ax.fill_between(range(len(mean)), p10, p90, color=color, alpha=0.2)
    ax.set(
        title="Layer allocation: mean + P10–90",
        xlabel="Decoder layer (0-based)",
        ylabel="Pruning rate within layer",
        ylim=(0, 1),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=9)
    ax = axes[2]
    for e, color in enumerate(COLORS):
        rows = [r for r in pair_rows if r["epsilon"] == EPS[e]]
        data = [[r[key] for r in rows] for key in ("jaccard", "layer_random_reference")]
        box = ax.boxplot(
            data, positions=[e - 0.16, e + 0.16], widths=0.24, whis=(10, 90), showfliers=False, patch_artist=True
        )
        box["boxes"][0].set_facecolor(color)
        box["boxes"][1].set_facecolor("#E2E8F0")
        box["boxes"][1].set_hatch("///")
        for patch in box["boxes"]:
            patch.set_alpha(0.8)
    ax.set_xticks(range(3), [f"{v:.2f}" for v in EPS])
    ax.set(
        title="Pairwise neuron-set overlap", xlabel="Response budget ε", ylabel="Jaccard similarity", ylim=(0, 1.02)
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#397DB5", label="Observed (left)"),
            Patch(facecolor="#E2E8F0", hatch="///", label="Layer-matched reference (right)"),
        ],
        fontsize=8,
        loc="lower right",
    )
    for ax in axes:
        ax.grid(axis="y", color="#E4EAF0", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    fig.text(
        0.055,
        0.14,
        "Boxes: median / P25–75; whiskers: P10–90. Left axis is focused. Layer bands show sample spread, not CIs.",
        color="#64748B",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.09,
        "Reference = E[intersection] / (Ka + Kb − E[intersection]), preserving each sample's deletion count in every layer.",
        color="#64748B",
        fontsize=10,
    )
    fig.text(
        0.055,
        0.045,
        "Cross-group sample pairs are descriptive and not independent replicates. This test does not measure accuracy or speed.",
        color="#64748B",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def run(model, dataset, output):
    name = f"{model}__{dataset}"
    source = BASE / "control_response_2026_10_frozen" / name
    manifest = json.loads((source / "manifest.json").read_text())
    widths = np.array(manifest["d_ff_by_layer"], dtype=int)
    offsets = np.r_[0, np.cumsum(widths)]
    total_neurons = int(sum(widths))
    references = []
    provenance = {}
    for epsilon in EPS:
        tag = f"{epsilon:.2f}".replace(".", "p")
        p = BASE / "nonuniform_budget_2026_10" / f"{name}__eps_{tag}" / "predictions.jsonl"
        payload = p.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        rows = [r for r in rows if r["condition"] == "N"]
        refs = {r["sample_index"]: r for r in rows}
        if len(rows) != 1000 or set(refs) != set(range(1000)):
            raise ValueError("Incomplete or duplicate reference")
        references.append(refs)
        provenance[str(p)] = hashlib.sha256(payload).hexdigest()
    masks = np.empty((3, 1000, (total_neurons + 7) // 8), dtype=np.uint8)
    counts = np.empty((3, 1000, len(widths)), dtype=np.int64)
    sample_rows = []
    files = sorted((source / "sample_statistics").glob("*.npz"))
    if len(files) != 1000 or manifest["status"] != "complete":
        raise ValueError("Incomplete frozen statistics")
    for i, p in enumerate(files):
        if int(p.stem) != i:
            raise ValueError("Unexpected sample index")
        with np.load(p, allow_pickle=False) as data:
            q = data["q"]
            sid = str(data["sample_id"].item())
        costs = []
        for l, w in enumerate(widths):
            v = q[l, :, :w].astype(np.float64)
            if not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
                raise ValueError("Invalid q")
            s = v.sum(axis=0)
            costs.append(s / s.sum() / len(widths))
        costs = np.concatenate(costs)
        order = np.argsort(costs, kind="stable")
        cumulative = np.cumsum(costs[order])
        for e, epsilon in enumerate(EPS):
            ref = references[e][i]
            k = int(np.searchsorted(cumulative, epsilon + 1e-12, side="right"))
            mask = np.zeros(total_neurons, dtype=bool)
            mask[order[:k]] = True
            layer_counts = np.add.reduceat(mask.astype(np.int64), offsets[:-1])
            if sid != ref["sample_id"] or digest_mask(mask, offsets) != ref["selection_sha256"]:
                raise ValueError(f"Mask/identity mismatch {name}/{i}/{epsilon}")
            if k != ref["selected_neurons"] or not np.array_equal(layer_counts, ref["selected_per_layer"]):
                raise ValueError("Count mismatch")
            counts[e, i] = layer_counts
            masks[e, i] = np.packbits(mask)
            sample_rows.append(
                dict(
                    sample_index=i,
                    sample_id=sid,
                    group_id=ref["group_id"],
                    epsilon=epsilon,
                    actual_ratio=k / total_neurons,
                )
            )
        if (i + 1) % 200 == 0:
            print(f"{name}: reconstructed and digest-verified {i + 1}/1000", flush=True)
    groups = [references[0][i]["group_id"] for i in range(1000)]
    pairs = sample_pairs(groups)
    pair_rows = []
    for e, epsilon in enumerate(EPS):
        for i, j in pairs:
            intersection = int(POPCOUNT[np.bitwise_and(masks[e, i], masks[e, j])].sum())
            metrics = overlap_metrics(intersection, counts[e, i], counts[e, j], widths)
            pair_rows.append(dict(epsilon=epsilon, index_a=i, index_b=j, intersection=intersection, **metrics))
    total = counts.sum(axis=2) / total_neurons
    layer_rates = counts / widths
    layer_rows = []
    summaries = []
    for e, epsilon in enumerate(EPS):
        for l in range(len(widths)):
            vals = layer_rates[e, :, l]
            layer_rows.append(
                dict(
                    epsilon=epsilon,
                    layer=l,
                    width=int(widths[l]),
                    mean=vals.mean(),
                    sd=vals.std(ddof=1),
                    p10=np.quantile(vals, 0.1),
                    p90=np.quantile(vals, 0.9),
                )
            )
        pr = [r for r in pair_rows if r["epsilon"] == epsilon]
        summaries.append(
            dict(
                model=model,
                dataset=dataset,
                epsilon=epsilon,
                samples=1000,
                pairs=len(pairs),
                total_mean=total[e].mean(),
                total_sd=total[e].std(ddof=1),
                total_p10=np.quantile(total[e], 0.1),
                total_p90=np.quantile(total[e], 0.9),
                mean_layer_sd=layer_rates[e].std(axis=0, ddof=1).mean(),
                max_layer_sd=layer_rates[e].std(axis=0, ddof=1).max(),
                jaccard_mean=np.mean([r["jaccard"] for r in pr]),
                jaccard_p10=np.quantile([r["jaccard"] for r in pr], 0.1),
                jaccard_p90=np.quantile([r["jaccard"] for r in pr], 0.9),
                layer_reference_mean=np.mean([r["layer_random_reference"] for r in pr]),
                adjusted_overlap_mean=np.nanmean([r["adjusted_overlap"] for r in pr]),
            )
        )
    save_csv(output / f"{name}__samples.csv", sample_rows)
    save_csv(output / f"{name}__layers.csv", layer_rows)
    save_csv(output / f"{name}__pairs.csv", pair_rows)
    save_csv(output / f"{name}__summary.csv", summaries)
    plot(name, total, layer_rates, pair_rows, output)
    (output / f"{name}.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                reference_sha256=provenance,
                mask_digests_verified=3000,
                pair_seed=20261022,
                cross_group_pairs=len(pairs),
                summary=summaries,
            ),
            indent=2,
        )
        + "\n"
    )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--replot-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    if args.workers > 1 and not args.replot_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for model in MODELS:
                if args.model and args.model != model:
                    continue
                for dataset in DATASETS:
                    name = f"{model}__{dataset}"
                    done = OUT / f"{name}.json"
                    if args.resume and done.exists():
                        metadata = json.loads(done.read_text())
                        for path, expected in metadata["reference_sha256"].items():
                            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                                raise ValueError("Changed reference")
                        summaries.extend(metadata["summary"])
                    else:
                        futures.append(pool.submit(run, model, dataset, OUT))
            for future in futures:
                summaries.extend(future.result())
    for model in MODELS:
        if args.workers > 1 and not args.replot_only:
            break
        if args.model and args.model != model:
            continue
        for dataset in DATASETS:
            if args.replot_only:
                name = f"{model}__{dataset}"
                metadata = json.loads((OUT / f"{name}.json").read_text())
                manifest = json.loads((Path(metadata["source"]) / "manifest.json").read_text())
                widths = np.asarray(manifest["d_ff_by_layer"])
                counts = []
                for path, expected in metadata["reference_sha256"].items():
                    payload = Path(path).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != expected:
                        raise ValueError("Reference changed since analysis")
                    rows = [json.loads(line) for line in payload.splitlines()]
                    rows = sorted((r for r in rows if r["condition"] == "N"), key=lambda r: r["sample_index"])
                    counts.append([r["selected_per_layer"] for r in rows])
                counts = np.asarray(counts)
                with (OUT / f"{name}__pairs.csv").open() as handle:
                    pairs = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
                plot(name, counts.sum(axis=2) / widths.sum(), counts / widths, pairs, OUT)
                summaries.extend(metadata["summary"])
            else:
                summaries.extend(run(model, dataset, OUT))
    if not args.model:
        save_csv(OUT / "summary.csv", summaries)
        with ZipFile(OUT / "sample_variability_16_figures.zip", "w", ZIP_DEFLATED) as archive:
            for path in sorted(OUT.iterdir()):
                if path.suffix in (".png", ".pdf", ".json", ".csv", ".md"):
                    archive.write(path, path.name)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
