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

"""Global per-sample cumulative costs, audited against frozen nonuniform pruning."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from report_cumulative_importance_2026_10 import DATASETS, MODELS, PercentFormatter, candidate_knee, plt


BASE = Path("artifacts/neuron_typing_generalization")
OUTPUT = Path("research/neuron_typing_generalization/reports/advisor_2026_09/figures/global_cumulative_importance")
EPSILONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def global_cumulative(q, widths):
    """Exact cumulative costs over real neurons; padding is never included."""
    if q.ndim != 3 or q.shape[:2] != (len(widths), 3):
        raise ValueError("Expected q[layers,3,max_width]")
    costs = []
    for layer, width in enumerate(widths):
        if not 0 < width <= q.shape[2]:
            raise ValueError("Invalid real layer width")
        values = q[layer, :, :width].astype(np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid observed q")
        scores = values.sum(axis=0)
        if scores.sum() <= 0:
            raise ValueError("Zero layer total")
        costs.append(scores / scores.sum() / len(widths))
    cumulative = np.r_[0.0, np.cumsum(np.sort(np.concatenate(costs)), dtype=np.float64)]
    if not np.isclose(cumulative[-1], 1, atol=1e-10, rtol=0):
        raise ValueError("Costs do not sum to one")
    return cumulative


def budget_count(cumulative, epsilon):
    """Match the prior experiment's exact discrete prefix and boundary tolerance."""
    return int(np.searchsorted(cumulative[1:], epsilon + 1e-12, side="right"))


def save_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(u, mean, p10, p90, model, dataset, count, stem):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 7.5))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.19, top=0.77)
    fig.suptitle("Global cumulative importance distribution", y=0.97, fontsize=20, fontweight="bold")
    fig.text(0.55, 0.90, f"{MODELS[model]} | {DATASETS[dataset]} | N = {count}", ha="center", color="#475569")
    fig.text(0.55, 0.855, "Globally sorted c = (qV + qT + qC) / layer total / L", ha="center", color="#64748B")
    ax.fill_between(u, p10, p90, color="#80BEE7", alpha=0.55, label="10–90% range across samples")
    ax.plot(u, mean, color="#397DB5", lw=2.7, label=r"Dataset mean $H_{\mathcal{D}}(u)$")
    ax.plot(u, u, "--", color="#94A3B8", lw=1.5, label="Uniform-cost reference")
    knee = candidate_knee(u, mean)
    x, y = u[knee], mean[knee]
    ax.plot([x, x], [y, x], ":", color="#D58A49", lw=2)
    ax.scatter([x], [y], color="#D58A49", s=55, zorder=5, label="Descriptive mean-curve knee")
    ax.annotate(
        f"Global fraction: {x:.1%}\nCumulative cost: {y:.1%}",
        (x, y),
        xytext=(0.08, 0.67),
        textcoords="axes fraction",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=.5", fc="white", ec="#E2E8F0"),
        arrowprops=dict(arrowstyle="->", color="#D58A49", lw=1.2),
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Lowest-cost neurons selected globally (u)",
        ylabel="Cumulative cost (equal-layer mean response share)",
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95, edgecolor="#E2E8F0")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(color="#E4EAF0", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    fig.text(
        0.12,
        0.075,
        "Band = sample variation, not a confidence interval; no artificial widening.",
        fontsize=9,
        color="#64748B",
    )
    fig.text(
        0.12,
        0.038,
        "The mean-curve knee is descriptive; safe pruning still requires accuracy tests.",
        fontsize=9,
        color="#64748B",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix("." + suffix), dpi=220, facecolor="white")
    plt.close(fig)


def report(model, dataset, output):
    name = f"{model}__{dataset}"
    source = BASE / "control_response_2026_10_frozen" / name
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    files = sorted((source / "sample_statistics").glob("*.npz"))
    if manifest["status"] != "complete" or manifest["samples"] != 1000 or len(files) != 1000:
        raise ValueError(f"Incomplete source {name}")
    if (manifest["model"], manifest["dataset"]) != (model, dataset):
        raise ValueError("Manifest identity mismatch")
    widths = list(map(int, manifest["d_ff_by_layer"]))
    references, provenance = {}, {}
    for epsilon in EPSILONS:
        tag = f"{epsilon:.2f}".replace(".", "p")
        path = BASE / "nonuniform_budget_2026_10" / f"{name}__eps_{tag}" / "predictions.jsonl"
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        rows = [row for row in rows if row["condition"] == "N"]
        refs = {row["sample_index"]: row for row in rows}
        if len(rows) != 1000 or len(refs) != 1000:
            raise ValueError(f"Incomplete or duplicate reference {path}")
        references[epsilon] = refs
        provenance[str(path)] = hashlib.sha256(payload).hexdigest()
    u = np.linspace(0, 1, 1001)
    curves, budget_rows, sample_ids = [], [], set()
    digest = hashlib.sha256()
    for index, path in enumerate(files):
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        with np.load(path, allow_pickle=False) as data:
            sample_id = str(data["sample_id"].item())
            cumulative = global_cumulative(data["q"], widths)
        if sample_id in sample_ids or int(path.stem) != index:
            raise ValueError("Duplicate identity or unexpected sample index")
        sample_ids.add(sample_id)
        total = len(cumulative) - 1
        curve = np.interp(u * total, np.arange(total + 1), cumulative)
        if (np.diff(curve) < -1e-12).any() or (curve > u + 1e-10).any():
            raise ValueError("Invalid cumulative curve")
        curves.append(curve)
        for epsilon in EPSILONS:
            k = budget_count(cumulative, epsilon)
            ref = references[epsilon][index]
            if ref["sample_id"] != sample_id or ref["selected_neurons"] != k:
                raise ValueError(f"Identity/count mismatch {name}/{index}/{epsilon}")
            if abs(ref["actual_score_share"] - cumulative[k]) > 1e-10:
                raise ValueError("Reference cost mismatch")
            budget_rows.append(
                dict(
                    sample_index=index,
                    sample_id=sample_id,
                    epsilon=epsilon,
                    selected_neurons=k,
                    actual_ratio=k / total,
                    actual_cost=cumulative[k],
                )
            )
        if (index + 1) % 100 == 0:
            print(f"{name}: {index + 1}/1000; six budgets audited", flush=True)
    curves = np.asarray(curves)
    mean = curves.mean(axis=0)
    p10, p90 = np.percentile(curves, (10, 90), axis=0)
    stem = output / name
    np.savetxt(
        stem.with_suffix(".csv"),
        np.column_stack([u, mean, p10, p90]),
        delimiter=",",
        header="u,dataset_mean,sample_p10,sample_p90",
        comments="",
    )
    save_csv(output / f"{name}__budgets.csv", budget_rows)
    knee = candidate_knee(u, mean)
    metadata = dict(
        model=model,
        dataset=dataset,
        samples=1000,
        total_neurons=sum(widths),
        widths=widths,
        grid_size=len(u),
        source_manifest=str(manifest_path),
        source_run_id=manifest["run_id"],
        statistics_sha256=digest.hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        reference_sha256=provenance,
        audited_budget_conditions=len(budget_rows),
        knee_u=float(u[knee]),
        knee_cost=float(mean[knee]),
        mean_band_width=float(np.mean(p90 - p10)),
    )
    plot(u, mean, p10, p90, model, dataset, 1000, stem)
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--replot-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for dataset in DATASETS:
            if (args.model and args.model != model) or (args.dataset and args.dataset != dataset):
                continue
            stem = args.output_dir / f"{model}__{dataset}"
            if args.replot_only:
                meta = json.loads(stem.with_suffix(".json").read_text())
                values = np.loadtxt(stem.with_suffix(".csv"), delimiter=",", skiprows=1)
                plot(*values.T, model, dataset, meta["samples"], stem)
            else:
                report(model, dataset, args.output_dir)
    if not args.model and not args.dataset:
        summaries = []
        for model in MODELS:
            for dataset in DATASETS:
                meta = json.loads((args.output_dir / f"{model}__{dataset}.json").read_text())
                summaries.append(
                    {
                        key: meta[key]
                        for key in (
                            "model",
                            "dataset",
                            "samples",
                            "total_neurons",
                            "audited_budget_conditions",
                            "knee_u",
                            "knee_cost",
                            "mean_band_width",
                        )
                    }
                )
        save_csv(args.output_dir / "summary.csv", summaries)
        with ZipFile(args.output_dir / "global_cumulative_16_figures.zip", "w", ZIP_DEFLATED) as archive:
            for path in sorted(args.output_dir.iterdir()):
                if path.suffix in (".png", ".pdf", ".csv", ".json"):
                    archive.write(path, path.name)
        print("Complete: 16 figures and 96,000 audited budget conditions", flush=True)


if __name__ == "__main__":
    main()
