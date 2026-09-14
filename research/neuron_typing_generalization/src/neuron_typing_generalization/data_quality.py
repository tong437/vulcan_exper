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

"""High-signal quality checks for frozen samples, pairs, and pinned source records."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .clevr_audit import program_edit_distance
from .manifests import normalize_question


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def pairing_violations(dataset: str, samples: pd.DataFrame, pairs: pd.DataFrame) -> list[dict[str, str]]:
    """Return stable pair-policy violations with pair IDs as inspectable evidence."""
    by_id = {str(row.sample_id): row for row in samples.itertuples()}
    violations = []

    def add(pair: Any, rule: str) -> None:
        violations.append(
            {"left_sample_id": str(pair.left_sample_id), "right_sample_id": str(pair.right_sample_id), "rule": rule}
        )

    seen_edges = set()
    for pair in pairs.itertuples():
        left_id = str(pair.left_sample_id)
        right_id = str(pair.right_sample_id)
        edge = tuple(sorted((left_id, right_id)))
        if left_id not in by_id or right_id not in by_id:
            add(pair, "pair_reference_exists")
            continue
        if left_id == right_id:
            add(pair, "no_self_pair")
        if edge in seen_edges:
            add(pair, "unique_unordered_pair")
        seen_edges.add(edge)
        left = by_id[left_id]
        right = by_id[right_id]
        if str(left.pair_group_id) != str(pair.pair_group_id) or str(right.pair_group_id) != str(pair.pair_group_id):
            add(pair, "pair_group_referential_integrity")
        if dataset == "vqav2":
            if normalize_question(str(left.question)) != normalize_question(str(right.question)):
                add(pair, "same_normalized_question")
            if str(left.image_id) == str(right.image_id):
                add(pair, "different_image")
        elif dataset == "gqa_balanced":
            if str(left.image_id) != str(right.image_id):
                add(pair, "same_image")
            if normalize_question(str(left.question)) == normalize_question(str(right.question)):
                add(pair, "different_normalized_question")
        elif dataset == "clevr" and str(pair.audit_status) == "passed":
            if str(left.image_id) != str(right.image_id):
                add(pair, "same_scene")
            distance = program_edit_distance(_metadata(left.metadata)["program"], _metadata(right.metadata)["program"])
            if distance != 1:
                add(pair, "program_edit_distance_one")
        elif dataset == "vizwiz_vqa":
            add(pair, "vizwiz_has_no_controlled_pairs")
    return violations


def profile_manifest(dataset: str, tier: str, samples: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    """Profile grain, completeness, duplication, distribution, and policy integrity."""
    required = ["sample_id", "image_id", "question", "image_locator", "dataset_revision"]
    null_counts = {column: int(samples[column].isna().sum()) for column in required}
    normalized = samples["question"].astype(str).map(normalize_question)
    answer_values = samples["answer"].fillna("<null>").astype(str)
    strata = samples["stratum"].fillna("<none>").astype(str)
    violations = pairing_violations(dataset, samples, pairs)
    return {
        "tier": tier,
        "grain": "one row per selected question/sample_id; explicit pair edges in pair_manifest",
        "samples": len(samples),
        "columns": len(samples.columns),
        "pairs": len(pairs),
        "unique_sample_ids": int(samples["sample_id"].astype(str).nunique()),
        "duplicate_sample_ids": int(samples["sample_id"].astype(str).duplicated().sum()),
        "exact_duplicate_rows": int(samples.duplicated().sum()),
        "required_null_counts": null_counts,
        "unique_images": int(samples["image_id"].astype(str).nunique()),
        "image_reuse_rate": float(1 - samples["image_id"].astype(str).nunique() / len(samples)),
        "unique_normalized_questions": int(normalized.nunique()),
        "normalized_question_collision_rows": int(normalized.duplicated(keep=False).sum()),
        "top_answer_counts": dict(Counter(answer_values).most_common(10)),
        "stratum_counts": dict(Counter(strata).most_common()),
        "pairing_violation_count": len(violations),
        "pairing_violations": violations[:25],
        "status": "passed" if not violations and not any(null_counts.values()) else "failed",
    }


def audit_source(config: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    """Check pinned revision/integrity and flag source-card identity conflicts."""
    snapshot_path = source_dir / "source_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    integrity_failures = []
    persisted = {item["relative_path"]: item for item in [*snapshot["files"], *snapshot["extracted_metadata"]]}
    for relative_path, item in persisted.items():
        path = source_dir / relative_path
        if not path.is_file():
            continue
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            integrity_failures.append(relative_path)
    readme = (source_dir / "README.md").read_text(encoding="utf-8", errors="replace")
    identity_conflicts = []
    if config["name"] == "clevr" and "CLEVR" not in readme.upper() and "visualqa.org" in readme:
        identity_conflicts.append(
            {
                "severity": "medium",
                "confidence": "high",
                "finding": "Pinned upstream CLEVR README contains VQAv2/visualqa statistics and no CLEVR identity.",
                "impact": "The README cannot substantiate CLEVR dataset facts; pinned config and official metadata must govern.",
                "remediation": "Retain the byte-exact README for provenance and label it as an upstream card defect.",
            }
        )
    return {
        "dataset_id": config["dataset_id"],
        "configured_revision": config["revision"],
        "snapshot_revision": snapshot["revision"],
        "revision_matches": snapshot["revision"] == config["revision"],
        "integrity_failures": integrity_failures,
        "identity_conflicts": identity_conflicts,
        "status": (
            "failed"
            if snapshot["revision"] != config["revision"] or integrity_failures
            else "passed_with_documented_source_issue"
            if identity_conflicts
            else "passed"
        ),
    }


def build_data_quality_report(experiment_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Build a machine-readable quality assessment across every frozen dataset tier."""
    datasets = []
    critical_failures = 0
    documented_issues = 0
    for config_path in sorted((experiment_root / "configs/datasets").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        name = str(config["name"])
        source = audit_source(config, artifact_root / "sources" / name)
        tiers = []
        for tier_dir in sorted((experiment_root / "manifests" / name).iterdir()):
            profile = profile_manifest(
                name,
                tier_dir.name,
                pd.read_parquet(tier_dir / "sample_manifest.parquet"),
                pd.read_parquet(tier_dir / "pair_manifest.parquet"),
            )
            tiers.append(profile)
            critical_failures += int(profile["status"] == "failed")
        critical_failures += int(source["status"] == "failed")
        documented_issues += len(source["identity_conflicts"])
        datasets.append({"dataset": name, "source": source, "tiers": tiers})
    return {
        "status": "failed" if critical_failures else "ready_with_documented_risks" if documented_issues else "ready",
        "critical_failures": critical_failures,
        "documented_issues": documented_issues,
        "datasets": datasets,
    }


def render_quality_report(report: dict[str, Any]) -> str:
    """Render compact stakeholder-facing evidence from the machine-readable assessment."""
    lines = ["# Frozen dataset quality assessment", "", f"Overall status: `{report['status']}`.", ""]
    for dataset in report["datasets"]:
        source = dataset["source"]
        lines.extend(
            [
                f"## {dataset['dataset']}",
                "",
                f"Source revision match: `{source['revision_matches']}`; integrity failures: "
                f"`{len(source['integrity_failures'])}`.",
                "",
            ]
        )
        for issue in source["identity_conflicts"]:
            lines.append(
                f"- **{issue['severity']} / {issue['confidence']} confidence:** {issue['finding']} "
                f"Impact: {issue['impact']} Remediation: {issue['remediation']}"
            )
        if source["identity_conflicts"]:
            lines.append("")
        lines.extend(
            [
                "| Tier | Samples | Pairs | Unique images | Duplicate IDs | Pair violations | Status |",
                "|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for tier in dataset["tiers"]:
            lines.append(
                f"| {tier['tier']} | {tier['samples']} | {tier['pairs']} | {tier['unique_images']} | "
                f"{tier['duplicate_sample_ids']} | {tier['pairing_violation_count']} | {tier['status']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Manifest grain, required-field completeness, sample-key uniqueness, pair referential integrity, and each "
            "dataset's frozen pairing rule were checked. Pending CLEVR tiers remain candidate evidence only; "
            "`audited_v1` is the controlled-analysis tier. VizWiz remains descriptive cohort data.",
            "",
        ]
    )
    return "\n".join(lines)
