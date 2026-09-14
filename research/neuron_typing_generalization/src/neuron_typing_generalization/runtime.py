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

"""Run identity, artifact layout, and provenance capture."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from .config import ExperimentSpec
from .contracts import STAGE_ORDER


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
RUN_SUBDIRECTORIES = (
    "activations",
    "checkpoints",
    "classifications",
    "distributions",
    "figures",
    "logs",
    "metrics",
    "samples",
    "scores",
    "stages",
)


@dataclass(frozen=True)
class RunContext:
    """Resolved identity and filesystem location for one experiment run."""

    run_id: str
    run_dir: Path
    spec: ExperimentSpec


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-_").lower()
    return normalized or "unnamed"


def _stable_payload(spec: ExperimentSpec) -> str:
    return json.dumps(spec.as_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_run_id(spec: ExperimentSpec) -> str:
    """Build a readable run ID whose suffix changes with any resolved configuration."""
    digest = hashlib.sha256(_stable_payload(spec).encode()).hexdigest()[:10]
    return f"{_slug(spec.model.name)}__{_slug(spec.dataset.name)}__seed{spec.seed}__{digest}"


def resolve_artifact_root(spec: ExperimentSpec) -> Path:
    """Resolve artifact root, allowing a machine-local environment override."""
    override = os.environ.get("NEURON_TYPING_ARTIFACT_ROOT")
    root = Path(override or spec.artifact_root).expanduser()
    return root.resolve() if root.is_absolute() else (REPOSITORY_ROOT / root).resolve()


def _git_revision() -> dict[str, str | bool | None]:
    def run(*args: str) -> str | None:
        result = subprocess.run(("git", *args), cwd=REPOSITORY_ROOT, capture_output=True, check=False, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"), "dirty": bool(status)}


def initialize_run(spec: ExperimentSpec) -> RunContext:
    """Create a run directory and immutable resolved configuration/provenance files."""
    run_id = build_run_id(spec)
    root = resolve_artifact_root(spec)
    run_dir = root / "runs" / _slug(spec.study) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdirectory in RUN_SUBDIRECTORIES:
        (run_dir / subdirectory).mkdir(exist_ok=True)

    sampling = spec.study_values.get("sampling", {})
    if sampling.get("require_frozen_manifest"):
        tier = str(sampling.get("manifest_tier", "pilot"))
        source_dir = EXPERIMENT_ROOT / "manifests" / spec.dataset.name / tier
        for filename in ("sample_manifest.parquet", "pair_manifest.parquet"):
            source = source_dir / filename
            digest_source = source.with_suffix(source.suffix + ".sha256")
            if not source.is_file() or not digest_source.is_file():
                raise FileNotFoundError(f"Frozen {tier} manifest is unavailable for {spec.dataset.name}: {source}")
            destination = run_dir / "samples" / filename
            digest_destination = destination.with_suffix(destination.suffix + ".sha256")
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"Run contains a different frozen manifest: {destination}")
            if not destination.exists():
                shutil.copy2(source, destination)
                shutil.copy2(digest_source, digest_destination)

    config_path = run_dir / "config.resolved.yaml"
    resolved = spec.as_dict()
    if config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if existing != resolved:
            raise RuntimeError(f"Run ID collision with a different resolved configuration: {run_dir}")
    else:
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle, allow_unicode=True, sort_keys=False)

    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        metadata = {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "git": _git_revision(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "stage_order": STAGE_ORDER,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return RunContext(run_id=run_id, run_dir=run_dir, spec=spec)
