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

"""CPU-readiness and no-full-download audits for frozen experiment inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .manifests import PAIR_COLUMNS, SAMPLE_COLUMNS, verify_frozen_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(
    config_path: Path,
    source_root: Path,
    manifest_root: Path,
    image_root: Path,
) -> dict[str, Any]:
    """Audit one dataset's revision, manifests, materialized images, and download boundary."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = str(config["name"])
    if config["images"].get("allow_full_download") is not False:
        raise RuntimeError(f"Full image download is not disabled for {name}.")
    if config["images"].get("materialize_selected_only") is not True:
        raise RuntimeError(f"Selected-only image materialization is not required for {name}.")

    source_dir = source_root / name
    snapshot = json.loads((source_dir / "source_snapshot.json").read_text(encoding="utf-8"))
    if snapshot["revision"] != config["revision"] or snapshot["allow_full_download"] is not False:
        raise RuntimeError(f"Source snapshot does not match pinned config for {name}.")
    for item in snapshot["extracted_metadata"]:
        path = source_dir / item["relative_path"]
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"Metadata integrity check failed: {path}")

    tiers = []
    expected_images: set[str] = set()
    for tier_dir in sorted((manifest_root / name).iterdir()):
        sample_path = tier_dir / "sample_manifest.parquet"
        pair_path = tier_dir / "pair_manifest.parquet"
        verify_frozen_manifest(sample_path, SAMPLE_COLUMNS)
        verify_frozen_manifest(pair_path, PAIR_COLUMNS)
        samples = pd.read_parquet(sample_path)
        expected_images.update(str(value) for value in samples["image_id"])
        pairs = pd.read_parquet(pair_path)
        if len(pairs) and (pairs["left_sample_id"] == pairs["right_sample_id"]).any():
            raise RuntimeError(f"Controlled manifest contains self-pairs: {pair_path}")
        tiers.append({"tier": tier_dir.name, "samples": len(samples), "pairs": len(pairs)})

    status = json.loads((image_root / name / "materialization_status.json").read_text(encoding="utf-8"))
    images = pd.read_parquet(image_root / name / "image_manifest.parquet")
    actual_images = {str(value) for value in images["image_id"]}
    if status["status"] != "complete" or status["full_download"] is not False or actual_images != expected_images:
        raise RuntimeError(f"Selected image materialization is incomplete for {name}.")
    if int(images["size"].sum()) > int(config["images"]["max_materialized_bytes"]):
        raise RuntimeError(f"Image materialization exceeds byte budget for {name}.")
    for item in images.to_dict(orient="records"):
        path = image_root / name / item["relative_path"]
        if path.stat().st_size != item["size"] or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"Image integrity check failed: {path}")
    return {
        "dataset": name,
        "revision": config["revision"],
        "metadata_bytes": sum(item["size"] for item in snapshot["extracted_metadata"]),
        "tiers": tiers,
        "materialized_images": len(images),
        "materialized_bytes": int(images["size"].sum()),
        "full_download": False,
    }


def audit_all(experiment_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Audit every configured dataset and return a serialization-ready report."""
    datasets = [
        audit_dataset(
            config_path,
            artifact_root / "sources",
            experiment_root / "manifests",
            artifact_root / "images",
        )
        for config_path in sorted((experiment_root / "configs/datasets").glob("*.yaml"))
    ]
    return {"status": "ready", "datasets": datasets}
