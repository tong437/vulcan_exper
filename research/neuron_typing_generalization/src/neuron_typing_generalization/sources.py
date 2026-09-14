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

"""Bounded acquisition of dataset documentation and question-only metadata."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DownloadRecord:
    """Provenance for one bounded downloaded file."""

    url: str
    relative_path: str
    size: int
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_limit(url: str, destination: Path, max_bytes: int) -> DownloadRecord:
    """Download one file atomically and abort before a configured byte limit is exceeded."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "neuron-typing-generalization/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > max_bytes:
            raise RuntimeError(f"Refusing {url}: Content-Length {content_length} exceeds {max_bytes} bytes.")
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            size = 0
            try:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise RuntimeError(f"Refusing {url}: streamed content exceeds {max_bytes} bytes.")
                    temporary.write(chunk)
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
    os.replace(temporary_path, destination)
    return DownloadRecord(url, destination.name, size, _sha256(destination))


def _extract_selected(archive: Path, destination: Path, predicates: tuple[str, ...], max_bytes: int) -> list[Path]:
    extracted = []
    total_size = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            normalized = Path(member.filename)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            if member.is_dir() or not any(fragment in member.filename for fragment in predicates):
                continue
            total_size += member.file_size
            if total_size > max_bytes:
                raise RuntimeError(f"Selected archive members exceed extraction limit {max_bytes} bytes.")
            target = destination / normalized.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(target)
    if not extracted:
        raise RuntimeError(f"No requested metadata members found in {archive}.")
    return extracted


def _gqa_parquet_url(dataset_id: str, config: str, split: str) -> tuple[str, int]:
    endpoint = "https://datasets-server.huggingface.co/parquet?dataset=" + urllib.parse.quote(dataset_id, safe="")
    with urllib.request.urlopen(endpoint, timeout=60) as response:
        payload = json.load(response)
    matches = [item for item in payload["parquet_files"] if item["config"] == config and item["split"] == split]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one question-only Parquet file for {dataset_id}/{config}/{split}.")
    return str(matches[0]["url"]), int(matches[0]["size"])


def fetch_dataset_source(config_path: Path, artifact_root: Path) -> Path:
    """Fetch one pinned source's README and non-image metadata within hard byte limits."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = str(config["name"])
    dataset_id = str(config["dataset_id"])
    revision = str(config["revision"])
    source_dir = artifact_root / "sources" / name
    source_dir.mkdir(parents=True, exist_ok=True)
    records: list[DownloadRecord] = []

    encoded_revision = urllib.parse.quote(revision, safe="")
    readme_url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{encoded_revision}/README.md"
    records.append(download_with_limit(readme_url, source_dir / "README.md", 2_000_000))
    api_url = f"https://huggingface.co/api/datasets/{dataset_id}/revision/{encoded_revision}"
    records.append(download_with_limit(api_url, source_dir / "repository.json", 5_000_000))

    metadata = config["metadata"]
    max_download = int(metadata["max_download_bytes"])
    if name == "vqav2":
        for field in ("questions_url", "annotations_url"):
            archive = source_dir / Path(str(metadata[field])).name
            record = download_with_limit(str(metadata[field]), archive, max_download)
            records.append(record)
            _extract_selected(archive, source_dir, (".json",), max_download * 4)
            archive.unlink()
    elif name == "gqa_balanced":
        parquet_url, advertised_size = _gqa_parquet_url(dataset_id, str(config["config"]), str(config["split"]))
        if advertised_size > max_download:
            raise RuntimeError(f"GQA metadata Parquet exceeds configured limit: {advertised_size} > {max_download}")
        records.append(download_with_limit(parquet_url, source_dir / "instructions.parquet", max_download))
    elif name == "clevr":
        archive_url = str(metadata["archive_url"])
        archive = source_dir / Path(archive_url).name
        records.append(download_with_limit(archive_url, archive, max_download))
        _extract_selected(
            archive,
            source_dir,
            ("CLEVR_val_questions.json", "CLEVR_val_scenes.json"),
            max_download * 4,
        )
        archive.unlink()
    elif name == "vizwiz_vqa":
        archive_url = str(metadata["annotations_url"])
        archive = source_dir / Path(archive_url).name
        records.append(download_with_limit(archive_url, archive, max_download))
        _extract_selected(archive, source_dir, ("val.json",), max_download * 4)
        archive.unlink()
    else:
        raise KeyError(f"No bounded metadata acquisition policy for {name}.")

    repository = json.loads((source_dir / "repository.json").read_text(encoding="utf-8"))
    resolved_revision = repository.get("sha")
    if resolved_revision != revision:
        raise RuntimeError(f"Pinned revision mismatch for {name}: expected {revision}, received {resolved_revision}.")
    snapshot = {
        "dataset": name,
        "dataset_id": dataset_id,
        "revision": revision,
        "allow_full_download": False,
        "files": [record.__dict__ for record in records],
        "extracted_metadata": [
            {"relative_path": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(source_dir.iterdir())
            if path.name not in {"source_snapshot.json"} and path.suffix in {".json", ".parquet"}
        ],
    }
    snapshot_path = source_dir / "source_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snapshot_path


def fetch_all_sources(config_dir: Path, artifact_root: Path) -> list[Path]:
    """Fetch every committed dataset config in stable filename order."""
    return [fetch_dataset_source(path, artifact_root) for path in sorted(config_dir.glob("*.yaml"))]
