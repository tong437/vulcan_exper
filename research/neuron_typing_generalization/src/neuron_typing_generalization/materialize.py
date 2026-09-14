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

"""Materialize only unique images referenced by a frozen pilot manifest."""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd
import yaml
from datasets import Image, load_from_disk
from PIL import Image as PILImage

from .sources import download_with_limit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_image(path: Path) -> tuple[int, int, str]:
    with PILImage.open(path) as image:
        width, height = image.size
        image_format = image.format or path.suffix.lstrip(".").upper()
        image.verify()
    return width, height, image_format


def _materialize_vqav2(frame: pd.DataFrame, config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    dataset_path = Path(config["images"]["local_dataset_path"])
    dataset = load_from_disk(str(dataset_path))["val"].cast_column("image", Image(decode=False))
    selected = {f"COCO_val2014_{int(image_id):012d}.jpg": str(image_id) for image_id in frame["image_id"].unique()}
    row_by_filename = {filename: index for index, filename in enumerate(dataset["file_name"]) if filename in selected}
    missing = sorted(set(selected) - set(row_by_filename))
    if missing:
        raise RuntimeError(f"Local COCO cache is missing {len(missing)} selected VQAv2 images.")

    records = []
    for filename in sorted(selected):
        image_value = dataset[row_by_filename[filename]]["image"]
        target = destination / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            if image_value.get("bytes") is not None:
                target.write_bytes(image_value["bytes"])
            else:
                source_path = Path(image_value["path"])
                target.write_bytes(source_path.read_bytes())
        width, height, image_format = _verify_image(target)
        records.append(
            {
                "image_id": selected[filename],
                "relative_path": filename,
                "size": target.stat().st_size,
                "sha256": _sha256(target),
                "width": width,
                "height": height,
                "format": image_format,
                "source": str(dataset_path),
            }
        )
    return records


def _gqa_cached_assets(config: dict[str, Any]) -> dict[str, str]:
    dataset_id = str(config["dataset_id"])
    image_config = str(config["image_config"])
    split = str(config["split"])
    assets: dict[str, str] = {}
    for offset in range(0, 398, 100):
        query = urllib.parse.urlencode(
            {"dataset": dataset_id, "config": image_config, "split": split, "offset": offset, "length": 100}
        )
        with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=120) as response:
            payload = json.load(response)
        for item in payload["rows"]:
            assets[str(item["row"]["id"])] = str(item["row"]["image"]["src"])
    return assets


def _materialize_gqa(frame: pd.DataFrame, config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    selected = sorted({str(value) for value in frame["image_id"]})
    assets = _gqa_cached_assets(config)
    missing = sorted(set(selected) - set(assets))
    if missing:
        raise RuntimeError(f"GQA image config is missing {len(missing)} selected image IDs.")
    records = []
    for image_id in selected:
        target = destination / f"{image_id}.jpg"
        if target.is_file():
            size = target.stat().st_size
            sha256 = _sha256(target)
        else:
            record = download_with_limit(assets[image_id], target, 25_000_000)
            size = record.size
            sha256 = record.sha256
        width, height, image_format = _verify_image(target)
        records.append(
            {
                "image_id": image_id,
                "relative_path": target.name,
                "size": size,
                "sha256": sha256,
                "width": width,
                "height": height,
                "format": image_format,
                "source": "huggingface_dataset_server_cached_asset",
            }
        )
    return records


def _materialize_remote_zip(frame: pd.DataFrame, config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    """Read selected ZIP members through HTTP ranges without downloading the archive."""
    archive_url = str(config["images"]["archive_url"])
    template = str(config["images"]["archive_member_template"])
    selected = sorted({str(value) for value in frame["image_id"]})
    member_by_id = {
        image_id: template.format(image_id=int(image_id) if image_id.isdigit() else image_id) for image_id in selected
    }
    byte_limit = int(config["images"]["max_materialized_bytes"])
    with fsspec.open(archive_url, "rb", block_size=1024 * 1024).open() as remote:
        with zipfile.ZipFile(remote) as archive:
            info_by_member = {info.filename: info for info in archive.infolist()}
            missing = sorted(set(member_by_id.values()) - set(info_by_member))
            if missing:
                raise RuntimeError(f"Remote archive is missing {len(missing)} selected image members.")
            selected_info = {image_id: info_by_member[member] for image_id, member in member_by_id.items()}
    projected_size = sum(info.file_size for info in selected_info.values())
    if projected_size > byte_limit:
        raise RuntimeError(f"Selected ZIP members exceed dataset byte limit: {projected_size} > {byte_limit}")

    def materialize_one(item: tuple[str, zipfile.ZipInfo]) -> dict[str, Any]:
        image_id, info = item
        target = destination / Path(info.filename).name
        if not target.is_file():
            target.write_bytes(_read_zip_member_range(archive_url, info))
        if target.stat().st_size != info.file_size:
            raise RuntimeError(f"Extracted size mismatch for {info.filename}.")
        if binascii.crc32(target.read_bytes()) & 0xFFFFFFFF != info.CRC:
            raise RuntimeError(f"Extracted CRC mismatch for {info.filename}.")
        width, height, image_format = _verify_image(target)
        return {
            "image_id": image_id,
            "relative_path": target.name,
            "size": target.stat().st_size,
            "sha256": _sha256(target),
            "width": width,
            "height": height,
            "format": image_format,
            "source": f"remote_zip_range:{archive_url}",
        }

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(materialize_one, selected_info.items()))


def _read_zip_member_range(archive_url: str, info: zipfile.ZipInfo) -> bytes:
    """Fetch and decompress one ZIP member from a bounded HTTP byte range."""
    overfetch = 65_535
    stop = info.header_offset + 30 + overfetch + info.compress_size - 1
    request = urllib.request.Request(
        archive_url,
        headers={
            "Range": f"bytes={info.header_offset}-{stop}",
            "User-Agent": "neuron-typing-generalization/1.0",
        },
    )
    payload = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206 or response.headers.get("Content-Range") is None:
                    raise RuntimeError(f"Archive server ignored bounded Range request for {info.filename}.")
                payload = response.read(30 + overfetch + info.compress_size)
            break
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(0.5 * 2**attempt)
    assert payload is not None
    if len(payload) < 30:
        raise RuntimeError(f"Truncated local ZIP header for {info.filename}.")
    signature, _, flag, compression, _, _, _, compressed_size, _, name_length, extra_length = struct.unpack(
        "<IHHHHHIIIHH", payload[:30]
    )
    if signature != 0x04034B50 or flag & 0x1:
        raise RuntimeError(f"Unsupported local ZIP header for {info.filename}.")
    if compressed_size not in (0, info.compress_size) or compression != info.compress_type:
        raise RuntimeError(f"ZIP central/local header mismatch for {info.filename}.")
    data_start = 30 + name_length + extra_length
    compressed = payload[data_start : data_start + info.compress_size]
    if len(compressed) != info.compress_size:
        raise RuntimeError(f"Truncated ZIP member data for {info.filename}.")
    if info.compress_type == zipfile.ZIP_STORED:
        value = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        value = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise RuntimeError(f"Unsupported ZIP compression {info.compress_type} for {info.filename}.")
    if len(value) != info.file_size or binascii.crc32(value) & 0xFFFFFFFF != info.CRC:
        raise RuntimeError(f"ZIP integrity check failed for {info.filename}.")
    return value


def materialize_selected_images(config_path: Path, manifest_path: Path, artifact_root: Path) -> Path:
    """Materialize one dataset using only frozen-manifest image references."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = str(config["name"])
    frame = pd.read_parquet(manifest_path)
    destination = artifact_root / "images" / name
    destination.mkdir(parents=True, exist_ok=True)
    status_path = destination / "materialization_status.json"
    if name == "vqav2":
        records = _materialize_vqav2(frame, config, destination)
    elif name == "gqa_balanced":
        records = _materialize_gqa(frame, config, destination)
    elif name in {"clevr", "vizwiz_vqa"}:
        records = _materialize_remote_zip(frame, config, destination)
    else:
        raise KeyError(f"No selected-image materializer for {name}.")
    total_bytes = sum(record["size"] for record in records)
    byte_limit = int(config["images"]["max_materialized_bytes"])
    if total_bytes > byte_limit:
        raise RuntimeError(f"Materialized images exceed dataset byte limit: {total_bytes} > {byte_limit}")
    pd.DataFrame(records).to_parquet(destination / "image_manifest.parquet", index=False)
    status = {
        "dataset": name,
        "status": "complete",
        "manifest": str(manifest_path),
        "unique_images": len(records),
        "total_bytes": total_bytes,
        "full_download": False,
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return status_path
