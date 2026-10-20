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

"""Convert advisor JSONL selections into run manifests and materialize selected images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Image, load_dataset


SPECS = {
    "mme": ("lmms-lab-encoder/MME", "d6c9023f017b564f7b3ccccf5348166bce8fdbcd", "default", "test"),
    "mmbench": ("lmms-lab-encoder/MMBench", "56ba1af8954932c4804bd3f522e05ed96e63b654", "en", "dev"),
    "textvqa": (
        "lmms-lab-encoder/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "default",
        "validation",
    ),
    "chartqa": (
        "lmms-lab-encoder/ChartQA",
        "9e63b7df1592a1c2158e735cc1725454aef0d6d9",
        "default",
        "test",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_frozen_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    path.with_suffix(path.suffix + ".sha256").write_text(_sha256(path) + "\n", encoding="utf-8")


def prepare(selection_root: Path, manifest_root: Path, artifact_root: Path) -> dict[str, Any]:
    """Prepare all selected manifests and image files."""
    report: dict[str, Any] = {"tier": "advisor_2026_09", "datasets": {}}
    for name, (hub_id, revision, config, split) in SPECS.items():
        selections = [json.loads(line) for line in (selection_root / f"{name}_1000.jsonl").open()]
        dataset = load_dataset(hub_id, config, split=split, revision=revision).cast_column(
            "image", Image(decode=False)
        )
        image_dir = artifact_root / "images" / name
        image_dir.mkdir(parents=True, exist_ok=True)
        sample_rows = []
        images: dict[str, dict[str, Any]] = {}
        for rank, selected in enumerate(selections):
            source_index = int(selected["source_index"])
            source = dataset[source_index]
            image = source["image"]
            data = image.get("bytes")
            if data is None:
                data = Path(image["path"]).read_bytes()
            image_sha256 = hashlib.sha256(data).hexdigest()
            if image_sha256 != selected["image_sha256"]:
                raise ValueError(f"Image digest changed for {name} source row {source_index}.")
            suffix = Path(str(image.get("path") or selected.get("image_path") or ".img")).suffix.lower() or ".img"
            relative_path = f"{image_sha256}{suffix}"
            destination = image_dir / relative_path
            if not destination.exists():
                destination.write_bytes(data)
            elif _sha256(destination) != image_sha256:
                raise RuntimeError(f"Conflicting materialized image: {destination}")
            images[image_sha256] = {
                "image_id": image_sha256,
                "relative_path": relative_path,
                "sha256": image_sha256,
            }
            answer = selected.get("answers", selected.get("answer"))
            sample_rows.append(
                {
                    "dataset": name,
                    "dataset_revision": revision,
                    "split": split,
                    "tier": "advisor_2026_09",
                    "sample_id": str(selected["sample_id"]),
                    "image_id": image_sha256,
                    "question": str(selected["question"]),
                    "answer": answer,
                    "pair_group_id": selected.get("pair_id"),
                    "stratum": selected.get("stratum", selected.get("stratum_sha256")),
                    "selection_seed": 42,
                    "selection_rank": rank,
                    "image_locator": f"source_index:{source_index}",
                    "image_sha256": image_sha256,
                    "metadata": {
                        "config": config,
                        "source_index": source_index,
                        "image_path": str(selected.get("image_path") or ""),
                    },
                }
            )
        manifest_dir = manifest_root / name / "advisor_2026_09"
        _write_frozen_parquet(sample_rows, manifest_dir / "sample_manifest.parquet")
        image_manifest = pd.DataFrame(sorted(images.values(), key=lambda row: row["image_id"]))
        image_manifest.to_parquet(image_dir / "image_manifest.parquet", index=False)
        report["datasets"][name] = {
            "samples": len(sample_rows),
            "unique_images": len(images),
            "sample_manifest_sha256": _sha256(manifest_dir / "sample_manifest.parquet"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.selection_root, args.manifest_root, args.artifact_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
