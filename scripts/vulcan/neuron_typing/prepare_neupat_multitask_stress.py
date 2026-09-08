# Copyright 2026 the LlamaFactory team.
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

"""Build leakage-safe COCO + VQA-Med splits for NeuPAT forgetting experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_COCO_FILE = "/root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/val.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe multimodal forgetting-stress splits.")
    parser.add_argument("--coco_file", default=DEFAULT_COCO_FILE)
    parser.add_argument("--vqa_med_train", default="datasets/vqa_med/train_cls.jsonl")
    parser.add_argument("--vqa_med_dev", default="datasets/vqa_med/val_cls.jsonl")
    parser.add_argument("--vqa_med_image_root", default="datasets/vqa_med")
    parser.add_argument("--neuron_typing_root", default="saves/neuron_typing")
    parser.add_argument("--pope_dir", default="saves/neuron_typing/pope_data")
    parser.add_argument("--output_dir", default="datasets/neupat_multitask_stress")
    parser.add_argument("--coco_train_samples", type=int, default=30000)
    parser.add_argument("--coco_dev_samples", type=int, default=3000)
    parser.add_argument("--vqa_med_train_samples", type=int, default=0, help="Zero uses every training row.")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON array or JSONL file: {path}")
    return value


def image_path_of(row: dict[str, Any], image_root: Path | None = None) -> Path:
    images = row.get("images") or []
    if len(images) != 1:
        raise ValueError(f"Expected exactly one image per row, got {len(images)}: {row}")
    path = Path(images[0])
    if not path.is_absolute():
        if image_root is None:
            raise ValueError(f"Relative image path has no image root: {path}")
        path = image_root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def validate_supervised_row(row: dict[str, Any]) -> None:
    messages = row.get("messages") or []
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError(f"Row has no terminal assistant response: {row}")
    answer = str(messages[-1].get("content") or "").strip()
    if not answer or answer.lower() == "none":
        raise ValueError(f"Row has an empty/None assistant response: {row}")


def canonicalize_row(row: dict[str, Any], image_path: Path, source: str) -> dict[str, Any]:
    output = dict(row)
    output["images"] = [str(image_path)]
    output["stress_source"] = source
    return output


def coco_manifest_indices(root: Path) -> tuple[set[int], list[str]]:
    indices: set[int] = set()
    manifests: list[str] = []
    if not root.is_dir():
        return indices, manifests
    for path in sorted(root.rglob("*manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        dataset = str(value.get("dataset") or "").lower()
        source_indices = value.get("source_indices")
        if "coco_captions_val" not in dataset or not isinstance(source_indices, list):
            continue
        indices.update(int(index) for index in source_indices)
        manifests.append(str(path.resolve()))
    return indices, manifests


def pope_image_names(pope_dir: Path) -> tuple[set[str], list[str]]:
    names: set[str] = set()
    sources: list[str] = []
    if not pope_dir.is_dir():
        return names, sources
    for path in sorted(pope_dir.glob("coco_pope_*.json")):
        rows = read_records(path)
        for row in rows:
            image = row.get("image") or row.get("file_name") or row.get("image_path")
            if image:
                names.add(Path(str(image)).name)
        sources.append(str(path.resolve()))
    return names, sources


def hash_rows(rows: list[dict[str, Any]], image_root: Path | None = None) -> tuple[list[Path], list[str]]:
    cache: dict[Path, str] = {}
    paths: list[Path] = []
    hashes: list[str] = []
    for row in rows:
        validate_supervised_row(row)
        path = image_path_of(row, image_root)
        if path not in cache:
            cache[path] = sha256_file(path)
        paths.append(path)
        hashes.append(cache[path])
    return paths, hashes


def deduplicated_indices(indices: list[int], hashes: list[str]) -> tuple[list[int], int]:
    seen: set[str] = set()
    output: list[int] = []
    for index in indices:
        if hashes[index] in seen:
            continue
        seen.add(hashes[index])
        output.append(index)
    return output, len(indices) - len(output)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_entry(file_name: str) -> dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def build_splits(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_paths = {
        "train": output_dir / "train.jsonl",
        "coco_dev": output_dir / "coco_dev.jsonl",
        "vqa_med_dev": output_dir / "vqa_med_dev.jsonl",
        "multitask_dev": output_dir / "multitask_dev.jsonl",
        "dataset_info": output_dir / "dataset_info.json",
        "manifest": output_dir / "split_manifest.json",
    }
    if not args.force and any(path.exists() for path in output_paths.values()):
        raise FileExistsError(f"Output already exists under {output_dir}; review it or pass --force.")

    coco_path = Path(args.coco_file)
    vqa_train_path = Path(args.vqa_med_train)
    vqa_dev_path = Path(args.vqa_med_dev)
    coco_rows = read_records(coco_path)
    vqa_train_rows = read_records(vqa_train_path)
    vqa_dev_rows = read_records(vqa_dev_path)
    coco_paths, coco_hashes = hash_rows(coco_rows)
    vqa_root = Path(args.vqa_med_image_root)
    vqa_train_paths, vqa_train_hashes = hash_rows(vqa_train_rows, vqa_root)
    vqa_dev_paths, vqa_dev_hashes = hash_rows(vqa_dev_rows, vqa_root)

    manifest_indices, manifest_sources = coco_manifest_indices(Path(args.neuron_typing_root))
    pope_names, pope_sources = pope_image_names(Path(args.pope_dir))
    invalid_manifest_indices = sorted(index for index in manifest_indices if index < 0 or index >= len(coco_rows))
    if invalid_manifest_indices:
        raise ValueError(f"COCO exclusion manifests contain out-of-range indices: {invalid_manifest_indices[:10]}")
    pope_indices = {index for index, path in enumerate(coco_paths) if path.name in pope_names}
    excluded_indices = manifest_indices | pope_indices
    candidate_indices = [index for index in range(len(coco_rows)) if index not in excluded_indices]
    candidate_indices, duplicate_candidates = deduplicated_indices(candidate_indices, coco_hashes)

    rng = random.Random(args.seed)
    rng.shuffle(candidate_indices)
    required = args.coco_train_samples + args.coco_dev_samples
    if required > len(candidate_indices):
        raise ValueError(f"Requested {required} COCO rows but only {len(candidate_indices)} leakage-safe rows remain.")
    coco_train_indices = candidate_indices[: args.coco_train_samples]
    coco_dev_indices = candidate_indices[args.coco_train_samples : required]

    vqa_train_indices = list(range(len(vqa_train_rows)))
    rng.shuffle(vqa_train_indices)
    if args.vqa_med_train_samples > 0:
        if args.vqa_med_train_samples > len(vqa_train_indices):
            raise ValueError("vqa_med_train_samples exceeds the available VQA-Med training rows.")
        vqa_train_indices = vqa_train_indices[: args.vqa_med_train_samples]

    coco_train = [canonicalize_row(coco_rows[i], coco_paths[i], "coco_caption") for i in coco_train_indices]
    coco_dev = [canonicalize_row(coco_rows[i], coco_paths[i], "coco_caption") for i in coco_dev_indices]
    vqa_train = [canonicalize_row(vqa_train_rows[i], vqa_train_paths[i], "vqa_med") for i in vqa_train_indices]
    vqa_dev = [canonicalize_row(row, path, "vqa_med") for row, path in zip(vqa_dev_rows, vqa_dev_paths)]
    multitask_train = coco_train + vqa_train
    multitask_dev = coco_dev + vqa_dev
    rng.shuffle(multitask_train)
    rng.shuffle(multitask_dev)

    coco_train_hashes = {coco_hashes[i] for i in coco_train_indices}
    coco_dev_hashes = {coco_hashes[i] for i in coco_dev_indices}
    excluded_hashes = {coco_hashes[i] for i in excluded_indices}
    selected_vqa_train_hashes = {vqa_train_hashes[i] for i in vqa_train_indices}
    all_vqa_dev_hashes = set(vqa_dev_hashes)
    overlap_checks = {
        "coco_train_vs_coco_dev": len(coco_train_hashes & coco_dev_hashes),
        "coco_train_vs_excluded": len(coco_train_hashes & excluded_hashes),
        "coco_dev_vs_excluded": len(coco_dev_hashes & excluded_hashes),
        "vqa_med_train_vs_dev": len(selected_vqa_train_hashes & all_vqa_dev_hashes),
        "all_train_vs_all_dev": len(
            (coco_train_hashes | selected_vqa_train_hashes) & (coco_dev_hashes | all_vqa_dev_hashes)
        ),
    }
    if any(overlap_checks.values()):
        raise RuntimeError(f"Image-content leakage detected: {overlap_checks}")

    write_jsonl(output_paths["train"], multitask_train)
    write_jsonl(output_paths["coco_dev"], coco_dev)
    write_jsonl(output_paths["vqa_med_dev"], vqa_dev)
    write_jsonl(output_paths["multitask_dev"], multitask_dev)
    dataset_info = {
        "neupat_multitask_train": dataset_entry(output_paths["train"].name),
        "neupat_coco_dev": dataset_entry(output_paths["coco_dev"].name),
        "neupat_vqa_med_dev": dataset_entry(output_paths["vqa_med_dev"].name),
        "neupat_multitask_dev": dataset_entry(output_paths["multitask_dev"].name),
    }
    output_paths["dataset_info"].write_text(
        json.dumps(dataset_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "sources": {
            "coco": str(coco_path.resolve()),
            "vqa_med_train": str(vqa_train_path.resolve()),
            "vqa_med_dev": str(vqa_dev_path.resolve()),
            "coco_exclusion_manifests": manifest_sources,
            "pope_files": pope_sources,
        },
        "counts": {
            "coco_source_rows": len(coco_rows),
            "coco_excluded_by_manifest": len(manifest_indices),
            "coco_excluded_by_pope": len(pope_indices),
            "coco_excluded_union": len(excluded_indices),
            "coco_duplicate_candidates_dropped": duplicate_candidates,
            "coco_train": len(coco_train),
            "coco_dev": len(coco_dev),
            "vqa_med_train": len(vqa_train),
            "vqa_med_dev": len(vqa_dev),
            "multitask_train": len(multitask_train),
            "multitask_dev": len(multitask_dev),
            "unused_safe_coco": len(candidate_indices) - required,
        },
        "overlap_checks": overlap_checks,
        "indices": {
            "coco_train": sorted(coco_train_indices),
            "coco_dev": sorted(coco_dev_indices),
            "coco_excluded": sorted(excluded_indices),
            "vqa_med_train": sorted(vqa_train_indices),
        },
        "outputs": {name: str(path.resolve()) for name, path in output_paths.items() if name != "manifest"},
    }
    output_paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    manifest = build_splits(parse_args())
    print(json.dumps({"counts": manifest["counts"], "overlap_checks": manifest["overlap_checks"]}, indent=2))


if __name__ == "__main__":
    main()
