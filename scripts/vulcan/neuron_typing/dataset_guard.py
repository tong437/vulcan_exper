"""Dataset slicing, manifesting, and isolation checks for neuron experiments."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _flatten_images(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if isinstance(value, dict):
        for key in ("path", "image", "file_name", "filename"):
            if value.get(key):
                return [str(value[key])]
        return []
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_images(item))
        return result
    return []


def normalize_image_id(value: str) -> str:
    """Use a stable split-aware basename as the image identifier."""
    return Path(value).name


def slice_dataset(
    dataset,
    sample_offset: int,
    max_samples: int | None,
    *,
    allow_short: bool = False,
):
    """Return a deterministic contiguous dataset slice and its source indices."""
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be non-negative, got {sample_offset}.")

    total = len(dataset)
    requested = total - sample_offset if max_samples is None else max_samples
    if requested < 0:
        raise ValueError(f"max_samples must be non-negative, got {requested}.")

    end = sample_offset + requested
    if end > total:
        if not allow_short:
            raise ValueError(
                f"Dataset slice [{sample_offset}:{end}] requires {requested} samples, "
                f"but the loaded dataset only has {total}. This commonly means a stale "
                "tokenized_path cache was reused. Rebuild the cache or pass --allow_short_dataset."
            )
        end = total

    indices = list(range(sample_offset, end))
    if hasattr(dataset, "select"):
        return dataset.select(indices), indices

    from torch.utils.data import Subset

    return Subset(dataset, indices), indices


def build_dataset_manifest(
    dataset,
    source_indices: list[int],
    *,
    role: str,
    dataset_name: str | None,
    tokenized_path: str | None,
    max_image_repeat: int = 5,
    allow_excessive_image_repeats: bool = False,
) -> dict[str, Any]:
    """Build a lightweight manifest and reject obviously corrupted image mappings."""
    image_ids_per_row: list[list[str]] = []
    counts: Counter[str] = Counter()

    for row_idx in range(len(dataset)):
        row = dataset[row_idx]
        ids = [normalize_image_id(path) for path in _flatten_images(row.get("images"))]
        image_ids_per_row.append(ids)
        counts.update(ids)

    max_repeat = max(counts.values(), default=0)
    repeated = sorted(
        ({"image_id": image_id, "count": count} for image_id, count in counts.items() if count > max_image_repeat),
        key=lambda item: (-item["count"], item["image_id"]),
    )
    if repeated and not allow_excessive_image_repeats:
        worst = repeated[0]
        raise ValueError(
            f"Dataset role {role!r} maps {worst['count']} rows to image {worst['image_id']!r}; "
            f"the allowed maximum is {max_image_repeat}. The source dataset or cache is likely corrupted. "
            "Use a corrected dataset, or explicitly pass --allow_excessive_image_repeats for a diagnostic run."
        )

    return {
        "role": role,
        "dataset": dataset_name,
        "tokenized_path": tokenized_path,
        "num_rows": len(dataset),
        "source_indices": source_indices,
        "image_ids": sorted(counts),
        "row_image_ids": image_ids_per_row,
        "num_unique_images": len(counts),
        "max_image_repeat": max_repeat,
        "rows_without_images": sum(not ids for ids in image_ids_per_row),
        "excessive_repeats": repeated[:20],
    }


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def assert_disjoint_manifests(
    current: dict[str, Any],
    other_paths: Iterable[str | Path],
) -> dict[str, Any]:
    """Raise when the current image set overlaps any calibration/typing manifest."""
    current_ids = set(current.get("image_ids", []))
    comparisons = []
    for path_value in other_paths:
        path = Path(path_value)
        other = json.loads(path.read_text(encoding="utf-8"))
        overlap = current_ids & set(other.get("image_ids", []))
        comparisons.append(
            {
                "manifest": str(path),
                "other_role": other.get("role"),
                "overlap_count": len(overlap),
                "overlap_examples": sorted(overlap)[:20],
            }
        )
        if overlap:
            raise ValueError(
                f"Data isolation failed: role {current.get('role')!r} overlaps {len(overlap)} images "
                f"with {path}. Examples: {sorted(overlap)[:5]}"
            )

    return {"is_isolated": True, "comparisons": comparisons}
