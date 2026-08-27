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

"""Collect paired correct-image/shuffled-image representations for Phase 4."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_guard import (  # noqa: E402
    assert_disjoint_manifests,
    build_dataset_manifest,
    normalize_image_id,
    save_manifest,
)
from evaluate_vqa import (  # noqa: E402
    _manifest_image_ids,
    load_binary_records,
    prepare_model_batch,
    select_binary_records,
)
from phase4_mapping_utils import (  # noqa: E402
    image_disjoint_split,
    masked_pool,
    split_local_derangement,
    verify_image_disjoint_split,
)

from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.train.vulcan import find_mlp_layers  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Phase-4 paired activation mapping data.")
    parser.add_argument("--config", required=True, help="LlamaFactory model/data YAML.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--vqa_file", required=True, help="POPE-compatible JSON/JSONL file.")
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--image_offset", type=int, default=0)
    parser.add_argument("--calibration_manifest", default=None)
    parser.add_argument("--typing_manifest", default=None)
    parser.add_argument("--exclude_manifest", action="append", default=[])
    parser.add_argument("--filter_manifest_overlaps", action="store_true")
    parser.add_argument("--require_data_isolation", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=6)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--validation_ratio", type=float, default=0.15)
    parser.add_argument("--split_seed", type=int, default=2027)
    parser.add_argument("--shuffled_image_seed", type=int, default=2028)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--feature_pool", choices=["mean", "max_abs", "mean_max_abs"], default="mean_max_abs")
    parser.add_argument("--activation_pool", choices=["mean", "max_abs"], default="mean")
    parser.add_argument("--storage_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if key != "resume"}


class PairedActivationCollector:
    """Pool bridge features and text-position FFN activations inside forward hooks."""

    def __init__(
        self,
        mlp_layers,
        *,
        image_token_id: int,
        excluded_text_token_ids: set[int],
        feature_pool: str,
        activation_pool: str,
    ):
        self.mlp_layers = mlp_layers
        self.image_token_id = image_token_id
        self.excluded_text_token_ids = excluded_text_token_ids
        self.feature_pool = feature_pool
        self.activation_pool = activation_pool
        self._visual_mask: torch.Tensor | None = None
        self._text_mask: torch.Tensor | None = None
        self._bridge_features: torch.Tensor | None = None
        self._question_features: torch.Tensor | None = None
        self._activations: dict[int, torch.Tensor] = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        self._hooks.append(self.mlp_layers[0].layer.register_forward_pre_hook(self._bridge_hook))
        for layer_ref in self.mlp_layers:
            self._hooks.append(
                layer_ref.mlp.down_proj.register_forward_pre_hook(self._make_activation_hook(layer_ref.index))
            )

    def _bridge_hook(self, module, args) -> None:
        del module
        hidden_states = args[0]
        if self._visual_mask is None or self._text_mask is None:
            raise RuntimeError("Collector token masks were not configured before the model forward.")
        self._bridge_features = masked_pool(hidden_states, self._visual_mask, self.feature_pool).detach().cpu()
        self._question_features = masked_pool(hidden_states, self._text_mask, self.feature_pool).detach().cpu()

    def _make_activation_hook(self, layer_index: int):
        def hook(module, args) -> None:
            del module
            if self._text_mask is None:
                raise RuntimeError("Collector text mask was not configured before the model forward.")
            self._activations[layer_index] = (
                masked_pool(args[0], self._text_mask, self.activation_pool).detach().cpu()
            )

        return hook

    def configure(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        visual_mask = input_ids == self.image_token_id
        text_mask = attention_mask.bool() & ~visual_mask
        for token_id in self.excluded_text_token_ids:
            text_mask &= input_ids != token_id
        if torch.any(visual_mask.sum(dim=1) == 0):
            raise ValueError("At least one Phase-4 sample has no visual tokens.")
        if torch.any(text_mask.sum(dim=1) == 0):
            raise ValueError("At least one Phase-4 sample has no non-visual prompt tokens.")
        self._visual_mask = visual_mask
        self._text_mask = text_mask
        self._bridge_features = None
        self._question_features = None
        self._activations.clear()

    def export(self) -> dict[str, Any]:
        if self._bridge_features is None or self._question_features is None:
            raise RuntimeError("The bridge hook did not run.")
        expected_layers = {layer.index for layer in self.mlp_layers}
        if set(self._activations) != expected_layers:
            raise RuntimeError(
                f"FFN hook coverage mismatch: captured={sorted(self._activations)}, expected={sorted(expected_layers)}"
            )
        return {
            "vision_features": self._bridge_features,
            "question_features": self._question_features,
            "activations": dict(self._activations),
        }

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class ChunkWriter:
    """Buffer CPU tensors and write contiguous, resumable Phase-4 chunks."""

    def __init__(
        self,
        chunk_dir: Path,
        *,
        chunk_size: int,
        storage_dtype: torch.dtype,
        start_index: int,
    ):
        self.chunk_dir = chunk_dir
        self.chunk_size = chunk_size
        self.storage_dtype = storage_dtype
        self.next_index = start_index
        self._vision: list[torch.Tensor] = []
        self._question: list[torch.Tensor] = []
        self._delta: dict[int, list[torch.Tensor]] = {}
        self._rows: list[dict[str, Any]] = []

    def add(self, correct: dict[str, Any], shuffled: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        if len(rows) != correct["vision_features"].shape[0] or len(rows) != shuffled["vision_features"].shape[0]:
            raise ValueError("Chunk metadata and captured batches are not aligned.")
        if set(correct["activations"]) != set(shuffled["activations"]):
            raise ValueError("Correct and shuffled forwards captured different decoder layers.")
        self._vision.append(correct["vision_features"].float())
        self._question.append(correct["question_features"].float())
        for layer_index in correct["activations"]:
            delta = correct["activations"][layer_index].float() - shuffled["activations"][layer_index].float()
            self._delta.setdefault(layer_index, []).append(delta)
        self._rows.extend(rows)
        self.flush(force=False)

    def flush(self, *, force: bool) -> None:
        if not self._rows or (len(self._rows) < self.chunk_size and not force):
            return
        vision = torch.cat(self._vision, dim=0)
        question = torch.cat(self._question, dim=0)
        delta = {layer: torch.cat(parts, dim=0) for layer, parts in self._delta.items()}
        write_count = len(self._rows) if force else self.chunk_size
        start = self.next_index
        end = start + write_count
        payload = {
            "start_index": start,
            "end_index": end,
            "vision_features": vision[:write_count].to(self.storage_dtype),
            "question_features": question[:write_count].to(self.storage_dtype),
            "delta_activations": {
                str(layer): values[:write_count].to(self.storage_dtype) for layer, values in delta.items()
            },
            "rows": self._rows[:write_count],
        }
        target = self.chunk_dir / f"chunk_{start:06d}_{end:06d}.pt"
        temporary = target.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(target)
        self.next_index = end

        self._vision = [vision[write_count:]] if write_count < len(vision) else []
        self._question = [question[write_count:]] if write_count < len(question) else []
        self._delta = (
            {layer: [values[write_count:]] for layer, values in delta.items()} if write_count < len(vision) else {}
        )
        self._rows = self._rows[write_count:]
        if not force and len(self._rows) >= self.chunk_size:
            self.flush(force=False)


def _completed_rows(chunk_dir: Path) -> int:
    expected_start = 0
    for path in sorted(chunk_dir.glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["start_index"]) != expected_start:
            raise ValueError(f"Non-contiguous Phase-4 chunks at {path}.")
        expected_start = int(payload["end_index"])
        if len(payload["rows"]) != expected_start - int(payload["start_index"]):
            raise ValueError(f"Invalid row count in Phase-4 chunk {path}.")
    return expected_start


def _build_shuffled_records(
    records: list[dict[str, Any]],
    splits: list[str],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    image_ids = [normalize_image_id(record["images"][0]) for record in records]
    mapping = split_local_derangement(image_ids, splits, seed=seed)
    image_paths = {normalize_image_id(record["images"][0]): record["images"][0] for record in records}
    shuffled = deepcopy(records)
    for record in shuffled:
        original_id = normalize_image_id(record["images"][0])
        target_id = mapping[original_id]
        record["original_image"] = record["images"][0]
        record["images"] = [image_paths[target_id]]
    return shuffled, mapping


def _resolve_image_token_id(model, tokenizer) -> int:
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None or image_token_id < 0:
        raise ValueError("Could not resolve the model image token id.")
    return int(image_token_id)


def collect(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or args.chunk_size < 1:
        raise ValueError("batch_size and chunk_size must be positive.")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    chunk_dir = output_dir / "chunks"
    state_path = output_dir / "collection_state.json"
    signature = _config_signature(args)
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(f"Phase-4 output already exists; pass --resume: {output_dir}")
        if previous["config"] != signature:
            differing = sorted(
                key
                for key in set(previous["config"]) | set(signature)
                if previous["config"].get(key) != signature.get(key)
            )
            raise ValueError(f"Resume configuration mismatch for keys: {differing}")
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume because collection_state.json is absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    comparison_paths = list(
        dict.fromkeys(
            path for path in (args.calibration_manifest, args.typing_manifest, *args.exclude_manifest) if path
        )
    )
    if args.require_data_isolation and not (args.calibration_manifest and args.typing_manifest):
        raise ValueError("Data isolation requires both calibration and typing manifests.")
    excluded_ids = _manifest_image_ids(comparison_paths) if args.filter_manifest_overlaps else set()
    all_records = load_binary_records(args.vqa_file, args.image_root)
    records, selection_summary = select_binary_records(
        all_records,
        excluded_image_ids=excluded_ids,
        image_offset=args.image_offset,
        max_images=args.max_images,
    )
    manifest = build_dataset_manifest(
        records,
        [record["source_index"] for record in records],
        role="phase4_mapping",
        dataset_name=args.vqa_file,
        tokenized_path=None,
        max_image_repeat=args.max_image_repeat,
    )
    isolation = assert_disjoint_manifests(manifest, comparison_paths) if comparison_paths else None
    manifest_path = output_dir / "sample_manifest.json"
    save_manifest(manifest, manifest_path)

    image_ids = [normalize_image_id(record["images"][0]) for record in records]
    splits = image_disjoint_split(
        image_ids,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        seed=args.split_seed,
    )
    split_verification = verify_image_disjoint_split(image_ids, splits)
    shuffled_records, shuffled_mapping = _build_shuffled_records(
        records,
        splits,
        seed=args.shuffled_image_seed,
    )
    split_payload = {
        "split_seed": args.split_seed,
        "shuffled_image_seed": args.shuffled_image_seed,
        "verification": split_verification,
        "image_to_split": dict(sorted(set(zip(image_ids, splits)))),
        "shuffled_image_mapping": shuffled_mapping,
    }
    (output_dir / "splits.json").write_text(
        json.dumps(split_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    completed = _completed_rows(chunk_dir)
    if completed > len(records):
        raise ValueError("Existing Phase-4 chunks contain more rows than the selected dataset.")
    state = {
        "complete": completed == len(records),
        "config": signature,
        "selected_rows": len(records),
        "completed_rows": completed,
        "selection_summary": selection_summary,
        "data_isolation": isolation,
        "split_verification": split_verification,
        "manifest": str(manifest_path),
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    if completed == len(records):
        return state

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config.update({"do_train": False, "do_eval": False, "do_predict": False})
    if args.model_name_or_path:
        config["model_name_or_path"] = args.model_name_or_path
    config.setdefault("output_dir", "saves/neuron_typing/phase4_tmp")
    model_args, _, _, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module.get("processor")
    if processor is None:
        raise RuntimeError("The configured model did not provide a multimodal processor.")
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    mlp_layers = find_mlp_layers(model)
    image_token_id = _resolve_image_token_id(model, tokenizer)
    excluded_text_ids = {
        int(token_id)
        for token_id in (
            getattr(model.config, "vision_start_token_id", None),
            getattr(model.config, "vision_end_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
        )
        if token_id is not None
    }
    collector = PairedActivationCollector(
        mlp_layers,
        image_token_id=image_token_id,
        excluded_text_token_ids=excluded_text_ids,
        feature_pool=args.feature_pool,
        activation_pool=args.activation_pool,
    )
    storage_dtype = torch.float16 if args.storage_dtype == "float16" else torch.float32
    writer = ChunkWriter(
        chunk_dir,
        chunk_size=args.chunk_size,
        storage_dtype=storage_dtype,
        start_index=completed,
    )
    try:
        with torch.no_grad():
            for start in range(completed, len(records), args.batch_size):
                end = min(start + args.batch_size, len(records))
                correct_batch = records[start:end]
                shuffled_batch = shuffled_records[start:end]

                correct_inputs = prepare_model_batch(processor, correct_batch, device)
                collector.configure(correct_inputs["input_ids"], correct_inputs.get("attention_mask"))
                model(**correct_inputs, use_cache=False)
                correct_capture = collector.export()

                shuffled_inputs = prepare_model_batch(processor, shuffled_batch, device)
                collector.configure(shuffled_inputs["input_ids"], shuffled_inputs.get("attention_mask"))
                model(**shuffled_inputs, use_cache=False)
                shuffled_capture = collector.export()

                rows = [
                    {
                        "row_index": row_index,
                        "source_index": record["source_index"],
                        "question_id": record["question_id"],
                        "image_id": image_ids[row_index],
                        "shuffled_image_id": normalize_image_id(shuffled_record["images"][0]),
                        "split": splits[row_index],
                        "question": record["question"],
                        "answer": record["answer"],
                    }
                    for row_index, record, shuffled_record in zip(
                        range(start, end),
                        correct_batch,
                        shuffled_batch,
                    )
                ]
                writer.add(correct_capture, shuffled_capture, rows)
                if end % max(args.chunk_size, args.batch_size) == 0 or end == len(records):
                    print(f"Phase 4 collection: {end}/{len(records)} rows", flush=True)
        writer.flush(force=True)
    finally:
        collector.close()

    state.update(
        {
            "complete": True,
            "completed_rows": len(records),
            "model": {
                "num_layers": len(mlp_layers),
                "intermediate_sizes": {
                    str(layer.index): int(layer.mlp.up_proj.weight.shape[0]) for layer in mlp_layers
                },
                "hidden_size": int(mlp_layers[0].mlp.up_proj.weight.shape[1]),
                "image_token_id": image_token_id,
            },
            "representation": {
                "vision": f"decoder layer-0 input, visual tokens, {args.feature_pool}",
                "question": f"decoder layer-0 input, non-visual prompt tokens, {args.feature_pool}",
                "target": (
                    "correct-image minus split-local-shuffled-image down_proj input "
                    f"over non-visual prompt tokens, {args.activation_pool}"
                ),
                "storage_dtype": args.storage_dtype,
            },
        }
    )
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    return state


def main() -> None:
    result = collect(parse_args())
    print(json.dumps({"complete": result["complete"], "output_dir": result["config"]["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
