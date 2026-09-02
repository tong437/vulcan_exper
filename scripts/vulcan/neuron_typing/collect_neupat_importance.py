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

"""Collect NeuPAT text/vision importance and allocate FFN neuron roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from collect_ffn_activations import (  # noqa: E402
    ActivationCollector,
    build_dataloader,
    load_config,
    parse_config_override,
)
from dataset_guard import save_manifest  # noqa: E402

from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_train_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.train.vulcan import allocate_neupat_roles, find_mlp_layers  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NeuPAT modality-associated FFN importance.")
    parser.add_argument("--vision_config", required=True, help="LlamaFactory config for the image-text probe set.")
    parser.add_argument("--text_config", required=True, help="LlamaFactory config for the text-only probe set.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vision_samples", type=int, default=2048)
    parser.add_argument("--text_samples", type=int, default=2048)
    parser.add_argument("--vision_offset", type=int, default=0)
    parser.add_argument("--text_offset", type=int, default=0)
    parser.add_argument("--tau_vision", type=float, default=0.8)
    parser.add_argument("--tau_text", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow_short_dataset", action="store_true")
    parser.add_argument("--max_image_repeat", type=int, default=6)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Apply the same key=value override to both probe configs. Repeatable.",
    )
    parser.add_argument(
        "--vision_override",
        action="append",
        default=[],
        help="Apply a key=value override only to the image-text probe config. Repeatable.",
    )
    parser.add_argument(
        "--text_override",
        action="append",
        default=[],
        help="Apply a key=value override only to the text-only probe config. Repeatable.",
    )
    return parser.parse_args()


def prepare_config(path: str, overrides: list[str]) -> dict[str, Any]:
    config = load_config(path)
    for override in overrides:
        key, value = parse_config_override(override)
        config[key] = value
    config["do_train"] = False
    config["do_eval"] = False
    config["do_predict"] = False
    config.setdefault("output_dir", "saves/neuron_typing/neupat_tmp")
    config.setdefault("preprocessing_num_workers", 8)
    return config


def _valid_token_mask(batch: dict[str, torch.Tensor], pad_token_id: int) -> torch.Tensor:
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None and attention_mask.ndim == 2:
        return attention_mask.bool()
    return batch["input_ids"] != pad_token_id


def collect_activation_rms(
    model: torch.nn.Module,
    dataloader,
    mlp_layers,
    *,
    device: torch.device,
    pad_token_id: int,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Collect RMS intermediate-channel activations over valid tokens."""
    collector = ActivationCollector(model, mlp_layers)
    sum_squares = {
        layer.index: torch.zeros(layer.mlp.up_proj.weight.shape[0], dtype=torch.float64) for layer in mlp_layers
    }
    valid_tokens = 0
    samples = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"]
                token_mask = _valid_token_mask(batch, pad_token_id)
                forward_inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
                model(**forward_inputs)
                captured = collector.get_captured()
                mask = token_mask.to(device)
                token_count = int(mask.sum().item())
                for layer_idx, activations in captured.items():
                    aligned = activations[:, : input_ids.shape[1], :].float()
                    selected = aligned[mask]
                    sum_squares[layer_idx] += selected.square().sum(dim=0).double().cpu()
                collector.clear()
                valid_tokens += token_count
                samples += int(input_ids.shape[0])
                if (batch_idx + 1) % 50 == 0:
                    print(f"  processed {samples} samples / {valid_tokens} valid tokens", flush=True)
    finally:
        collector.remove_hooks()
    if valid_tokens == 0:
        raise ValueError("NeuPAT probing found no valid tokens.")
    rms = {layer_idx: (values / valid_tokens).sqrt().float() for layer_idx, values in sum_squares.items()}
    return rms, samples, valid_tokens


def build_neupat_outputs(
    mlp_layers,
    text_rms: dict[int, torch.Tensor],
    vision_rms: dict[int, torch.Tensor],
    *,
    tau_text: float,
    tau_vision: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifact_layers: dict[str, Any] = {}
    total_counts = dict.fromkeys(("language", "multimodal", "shared", "reserve"), 0)
    per_layer_summary: dict[str, Any] = {}
    for layer_ref in mlp_layers:
        layer_idx = layer_ref.index
        output_norm = layer_ref.mlp.down_proj.weight.detach().float().norm(dim=0).cpu()
        text_importance = output_norm * text_rms[layer_idx]
        vision_importance = output_norm * vision_rms[layer_idx]
        roles = allocate_neupat_roles(
            text_importance,
            vision_importance,
            tau_text=tau_text,
            tau_vision=tau_vision,
        )
        role_by_neuron = [""] * output_norm.numel()
        layer_row: dict[str, Any] = {
            "intermediate_size": output_norm.numel(),
            "text_important": roles["text_important"].tolist(),
            "vision_important": roles["vision_important"].tolist(),
        }
        layer_counts: dict[str, int] = {}
        for role in ("language", "multimodal", "shared", "reserve"):
            indices = roles[role].tolist()
            layer_row[role] = indices
            layer_counts[role] = len(indices)
            total_counts[role] += len(indices)
            for neuron_idx in indices:
                role_by_neuron[neuron_idx] = role
        artifact_layers[str(layer_idx)] = layer_row
        per_layer_summary[str(layer_idx)] = layer_counts
        text_denom = float(text_importance.sum().item())
        vision_denom = float(vision_importance.sum().item())
        for neuron_idx in range(output_norm.numel()):
            role = role_by_neuron[neuron_idx]
            rows.append(
                {
                    "layer": layer_idx,
                    "neuron_idx": neuron_idx,
                    "neupat_text_rms": float(text_rms[layer_idx][neuron_idx]),
                    "neupat_vision_rms": float(vision_rms[layer_idx][neuron_idx]),
                    "neupat_output_norm": float(output_norm[neuron_idx]),
                    "neupat_text_importance": float(text_importance[neuron_idx]),
                    "neupat_vision_importance": float(vision_importance[neuron_idx]),
                    "neupat_text_mass": float(text_importance[neuron_idx] / max(text_denom, 1e-12)),
                    "neupat_vision_mass": float(vision_importance[neuron_idx] / max(vision_denom, 1e-12)),
                    "neupat_preference": float(vision_importance[neuron_idx] - text_importance[neuron_idx]),
                    "neupat_overall_importance": float(
                        (vision_importance[neuron_idx] + text_importance[neuron_idx]) / 2
                    ),
                    "neupat_role": role,
                    **{f"neupat_{name}": role == name for name in ("language", "multimodal", "shared", "reserve")},
                }
            )
    artifact = {
        "artifact_version": 1,
        "method": "neupat",
        "tau_text": tau_text,
        "tau_vision": tau_vision,
        "layers": artifact_layers,
    }
    summary = {"total_counts": total_counts, "per_layer_counts": per_layer_summary}
    return pd.DataFrame(rows), artifact, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    vision_config = prepare_config(args.vision_config, [*args.override, *args.vision_override])
    text_config = prepare_config(args.text_config, [*args.override, *args.text_override])
    vision_model_path = vision_config.get("model_name_or_path")
    if text_config.get("model_name_or_path") != vision_model_path:
        raise ValueError("NeuPAT text and vision probe configs must use the same model_name_or_path.")
    if text_config.get("template") != vision_config.get("template"):
        raise ValueError("NeuPAT text and vision probe configs must use the same chat template.")

    model_args, data_args, _, finetuning_args, _ = get_train_args(vision_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    mlp_layers = find_mlp_layers(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_specs = {
        "vision": (vision_config, args.vision_offset, args.vision_samples),
        "text": (text_config, args.text_offset, args.text_samples),
    }
    probe_rms: dict[str, dict[int, torch.Tensor]] = {}
    probe_stats: dict[str, Any] = {}
    for probe_name, (config, offset, sample_count) in probe_specs.items():
        print(f"Collecting NeuPAT {probe_name} probe statistics", flush=True)
        dataloader, manifest = build_dataloader(
            config,
            model,
            tokenizer_module,
            template,
            args.batch_size,
            args.num_workers,
            args.seed,
            offset,
            sample_count,
            args.allow_short_dataset,
            args.max_image_repeat,
            args.allow_excessive_image_repeats,
            "typing",
        )
        manifest["role"] = f"neupat_{probe_name}_probe"
        save_manifest(manifest, output_dir / f"{probe_name}_manifest.json")
        rms, processed_samples, valid_tokens = collect_activation_rms(
            model,
            dataloader,
            mlp_layers,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
        )
        probe_rms[probe_name] = rms
        probe_stats[probe_name] = {"samples": processed_samples, "valid_tokens": valid_tokens}

    score_table, artifact, summary = build_neupat_outputs(
        mlp_layers,
        probe_rms["text"],
        probe_rms["vision"],
        tau_text=args.tau_text,
        tau_vision=args.tau_vision,
    )
    score_path = output_dir / "neupat_scores.parquet"
    score_table.to_parquet(score_path, index=False)
    artifact["score_file"] = str(score_path.resolve())
    artifact["score_sha256"] = sha256_file(score_path)
    artifact["model_name_or_path"] = vision_model_path
    artifact["probe_stats"] = probe_stats
    write_json(output_dir / "neupat_roles.json", artifact)
    write_json(output_dir / "neupat_summary.json", {**summary, "probe_stats": probe_stats})
    print(json.dumps({"score_file": str(score_path), "summary": summary["total_counts"]}, indent=2))


if __name__ == "__main__":
    main()
