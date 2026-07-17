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

"""Calibrate per-neuron, per-modality quantile thresholds.

Uses streaming quantile estimation to avoid storing full activations.
Output: neuron_quantiles.pt with per-neuron T_visual and T_text.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import (
    SFTDataCollatorWith4DAttentionMask,
    get_dataset,
    get_template_and_fix_tokenizer,
)
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.vulcan import find_mlp_layers

from dataset_guard import build_dataset_manifest, save_manifest, slice_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate per-neuron quantile thresholds.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--max_samples", type=int, default=500, help="Calibration samples.")
    parser.add_argument("--sample_offset", type=int, default=0, help="Start index in the tokenized dataset.")
    parser.add_argument("--allow_short_dataset", action="store_true",
                        help="Allow fewer rows than requested (diagnostics only).")
    parser.add_argument("--max_image_repeat", type=int, default=5)
    parser.add_argument("--allow_excessive_image_repeats", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--quantiles", default="0.95,0.97,0.99",
                        help="Comma-separated quantiles to compute.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Config overrides must use key=value syntax, got: {override}")
    key, value = override.split("=", maxsplit=1)
    return key.strip(), yaml.safe_load(value)


def build_dataloader(
    train_config: dict[str, Any],
    model,
    tokenizer_module,
    template,
    batch_size: int,
    num_workers: int,
    seed: int,
    sample_offset: int,
    max_samples: int,
    allow_short_dataset: bool,
    max_image_repeat: int,
    allow_excessive_image_repeats: bool,
) -> tuple[DataLoader, dict[str, Any]]:
    model_args, data_args, training_args, _, _ = get_train_args(train_config)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        pad_to_multiple_of=None,
        label_pad_token_id=(
            IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer_module["tokenizer"].pad_token_id
        ),
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=data_args.neat_packing,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )
    dataset, source_indices = slice_dataset(
        dataset_module["train_dataset"], sample_offset, max_samples, allow_short=allow_short_dataset
    )
    manifest = build_dataset_manifest(
        dataset,
        source_indices,
        role="calibration",
        dataset_name=str(data_args.dataset),
        tokenized_path=str(data_args.tokenized_path) if data_args.tokenized_path else None,
        max_image_repeat=max_image_repeat,
        allow_excessive_image_repeats=allow_excessive_image_repeats,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        shuffle=True,
        generator=generator,
    )
    return dataloader, manifest


class StreamingQuantile:
    """Streaming quantile estimation using reservoir sampling.

    Stores a fixed-size reservoir of activation values per neuron,
    then computes quantiles from the reservoir.
    """

    def __init__(self, num_neurons: int, reservoir_size: int = 10000, device: str = "cpu"):
        self.num_neurons = num_neurons
        self.reservoir_size = reservoir_size
        self.device = device

        self.visual_reservoir = torch.zeros(num_neurons, reservoir_size, dtype=torch.float32)
        self.text_reservoir = torch.zeros(num_neurons, reservoir_size, dtype=torch.float32)
        self.visual_count = torch.zeros(num_neurons, dtype=torch.long)
        self.text_count = torch.zeros(num_neurons, dtype=torch.long)
        self.visual_seen = 0
        self.text_seen = 0

    def update(self, activations: torch.Tensor, visual_mask: torch.Tensor, text_mask: torch.Tensor):
        """Update reservoir with new activations.

        Args:
            activations: [seq_len, num_neurons] normalized activations
            visual_mask: [seq_len] bool mask for visual tokens
            text_mask: [seq_len] bool mask for text tokens
        """
        if visual_mask.any():
            v_acts = activations[visual_mask]
            n_visual = v_acts.shape[0]
            for i in range(n_visual):
                self._update_reservoir(
                    self.visual_reservoir, self.visual_count,
                    v_acts[i], self.visual_seen
                )
                self.visual_seen += 1

        if text_mask.any():
            t_acts = activations[text_mask]
            n_text = t_acts.shape[0]
            for i in range(n_text):
                self._update_reservoir(
                    self.text_reservoir, self.text_count,
                    t_acts[i], self.text_seen
                )
                self.text_seen += 1

    def _update_reservoir(self, reservoir: torch.Tensor, counts: torch.Tensor,
                          values: torch.Tensor, seen: int):
        """Update reservoir for one token's activations."""
        idx = seen % self.reservoir_size
        if seen < self.reservoir_size:
            reservoir[:, idx] = values
            counts += 1
        else:
            replace_idx = torch.randint(0, seen + 1, (1,)).item()
            if replace_idx < self.reservoir_size:
                reservoir[:, replace_idx] = values

    def compute_quantiles(self, quantiles: list[float]) -> dict[str, torch.Tensor]:
        """Compute quantiles from reservoir.

        Returns:
            dict with 'visual' and 'text' tensors of shape [num_neurons, len(quantiles)]
        """
        results = {}
        for modality, reservoir, count in [
            ("visual", self.visual_reservoir, self.visual_count),
            ("text", self.text_reservoir, self.text_count)
        ]:
            quantile_values = torch.zeros(self.num_neurons, len(quantiles))
            for j in range(self.num_neurons):
                n = min(count[j].item(), self.reservoir_size)
                if n > 0:
                    sorted_vals = reservoir[j, :n].sort().values
                    for qi, q in enumerate(quantiles):
                        idx = int(q * (n - 1))
                        idx = min(idx, n - 1)
                        quantile_values[j, qi] = sorted_vals[idx]
            results[modality] = quantile_values
        return results


class ActivationCollector:
    """Collects FFN activations via forward-pre-hook on down_proj."""

    def __init__(self, model, mlp_layers):
        self.model = model
        self.mlp_layers = mlp_layers
        self._hooks = []
        self._act_store: dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, args, output):
            self._act_store[layer_idx] = args[0].detach()
        return hook_fn

    def _register_hooks(self):
        for layer_ref in self.mlp_layers:
            h = layer_ref.mlp.down_proj.register_forward_hook(self._make_hook(layer_ref.index))
            self._hooks.append(h)

    def get_captured(self) -> dict[int, torch.Tensor]:
        return dict(self._act_store)

    def clear(self):
        self._act_store.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def build_token_masks(
    input_ids: torch.Tensor,
    image_token_id: int,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build visual and caption masks."""
    visual_mask = input_ids == image_token_id
    if labels is not None:
        caption_mask = (labels != IGNORE_INDEX) & (~visual_mask)
    else:
        caption_mask = torch.zeros_like(visual_mask)
    return visual_mask, caption_mask


def calibrate(
    model,
    dataloader: DataLoader,
    mlp_layers: list,
    image_token_id: int,
    global_max: dict[int, torch.Tensor],
    max_samples: int,
    device: torch.device,
    quantiles: list[float],
) -> dict[int, dict[str, torch.Tensor]]:
    """Run calibration to estimate per-neuron quantile thresholds.

    Returns:
        dict mapping layer_idx to dict with:
            - 'visual': [num_neurons, len(quantiles)] quantile values
            - 'text': [num_neurons, len(quantiles)] quantile values
    """
    intermediate_sizes = {ref.index: int(ref.mlp.up_proj.weight.shape[0]) for ref in mlp_layers}

    reservoirs = {
        layer_idx: StreamingQuantile(size, reservoir_size=10000)
        for layer_idx, size in intermediate_sizes.items()
    }

    collector = ActivationCollector(model, mlp_layers)
    model.eval()
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)

            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]

            forward_inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels",)}
            _ = model(**forward_inputs)

            visual_mask, caption_mask = build_token_masks(input_ids, image_token_id, labels)

            captured = collector.get_captured()
            for layer_idx, act in captured.items():
                act_seq = act[:, :seq_len, :].float()
                gmax = global_max[layer_idx].to(act_seq.device).float().clamp(min=1e-4)

                for b in range(batch_size):
                    a = act_seq[b]
                    a_norm = torch.clamp(a, min=0) / gmax * 10.0

                    vm = visual_mask[b].cpu()
                    cm = caption_mask[b].cpu()
                    a_cpu = a_norm.cpu()

                    reservoirs[layer_idx].update(a_cpu, vm, cm)

            collector.clear()
            sample_count += batch_size

            if (batch_idx + 1) % 50 == 0:
                print(f"  Calibration: processed {sample_count} samples", flush=True)

    collector.remove_hooks()

    results = {}
    for layer_idx, reservoir in reservoirs.items():
        q_values = reservoir.compute_quantiles(quantiles)
        results[layer_idx] = {
            "visual": q_values["visual"],
            "text": q_values["text"],
            "visual_seen": reservoir.visual_seen,
            "text_seen": reservoir.text_seen,
        }

    return results


def save_calibration(
    output_dir: str,
    calibration: dict[int, dict[str, torch.Tensor]],
    quantiles: list[float],
    config: dict,
):
    """Save calibration results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    quantile_tensors = {}
    for layer_idx, data in calibration.items():
        quantile_tensors[layer_idx] = {
            "visual": data["visual"],
            "text": data["text"],
        }

    torch.save(quantile_tensors, output_path / "neuron_quantiles.pt")
    print(f"Saved neuron_quantiles.pt")

    import json
    summary = {}
    for layer_idx, data in calibration.items():
        summary[f"layer_{layer_idx}"] = {
            "visual_seen": data["visual_seen"],
            "text_seen": data["text_seen"],
            "visual_q97_mean": data["visual"][:, 1].mean().item() if data["visual"].shape[1] > 1 else 0,
            "text_q97_mean": data["text"][:, 1].mean().item() if data["text"].shape[1] > 1 else 0,
        }

    with open(output_path / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    config["quantiles"] = quantiles
    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def print_summary(calibration: dict[int, dict[str, torch.Tensor]], quantiles: list[float]):
    """Print calibration summary."""
    print(f"\n{'='*60}")
    print("Calibration Summary")
    print(f"{'='*60}")

    print(f"\nQuantiles computed: {quantiles}")
    print(f"\n{'Layer':>6} | {'V_seen':>8} {'T_seen':>8} | ", end="")
    for q in quantiles:
        print(f"V_q{int(q*100):02d}_mean {'T_q' + str(int(q*100)).zfill(2) + '_mean':>10} | ", end="")
    print()
    print("-" * 80)

    for layer_idx in sorted(calibration.keys()):
        data = calibration[layer_idx]
        v_seen = data["visual_seen"]
        t_seen = data["text_seen"]
        print(f"  {layer_idx:>4} | {v_seen:>8} {t_seen:>8} | ", end="")
        for qi, q in enumerate(quantiles):
            v_mean = data["visual"][:, qi].mean().item()
            t_mean = data["text"][:, qi].mean().item()
            print(f"{v_mean:>10.3f} {t_mean:>10.3f} | ", end="")
        print()


def main() -> None:
    args = parse_args()
    train_config = load_config(args.config)

    train_config["do_train"] = False
    train_config["do_eval"] = False
    train_config["do_predict"] = False
    train_config.setdefault("output_dir", "saves/neuron_typing/calibration_tmp")

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(train_config)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    mlp_layers = find_mlp_layers(model)
    print(f"Found {len(mlp_layers)} MLP layers")

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None and tokenizer_module.get("processor") is not None:
        image_token_id = getattr(tokenizer_module["processor"], "image_token_id", None)
    if image_token_id is None:
        image_token_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    print(f"Image token ID: {image_token_id}")

    global_max_path = Path(args.output_dir).parent / "activations" / "global_max.pt"
    if not global_max_path.exists():
        print(f"ERROR: global_max.pt not found at {global_max_path}")
        print("Run Pass 1 first to collect global_max.")
        sys.exit(1)

    global_max = torch.load(global_max_path, map_location="cpu")
    print(f"Loaded global_max from {global_max_path}")

    quantiles = [float(q) for q in args.quantiles.split(",")]
    print(f"Quantiles to compute: {quantiles}")

    dataloader, dataset_manifest = build_dataloader(
        train_config, model, tokenizer_module, template,
        args.batch_size, args.num_workers, args.seed,
        args.sample_offset, args.max_samples, args.allow_short_dataset,
        args.max_image_repeat, args.allow_excessive_image_repeats,
    )
    manifest_path = Path(args.output_dir) / "sample_manifest.json"
    save_manifest(dataset_manifest, manifest_path)

    print(f"\n{'='*60}")
    print(f"Running calibration on {args.max_samples} samples")
    print(f"{'='*60}")

    calibration = calibrate(
        model, dataloader, mlp_layers, image_token_id,
        global_max, args.max_samples, device, quantiles
    )

    config = {
        "model_name": model_args.model_name_or_path,
        "max_samples": args.max_samples,
        "sample_offset": args.sample_offset,
        "actual_samples": dataset_manifest["num_rows"],
        "dataset_manifest": str(manifest_path),
        "image_token_id": image_token_id,
        "num_mlp_layers": len(mlp_layers),
        "intermediate_size": int(mlp_layers[0].mlp.up_proj.weight.shape[0]) if mlp_layers else 0,
    }

    save_calibration(args.output_dir, calibration, quantiles, config)
    print_summary(calibration, quantiles)

    print(f"\nCalibration saved to {args.output_dir}")


if __name__ == "__main__":
    main()
