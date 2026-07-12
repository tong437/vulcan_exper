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

"""Two-pass FFN activation collection for neuron typing.

Supports two threshold modes:
- fixed: Use fixed thresholds (T_visual, T_text) - for comparison with paper
- quantile: Use per-neuron, per-modality quantile thresholds (recommended)

Pass 1: Global max activation per neuron across all samples.
Pass 2: Per-sample visual/text activation count for neuron classification.
"""

import argparse
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FFN activations for neuron typing.")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--output_dir", required=True, help="Directory to save activation stats.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to process.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold_mode", choices=["fixed", "quantile"], default="quantile",
                        help="Threshold mode: fixed (paper) or quantile (recommended)")
    parser.add_argument("--t_visual", type=float, default=2.0,
                        help="Fixed visual activation threshold (only for fixed mode).")
    parser.add_argument("--t_text", type=float, default=3.0,
                        help="Fixed text activation threshold (only for fixed mode).")
    parser.add_argument("--quantile_path", type=str, default=None,
                        help="Path to neuron_quantiles.pt from calibration (for quantile mode).")
    parser.add_argument("--quantile_idx_visual", type=int, default=1,
                        help="Quantile index for visual threshold (0=q95, 1=q97, 2=q99).")
    parser.add_argument("--quantile_idx_text", type=int, default=0,
                        help="Quantile index for text threshold (0=q95, 1=q97, 2=q99).")
    parser.add_argument("--visual_ratio", type=float, default=0.005,
                        help="Min ratio of visual tokens that must exceed threshold.")
    parser.add_argument("--visual_min_count", type=int, default=4,
                        help="Min absolute visual token count.")
    parser.add_argument("--text_ratio", type=float, default=0.05,
                        help="Min ratio of text tokens that must exceed threshold.")
    parser.add_argument("--text_min_count", type=int, default=1,
                        help="Min absolute text token count.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K samples per neuron.")
    parser.add_argument("--sample_score_top_m", type=int, default=5,
                        help="Top-m tokens for sample score (more stable than single max).")
    parser.add_argument("--pilot", action="store_true", help="Pilot mode: only Pass 1 for threshold calibration.")
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_config_override(override: str) -> tuple[str, Any]:
    if "=" not in override:
        raise ValueError(f"Config overrides must use key=value syntax, got: {override}")
    key, value = override.split("=", maxsplit=1)
    key = key.strip()
    if not key:
        raise ValueError(f"Config override key cannot be empty: {override}")
    return key, yaml.safe_load(value)


def build_dataloader(
    train_config: dict[str, Any],
    model,
    tokenizer_module,
    template,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
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
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset_module["train_dataset"],
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=num_workers,
        shuffle=True,
        generator=generator,
    )


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build visual, caption, and ignore masks.

    Returns:
        visual_mask: [batch, seq] bool - image tokens
        caption_mask: [batch, seq] bool - caption/answer tokens
        ignore_mask: [batch, seq] bool - everything else
    """
    visual_mask = input_ids == image_token_id

    if labels is not None:
        caption_mask = (labels != IGNORE_INDEX) & (~visual_mask)
    else:
        caption_mask = torch.zeros_like(visual_mask)

    ignore_mask = ~visual_mask & ~caption_mask
    return visual_mask, caption_mask, ignore_mask


def compute_required_counts(
    num_visual_tokens: int,
    num_text_tokens: int,
    visual_ratio: float,
    visual_min_count: int,
    text_ratio: float,
    text_min_count: int,
) -> tuple[int, int]:
    """Compute ratio-based required counts.

    Args:
        num_visual_tokens: Number of visual tokens in this sample
        num_text_tokens: Number of text tokens in this sample
        visual_ratio: Min ratio of visual tokens
        visual_min_count: Min absolute visual count
        text_ratio: Min ratio of text tokens
        text_min_count: Min absolute text count

    Returns:
        (visual_required, text_required)
    """
    visual_required = max(visual_min_count, math.ceil(visual_ratio * num_visual_tokens))
    text_required = max(text_min_count, math.ceil(text_ratio * num_text_tokens))
    return visual_required, text_required


def compute_sample_score_top_m(
    activations: torch.Tensor,
    token_mask: torch.Tensor,
    top_m: int,
) -> torch.Tensor:
    """Compute sample score using top-m mean activation.

    More stable than single max which is sensitive to outliers.

    Args:
        activations: [seq_len, num_neurons] normalized activations
        token_mask: [seq_len] bool mask for relevant tokens
        top_m: Number of top tokens to average

    Returns:
        [num_neurons] tensor of sample scores
    """
    if not token_mask.any():
        return torch.zeros(activations.shape[1])

    selected = activations[token_mask]
    if selected.shape[0] <= top_m:
        return selected.mean(dim=0)

    topk_values = torch.topk(selected, k=top_m, dim=0).values
    return topk_values.mean(dim=0)


def pass1_global_max(
    model,
    dataloader: DataLoader,
    mlp_layers: list,
    image_token_id: int,
    max_samples: int,
    device: torch.device,
) -> tuple[dict[int, torch.Tensor], int]:
    """Pass 1: Collect global max activation per neuron.

    Returns:
        global_max: dict[layer_idx -> [intermediate_size]] tensor
        sample_count: number of processed samples
    """
    collector = ActivationCollector(model, mlp_layers)
    intermediate_sizes = {ref.index: int(ref.mlp.up_proj.weight.shape[0]) for ref in mlp_layers}

    global_max = {
        layer_idx: torch.zeros(size, device="cpu", dtype=torch.float32)
        for layer_idx, size in intermediate_sizes.items()
    }

    model.eval()
    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break

            input_ids = batch["input_ids"].to(device)
            batch_size = input_ids.shape[0]

            forward_inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels",)}
            _ = model(**forward_inputs)

            captured = collector.get_captured()
            for layer_idx, act in captured.items():
                act_seq = act[:, :input_ids.shape[1], :].float()
                act_pos = torch.clamp(act_seq, min=0)
                batch_max = act_pos.amax(dim=1)
                global_max[layer_idx] = torch.maximum(global_max[layer_idx], batch_max.cpu())

            collector.clear()
            sample_count += batch_size

            if (batch_idx + 1) % 50 == 0:
                print(f"  Pass 1: processed {sample_count} samples", flush=True)

    collector.remove_hooks()
    return global_max, sample_count


def load_quantile_thresholds(
    quantile_path: str,
    quantile_idx_visual: int,
    quantile_idx_text: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Load per-neuron quantile thresholds from calibration.

    Supports asymmetric quantile indices (e.g., q97 for visual, q95 for text)
    to calibrate for the visual/text token count disparity.

    Returns:
        dict mapping layer_idx to dict with 'visual' and 'text' tensors [num_neurons]
    """
    data = torch.load(quantile_path, map_location="cpu")

    thresholds = {}
    for layer_key, layer_data in data.items():
        layer_idx = int(layer_key) if isinstance(layer_key, (int, str)) else layer_key
        thresholds[layer_idx] = {
            "visual": layer_data["visual"][:, quantile_idx_visual],
            "text": layer_data["text"][:, quantile_idx_text],
        }

    return thresholds


def pass2_classify_neurons(
    model,
    dataloader: DataLoader,
    mlp_layers: list,
    image_token_id: int,
    global_max: dict[int, torch.Tensor],
    max_samples: int,
    device: torch.device,
    threshold_mode: str,
    t_visual: float,
    t_text: float,
    quantile_thresholds: dict[int, dict[str, torch.Tensor]] | None,
    visual_ratio: float,
    visual_min_count: int,
    text_ratio: float,
    text_min_count: int,
    top_k: int,
    sample_score_top_m: int,
) -> dict[str, Any]:
    """Pass 2: Classify neurons per sample and compute type scores.

    Batch-level vectorized: processes all samples in a batch at once,
    eliminating the per-sample Python loop.

    Returns:
        results: dict with neuron type scores and top-K samples
    """
    collector = ActivationCollector(model, mlp_layers)
    intermediate_sizes = {ref.index: int(ref.mlp.up_proj.weight.shape[0]) for ref in mlp_layers}

    neuron_scores = {
        layer_idx: {
            "unknown_count": torch.zeros(size, dtype=torch.long),
            "total_samples": 0,
            # Separate top-K for visual and text modalities
            "top_k_visual_scores": torch.full((top_k, size), -1.0),
            "top_k_visual_samples": torch.full((top_k, size), -1, dtype=torch.long),
            "top_k_text_scores": torch.full((top_k, size), -1.0),
            "top_k_text_samples": torch.full((top_k, size), -1, dtype=torch.long),
            # Per-modality-topK classification counts
            "visual_count_from_vis_topk": torch.zeros(size, dtype=torch.long),
            "text_count_from_txt_topk": torch.zeros(size, dtype=torch.long),
            "multimodal_count_from_joint": torch.zeros(size, dtype=torch.long),
        }
        for layer_idx, size in intermediate_sizes.items()
    }

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

            visual_mask, caption_mask, ignore_mask = build_token_masks(input_ids, image_token_id, labels)
            combined_mask = visual_mask | caption_mask  # [B, S]

            # Per-sample token counts for ratio-based thresholds: [B]
            num_visual_per_sample = visual_mask.sum(dim=1)  # [B]
            num_text_per_sample = caption_mask.sum(dim=1)  # [B]
            vis_req = torch.clamp(
                torch.maximum(
                    torch.full((batch_size,), visual_min_count, dtype=torch.long, device=device),
                    torch.ceil(num_visual_per_sample.float() * visual_ratio).long(),
                ),
                min=1,
            )  # [B]
            txt_req = torch.clamp(
                torch.maximum(
                    torch.full((batch_size,), text_min_count, dtype=torch.long, device=device),
                    torch.ceil(num_text_per_sample.float() * text_ratio).long(),
                ),
                min=1,
            )  # [B]

            captured = collector.get_captured()
            for layer_idx, act in captured.items():
                act_seq = act[:, :seq_len, :].float()  # [B, S, D]
                gmax = global_max[layer_idx].to(act_seq.device).float().clamp(min=1e-4)  # [D]
                size = intermediate_sizes[layer_idx]

                # Thresholds
                if threshold_mode == "quantile" and quantile_thresholds is not None:
                    t_v = quantile_thresholds[layer_idx]["visual"].to(act_seq.device)  # [D]
                    t_t = quantile_thresholds[layer_idx]["text"].to(act_seq.device)    # [D]
                else:
                    t_v = t_visual
                    t_t = t_text

                # Normalize: [B, S, D]
                a_norm = torch.clamp(act_seq, min=0) / gmax * 10.0

                # Per-sample per-neuron counts: [B, D]
                # For visual: count visual tokens exceeding threshold per neuron
                if threshold_mode == "quantile":
                    # a_norm[:, :, j] > t_v[j] for visual tokens
                    # Expand masks: [B, S, 1] to broadcast with [B, S, D]
                    vm_expanded = visual_mask.unsqueeze(-1)  # [B, S, 1]
                    cm_expanded = caption_mask.unsqueeze(-1)  # [B, S, 1]

                    v_exceed = (a_norm > t_v) & vm_expanded  # [B, S, D]
                    t_exceed = (a_norm > t_t) & cm_expanded  # [B, S, D]

                    v_count = v_exceed.sum(dim=1)  # [B, D]
                    t_count = t_exceed.sum(dim=1)  # [B, D]
                else:
                    vm_expanded = visual_mask.unsqueeze(-1)
                    cm_expanded = caption_mask.unsqueeze(-1)

                    v_exceed = (a_norm > t_v) & vm_expanded
                    t_exceed = (a_norm > t_t) & cm_expanded

                    v_count = v_exceed.sum(dim=1)  # [B, D]
                    t_count = t_exceed.sum(dim=1)  # [B, D]

                # Per-sample classification: [B, D]
                vis_req_expanded = vis_req.unsqueeze(1)  # [B, 1]
                txt_req_expanded = txt_req.unsqueeze(1)  # [B, 1]

                is_visual = (v_count > vis_req_expanded) & (t_count <= txt_req_expanded)
                is_text = (v_count <= vis_req_expanded) & (t_count > txt_req_expanded)
                is_multimodal = (v_count > vis_req_expanded) & (t_count > txt_req_expanded)
                is_unknown = (v_count <= vis_req_expanded) & (t_count <= txt_req_expanded)

                # Accumulate counts
                scores = neuron_scores[layer_idx]
                scores["unknown_count"] += is_unknown.sum(dim=0).cpu().long()
                scores["total_samples"] += batch_size

                # Per-sample modality-specific scores and top-K update
                for b in range(batch_size):
                    vm_b = visual_mask[b]   # [S]
                    cm_b = caption_mask[b]  # [S]

                    # Visual score: top-m mean over visual tokens
                    if vm_b.any():
                        v_selected = a_norm[b, vm_b]  # [n_vis, D]
                        if v_selected.shape[0] <= sample_score_top_m:
                            visual_score = v_selected.mean(dim=0)
                        else:
                            visual_score = torch.topk(v_selected, k=sample_score_top_m, dim=0).values.mean(dim=0)
                    else:
                        visual_score = torch.zeros(size, device=a_norm.device)

                    # Text score: top-m mean over text tokens
                    if cm_b.any():
                        t_selected = a_norm[b, cm_b]  # [n_txt, D]
                        if t_selected.shape[0] <= sample_score_top_m:
                            text_score = t_selected.mean(dim=0)
                        else:
                            text_score = torch.topk(t_selected, k=sample_score_top_m, dim=0).values.mean(dim=0)
                    else:
                        text_score = torch.zeros(size, device=a_norm.device)

                    sample_idx = sample_count + b
                    _separate_topk_update(
                        scores, visual_score.cpu(), text_score.cpu(), sample_idx,
                        is_visual[b].cpu(), is_text[b].cpu(), is_multimodal[b].cpu(),
                        top_k,
                    )

            collector.clear()
            sample_count += batch_size

            if (batch_idx + 1) % 50 == 0:
                print(f"  Pass 2: processed {sample_count} samples", flush=True)

    collector.remove_hooks()

    results = {}
    for layer_idx, scores in neuron_scores.items():
        total = scores["total_samples"]
        if total == 0:
            continue

        gmax = global_max[layer_idx].float()
        dead_mask = (gmax <= 1e-6).tolist()

        # Compute p_type from separate modality-specific top-K lists
        # p_visual: from visual_topK, fraction where visual_on AND NOT text_on
        # p_text: from text_topK, fraction where text_on AND NOT visual_on
        # p_multimodal: from union of visual_topK and text_topK, fraction where both_on
        vis_topk_valid = (scores["top_k_visual_scores"] >= 0).sum(dim=0).float().clamp(min=1)  # [D]
        txt_topk_valid = (scores["top_k_text_scores"] >= 0).sum(dim=0).float().clamp(min=1)    # [D]

        p_visual = (scores["visual_count_from_vis_topk"] / vis_topk_valid).tolist()
        p_text = (scores["text_count_from_txt_topk"] / txt_topk_valid).tolist()
        # multimodal: use the union of both top-K lists
        joint_valid = (vis_topk_valid + txt_topk_valid).clamp(min=1)
        p_multimodal = (scores["multimodal_count_from_joint"] / joint_valid).tolist()
        p_unknown = (scores["unknown_count"] / total).tolist()

        results[layer_idx] = {
            "p_visual": p_visual,
            "p_text": p_text,
            "p_multimodal": p_multimodal,
            "p_unknown": p_unknown,
            "total_samples": total,
            "dead_mask": dead_mask,
            "top_k_visual": _tensor_to_topk_list(scores["top_k_visual_scores"], scores["top_k_visual_samples"]),
            "top_k_text": _tensor_to_topk_list(scores["top_k_text_scores"], scores["top_k_text_samples"]),
        }

    return results


def _separate_topk_update(
    scores: dict,
    visual_score: torch.Tensor,
    text_score: torch.Tensor,
    sample_idx: int,
    is_visual: torch.Tensor,
    is_text: torch.Tensor,
    is_multimodal: torch.Tensor,
    top_k: int,
):
    """Update separate visual/text top-K lists and classify samples within each.

    For visual_topK: update by visual_score, then check text_on to classify.
    For text_topK: update by text_score, then check visual_on to classify.
    """
    # Update visual top-K for all neurons (not just visual_on ones)
    _update_single_topk(scores, visual_score, sample_idx, "top_k_visual_scores", "top_k_visual_samples", top_k)
    # Update text top-K for all neurons
    _update_single_topk(scores, text_score, sample_idx, "top_k_text_scores", "top_k_text_samples", top_k)

    # Classify samples that entered visual_topK
    # For each neuron, check if this sample is in its visual_topK
    vis_in_topk = _is_in_topk(scores["top_k_visual_samples"], sample_idx)  # [D]
    if vis_in_topk.any():
        # In visual topK: visual_on AND NOT text_on → visual; both → multimodal
        vis_on = is_visual | is_multimodal  # visual activation present
        txt_on = is_text | is_multimodal    # text activation present
        scores["visual_count_from_vis_topk"] += (vis_in_topk & vis_on & ~txt_on).long()
        scores["multimodal_count_from_joint"] += (vis_in_topk & vis_on & txt_on).long()

    # Classify samples that entered text_topK
    txt_in_topk = _is_in_topk(scores["top_k_text_samples"], sample_idx)  # [D]
    if txt_in_topk.any():
        vis_on = is_visual | is_multimodal
        txt_on = is_text | is_multimodal
        scores["text_count_from_txt_topk"] += (txt_in_topk & txt_on & ~vis_on).long()
        scores["multimodal_count_from_joint"] += (txt_in_topk & vis_on & txt_on).long()


def _update_single_topk(
    scores: dict,
    new_score: torch.Tensor,
    sample_idx: int,
    score_key: str,
    sample_key: str,
    top_k: int,
):
    """Update one top-K list with a new sample's scores for all neurons at once."""
    # For each neuron, try to insert new_score into the top-K
    old_scores = scores[score_key]  # [K, D]
    # Find the minimum score in each neuron's top-K
    min_scores = old_scores[-1]  # [D] (last row = smallest since sorted desc)
    # Only update neurons where new_score > min_score (or where there's empty space)
    can_update = (new_score > min_scores) | (old_scores[0] < 0)  # [D]

    if not can_update.any():
        return

    neuron_indices = can_update.nonzero(as_tuple=True)[0]
    ns = new_score[neuron_indices]
    si = torch.full_like(neuron_indices, sample_idx)

    old_s = scores[score_key][:, neuron_indices]
    old_si = scores[sample_key][:, neuron_indices]

    combined_s = torch.cat([old_s, ns.unsqueeze(0)], dim=0)
    combined_si = torch.cat([old_si, si.unsqueeze(0)], dim=0)

    topk_s, topk_idx = torch.topk(combined_s, k=top_k, dim=0, sorted=True)
    topk_si = combined_si.gather(0, topk_idx)

    scores[score_key][:, neuron_indices] = topk_s
    scores[sample_key][:, neuron_indices] = topk_si


def _is_in_topk(samples_tensor: torch.Tensor, sample_idx: int) -> torch.Tensor:
    """Check if sample_idx appears in any neuron's top-K list.

    Args:
        samples_tensor: [K, D] tensor of sample indices
        sample_idx: the sample index to look for

    Returns:
        [D] bool tensor
    """
    return (samples_tensor == sample_idx).any(dim=0)


def _tensor_to_topk_list(
    scores_tensor: torch.Tensor,
    samples_tensor: torch.Tensor,
) -> list[list[tuple[float, int]]]:
    """Convert top-K tensors to list-of-lists-of-tuples format.

    Args:
        scores_tensor: [top_k, num_neurons]
        samples_tensor: [top_k, num_neurons]

    Returns:
        list of num_neurons entries, each is list of (score, sample_idx) tuples
    """
    num_neurons = scores_tensor.shape[1]
    result = []
    for j in range(num_neurons):
        entries = []
        for k in range(scores_tensor.shape[0]):
            s = scores_tensor[k, j].item()
            if s < 0:
                break
            entries.append((s, samples_tensor[k, j].item()))
        result.append(entries)
    return result


def save_results(
    output_dir: str,
    global_max: dict[int, torch.Tensor],
    neuron_scores: dict[str, Any],
    config: dict[str, Any],
):
    """Save activation collection results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    global_max_path = output_path / "global_max.pt"
    torch.save({k: v.half() for k, v in global_max.items()}, global_max_path)
    print(f"Saved global_max to {global_max_path}")

    scores_path = output_path / "neuron_scores.json"
    serializable_scores = {}
    for layer_idx, scores in neuron_scores.items():
        layer_key = f"layer_{layer_idx}"
        serializable_scores[layer_key] = {
            "p_visual": scores["p_visual"],
            "p_text": scores["p_text"],
            "p_multimodal": scores["p_multimodal"],
            "p_unknown": scores["p_unknown"],
            "total_samples": scores["total_samples"],
        }

    with open(scores_path, "w") as f:
        json.dump(serializable_scores, f, indent=2)
    print(f"Saved neuron_scores to {scores_path}")

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main() -> None:
    args = parse_args()
    train_config = load_config(args.config)
    for override in args.overrides:
        key, value = parse_config_override(override)
        train_config[key] = value

    train_config["do_train"] = False
    train_config["do_eval"] = False
    train_config["do_predict"] = False
    train_config["max_samples"] = args.max_samples
    train_config.setdefault("output_dir", "saves/neuron_typing/tmp")
    train_config.setdefault("preprocessing_num_workers", 8)

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

    print(f"\n{'='*60}")
    print("Building dataset (one-time)")
    print(f"{'='*60}")
    dataloader = build_dataloader(
        train_config,
        model,
        tokenizer_module,
        template,
        args.batch_size,
        args.num_workers,
        args.seed,
    )

    print(f"\n{'='*60}")
    print("Pass 1: Global Max Collection")
    print(f"{'='*60}")
    global_max, sample_count = pass1_global_max(
        model, dataloader, mlp_layers, image_token_id, args.max_samples, device
    )
    print(f"Pass 1 complete: {sample_count} samples processed")

    if args.pilot:
        pilot_output = Path(args.output_dir) / "pilot"
        pilot_output.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.half() for k, v in global_max.items()}, pilot_output / "global_max.pt")

        stats = {}
        for layer_idx, gmax in global_max.items():
            stats[f"layer_{layer_idx}"] = {
                "mean": gmax.mean().item(),
                "std": gmax.std().item(),
                "max": gmax.max().item(),
                "min": gmax.min().item(),
                "median": gmax.median().item(),
            }
        with open(pilot_output / "activation_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Pilot results saved to {pilot_output}")
        return

    quantile_thresholds = None
    if args.threshold_mode == "quantile":
        if args.quantile_path is None:
            args.quantile_path = str(Path(args.output_dir).parent / "calibration" / "neuron_quantiles.pt")
        if not Path(args.quantile_path).exists():
            print(f"ERROR: neuron_quantiles.pt not found at {args.quantile_path}")
            print("Run calibrate_thresholds.py first.")
            sys.exit(1)
        quantile_thresholds = load_quantile_thresholds(args.quantile_path, args.quantile_idx_visual, args.quantile_idx_text)
        print(f"Loaded quantile thresholds from {args.quantile_path} (vis_idx={args.quantile_idx_visual}, txt_idx={args.quantile_idx_text})")

    print(f"\n{'='*60}")
    print(f"Pass 2: Neuron Classification (mode={args.threshold_mode})")
    print(f"{'='*60}")
    neuron_scores = pass2_classify_neurons(
        model,
        dataloader,
        mlp_layers,
        image_token_id,
        global_max,
        args.max_samples,
        device,
        args.threshold_mode,
        args.t_visual,
        args.t_text,
        quantile_thresholds,
        args.visual_ratio,
        args.visual_min_count,
        args.text_ratio,
        args.text_min_count,
        args.top_k,
        args.sample_score_top_m,
    )
    print(f"Pass 2 complete: classified neurons for {len(neuron_scores)} layers")

    config = {
        "model_name": model_args.model_name_or_path,
        "max_samples": args.max_samples,
        "threshold_mode": args.threshold_mode,
        "t_visual": args.t_visual,
        "t_text": args.t_text,
        "quantile_path": args.quantile_path,
        "quantile_idx_visual": args.quantile_idx_visual,
        "quantile_idx_text": args.quantile_idx_text,
        "visual_ratio": args.visual_ratio,
        "visual_min_count": args.visual_min_count,
        "text_ratio": args.text_ratio,
        "text_min_count": args.text_min_count,
        "top_k": args.top_k,
        "sample_score_top_m": args.sample_score_top_m,
        "image_token_id": image_token_id,
        "num_mlp_layers": len(mlp_layers),
        "intermediate_size": int(mlp_layers[0].mlp.up_proj.weight.shape[0]) if mlp_layers else 0,
        "actual_samples": sample_count,
    }
    save_results(args.output_dir, global_max, neuron_scores, config)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
