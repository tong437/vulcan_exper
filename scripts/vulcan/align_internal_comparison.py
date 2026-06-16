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

"""Internal comparison: align vs no-align neuron activation drift.

Runs the same val samples through both models and computes:
1. Per-layer hard_topk_iou for each model (visual vs text within model)
2. Per-layer top20_set_change = 1 - IoU(top20_align, top20_noalign) for visual and text
3. Per-layer FFN activation cosine / L2 drift between models
4. Per-layer parameter delta norm (gate_proj, up_proj, down_proj)
5. Summary focusing on last 1/2 layers

Usage:
    python scripts/vulcan/align_internal_comparison.py \
        --no-align-path saves/qwen35-0_8b-vqa-med-cls/full/sft-continuation-noalign-lr3e6-200steps/checkpoint-200 \
        --align-path saves/qwen35-0_8b-vqa-med-cls/full/align-top20-q80-temp002-lam005-lr3e6/checkpoint-200 \
        --val-data datasets/vqa_med/val_cls.jsonl \
        --image-dir datasets/vqa_med/images \
        --num-samples 100 \
        --quantile 0.8 \
        --output align_internal_comparison.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Align vs no-align internal neuron comparison")
    parser.add_argument("--no-align-path", type=str, required=True, help="Path to no-align checkpoint")
    parser.add_argument("--align-path", type=str, required=True, help="Path to align checkpoint")
    parser.add_argument("--val-data", type=str, default="datasets/vqa_med/val_cls.jsonl")
    parser.add_argument("--image-dir", type=str, default="datasets/vqa_med/images")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of val samples to use")
    parser.add_argument("--quantile", type=float, default=0.8, help="Top-k quantile threshold")
    parser.add_argument("--temperature", type=float, default=0.02, help="Soft top-k temperature")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="align_internal_comparison.json")
    return parser.parse_args()


def load_val_data(val_data_path: str, num_samples: int) -> list[dict]:
    """Load val JSONL data."""
    samples = []
    with open(val_data_path) as f:
        for line in f:
            if len(samples) >= num_samples:
                break
            samples.append(json.loads(line))
    return samples


def find_mlp_layers(model):
    """Find MLP layers from the model."""
    from llamafactory.train.vulcan.modeling import find_mlp_layers as _find
    return _find(model)


def register_activation_hooks(model, mlp_layers):
    """Register forward hooks on MLP down_proj to capture gated activations."""
    act_store = {}

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            act_store[layer_idx] = args[0].detach()
        return hook_fn

    hooks = []
    for layer_ref in mlp_layers:
        h = layer_ref.mlp.down_proj.register_forward_hook(make_hook(layer_ref.index))
        hooks.append(h)

    return act_store, hooks


def get_image_token_id(processor):
    """Get the image token ID from the processor."""
    # For Qwen3.5-VL, the image token is typically 151655
    tokenizer = processor.tokenizer
    img_token = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if img_token != tokenizer.unk_token_id:
        return img_token
    # Fallback: try the vision config
    return 151655


def process_sample(sample, processor, image_dir: str, image_token_id: int, device: str):
    """Process a single val sample into model inputs with labels for answer tokens."""
    messages = sample["messages"]
    image_path = sample.get("images", [None])[0]

    # Get the answer text from assistant message
    answer_text = ""
    for msg in messages:
        if msg["role"] == "assistant":
            answer_text = msg["content"]
            break

    # Build full conversation (user only, for input)
    content_parts = []
    for msg in messages:
        if msg["role"] == "user":
            text = msg["content"].replace("<image>", "").strip()
            if image_path:
                content_parts.append({"type": "image", "image": f"file://{Path(image_dir) / image_path}"})
            content_parts.append({"type": "text", "text": text})

    conversation = [{"role": "user", "content": content_parts}]
    full_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)

    image = None
    if image_path:
        img_full_path = Path(image_dir) / image_path
        if img_full_path.exists():
            image = Image.open(img_full_path).convert("RGB")

    inputs = processor(
        text=[full_text],
        images=[image] if image else None,
        return_tensors="pt",
        padding=True,
    )

    # Build labels: mark answer tokens, mask everything else with -100
    # Tokenize answer separately to find its token count
    answer_tokens = processor.tokenizer.encode(answer_text, add_special_tokens=False)
    answer_len = len(answer_tokens)

    labels = inputs["input_ids"].clone()
    # Mask everything except the last answer_len tokens
    labels[:, :-answer_len] = -100

    inputs["labels"] = labels
    return {k: v.to(device) for k, v in inputs.items()}


def build_token_masks(input_ids, image_token_id, labels=None):
    """Build visual and text (qa) token masks from input_ids.

    visual_mask: tokens that are image tokens
    text_mask: all non-visual tokens (question + answer), matching align_text_mode=qa
    """
    visual_mask = input_ids == image_token_id
    text_mask = ~visual_mask
    return visual_mask, text_mask


def soft_topk_mask(pooled, quantile, temperature):
    """Compute soft top-k mask."""
    pooled_f32 = pooled.float()
    tau = torch.quantile(pooled_f32, quantile)
    return torch.sigmoid((pooled_f32 - tau) / temperature)


def hard_topk_mask(pooled, quantile):
    """Compute hard top-k mask."""
    pooled_f32 = pooled.float()
    topk = max(1, math.ceil((1.0 - quantile) * pooled_f32.numel()))
    topk = min(topk, pooled_f32.numel())
    topk_indices = torch.topk(pooled_f32, k=topk, sorted=False).indices
    hard_mask = torch.zeros_like(pooled_f32, dtype=torch.bool)
    hard_mask.scatter_(dim=0, index=topk_indices, value=True)
    return hard_mask


def compute_hard_iou(mask_a, mask_b):
    """Compute IoU between two boolean masks."""
    intersection = (mask_a & mask_b).sum(dtype=torch.float32)
    union = (mask_a | mask_b).sum(dtype=torch.float32)
    return (intersection / union.clamp_min(1.0)).item()


def collect_activations(model, processor, samples, image_dir, image_token_id, device, quantile, temperature):
    """Run samples through model and collect per-layer pooled activations."""
    mlp_layers = find_mlp_layers(model)
    act_store, hooks = register_activation_hooks(model, mlp_layers)

    # Accumulate pooled activations per layer
    # For each layer: visual_pooled_sum, text_pooled_sum, count
    layer_visual_sums = {}
    layer_text_sums = {}
    layer_visual_masks = {}  # hard masks per sample for set change computation
    layer_text_masks = {}
    sample_count = 0

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            try:
                inputs = process_sample(sample, processor, image_dir, image_token_id, device)
            except Exception as e:
                print(f"  Skipping sample {i}: {e}", file=sys.stderr)
                continue

            input_ids = inputs["input_ids"]
            # Forward pass (without labels to avoid loss computation)
            forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**forward_inputs)
            seq_len = input_ids.shape[1]

            # For text_mode=qa: text = all non-visual tokens (question + answer)
            visual_mask, text_mask = build_token_masks(input_ids, image_token_id)

            if visual_mask.sum() == 0 or text_mask.sum() == 0:
                continue

            for layer_idx, act in act_store.items():
                act_seq = act[:, :seq_len, :].float().abs()

                # Pool visual tokens
                v_tokens = act_seq[visual_mask]
                if v_tokens.numel() > 0:
                    v_pooled = v_tokens.mean(dim=0)
                    if layer_idx not in layer_visual_sums:
                        layer_visual_sums[layer_idx] = v_pooled
                    else:
                        layer_visual_sums[layer_idx] += v_pooled
                    # Store hard mask for this sample
                    hm = hard_topk_mask(v_pooled, quantile)
                    layer_visual_masks.setdefault(layer_idx, []).append(hm.cpu())

                # Pool text tokens
                t_tokens = act_seq[text_mask]
                if t_tokens.numel() > 0:
                    t_pooled = t_tokens.mean(dim=0)
                    if layer_idx not in layer_text_sums:
                        layer_text_sums[layer_idx] = t_pooled
                    else:
                        layer_text_sums[layer_idx] += t_pooled
                    hm = hard_topk_mask(t_pooled, quantile)
                    layer_text_masks.setdefault(layer_idx, []).append(hm.cpu())

            act_store.clear()
            sample_count += 1

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")

    for h in hooks:
        h.remove()

    # Average pooled activations
    layer_visual_avg = {}
    layer_text_avg = {}
    for layer_idx in layer_visual_sums:
        layer_visual_avg[layer_idx] = layer_visual_sums[layer_idx] / sample_count
    for layer_idx in layer_text_sums:
        layer_text_avg[layer_idx] = layer_text_sums[layer_idx] / sample_count

    return {
        "visual_avg": layer_visual_avg,
        "text_avg": layer_text_avg,
        "visual_masks": layer_visual_masks,
        "text_masks": layer_text_masks,
        "mlp_layers": mlp_layers,
        "sample_count": sample_count,
    }


def compute_within_model_metrics(activations, quantile, temperature):
    """Compute per-layer hard_topk_iou (visual vs text) within a single model."""
    results = {}
    for layer_idx in sorted(activations["visual_avg"].keys()):
        v_pooled = activations["visual_avg"][layer_idx]
        t_pooled = activations["text_avg"][layer_idx]

        v_hard = hard_topk_mask(v_pooled, quantile)
        t_hard = hard_topk_mask(t_pooled, quantile)
        hard_iou = compute_hard_iou(v_hard, t_hard)

        v_soft = soft_topk_mask(v_pooled, quantile, temperature)
        t_soft = soft_topk_mask(t_pooled, quantile, temperature)
        soft_iou = ((v_soft * t_soft).sum() / (v_soft.sum() + t_soft.sum() - (v_soft * t_soft).sum() + 1e-8)).item()

        results[layer_idx] = {
            "hard_topk_iou": hard_iou,
            "soft_iou": soft_iou,
            "v_mask_mean": v_soft.mean().item(),
            "t_mask_mean": t_soft.mean().item(),
        }
    return results


def compute_cross_model_metrics(align_act, noalign_act, quantile):
    """Compute top20 set change between align and no-align models."""
    results = {}
    common_layers = sorted(set(align_act["visual_avg"].keys()) & set(noalign_act["visual_avg"].keys()))

    for layer_idx in common_layers:
        layer_result = {}

        # Visual top20 set change
        v_align = align_act["visual_avg"][layer_idx]
        v_noalign = noalign_act["visual_avg"][layer_idx]
        v_mask_align = hard_topk_mask(v_align, quantile)
        v_mask_noalign = hard_topk_mask(v_noalign, quantile)
        v_iou = compute_hard_iou(v_mask_align, v_mask_noalign)
        layer_result["visual_top20_set_change"] = 1.0 - v_iou
        layer_result["visual_top20_iou"] = v_iou

        # Text top20 set change
        t_align = align_act["text_avg"][layer_idx]
        t_noalign = noalign_act["text_avg"][layer_idx]
        t_mask_align = hard_topk_mask(t_align, quantile)
        t_mask_noalign = hard_topk_mask(t_noalign, quantile)
        t_iou = compute_hard_iou(t_mask_align, t_mask_noalign)
        layer_result["text_top20_set_change"] = 1.0 - t_iou
        layer_result["text_top20_iou"] = t_iou

        # FFN activation drift (cosine + L2)
        v_cos = torch.nn.functional.cosine_similarity(
            v_align.float().unsqueeze(0), v_noalign.float().unsqueeze(0)
        ).item()
        v_l2 = (v_align.float() - v_noalign.float()).norm().item()
        t_cos = torch.nn.functional.cosine_similarity(
            t_align.float().unsqueeze(0), t_noalign.float().unsqueeze(0)
        ).item()
        t_l2 = (t_align.float() - t_noalign.float()).norm().item()

        layer_result["visual_act_cosine"] = v_cos
        layer_result["visual_act_l2"] = v_l2
        layer_result["text_act_cosine"] = t_cos
        layer_result["text_act_l2"] = t_l2

        # Visual-text co-movement: did both top20 sets shift in the same direction?
        v_shared = (v_mask_align & v_mask_noalign).sum().float().item()
        t_shared = (t_mask_align & t_mask_noalign).sum().float().item()
        v_total = (v_mask_align | v_mask_noalign).sum().float().item()
        t_total = (t_mask_align | t_mask_noalign).sum().float().item()
        layer_result["visual_stable_ratio"] = v_shared / max(v_total, 1)
        layer_result["text_stable_ratio"] = t_shared / max(t_total, 1)

        results[layer_idx] = layer_result

    return results


def compute_parameter_delta(align_path, noalign_path, mlp_layer_names):
    """Compute per-layer parameter delta norms for gate_proj, up_proj, down_proj."""
    align_weights = load_file(str(Path(align_path) / "model.safetensors"))
    noalign_weights = load_file(str(Path(noalign_path) / "model.safetensors"))

    results = {}
    proj_names = ["gate_proj", "up_proj", "down_proj"]

    for layer_name in mlp_layer_names:
        layer_result = {}
        for proj_name in proj_names:
            weight_key = f"{layer_name}.mlp.{proj_name}.weight"
            if weight_key in align_weights and weight_key in noalign_weights:
                a = align_weights[weight_key].float()
                n = noalign_weights[weight_key].float()
                delta = a - n
                layer_result[proj_name] = {
                    "delta_norm": delta.norm().item(),
                    "delta_norm_rel": (delta.norm() / n.norm()).item(),
                    "delta_max_abs": delta.abs().max().item(),
                    "delta_mean_abs": delta.abs().mean().item(),
                    "cosine_sim": torch.nn.functional.cosine_similarity(
                        a.flatten().unsqueeze(0), n.flatten().unsqueeze(0)
                    ).item(),
                }
        results[layer_name] = layer_result

    return results


def summarize_half_layers(per_layer_metrics, total_layers, label=""):
    """Focus on last 1/2 layers."""
    half_start = total_layers // 2
    half_results = {}
    for layer_idx, values in per_layer_metrics.items():
        if layer_idx >= half_start:
            half_results[layer_idx] = values

    if not half_results:
        return {}

    # Compute averages for numeric values
    if isinstance(next(iter(half_results.values())), dict):
        avg = {}
        for key in next(iter(half_results.values())):
            vals = [v[key] for v in half_results.values() if isinstance(v[key], (int, float))]
            if vals:
                avg[f"mean_{key}"] = sum(vals) / len(vals)
        return avg
    return half_results


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Align vs No-Align Internal Neuron Comparison")
    print("=" * 60)

    # Load val data
    print(f"\nLoading {args.num_samples} val samples from {args.val_data}")
    samples = load_val_data(args.val_data, args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Load processor (shared)
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(args.no_align_path, trust_remote_code=True)
    image_token_id = get_image_token_id(processor)
    print(f"Image token ID: {image_token_id}")

    # ── Load no-align model ──
    print("\n" + "=" * 60)
    print("Loading NO-ALIGN model...")
    print(f"  Path: {args.no_align_path}")
    noalign_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.no_align_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    noalign_model.eval()

    print("\nCollecting activations from NO-ALIGN model...")
    noalign_act = collect_activations(
        noalign_model, processor, samples, args.image_dir,
        image_token_id, device, args.quantile, args.temperature
    )
    print(f"  Collected from {noalign_act['sample_count']} samples, {len(noalign_act['visual_avg'])} layers")

    # Free no-align model
    noalign_mlp_layers = noalign_act["mlp_layers"]
    del noalign_model
    torch.cuda.empty_cache()

    # ── Load align model ──
    print("\n" + "=" * 60)
    print("Loading ALIGN model...")
    print(f"  Path: {args.align_path}")
    align_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.align_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    align_model.eval()

    print("\nCollecting activations from ALIGN model...")
    align_act = collect_activations(
        align_model, processor, samples, args.image_dir,
        image_token_id, device, args.quantile, args.temperature
    )
    print(f"  Collected from {align_act['sample_count']} samples, {len(align_act['visual_avg'])} layers")

    align_mlp_layers = align_act["mlp_layers"]
    del align_model
    torch.cuda.empty_cache()

    # ── Compute metrics ──
    print("\n" + "=" * 60)
    print("Computing metrics...")

    # 1. Within-model metrics
    print("\n[1] Within-model hard_topk_iou (visual vs text)")
    noalign_within = compute_within_model_metrics(noalign_act, args.quantile, args.temperature)
    align_within = compute_within_model_metrics(align_act, args.quantile, args.temperature)

    # 2. Cross-model metrics
    print("[2] Cross-model top20 set change")
    cross_metrics = compute_cross_model_metrics(align_act, noalign_act, args.quantile)

    # 3. Parameter delta
    print("[3] Parameter delta norms")
    noalign_layer_names = [ref.name for ref in noalign_mlp_layers]
    param_delta = compute_parameter_delta(args.align_path, args.no_align_path, noalign_layer_names)

    # ── Build results ──
    total_layers = len(noalign_mlp_layers)
    half_start = total_layers // 2

    results = {
        "config": {
            "no_align_path": args.no_align_path,
            "align_path": args.align_path,
            "num_samples": len(samples),
            "quantile": args.quantile,
            "temperature": args.temperature,
            "total_mlp_layers": total_layers,
            "half_start_layer": half_start,
        },
        "per_layer": {},
    }

    for layer_idx in range(total_layers):
        layer_key = f"layer_{layer_idx}"
        entry = {"layer_idx": layer_idx, "is_second_half": layer_idx >= half_start}

        if layer_idx in noalign_within:
            entry["noalign_within"] = noalign_within[layer_idx]
        if layer_idx in align_within:
            entry["align_within"] = align_within[layer_idx]
        if layer_idx in cross_metrics:
            entry["cross_model"] = cross_metrics[layer_idx]
        if noalign_layer_names[layer_idx] in param_delta:
            entry["param_delta"] = param_delta[noalign_layer_names[layer_idx]]

        results["per_layer"][layer_key] = entry

    # ── Summary: last 1/2 layers ──
    summary = {"second_half_avg": {}}

    # Within-model IoU
    noalign_half_ious = [noalign_within[i]["hard_topk_iou"] for i in range(half_start, total_layers) if i in noalign_within]
    align_half_ious = [align_within[i]["hard_topk_iou"] for i in range(half_start, total_layers) if i in align_within]
    if noalign_half_ious:
        summary["second_half_avg"]["noalign_hard_topk_iou_mean"] = sum(noalign_half_ious) / len(noalign_half_ious)
    if align_half_ious:
        summary["second_half_avg"]["align_hard_topk_iou_mean"] = sum(align_half_ious) / len(align_half_ious)

    # Cross-model set change
    v_changes = [cross_metrics[i]["visual_top20_set_change"] for i in range(half_start, total_layers) if i in cross_metrics]
    t_changes = [cross_metrics[i]["text_top20_set_change"] for i in range(half_start, total_layers) if i in cross_metrics]
    if v_changes:
        summary["second_half_avg"]["visual_top20_set_change_mean"] = sum(v_changes) / len(v_changes)
    if t_changes:
        summary["second_half_avg"]["text_top20_set_change_mean"] = sum(t_changes) / len(t_changes)

    # Activation drift
    v_cos_vals = [cross_metrics[i]["visual_act_cosine"] for i in range(half_start, total_layers) if i in cross_metrics]
    t_cos_vals = [cross_metrics[i]["text_act_cosine"] for i in range(half_start, total_layers) if i in cross_metrics]
    v_l2_vals = [cross_metrics[i]["visual_act_l2"] for i in range(half_start, total_layers) if i in cross_metrics]
    t_l2_vals = [cross_metrics[i]["text_act_l2"] for i in range(half_start, total_layers) if i in cross_metrics]
    if v_cos_vals:
        summary["second_half_avg"]["visual_act_cosine_mean"] = sum(v_cos_vals) / len(v_cos_vals)
    if t_cos_vals:
        summary["second_half_avg"]["text_act_cosine_mean"] = sum(t_cos_vals) / len(t_cos_vals)
    if v_l2_vals:
        summary["second_half_avg"]["visual_act_l2_mean"] = sum(v_l2_vals) / len(v_l2_vals)
    if t_l2_vals:
        summary["second_half_avg"]["text_act_l2_mean"] = sum(t_l2_vals) / len(t_l2_vals)

    # Parameter delta
    delta_norms = []
    for layer_name in noalign_layer_names[half_start:]:
        if layer_name in param_delta:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if proj in param_delta[layer_name]:
                    delta_norms.append(param_delta[layer_name][proj]["delta_norm_rel"])
    if delta_norms:
        summary["second_half_avg"]["param_delta_norm_rel_mean"] = sum(delta_norms) / len(delta_norms)

    # Co-movement: did visual and text top20 sets shift together?
    v_stable = [cross_metrics[i]["visual_stable_ratio"] for i in range(half_start, total_layers) if i in cross_metrics]
    t_stable = [cross_metrics[i]["text_stable_ratio"] for i in range(half_start, total_layers) if i in cross_metrics]
    if v_stable:
        summary["second_half_avg"]["visual_stable_ratio_mean"] = sum(v_stable) / len(v_stable)
    if t_stable:
        summary["second_half_avg"]["text_stable_ratio_mean"] = sum(t_stable) / len(t_stable)

    results["summary"] = summary

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ── Print human-readable summary ──
    print("\n" + "=" * 60)
    print("SUMMARY (last 1/2 layers)")
    print("=" * 60)
    for key, val in sorted(summary["second_half_avg"].items()):
        print(f"  {key}: {val:.6f}")

    print("\n" + "-" * 60)
    print("PER-LAYER DETAIL (last 1/2 layers)")
    print("-" * 60)
    print(f"{'Layer':>6} | {'noA_iou':>8} {'A_iou':>8} | {'V_setΔ':>8} {'T_setΔ':>8} | {'V_cos':>8} {'T_cos':>8} | {'V_L2':>8} {'T_L2':>8} | {'δ_rel':>8}")
    print("-" * 100)
    for layer_idx in range(half_start, total_layers):
        layer_key = f"layer_{layer_idx}"
        entry = results["per_layer"][layer_key]

        noa_iou = entry.get("noalign_within", {}).get("hard_topk_iou", float("nan"))
        a_iou = entry.get("align_within", {}).get("hard_topk_iou", float("nan"))
        v_set = entry.get("cross_model", {}).get("visual_top20_set_change", float("nan"))
        t_set = entry.get("cross_model", {}).get("text_top20_set_change", float("nan"))
        v_cos = entry.get("cross_model", {}).get("visual_act_cosine", float("nan"))
        t_cos = entry.get("cross_model", {}).get("text_act_cosine", float("nan"))
        v_l2 = entry.get("cross_model", {}).get("visual_act_l2", float("nan"))
        t_l2 = entry.get("cross_model", {}).get("text_act_l2", float("nan"))

        # Average relative delta across projections
        pd = entry.get("param_delta", {})
        delta_vals = [pd[p]["delta_norm_rel"] for p in ["gate_proj", "up_proj", "down_proj"] if p in pd]
        delta_rel = sum(delta_vals) / len(delta_vals) if delta_vals else float("nan")

        print(f"  {layer_idx:>4} | {noa_iou:>8.4f} {a_iou:>8.4f} | {v_set:>8.4f} {t_set:>8.4f} | {v_cos:>8.4f} {t_cos:>8.4f} | {v_l2:>8.2f} {t_l2:>8.2f} | {delta_rel:>8.6f}")


if __name__ == "__main__":
    main()
