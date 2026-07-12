"""Pilot typed ablation: zero out neuron groups, compare generated captions.

Usage:
    python scripts/vulcan/ablation/run_pilot_ablation.py \
        --neuron_scores saves/neuron_typing/phase1_2k/scores/neuron_type_scores.parquet \
        --config scripts/vulcan/neuron_typing/configs/pilot_coco.yaml \
        --output_dir saves/ablation/pilot \
        --ablation_count 50 \
        --eval_samples 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.vulcan import find_mlp_layers


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot typed ablation.")
    parser.add_argument("--neuron_scores", required=True, help="Path to neuron_type_scores.parquet")
    parser.add_argument("--config", required=True, help="LlamaFactory SFT YAML config.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--ablation_count", type=int, default=50, help="Neurons to ablate per type per layer.")
    parser.add_argument("--eval_samples", type=int, default=20, help="Samples for evaluation.")
    parser.add_argument("--high_conf_threshold", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_neurons(df, group, n_per_layer, threshold, rng):
    """Select top-n neurons per layer for a given type."""
    selected = []
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        if group == "visual":
            candidates = layer_df[layer_df["p_visual"] >= threshold].nlargest(n_per_layer, "p_visual")
        elif group == "text":
            candidates = layer_df[layer_df["p_text"] >= threshold].nlargest(n_per_layer, "p_text")
        elif group == "multimodal":
            candidates = layer_df[layer_df["p_multimodal"] >= threshold].nlargest(n_per_layer, "p_multimodal")
        elif group == "unknown":
            candidates = layer_df[layer_df["p_unknown"] >= threshold].nlargest(n_per_layer, "p_unknown")
        elif group == "random":
            candidates = layer_df.sample(n=min(n_per_layer, len(layer_df)), random_state=rng)
        else:
            raise ValueError(f"Unknown group: {group}")
        for _, row in candidates.iterrows():
            selected.append((int(row["layer"]), int(row["neuron_idx"])))
    return selected


def build_ablation_plan(neuron_list):
    """Convert (layer, neuron) list to per-layer dict."""
    plan = {}
    for layer_idx, neuron_idx in neuron_list:
        plan.setdefault(layer_idx, []).append(neuron_idx)
    return plan


def register_hooks(model, plan):
    """Register forward-pre-hooks to zero out neurons."""
    mlp_layers = find_mlp_layers(model)
    hooks = []
    for layer_ref in mlp_layers:
        if layer_ref.index in plan:
            indices = plan[layer_ref.index]
            def make_hook(idxs):
                def hook(module, args):
                    x = args[0]
                    x[..., idxs] = 0
                    return (x,)
                return hook
            h = layer_ref.mlp.down_proj.register_forward_pre_hook(make_hook(indices))
            hooks.append(h)
    return hooks


def load_eval_data(json_path, max_samples):
    """Load raw COCO caption data for evaluation."""
    with open(json_path) as f:
        data = json.load(f)
    return data[:max_samples]


def generate_captions(model, processor, eval_data, max_samples, device):
    """Generate captions for evaluation samples."""
    model.eval()
    results = []
    with torch.no_grad():
        for i, sample in enumerate(eval_data):
            if i >= max_samples:
                break
            # Extract prompt and reference from messages format
            messages = sample["messages"]
            prompt_text = messages[0]["content"]  # user message
            ref_text = messages[1]["content"]      # assistant message
            image_path = sample["images"][0] if sample.get("images") else None

            from PIL import Image
            image = Image.open(image_path).convert("RGB") if image_path else None

            # Replace <image> with vision tokens for Qwen-VL
            vision_prompt = prompt_text.replace(
                "<image>",
                "<|vision_start|><|image_pad|><|vision_end|>"
            )
            messages = [{"role": "user", "content": vision_prompt}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image] if image else None,
                return_tensors="pt",
                padding=True,
            ).to(device)
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            pred = processor.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append({"pred": pred.strip(), "ref": ref_text.strip()})
    return results


def compute_metrics(results):
    """Compute simple overlap metrics."""
    exact_match = sum(1 for r in results if r["pred"].lower() == r["ref"].lower()) / max(len(results), 1)
    avg_len = np.mean([len(r["pred"].split()) for r in results])
    # Token overlap (Jaccard)
    overlaps = []
    for r in results:
        pred_tokens = set(r["pred"].lower().split())
        ref_tokens = set(r["ref"].lower().split())
        if pred_tokens | ref_tokens:
            overlaps.append(len(pred_tokens & ref_tokens) / len(pred_tokens | ref_tokens))
    return {
        "exact_match": exact_match,
        "avg_pred_length": avg_len,
        "mean_token_overlap": np.mean(overlaps) if overlaps else 0,
        "n_samples": len(results),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # Load neuron scores
    df = pd.read_parquet(args.neuron_scores)
    print(f"Loaded {len(df)} neurons")

    # Load model
    config = yaml.safe_load(open(args.config))
    config["do_train"] = False
    config["max_samples"] = args.eval_samples
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    tok = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tok, data_args)
    model = load_model(tok, model_args, finetuning_args, is_trainable=False)
    device = torch.device("cuda")
    model.to(device)

    processor = tokenizer_module.get("processor", tok)

    # Load raw eval data
    eval_json = "/root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/val.json"
    eval_data = load_eval_data(eval_json, args.eval_samples)

    # Baseline
    print(f"\n{'='*60}")
    print("Baseline (no ablation)")
    print(f"{'='*60}")
    baseline_results = generate_captions(model, processor, eval_data, args.eval_samples, device)
    baseline_metrics = compute_metrics(baseline_results)
    print(f"  {baseline_metrics}")

    # Ablation groups
    groups = ["visual", "text", "multimodal", "unknown", "random"]
    all_results = {"baseline": {"metrics": baseline_metrics, "samples": baseline_results[:5]}}

    for group in groups:
        print(f"\n{'='*60}")
        print(f"Ablation: {group} (top {args.ablation_count}/layer)")
        print(f"{'='*60}")

        neurons = select_neurons(df, group, args.ablation_count, args.high_conf_threshold, rng)
        plan = build_ablation_plan(neurons)
        n_layers = len(plan)
        n_total = sum(len(v) for v in plan.values())
        print(f"  Selected {n_total} neurons across {n_layers} layers")

        hooks = register_hooks(model, plan)
        ablated_results = generate_captions(model, processor, eval_data, args.eval_samples, device)
        for h in hooks:
            h.remove()

        ablated_metrics = compute_metrics(ablated_results)
        delta = {k: ablated_metrics[k] - baseline_metrics[k] for k in ["exact_match", "mean_token_overlap"]}

        print(f"  Metrics: {ablated_metrics}")
        print(f"  Delta:   exact_match={delta['exact_match']:+.3f}  overlap={delta['mean_token_overlap']:+.3f}")

        all_results[group] = {
            "metrics": ablated_metrics,
            "delta": delta,
            "n_neurons": n_total,
            "n_layers": n_layers,
            "samples": ablated_results[:5],
        }

    # Save
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Group':<15} {'Neurons':>8} {'Overlap':>8} {'Delta':>8}")
    print("-" * 45)
    for group in ["baseline"] + groups:
        r = all_results[group]
        n = r.get("n_neurons", 0)
        overlap = r["metrics"]["mean_token_overlap"]
        delta = r.get("delta", {}).get("mean_token_overlap", 0)
        print(f"{group:<15} {n:>8} {overlap:>8.3f} {delta:>+8.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
