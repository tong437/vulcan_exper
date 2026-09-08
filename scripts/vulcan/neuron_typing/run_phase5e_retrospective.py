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

"""Phase 5E-R: reevaluate existing physical checkpoints under the semantic contract."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase5e_proxy import decode_gold_caption, evaluate_gold_proxy  # noqa: E402
from phase5e_semantics import REFERENCE_CAPTION, generate_short_caption, semantic_contract  # noqa: E402
from run_phase2_ablation import build_dataloader, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, valid_next_token_tensors, write_json  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle  # noqa: E402

from llamafactory.train.vulcan.modeling import find_mlp_layers, get_intermediate_size  # noqa: E402


DEFAULT_CHECKPOINTS = {
    "core13": "saves/neuron_typing/phase5_single_sample/sample_2500_core13_structural_best/model",
    "physical_100": "saves/neuron_typing/phase5_single_sample/sample_2500_phase5d_robust_structural_100_r2/model",
    "physical_250": "saves/neuron_typing/phase5_single_sample/sample_2500_phase5d_robust_structural_250_r1/model",
    "physical_750": "saves/neuron_typing/phase5_single_sample/sample_2500_phase5c_screen_750_r0/model",
    "physical_1000": "saves/neuron_typing/phase5_single_sample/sample_2500_phase5d_robust_structural_1000_r3/model",
}


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Checkpoint must use NAME=PATH syntax, got {value!r}.")
    name, path = value.split("=", maxsplit=1)
    name = name.strip()
    if not name or not path.strip():
        raise ValueError(f"Checkpoint must use non-empty NAME=PATH syntax, got {value!r}.")
    return name, Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrospectively score physical models under Phase 5E semantics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--original_model_path", default=None)
    parser.add_argument("--checkpoint", action="append", default=[], help="Repeat NAME=PATH for each physical model.")
    parser.add_argument("--include_default_checkpoints", action="store_true")
    parser.add_argument("--sample_offset", type=int, default=0, help="Index in the frozen one-row Phase 5E dataset.")
    parser.add_argument("--reference_caption", default=REFERENCE_CAPTION)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2057)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def build_reference_trace(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    labels = batch["labels"]
    outputs = model(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False)
    logits, valid_labels = valid_next_token_tensors(outputs.logits, labels)
    logits = logits.detach().cpu()
    valid_labels = valid_labels.detach().cpu()
    return {
        "valid_logits": logits,
        "valid_labels": valid_labels,
        "nll": float(F.cross_entropy(logits, valid_labels)),
        "num_label_tokens": int(valid_labels.numel()),
    }


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _model_structure(model: torch.nn.Module) -> dict[str, Any]:
    widths = {str(layer.index): get_intermediate_size(layer.mlp) for layer in find_mlp_layers(model)}
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters": parameters, "ffn_widths": widths, "total_ffn_neurons": sum(widths.values())}


def _load_one(
    config_path: str | Path,
    model_path: str | Path,
    args: argparse.Namespace,
    device: torch.device,
):
    model, tokenizer_module, template, config = load_model_bundle(
        config_path,
        model_path,
        device,
        trust_remote_code=True,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    if config.get("template") != "qwen3_5" or config.get("enable_thinking") is not False:
        raise ValueError("Phase 5E retrospective requires template=qwen3_5 and enable_thinking=false.")
    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=args.num_workers,
        sample_offset=args.sample_offset,
        max_samples=1,
        allow_short_dataset=False,
        max_image_repeat=5,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    batch = move_batch_to_device(next(iter(dataloader)), device)
    return model, tokenizer_module["tokenizer"], batch, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from run_phase2_ablation import load_yaml

    base_config = load_yaml(args.config)
    original_path = Path(args.original_model_path or base_config["model_name_or_path"])
    checkpoints: dict[str, Path] = {"original": original_path}
    if args.include_default_checkpoints:
        for name, path in DEFAULT_CHECKPOINTS.items():
            candidate = ROOT_DIR / path
            if candidate.is_dir():
                checkpoints[name] = candidate
    for spec in args.checkpoint:
        name, path = parse_checkpoint_spec(spec)
        if name == "original":
            raise ValueError("The checkpoint name 'original' is reserved for --original_model_path.")
        checkpoints[name] = path
    missing = {name: str(path) for name, path in checkpoints.items() if not path.is_dir()}
    if missing:
        raise FileNotFoundError(f"Physical checkpoint directories do not exist: {missing}.")

    output_path = Path(args.output_file)
    identity = {
        "config_path": args.config,
        "original_model_path": str(original_path),
        "sample_offset": args.sample_offset,
        "reference_caption": args.reference_caption,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    if output_path.is_file() and not args.resume:
        raise FileExistsError(f"Phase 5E retrospective output exists: {output_path}. Pass --resume to append.")
    if output_path.is_file():
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if result.get("config") != identity:
            raise ValueError("Cannot resume a Phase 5E retrospective with changed identity fields.")
        result["complete"] = False
    else:
        result = {
            "complete": False,
            "interpretation": (
                "Automatic semantic pass is a retrospective screen. Final frontier claims require explicit human "
                "confirmation; parameter reductions are relative to the unpruned reference loaded in this run."
            ),
            "config": identity,
            "semantic_contract": semantic_contract(),
            "reference": {},
            "checkpoints": {},
            "best_automatic_semantic_pass": None,
        }
    write_json(output_path, result)

    reference_trace = None
    reference_structure = None
    reference_gold_ids = None
    for name, model_path in checkpoints.items():
        if name in result["checkpoints"] and name != "original":
            print(f"Phase 5E-R skipping completed checkpoint {name}", flush=True)
            continue
        print(f"Phase 5E-R evaluating {name}: {model_path}", flush=True)
        model = None
        try:
            model, tokenizer, batch, manifest = _load_one(args.config, model_path, args, device)
            prompt_inputs, prompt_length = build_prompt_inputs(batch)
            if name == "original":
                reference_trace = build_reference_trace(model, batch)
                reference_structure = _model_structure(model)
                reference_gold_ids = reference_trace["valid_labels"].tolist()
                decoded = decode_gold_caption(tokenizer, reference_trace["valid_labels"])
                if " ".join(decoded.lower().split()).rstrip(" .") != " ".join(
                    args.reference_caption.lower().split()
                ).rstrip(" ."):
                    raise ValueError(
                        f"Frozen gold caption mismatch: expected {args.reference_caption!r}, decoded {decoded!r}."
                    )
                result["reference"] = {
                    "gold_caption": decoded,
                    "gold_token_ids": reference_gold_ids,
                    "num_gold_tokens": reference_trace["num_label_tokens"],
                    "prompt_tokens": prompt_length,
                    "sample_manifest": manifest,
                    "structure": reference_structure,
                }
            elif reference_trace is None or reference_structure is None or reference_gold_ids is None:
                raise RuntimeError("The original model must be evaluated before physical checkpoints.")
            candidate_gold_ids = build_reference_trace(model, batch)["valid_labels"].tolist()
            if candidate_gold_ids != reference_gold_ids:
                raise RuntimeError(f"Checkpoint {name} does not reproduce the frozen gold-caption tokenization.")
            structure = _model_structure(model)
            proxy = evaluate_gold_proxy(model, batch, reference_trace)
            generation = generate_short_caption(model, tokenizer, prompt_inputs, max_new_tokens=args.max_new_tokens)
            deleted_neurons = reference_structure["total_ffn_neurons"] - structure["total_ffn_neurons"]
            removed_parameters = reference_structure["parameters"] - structure["parameters"]
            row = {
                "name": name,
                "model_path": str(model_path),
                "structure": {
                    **structure,
                    "deleted_neurons": deleted_neurons,
                    "removed_parameters": removed_parameters,
                    "removed_parameter_ratio": removed_parameters / reference_structure["parameters"],
                    "deleted_by_layer": {
                        layer: reference_structure["ffn_widths"][layer] - width
                        for layer, width in structure["ffn_widths"].items()
                    },
                },
                "gold_proxy": proxy,
                "generation": generation,
                "automatic_semantic_pass": generation["semantic"]["automatic_pass"],
                "human_confirmation": None,
                "reload_evaluated": True,
            }
            result["checkpoints"][name] = row
            current = result["best_automatic_semantic_pass"]
            if row["automatic_semantic_pass"] and (current is None or deleted_neurons > current["deleted_neurons"]):
                result["best_automatic_semantic_pass"] = {
                    "checkpoint": name,
                    "deleted_neurons": deleted_neurons,
                    "removed_parameters": removed_parameters,
                    "final_caption": generation["final_caption"],
                    "human_confirmation": None,
                }
            write_json(output_path, result)
        finally:
            model = None
            batch = None
            prompt_inputs = None
            tokenizer = None
            _release_device_cache()

    result["complete"] = True
    write_json(output_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(json.dumps({"complete": result["complete"], "best": result["best_automatic_semantic_pass"]}, indent=2))


if __name__ == "__main__":
    main()
