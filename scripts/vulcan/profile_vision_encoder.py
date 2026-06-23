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

"""Profile the inference cost of an uncompressed vision encoder.

This script measures:
  * parameters and weight bytes owned by the vision encoder, projector, and LLM;
  * input preparation time (tokenization, image processing, and device transfer);
  * multimodal prefill latency and peak CUDA memory;
  * end-to-end generation latency and peak CUDA memory;
  * vision-pipeline and projector latency through forward hooks.

Example:
    WANDB_DISABLED=true python scripts/vulcan/profile_vision_encoder.py \
        --model_name_or_path saves/qwen35-0_8b-vqa-rad/full/sft-strong \
        --image_path datasets/vqa_rad/images/synpic29265.jpg \
        --template qwen3_5_nothink \
        --warmup 10 \
        --iterations 50 \
        --max_new_tokens 1 \
        --output_path results/vision_profile.json
"""

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llamafactory.chat.hf_engine import HuggingfaceEngine  # noqa: E402
from llamafactory.data import get_template_and_fix_tokenizer  # noqa: E402
from llamafactory.hparams import get_infer_args  # noqa: E402
from llamafactory.model import load_model, load_tokenizer  # noqa: E402
from llamafactory.model.model_utils.visual import COMPOSITE_MODELS  # noqa: E402


@dataclass
class Sample:
    messages: list[dict[str, str]]
    images: list[Image.Image]
    system: str | None
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the cost of an uncompressed VLM vision encoder.")
    parser.add_argument("--model_name_or_path", required=True, help="Base model, merged checkpoint, or HF model ID.")
    parser.add_argument("--adapter_name_or_path", default=None, help="Optional single LoRA adapter.")
    parser.add_argument("--template", default="qwen3_5_nothink", help="LlamaFactory prompt template.")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote model code.")
    parser.add_argument("--infer_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument("--image_min_pixels", type=int, default=1024)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image_path", help="Path to one image.")
    input_group.add_argument("--dataset_jsonl", help="JSONL sample source with LlamaFactory messages/images fields.")
    parser.add_argument("--sample_index", type=int, default=0, help="Zero-based row index for --dataset_jsonl.")
    parser.add_argument("--dataset_root", default=None, help="Root used to resolve relative image paths in JSONL.")
    parser.add_argument("--question", default="Describe the image.", help="Question used with --image_path.")
    parser.add_argument("--system_prompt", default=None)

    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--skip_generation", action="store_true", help="Only profile input preparation and prefill.")
    parser.add_argument("--output_path", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_sample(args: argparse.Namespace) -> Sample:
    if args.image_path is not None:
        image_path = Path(args.image_path).expanduser().resolve()
        image = Image.open(image_path).convert("RGB")
        return Sample(
            messages=[{"role": "user", "content": args.question}],
            images=[image],
            system=args.system_prompt,
            source=str(image_path),
        )

    dataset_path = Path(args.dataset_jsonl).expanduser().resolve()
    record = None
    with dataset_path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index == args.sample_index:
                record = json.loads(line)
                break

    if record is None:
        raise IndexError(f"Sample index {args.sample_index} does not exist in {dataset_path}.")

    messages = record.get("messages") or record.get("conversations")
    if not messages:
        raise ValueError("The selected JSONL row has no `messages` or `conversations` field.")

    system = args.system_prompt
    prompt_messages = []
    for message in messages:
        role = message.get("role") or message.get("from")
        content = message.get("content") or message.get("value")
        if role in {"system"}:
            system = system or content
        elif role in {"user", "human"}:
            prompt_messages.append({"role": "user", "content": content})
        elif role in {"assistant", "gpt"}:
            break

    if not prompt_messages:
        raise ValueError("The selected JSONL row has no user message before the first assistant response.")

    dataset_root = Path(args.dataset_root).expanduser() if args.dataset_root else dataset_path.parent
    image_paths = record.get("images") or []
    images = []
    for image_name in image_paths:
        image_path = Path(image_name).expanduser()
        if not image_path.is_absolute():
            image_path = dataset_root / image_path
        images.append(Image.open(image_path).convert("RGB"))

    if not images:
        raise ValueError("The selected JSONL row has no images.")

    return Sample(
        messages=prompt_messages,
        images=images,
        system=system,
        source=f"{dataset_path}#{args.sample_index}",
    )


def _find_named_module(model: torch.nn.Module, path: str) -> tuple[str, torch.nn.Module] | None:
    named_modules = dict(model.named_modules())
    if path in named_modules:
        return path, named_modules[path]

    matches = [(name, module) for name, module in named_modules.items() if name.endswith(f".{path}")]
    if not matches:
        return None

    return min(matches, key=lambda item: len(item[0]))


def _common_module_path(paths: list[str]) -> str:
    if len(paths) == 1:
        return paths[0]

    split_paths = [path.split(".") for path in paths]
    common = []
    for parts in zip(*split_paths):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])

    return ".".join(common)


def _expand_executable_modules(name: str, module: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
        return [(f"{name}.{index}", child) for index, child in enumerate(module)]
    return [(name, module)]


def _resolve_modules(model: torch.nn.Module, paths: list[str]) -> list[tuple[str, torch.nn.Module]]:
    resolved = []
    seen = set()
    for path in paths:
        match = _find_named_module(model, path)
        if match is None:
            continue
        for name, module in _expand_executable_modules(*match):
            if id(module) not in seen:
                resolved.append((name, module))
                seen.add(id(module))
    return resolved


def _parameter_ids_for_paths(model: torch.nn.Module, paths: list[str]) -> set[int]:
    parameter_ids = set()
    for name, parameter in model.named_parameters():
        if any(name == path or name.startswith(f"{path}.") or f".{path}." in f".{name}." for path in paths):
            parameter_ids.add(id(parameter))
    return parameter_ids


def _parameter_summary(
    model: torch.nn.Module, vision_paths: list[str], projector_paths: list[str], language_paths: list[str]
) -> dict[str, dict[str, float | int]]:
    parameters = {id(parameter): parameter for parameter in model.parameters()}
    projector_ids = _parameter_ids_for_paths(model, projector_paths)
    vision_ids = _parameter_ids_for_paths(model, vision_paths) - projector_ids
    language_ids = _parameter_ids_for_paths(model, language_paths) - projector_ids - vision_ids
    total_ids = set(parameters)
    groups = {
        "vision_encoder": vision_ids,
        "projector": projector_ids,
        "language_model": language_ids,
        "other": total_ids - vision_ids - projector_ids - language_ids,
        "total": total_ids,
    }
    total_parameters = sum(parameters[parameter_id].numel() for parameter_id in total_ids)
    total_bytes = sum(
        parameters[parameter_id].numel() * parameters[parameter_id].element_size() for parameter_id in total_ids
    )
    summary = {}
    for group, parameter_ids in groups.items():
        count = sum(parameters[parameter_id].numel() for parameter_id in parameter_ids)
        num_bytes = sum(
            parameters[parameter_id].numel() * parameters[parameter_id].element_size()
            for parameter_id in parameter_ids
        )
        summary[group] = {
            "parameters": count,
            "parameter_ratio": count / total_parameters if total_parameters else 0.0,
            "weight_bytes": num_bytes,
            "weight_mib": num_bytes / 2**20,
            "weight_byte_ratio": num_bytes / total_bytes if total_bytes else 0.0,
        }
    return summary


class ModuleTimer:
    def __init__(self, use_cuda: bool) -> None:
        self.use_cuda = use_cuda
        self.handles = []
        self.records: dict[str, list[Any]] = defaultdict(list)
        self.starts: dict[int, list[Any]] = defaultdict(list)

    def register(self, group: str, modules: list[tuple[str, torch.nn.Module]]) -> None:
        for _, module in modules:
            module_id = id(module)

            def pre_hook(_module, _args, *, key=module_id):
                start = torch.cuda.Event(enable_timing=True) if self.use_cuda else time.perf_counter()
                if self.use_cuda:
                    start.record()
                self.starts[key].append(start)

            def post_hook(_module, _args, _output, *, key=module_id, group_name=group):
                start = self.starts[key].pop()
                if self.use_cuda:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    self.records[group_name].append((start, end))
                else:
                    self.records[group_name].append((time.perf_counter() - start) * 1000)

            self.handles.append(module.register_forward_pre_hook(pre_hook))
            self.handles.append(module.register_forward_hook(post_hook))

    def reset(self) -> None:
        self.records.clear()
        self.starts.clear()

    def elapsed_ms(self) -> dict[str, float]:
        if self.use_cuda:
            torch.cuda.synchronize()
        elapsed = {}
        for group, records in self.records.items():
            if self.use_cuda:
                elapsed[group] = sum(start.elapsed_time(end) for start, end in records)
            else:
                elapsed[group] = sum(records)
        return elapsed

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _synchronize(use_cuda: bool) -> None:
    if use_cuda:
        torch.cuda.synchronize()


def _timed_call(function, use_cuda: bool) -> tuple[Any, float]:
    _synchronize(use_cuda)
    if use_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        end.record()
        torch.cuda.synchronize()
        return output, start.elapsed_time(end)

    start_time = time.perf_counter()
    output = function()
    return output, (time.perf_counter() - start_time) * 1000


def _memory_snapshot(use_cuda: bool) -> dict[str, float]:
    if not use_cuda:
        return {}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        index = math.ceil(fraction * len(ordered)) - 1
        return ordered[max(index, 0)]

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": percentile(0.9),
        "max": ordered[-1],
    }


def _summarize_runs(runs: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    keys = sorted({key for run in runs for key in run})
    return {key: _stats([run[key] for run in runs if key in run]) for key in keys}


def _prepare_forward_kwargs(gen_kwargs: dict[str, Any]) -> dict[str, Any]:
    forward_kwargs = {
        key: value
        for key, value in gen_kwargs.items()
        if key not in {"generation_config", "tokenizer", "inputs"}
    }
    if "input_ids" not in forward_kwargs and "inputs" in gen_kwargs:
        forward_kwargs["input_ids"] = gen_kwargs["inputs"]
    forward_kwargs["use_cache"] = False
    return forward_kwargs


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("`warmup` must be non-negative and `iterations` must be positive.")

    sample = load_sample(args)
    infer_args = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "trust_remote_code": args.trust_remote_code,
        "infer_dtype": args.infer_dtype,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    if args.adapter_name_or_path:
        infer_args["adapter_name_or_path"] = [args.adapter_name_or_path]

    model_args, data_args, finetuning_args, generating_args = get_infer_args(infer_args)
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    tokenizer.padding_side = "left"
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    model.eval()
    use_cuda = any(parameter.is_cuda for parameter in model.parameters())
    model_memory_after_load = _memory_snapshot(use_cuda)

    model_type = getattr(model.config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        raise ValueError(f"Model type {model_type!r} is not registered in COMPOSITE_MODELS.")
    composite = COMPOSITE_MODELS[model_type]
    vision_paths = composite.vision_model_keys
    projector_paths = composite.projector_keys
    language_paths = composite.language_model_keys

    vision_root_path = _common_module_path(vision_paths)
    vision_pipeline_modules = _resolve_modules(model, [vision_root_path]) if vision_root_path else []
    vision_component_modules = _resolve_modules(model, vision_paths)
    projector_modules = _resolve_modules(model, projector_paths)
    if not vision_pipeline_modules:
        raise ValueError(f"Could not resolve a vision pipeline module from paths: {vision_paths}.")

    pipeline_names = [name for name, _ in vision_pipeline_modules]
    projector_names = [name for name, _ in projector_modules]
    projector_is_nested = any(
        projector_name.startswith(f"{pipeline_name}.")
        for pipeline_name in pipeline_names
        for projector_name in projector_names
    )

    timer = ModuleTimer(use_cuda=use_cuda)
    timer.register("vision_pipeline_ms", vision_pipeline_modules)
    timer.register("vision_components_ms", vision_component_modules)
    timer.register("projector_ms", projector_modules)

    def prepare_inputs() -> dict[str, Any]:
        gen_kwargs, _ = HuggingfaceEngine._process_args(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            template=template,
            generating_args=generating_args.to_dict(),
            messages=copy.deepcopy(sample.messages),
            system=sample.system,
            images=sample.images,
            input_kwargs={"max_new_tokens": args.max_new_tokens, "do_sample": False},
        )
        return gen_kwargs

    preparation_runs = []
    gen_kwargs = None
    for index in range(args.warmup + args.iterations):
        gen_kwargs, elapsed_ms = _timed_call(prepare_inputs, use_cuda)
        if index >= args.warmup:
            preparation_runs.append(elapsed_ms)

    assert gen_kwargs is not None
    forward_kwargs = _prepare_forward_kwargs(gen_kwargs)
    prompt_tokens = int(forward_kwargs["input_ids"].shape[-1])

    def run_benchmark(function) -> list[dict[str, float]]:
        runs = []
        for index in range(args.warmup + args.iterations):
            timer.reset()
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated() / 2**20 if use_cuda else 0.0
            with torch.inference_mode():
                _, total_ms = _timed_call(function, use_cuda)
            module_times = timer.elapsed_ms()
            if index < args.warmup:
                continue

            run = {"total_ms": total_ms, **module_times}
            projector_ms = module_times.get("projector_ms", 0.0)
            vision_pipeline_ms = module_times.get("vision_pipeline_ms", 0.0)
            nested_projector_ms = projector_ms if projector_is_nested else 0.0
            run["vision_encoder_estimated_ms"] = max(vision_pipeline_ms - nested_projector_ms, 0.0)
            run["vision_encoder_ratio"] = run["vision_encoder_estimated_ms"] / total_ms if total_ms else 0.0
            if use_cuda:
                run["memory_before_mib"] = memory_before
                run["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                run["peak_increment_mib"] = max(run["peak_allocated_mib"] - memory_before, 0.0)
            runs.append(run)
        return runs

    prefill_runs = run_benchmark(lambda: model(**forward_kwargs))
    generation_runs = []
    if not args.skip_generation:
        generation_runs = run_benchmark(lambda: model.generate(**gen_kwargs))

    timer.close()
    result = {
        "configuration": {
            "model_name_or_path": args.model_name_or_path,
            "model_type": model_type,
            "template": args.template,
            "dtype": str(model.dtype),
            "source": sample.source,
            "num_images": len(sample.images),
            "image_sizes": [list(image.size) for image in sample.images],
            "image_min_pixels": args.image_min_pixels,
            "image_max_pixels": args.image_max_pixels,
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "device": str(next(model.parameters()).device),
            "cuda_device": torch.cuda.get_device_name() if use_cuda else None,
        },
        "resolved_modules": {
            "vision_paths": vision_paths,
            "vision_pipeline": pipeline_names,
            "vision_components": [name for name, _ in vision_component_modules],
            "projector_paths": projector_paths,
            "projectors": projector_names,
            "projector_is_nested_in_vision_pipeline": projector_is_nested,
            "language_paths": language_paths,
        },
        "parameters": _parameter_summary(model, vision_paths, projector_paths, language_paths),
        "input_preparation_ms": _stats(preparation_runs),
        "prefill": _summarize_runs(prefill_runs),
        "generation": _summarize_runs(generation_runs),
        "model_memory_after_load": model_memory_after_load,
        "notes": [
            "`vision_pipeline_ms` measures the common executable vision module and can include a nested projector.",
            "`vision_encoder_estimated_ms` subtracts nested projector time from `vision_pipeline_ms`.",
            "`vision_components_ms` is a diagnostic sum over registered vision submodules and may omit functional ops.",
            "Input preparation includes prompt tokenization, image processing, and tensor transfer to the model device.",
            "Peak memory is process-level CUDA memory during the measured call, not vision-exclusive activation memory.",
        ],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_path:
        output_path = Path(args.output_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"Saved profile to {output_path.resolve()}.")


if __name__ == "__main__":
    main()
