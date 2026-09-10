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

"""Run the frozen Phase 6A prompt and image-counterfactual audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageEnhance, ImageOps, ImageStat


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase5e_semantics import REFERENCE_CAPTION, generate_short_caption  # noqa: E402
from phase6a_robustness import AUDIT_VERSION, evaluate_robustness_case, summarize_model_cases  # noqa: E402
from run_phase2_ablation import build_dataloader, load_yaml, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle  # noqa: E402

from llamafactory.train.vulcan.modeling import find_mlp_layers, get_intermediate_size  # noqa: E402


TARGET_IMAGE = Path("/root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images/COCO_val2014_000000093736.jpg")
BIKE_ONLY_IMAGE = Path("/root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images/COCO_val2014_000000203564.jpg")
TRAIN_ONLY_IMAGE = Path("/root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images/COCO_val2014_000000232538.jpg")
UNRELATED_IMAGE = Path("/root/autodl-pub-RTX4090-hdd-1/datasets/coco-caption-lf/images/COCO_val2014_000000322141.jpg")
CANONICAL_PROMPT = (
    "Write one concise English caption for this image in one complete sentence. Output only the caption."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Phase 5E's 62,950-neuron model for prompt robustness.")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--pruned_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--original_model_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--preprocessing_num_workers", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2060)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_rgb(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", compress_level=9)


def prepare_derived_images(output_dir: Path) -> dict[str, Path]:
    """Create deterministic benign and semantic-removal interventions."""
    derived_dir = output_dir / "audit_inputs" / "derived_images"
    with Image.open(TARGET_IMAGE) as source:
        image = source.convert("RGB")
        width, height = image.size
        _save_rgb(ImageOps.mirror(image), derived_dir / "target_hflip.png")
        _save_rgb(ImageEnhance.Brightness(image).enhance(0.65), derived_dir / "target_brightness_065.png")
        crop_box = (round(width * 0.05), round(height * 0.05), round(width * 0.95), round(height * 0.95))
        mild_crop = image.crop(crop_box).resize(image.size, Image.Resampling.BICUBIC)
        _save_rgb(mild_crop, derived_dir / "target_center_crop_90.png")

        train_crop = image.crop((round(width * 0.52), 0, width, height)).resize(image.size, Image.Resampling.BICUBIC)
        _save_rgb(train_crop, derived_dir / "counterfactual_train_only_crop.png")

        occluded = image.copy()
        background = image.crop((round(width * 0.58), round(height * 0.35), width, height))
        fill = tuple(round(value) for value in ImageStat.Stat(background).median[:3])
        occlusion = Image.new("RGB", (round(width * 0.60), round(height * 0.72)), fill)
        occluded.paste(occlusion, (0, round(height * 0.28)))
        _save_rgb(occluded, derived_dir / "counterfactual_bicycles_occluded.png")
        _save_rgb(Image.new("RGB", image.size, (127, 127, 127)), derived_dir / "counterfactual_blank.png")

    return {
        "target": TARGET_IMAGE,
        "target_hflip": derived_dir / "target_hflip.png",
        "target_brightness_065": derived_dir / "target_brightness_065.png",
        "target_center_crop_90": derived_dir / "target_center_crop_90.png",
        "counterfactual_train_only_crop": derived_dir / "counterfactual_train_only_crop.png",
        "counterfactual_bicycles_occluded": derived_dir / "counterfactual_bicycles_occluded.png",
        "counterfactual_blank": derived_dir / "counterfactual_blank.png",
        "bike_only": BIKE_ONLY_IMAGE,
        "train_only": TRAIN_ONLY_IMAGE,
        "unrelated": UNRELATED_IMAGE,
    }


def frozen_cases(images: dict[str, Path]) -> list[dict[str, Any]]:
    prompt_cases = [
        ("prompt_canonical", CANONICAL_PROMPT),
        ("prompt_generic", "Describe this image in one concise English sentence. Output only the sentence."),
        ("prompt_alt_text", "Provide factual one-sentence English alt text for the image, with no preamble."),
        ("prompt_main_scene", "Summarize the main objects and setting shown here in one complete English sentence."),
        ("prompt_photo_question", "What does this photograph show? Answer with one concise English sentence only."),
    ]
    cases = [
        {
            "case_id": case_id,
            "group": "prompt_primary",
            "prompt": prompt,
            "image": str(images["target"]),
            "expected_profile": "target_semantics",
            "answer_leakage": False,
        }
        for case_id, prompt in prompt_cases
    ]
    cases.extend(
        [
            {
                "case_id": "prompt_relation_diagnostic",
                "group": "prompt_diagnostic",
                "prompt": "Where have the bicycles been placed? Answer in one complete English sentence.",
                "image": str(images["target"]),
                "expected_profile": "target_semantics",
                "answer_leakage": True,
            },
            *[
                {
                    "case_id": case_id,
                    "group": "benign_image",
                    "prompt": CANONICAL_PROMPT,
                    "image": str(images[image_key]),
                    "expected_profile": "target_semantics",
                    "answer_leakage": False,
                }
                for case_id, image_key in (
                    ("benign_hflip", "target_hflip"),
                    ("benign_brightness_065", "target_brightness_065"),
                    ("benign_center_crop_90", "target_center_crop_90"),
                )
            ],
            {
                "case_id": "counterfactual_train_crop",
                "group": "counterfactual",
                "prompt": CANONICAL_PROMPT,
                "image": str(images["counterfactual_train_only_crop"]),
                "expected_profile": "train_without_bicycle",
                "answer_leakage": False,
            },
            {
                "case_id": "counterfactual_bicycles_occluded",
                "group": "counterfactual",
                "prompt": CANONICAL_PROMPT,
                "image": str(images["counterfactual_bicycles_occluded"]),
                "expected_profile": "train_without_bicycle",
                "answer_leakage": False,
            },
            {
                "case_id": "counterfactual_bike_only",
                "group": "counterfactual",
                "prompt": CANONICAL_PROMPT,
                "image": str(images["bike_only"]),
                "expected_profile": "bicycle_without_train",
                "answer_leakage": False,
            },
            {
                "case_id": "counterfactual_train_only",
                "group": "counterfactual",
                "prompt": CANONICAL_PROMPT,
                "image": str(images["train_only"]),
                "expected_profile": "train_without_bicycle",
                "answer_leakage": False,
            },
            {
                "case_id": "counterfactual_unrelated",
                "group": "counterfactual",
                "prompt": CANONICAL_PROMPT,
                "image": str(images["unrelated"]),
                "expected_profile": "neither_bicycle_nor_train",
                "answer_leakage": False,
            },
            {
                "case_id": "counterfactual_blank_image",
                "group": "counterfactual",
                "prompt": CANONICAL_PROMPT,
                "image": str(images["counterfactual_blank"]),
                "expected_profile": "no_original_scene",
                "answer_leakage": False,
            },
        ]
    )
    return cases


def prepare_dataset(output_dir: Path, cases: list[dict[str, Any]]) -> tuple[Path, Path]:
    dataset_dir = output_dir / "audit_inputs" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        content = case["prompt"]
        row: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": f"<image>\n{content}" if case["image"] else content},
                {"role": "assistant", "content": REFERENCE_CAPTION},
            ]
        }
        if case["image"]:
            row["images"] = [case["image"]]
        rows.append(row)
    dataset_path = dataset_dir / "phase6a_cases.json"
    write_json(dataset_path, rows)
    write_json(
        dataset_dir / "dataset_info.json",
        {
            "phase6a_prompt_counterfactual": {
                "file_name": dataset_path.name,
                "formatting": "sharegpt",
                "columns": {"messages": "messages", "images": "images"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                },
            }
        },
    )
    write_json(dataset_dir / "case_manifest.json", {"audit_version": AUDIT_VERSION, "cases": cases})
    return dataset_dir, dataset_path


def _model_structure(model: torch.nn.Module) -> dict[str, Any]:
    widths = {str(layer.index): get_intermediate_size(layer.mlp) for layer in find_mlp_layers(model)}
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "ffn_widths": widths,
        "total_ffn_neurons": sum(widths.values()),
    }


def _release_device_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_model(
    *,
    name: str,
    model_path: Path,
    config_path: Path,
    dataset_dir: Path,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    print(f"Phase 6A evaluating {name}: {model_path}", flush=True)
    model, tokenizer_module, template, config = load_model_bundle(
        config_path,
        model_path,
        device,
        trust_remote_code=name != "original",
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    config.update(
        {
            "dataset_dir": str(dataset_dir),
            "dataset": "phase6a_prompt_counterfactual",
            "eval_dataset": None,
            "tokenized_path": None,
            "max_samples": len(cases),
            "overwrite_cache": True,
        }
    )
    dataloader, dataset_manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=args.num_workers,
        sample_offset=0,
        max_samples=len(cases),
        allow_short_dataset=False,
        max_image_repeat=len(cases),
        allow_excessive_image_repeats=True,
        dataset_stage="sft",
    )
    rows = []
    for case, batch in zip(cases, dataloader, strict=True):
        batch = move_batch_to_device(batch, device)
        prompt_inputs, prompt_tokens = build_prompt_inputs(batch)
        generation = generate_short_caption(
            model,
            tokenizer_module["tokenizer"],
            prompt_inputs,
            max_new_tokens=args.max_new_tokens,
        )
        evaluation = evaluate_robustness_case(
            generation["raw_text"],
            expected_profile=case["expected_profile"],
            terminated_normally=generation["stop"]["terminated_normally"],
            hit_token_limit=generation["stop"]["hit_token_limit"],
        )
        rows.append(
            {
                **case,
                "prompt_tokens": prompt_tokens,
                "generation": generation,
                "evaluation": evaluation,
            }
        )
        print(
            json.dumps(
                {
                    "model": name,
                    "case": case["case_id"],
                    "passed": evaluation["passed"],
                    "caption": generation["final_caption"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    if len(rows) != len(cases):
        raise RuntimeError(f"Expected {len(cases)} audit rows, evaluated {len(rows)}.")
    return {
        "name": name,
        "model_path": str(model_path),
        "structure": _model_structure(model),
        "dataset_manifest": dataset_manifest,
        "cases": rows,
        "summary": summarize_model_cases(rows),
    }


def build_comparison(original: dict[str, Any], pruned: dict[str, Any]) -> dict[str, Any]:
    original_by_id = {row["case_id"]: row for row in original["cases"]}
    rows = []
    for row in pruned["cases"]:
        reference = original_by_id[row["case_id"]]
        rows.append(
            {
                "case_id": row["case_id"],
                "group": row["group"],
                "original_passed": reference["evaluation"]["passed"],
                "pruned_passed": row["evaluation"]["passed"],
                "retained_on_reference_pass": bool(
                    not reference["evaluation"]["passed"] or row["evaluation"]["passed"]
                ),
                "original_contract_leak_changed": (
                    reference["evaluation"]["original_contract_reproduced"]
                    != row["evaluation"]["original_contract_reproduced"]
                ),
            }
        )
    eligible = [row for row in rows if row["original_passed"]]
    retained = sum(row["pruned_passed"] for row in eligible)
    return {
        "cases": rows,
        "reference_eligible_cases": len(eligible),
        "pruned_passes_on_reference_eligible_cases": retained,
        "paired_retention_rate": retained / len(eligible) if eligible else None,
        "structure_delta": {
            "deleted_ffn_neurons": (
                original["structure"]["total_ffn_neurons"] - pruned["structure"]["total_ffn_neurons"]
            ),
            "removed_parameters": original["structure"]["parameters"] - pruned["structure"]["parameters"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be positive.")
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "phase6a_results.json"
    if result_path.exists() and not args.resume:
        raise FileExistsError(f"Phase 6A result already exists: {result_path}. Pass --resume to reuse completed arms.")

    config_path = Path(args.config).resolve()
    base_config = load_yaml(config_path)
    original_path = Path(args.original_model_path or base_config["model_name_or_path"]).resolve()
    pruned_path = Path(args.pruned_model_path).resolve()
    missing = [str(path) for path in (config_path, original_path, pruned_path, TARGET_IMAGE) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Phase 6A inputs do not exist: {missing}.")

    images = prepare_derived_images(output_dir)
    cases = frozen_cases(images)
    dataset_dir, dataset_path = prepare_dataset(output_dir, cases)
    identity = {
        "audit_version": AUDIT_VERSION,
        "config_path": str(config_path),
        "original_model_path": str(original_path),
        "pruned_model_path": str(pruned_path),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "case_manifest_sha256": sha256_file(dataset_dir / "case_manifest.json"),
        "dataset_sha256": sha256_file(dataset_path),
        "input_sha256": {key: sha256_file(path) for key, path in images.items()},
    }
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["config"] != identity:
            raise ValueError("Cannot resume Phase 6A with changed frozen inputs or identity fields.")
        result["complete"] = False
    else:
        result = {
            "complete": False,
            "interpretation": (
                "This audit distinguishes semantic retention on prompt/benign variants from sensitivity to inputs "
                "that should change the answer. It is a fixed-case audit, not a population-level robustness estimate."
            ),
            "config": identity,
            "pre_registered_gates": {
                "canonical_target_pass": "required",
                "prompt_primary": "at least 4/5 leak-free prompt variants",
                "benign_image": "at least 2/3 meaning-preserving transformations",
                "counterfactual_leakage": "0/6 may reproduce the original bicycle-in-train relation",
                "counterfactual_facts": "at least 5/6 must satisfy frozen fact checks",
                "output_hygiene": "all 15 outputs must terminate cleanly without repetition or role garbage",
            },
            "cases": cases,
            "models": {},
            "comparison": None,
        }
        write_json(result_path, result)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, model_path in (("original", original_path), ("pruned_62950", pruned_path)):
        if name in result["models"]:
            continue
        model_result = None
        try:
            model_result = evaluate_model(
                name=name,
                model_path=model_path,
                config_path=config_path,
                dataset_dir=dataset_dir,
                cases=cases,
                args=args,
                device=device,
            )
            result["models"][name] = model_result
            write_json(result_path, result)
        finally:
            model_result = None
            _release_device_cache()

    for model_result in result["models"].values():
        model_result["summary"] = summarize_model_cases(model_result["cases"])
    result["comparison"] = build_comparison(result["models"]["original"], result["models"]["pruned_62950"])
    result["complete"] = True
    write_json(result_path, result)
    return result


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "original": result["models"]["original"]["summary"],
                "pruned_62950": result["models"]["pruned_62950"]["summary"],
                "comparison": result["comparison"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
