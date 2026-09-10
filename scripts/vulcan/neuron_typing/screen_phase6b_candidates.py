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

"""Generate deterministic base-model captions for Phase 6B candidate selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from phase5e_semantics import generate_short_caption  # noqa: E402
from run_phase2_ablation import build_dataloader, load_yaml, move_batch_to_device  # noqa: E402
from run_phase5_single_sample_frontier import build_prompt_inputs, write_json  # noqa: E402
from verify_phase5_structural_equivalence import load_model_bundle  # noqa: E402


CANONICAL_PROMPT = (
    "Write one concise English caption for this image in one complete sentence. Output only the caption."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen diverse COCO candidates before Phase 6B mask search.")
    parser.add_argument("--candidate_file", default="data/phase6b_candidates/candidates.json")
    parser.add_argument("--config", default="scripts/vulcan/neuron_typing/configs/phase5e_coco.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2061)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_dataset(output_dir: Path, candidates: list[dict[str, Any]]) -> Path:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"<image>\n{CANONICAL_PROMPT}"},
                {"role": "assistant", "content": candidate["gold_caption"]},
            ],
            "images": [candidate["image"]],
        }
        for candidate in candidates
    ]
    write_json(dataset_dir / "candidates.json", rows)
    write_json(
        dataset_dir / "dataset_info.json",
        {
            "phase6b_candidates": {
                "file_name": "candidates.json",
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
    return dataset_dir


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    candidate_path = Path(args.candidate_file).resolve()
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if len({row["sample_id"] for row in candidates}) != len(candidates):
        raise ValueError("Phase 6B candidate sample IDs must be unique.")
    missing_images = [row["image"] for row in candidates if not Path(row["image"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing Phase 6B candidate images: {missing_images}.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "candidate_screen.json"
    if result_path.exists():
        raise FileExistsError(f"Candidate screen already exists: {result_path}.")
    dataset_dir = prepare_dataset(output_dir, candidates)
    config_path = Path(args.config).resolve()
    base_config = load_yaml(config_path)
    model_path = Path(args.model_name_or_path or base_config["model_name_or_path"]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer_module, template, config = load_model_bundle(
        config_path,
        model_path,
        device,
        trust_remote_code=False,
        preprocessing_num_workers=1,
    )
    config.update(
        {
            "dataset_dir": str(dataset_dir),
            "dataset": "phase6b_candidates",
            "eval_dataset": None,
            "tokenized_path": None,
            "max_samples": len(candidates),
            "overwrite_cache": True,
        }
    )
    dataloader, manifest = build_dataloader(
        config,
        model,
        tokenizer_module,
        template,
        batch_size=1,
        num_workers=0,
        sample_offset=0,
        max_samples=len(candidates),
        allow_short_dataset=False,
        max_image_repeat=1,
        allow_excessive_image_repeats=False,
        dataset_stage="sft",
    )
    rows = []
    for candidate, batch in zip(candidates, dataloader, strict=True):
        batch = move_batch_to_device(batch, device)
        prompt_inputs, prompt_tokens = build_prompt_inputs(batch)
        generation = generate_short_caption(
            model,
            tokenizer_module["tokenizer"],
            prompt_inputs,
            max_new_tokens=args.max_new_tokens,
        )
        row = {**candidate, "prompt_tokens": prompt_tokens, "generation": generation}
        rows.append(row)
        print(
            json.dumps(
                {"sample_id": candidate["sample_id"], "caption": generation["final_caption"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    result = {
        "complete": True,
        "config": {
            "candidate_file": str(candidate_path),
            "candidate_file_sha256": sha256_file(candidate_path),
            "config_path": str(config_path),
            "model_path": str(model_path),
            "prompt": CANONICAL_PROMPT,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "dataset_manifest": manifest,
        "candidates": rows,
    }
    write_json(result_path, result)


if __name__ == "__main__":
    main()
