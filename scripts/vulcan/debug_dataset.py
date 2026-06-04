import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.converter import SharegptDatasetConverter
from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer


def main() -> None:
    args = dict(
        model_name_or_path="/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b",
        dataset_dir="datasets/vqa_rad",
        dataset="vqa_rad_train_yesno",
        template="qwen3_5_nothink",
        trust_remote_code=True,
        stage="sft",
    )
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    from llamafactory.data.loader import get_dataset_list
    dataset_attrs = get_dataset_list(data_args.dataset, data_args.dataset_dir)
    print(f"Dataset attrs: {dataset_attrs}")
    for attr in dataset_attrs:
        print(f"  formatting={attr.formatting}, system_tag={attr.system_tag}, user_tag={attr.user_tag}, assistant_tag={attr.assistant_tag}")

    dataset_attr = dataset_attrs[0]
    converter = SharegptDatasetConverter(dataset_attr, data_args)

    data_path = Path("datasets/vqa_rad/train.jsonl")
    records = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"\nConverting {len(records)} records...")
    broken = 0
    ok = 0
    for i, record in enumerate(records[:5]):
        result = converter(record)
        prompt = result.get("_prompt", [])
        response = result.get("_response", [])
        system = result.get("_system", "")
        images = result.get("_images", [])
        print(f"  [{i}] prompt_len={len(prompt)}, response_len={len(response)}, system={system!r}, images={images}")
        if len(prompt) == 0 and len(response) == 0:
            broken += 1
        else:
            ok += 1

    for record in records:
        result = converter(record)
        prompt = result.get("_prompt", [])
        response = result.get("_response", [])
        if len(prompt) == 0 and len(response) == 0:
            broken += 1
        else:
            ok += 1

    print(f"\nTotal: {ok} ok, {broken} broken")

    print("\n=== Testing preprocess_dataset ===")
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    processor_obj = SupervisedDatasetProcessor(
        template=template, tokenizer=tokenizer, processor=processor, data_args=data_args
    )

    converted = [converter(r) for r in records[:5]]
    batch = {
        "_prompt": [c["_prompt"] for c in converted],
        "_response": [c["_response"] for c in converted],
        "_system": [c["_system"] for c in converted],
        "_tools": [c["_tools"] for c in converted],
        "_images": [c["_images"] for c in converted],
        "_videos": [c["_videos"] for c in converted],
        "_audios": [c["_audios"] for c in converted],
    }
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  _prompt[0] len: {len(batch['_prompt'][0])}")
    print(f"  _response[0] len: {len(batch['_response'][0])}")
    print(f"  _system[0]: {batch['_system'][0]!r}")
    print(f"  _images[0]: {batch['_images'][0]}")

    result = processor_obj.preprocess_dataset(batch)
    print(f"  Result keys: {list(result.keys())}")
    print(f"  Result len: {len(result.get('input_ids', []))}")


if __name__ == "__main__":
    main()
