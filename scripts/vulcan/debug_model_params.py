import importlib

from transformers import AutoConfig, AutoProcessor


config = AutoConfig.from_pretrained(
    "/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b",
    trust_remote_code=True,
)

print(f"model_type: {config.model_type}")
print(f"architectures: {config.architectures}")
print(f"has vision_config: {hasattr(config, 'vision_config')}")
if hasattr(config, "vision_config"):
    print(f"vision_config type: {type(config.vision_config)}")
    print(f"vision_config: {config.vision_config}")

print("\n=== Trying AutoProcessor ===")
try:
    processor = AutoProcessor.from_pretrained(
        "/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b",
        trust_remote_code=True,
    )
    print(f"Processor type: {type(processor)}")
    print(f"Processor attributes: {[a for a in dir(processor) if not a.startswith('_')][:20]}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Trying correct model class ===")
try:
    mod = importlib.import_module("transformers")
    model_cls = getattr(mod, config.architectures[0], None)
    if model_cls is None:
        print(f"Class {config.architectures[0]} not found in transformers")
        print("Trying trust_remote_code model...")
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            "/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b",
            trust_remote_code=True,
            torch_dtype="auto",
        )
    else:
        model = model_cls.from_pretrained(
            "/root/autodl-pub-RTX4090-hdd-1/models/qwen3.5-0.8b",
            trust_remote_code=True,
            torch_dtype="auto",
        )

    all_names = [n for n, _ in model.named_parameters()]
    print(f"Total params: {len(all_names)}")

    vision_params = [n for n in all_names if "visual" in n or "vision" in n]
    proj_params = [n for n in all_names if "merger" in n or "projector" in n]
    print(f"Vision params: {len(vision_params)}")
    if vision_params:
        for n in vision_params[:5]:
            print(f"  {n}")
        print(f"  ... total: {len(vision_params)}")

    print(f"Projector params: {len(proj_params)}")
    if proj_params:
        for n in proj_params[:5]:
            print(f"  {n}")

    print("\n=== Top-level modules ===")
    for name, _ in model.named_children():
        print(f"  {name}")

except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
