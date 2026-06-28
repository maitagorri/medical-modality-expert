"""
Modality-expert inference pipeline.

Loads Qwen3-VL-2B once with both LoRA adapters registered. Routes the input image
to the correct specialist adapter using the base model (adapter disabled), then
switches to the specialist adapter for the final answer.

No second model load — PEFT multi-adapter switching is zero-cost.

Usage:
  uv run python scripts/inference.py --image path/to/image.png
  uv run python scripts/inference.py --image path/to/image.png --query "Is there effusion?"
  uv run python scripts/inference.py --demo
"""

import argparse
import json
import os
import torch
from pathlib import Path

os.environ["USE_MODELSCOPE_HUB"] = "False"

REPO = Path(__file__).parent.parent

BASE_MODEL = (
    "/home/maita/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-VL-2B-Instruct"
    "/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)

# Fill in best checkpoint path after each SFT run completes.
# Set to None until the adapter exists.
ADAPTERS: dict[str, Path | None] = {
    "xray": REPO / "outputs/cxr_sft/v0-20260627-221625/checkpoint-78",
    "ecg":  REPO / "outputs/ecg_sft/v0-20260628-015306/checkpoint-30",
}

ROUTER_PROMPT = (
    "What medical imaging modality does this image show? "
    "Answer with exactly one word: xray, ecg, or other."
)

DEFAULT_QUERY: dict[str, str] = {
    "xray": "Does this chest X-ray show any abnormal findings?",
    "ecg":  "What is the primary cardiac diagnosis for this 12-lead ECG? "
            "Answer with exactly one of: NORM, MI, STTC, CD, HYP.",
    "other": "Describe what you see in this medical image.",
}

MAX_NEW_TOKENS = 128


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_and_processor():
    """Load base model then register all available adapters."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    print("Loading base model ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float32
    )

    available = {k: v for k, v in ADAPTERS.items() if v is not None and Path(v).exists()}
    if not available:
        print("No SFT adapters found — running base model only.")
        model.eval()
        return model, processor, set()

    first_key, first_path = next(iter(available.items()))
    print(f"Loading adapter: {first_key} ({first_path})")
    model = PeftModel.from_pretrained(model, str(first_path), adapter_name=first_key)

    for key, path in list(available.items())[1:]:
        print(f"Loading adapter: {key} ({path})")
        model.load_adapter(str(path), adapter_name=key)

    model.eval()
    return model, processor, set(available.keys())


# ── Inference helpers ─────────────────────────────────────────────────────────

def _generate(messages: list[dict], model, processor, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        return_tensors="pt",
    )
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _image_messages(image_path: Path, text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": text},
            ],
        }
    ]


# ── Pipeline ──────────────────────────────────────────────────────────────────

def detect_modality(image_path: Path, model, processor, loaded_adapters: set) -> str:
    messages = _image_messages(image_path, ROUTER_PROMPT)
    if loaded_adapters:
        with model.disable_adapter():
            raw = _generate(messages, model, processor, max_new_tokens=8)
    else:
        raw = _generate(messages, model, processor, max_new_tokens=8)
    token = raw.lower().strip().rstrip(".")
    for key in ADAPTERS:
        if key in token:
            return key
    return "other"


def run(image_path: Path, query: str | None, model, processor, loaded_adapters: set) -> dict:
    modality = detect_modality(image_path, model, processor, loaded_adapters)
    query = query or DEFAULT_QUERY.get(modality, DEFAULT_QUERY["other"])
    messages = _image_messages(image_path, query)

    if modality in loaded_adapters:
        model.set_adapter(modality)
        response = _generate(messages, model, processor)
        adapter_used = str(ADAPTERS[modality])
    else:
        if loaded_adapters:
            with model.disable_adapter():
                response = _generate(messages, model, processor)
        else:
            response = _generate(messages, model, processor)
        adapter_used = "none (base model)"

    return {"modality": modality, "adapter": adapter_used, "query": query, "response": response}


# ── Demo ──────────────────────────────────────────────────────────────────────

def _first_example(jsonl_path: Path) -> tuple[Path, str] | None:
    if not jsonl_path.exists():
        return None
    entry = json.loads(jsonl_path.read_text().splitlines()[0])
    content = entry["messages"][0]["content"]
    image = next(c["image"] for c in content if c["type"] == "image")
    query = next(c["text"]  for c in content if c["type"] == "text")
    return Path(image), query


def run_demo(model, processor, loaded_adapters: set) -> None:
    examples = [
        ("CXR", REPO / "data/processed/chexpert_plus/test_sft.jsonl"),
        ("ECG", REPO / "data/processed/ptbxl/test.jsonl"),
    ]
    print("=" * 60)
    print("  Demo — modality-expert pipeline")
    print("=" * 60)
    for label, jsonl_path in examples:
        ex = _first_example(jsonl_path)
        if ex is None:
            print(f"\n[{label}] Skipped — {jsonl_path.relative_to(REPO)} not found")
            continue
        image_path, query = ex
        print(f"\n[{label}]  {image_path.name}")
        result = run(image_path, query, model, processor, loaded_adapters)
        print(f"  Modality: {result['modality']}")
        print(f"  Adapter:  {result['adapter']}")
        print(f"  Query:    {result['query'][:80]}")
        print(f"  Response: {result['response']}")
    print("\n" + "=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if not args.demo and args.image is None:
        parser.error("Provide --image or --demo")

    model, processor, loaded_adapters = load_model_and_processor()

    if args.demo:
        run_demo(model, processor, loaded_adapters)
        return

    if not args.image.exists():
        parser.error(f"Image not found: {args.image}")

    result = run(args.image, args.query, model, processor, loaded_adapters)
    print(f"\nModality: {result['modality']}")
    print(f"Adapter:  {result['adapter']}")
    print(f"Query:    {result['query']}")
    print(f"Response: {result['response']}")


if __name__ == "__main__":
    main()
