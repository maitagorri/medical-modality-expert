"""Run inference with a fine-tuned Qwen3-VL checkpoint."""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, help="Optional image path for VL inference")
    p.add_argument("--prompt", type=str, default="Describe this medical image.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    messages = [{"role": "user", "content": args.prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    response = processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
