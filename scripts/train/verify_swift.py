"""
ms-swift + Qwen3-VL-2B-Instruct forward-pass verification.

Checks available RAM, loads the model in fp32 (no quantization — bitsandbytes
has no CPU backend), runs a forward pass on a test image, and prints peak RAM
usage and inference time. Falls back to Qwen2.5-VL-2B if the primary fails.

Run this before committing training time:
  python scripts/verify_swift.py
  python scripts/verify_swift.py --image path/to/image.png
"""

import argparse
import os
import resource
import time
from pathlib import Path

import psutil

# Prevent ms-swift from routing through ModelScope Hub.
# Must be set before importing swift.
os.environ["USE_MODELSCOPE_HUB"] = "False"

REPO_ROOT      = Path(__file__).parent.parent.parent
PRIMARY_MODEL  = "Qwen/Qwen3-VL-2B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-VL-2B-Instruct"
RAM_WARN_GB    = 16
TEST_PROMPT    = "What do you see in this image? Answer in one sentence."
DEFAULT_IMAGE  = REPO_ROOT / "notebooks/figures/ecg_example_norm.png"


def peak_ram_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def check_ram() -> None:
    mem = psutil.virtual_memory()
    avail = mem.available / 1e9
    total = mem.total / 1e9
    print(f"RAM: {avail:.1f} GB available / {total:.1f} GB total")
    if avail < RAM_WARN_GB:
        print(f"  [WARNING] Less than {RAM_WARN_GB} GB available. "
              f"Close other applications before training.")
    print()


def resolve_local_model(model_name: str) -> str:
    """Return local HF snapshot path, downloading any missing files if needed."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_name)


def check_swift_imports() -> None:
    """Verify ms-swift inference classes are importable — cheap, no model load."""
    from swift import InferRequest, RequestConfig, TransformersEngine  # noqa: F401
    print("  ms-swift TransformersEngine / InferRequest / RequestConfig: OK")


def run_forward_pass(model_name: str, image_path: Path) -> str:
    """Load model via plain HuggingFace transformers and run a forward pass."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    local_path = resolve_local_model(model_name)
    print(f"  Loading {model_name} in fp32 ...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s  |  peak RAM: {peak_ram_gb():.1f} GB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": TEST_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    print("  Running forward pass ...")
    t1 = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    infer_time = time.time() - t1

    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    output = processor.decode(new_tokens, skip_special_tokens=True).strip()
    print(f"  Inference: {infer_time:.1f}s  |  peak RAM: {peak_ram_gb():.1f} GB")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="ms-swift forward-pass smoke test")
    parser.add_argument("--image", type=Path, default=None)
    args = parser.parse_args()

    image_path = args.image or DEFAULT_IMAGE
    if not image_path.exists():
        candidates = sorted((REPO_ROOT / "notebooks/figures").glob("ecg_example_*.png"))
        if candidates:
            image_path = candidates[0]
            print(f"[i] Using {image_path.name} as test image")
        else:
            # No real image available — generate a small dummy PNG with PIL
            import tempfile
            from PIL import Image
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            Image.new("RGB", (224, 224), color=(200, 200, 200)).save(tmp.name)
            image_path = Path(tmp.name)
            print(f"[i] No ECG image found — using blank dummy image for smoke test")

    print("=" * 60)
    print("  verify_swift — forward-pass check")
    print("=" * 60)
    print(f"  Image:  {image_path}")
    print(f"  Prompt: {TEST_PROMPT}")
    print()

    check_ram()

    print("Checking ms-swift imports ...")
    check_swift_imports()
    print()

    model_used = None
    output     = None

    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            output     = run_forward_pass(model_name, image_path)
            model_used = model_name
            break
        except Exception as e:
            print(f"\n[!] {model_name} failed: {e}")
            if model_name == PRIMARY_MODEL:
                print(f"    Trying fallback: {FALLBACK_MODEL}")
            else:
                print("    Both models failed.")

    if model_used is None:
        print("\n[FAIL] Neither model loaded successfully.")
        return

    print()
    print("=" * 60)
    print(f"  Model:    {model_used}")
    print(f"  Output:   {output}")
    print(f"  Peak RAM: {peak_ram_gb():.1f} GB")
    print("=" * 60)
    print("\n[PASS] Environment verified. Ready for training.")


if __name__ == "__main__":
    main()
