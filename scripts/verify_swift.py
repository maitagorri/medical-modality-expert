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

# Force ms-swift to use HuggingFace Hub instead of ModelScope,
# so the locally cached model is found immediately.
os.environ.setdefault("USE_MODELSCOPE_HUB", "0")

REPO_ROOT      = Path(__file__).parent.parent
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


def run_forward_pass(model_name: str, image_path: Path) -> str:
    from swift import InferRequest, RequestConfig, TransformersEngine

    print(f"Loading {model_name} in fp32 ...")
    t0 = time.time()
    engine = TransformersEngine(
        model_name,
        torch_dtype="float32",
    )
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
    config   = RequestConfig(max_tokens=64, temperature=0.0)
    request  = InferRequest(messages=messages)

    print("  Running forward pass ...")
    t1 = time.time()
    response  = engine.infer([request], request_config=config)[0]
    infer_time = time.time() - t1

    output = response.choices[0].message.content.strip()
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
