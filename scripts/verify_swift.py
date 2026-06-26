"""
ms-swift environment smoke test.

Loads Qwen3-VL-2B-Instruct (4-bit quantization), runs a forward pass on a test
image and text prompt, prints the output and peak memory usage.

If Qwen3-VL-2B-Instruct fails, falls back to Qwen2.5-VL-2B-Instruct and reports
which model worked.

Usage:
  python scripts/verify_swift.py
  python scripts/verify_swift.py --image path/to/image.png
"""

import argparse
import resource
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

PRIMARY_MODEL  = "Qwen/Qwen3-VL-2B-Instruct"
FALLBACK_MODEL = "Qwen/Qwen2.5-VL-2B-Instruct"

TEST_PROMPT = "What do you see in this image? Answer in one sentence."

# Use an existing ECG figure as the test image if no --image is given
DEFAULT_IMAGE = REPO_ROOT / "notebooks/figures/ecg_example_norm.png"


def peak_memory_mb() -> float:
    """Peak RSS in MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_inference(model_name: str, image_path: Path, prompt: str) -> str:
    """Load model via ms-swift and run one inference call. Returns generated text."""
    from swift.llm import InferRequest, PtEngine, RequestConfig, get_model_tokenizer
    from swift.llm.utils import Messages

    print(f"\nLoading {model_name} ...")
    t0 = time.time()

    engine = PtEngine(
        model_name,
        quantization_bit=4,
        torch_dtype="float32",
    )

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s  |  peak RAM: {peak_memory_mb():.0f} MB")

    messages: Messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    request = InferRequest(messages=messages)
    config  = RequestConfig(max_tokens=128, temperature=0.0)

    print(f"  Running forward pass ...")
    t1 = time.time()
    responses = engine.infer([request], request_config=config)
    infer_time = time.time() - t1

    output = responses[0].choices[0].message.content
    print(f"  Inference: {infer_time:.1f}s  |  peak RAM: {peak_memory_mb():.0f} MB")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="ms-swift + Qwen3-VL smoke test")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE,
                        help="Image to run inference on (default: notebooks/figures/ecg_example_norm.png)")
    args = parser.parse_args()

    image_path = args.image
    if not image_path.exists():
        # Try any ECG figure we have
        candidates = list((REPO_ROOT / "notebooks/figures").glob("ecg_example_*.png"))
        if candidates:
            image_path = candidates[0]
            print(f"[i] Default image not found, using {image_path.name}")
        else:
            print(f"[!] No test image found at {image_path}")
            print(f"    Run scripts/eda_ecg.py first to generate ECG figures, or pass --image <path>")
            return

    print("=" * 60)
    print("  ms-swift / Qwen3-VL smoke test")
    print("=" * 60)
    print(f"  Image:  {image_path}")
    print(f"  Prompt: {TEST_PROMPT}")

    model_used = None
    output = None

    for model_name in (PRIMARY_MODEL, FALLBACK_MODEL):
        try:
            output = run_inference(model_name, image_path, TEST_PROMPT)
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

    print("\n" + "=" * 60)
    print(f"  Model:  {model_used}")
    print(f"  Output: {output}")
    print(f"  Peak RAM: {peak_memory_mb():.0f} MB")
    print("=" * 60)
    print("\n[PASS] Environment verified. Ready for training.")


if __name__ == "__main__":
    main()
