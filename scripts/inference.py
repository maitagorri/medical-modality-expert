"""
Full modality-expert inference pipeline.

Detects the imaging modality from the input image using the base model (zero-shot),
loads the appropriate LoRA adapter, and runs the specialist model.

Usage:
  python scripts/inference.py --image path/to/image.png
  python scripts/inference.py --image path/to/image.png --query "Is there pleural effusion?"
  python scripts/inference.py --demo
"""

import argparse
import json
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

# Adapter registry: modality keyword → adapter directory.
# Update paths after training completes.
ADAPTER_MAP: dict[str, str] = {
    "xray": str(REPO_ROOT / "outputs/cxr_sft"),
    "ecg":  str(REPO_ROOT / "outputs/ecg_sft"),
}

ROUTER_PROMPT = (
    "What medical imaging modality does this image show? "
    "Answer with exactly one word from this list: xray, ecg, echo, ct, mri, ultrasound, other."
)

DEFAULT_QUERIES: dict[str, str] = {
    "xray": "Does this chest X-ray show any abnormal findings?",
    "ecg":  "What is the primary cardiac diagnosis for this 12-lead ECG? "
            "Answer with exactly one of: NORM, MI, STTC, CD, HYP.",
    "other": "Describe what you see in this medical image.",
}


# ── Engine helpers ────────────────────────────────────────────────────────────

def _make_engine(adapter_path: str | None = None):
    from swift import TransformersEngine
    kwargs: dict = dict(model=BASE_MODEL, dtype="float32")
    if adapter_path:
        kwargs["adapters"] = [adapter_path]
    return TransformersEngine(**kwargs)


def _infer(engine, messages: list[dict], max_tokens: int = 64) -> str:
    from swift import InferRequest, RequestConfig
    config   = RequestConfig(max_tokens=max_tokens, temperature=0.0)
    response = engine.infer([InferRequest(messages=messages)], request_config=config)[0]
    return response.choices[0].message.content.strip()


# ── Modality routing ──────────────────────────────────────────────────────────

def detect_modality(image_path: Path, engine) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": ROUTER_PROMPT},
            ],
        }
    ]
    raw = _infer(engine, messages, max_tokens=8).lower().strip().rstrip(".")
    # Normalise to a known key
    for key in ADAPTER_MAP:
        if key in raw:
            return key
    return raw if raw else "other"


# ── Specialist inference ──────────────────────────────────────────────────────

def run_specialist(image_path: Path, query: str, modality: str, router_engine) -> tuple[str, bool]:
    """Returns (response_text, used_adapter)."""
    adapter_path = ADAPTER_MAP.get(modality)

    if adapter_path and Path(adapter_path).exists():
        specialist = _make_engine(adapter_path)
        used_adapter = True
    else:
        # Adapter not trained yet — fall back to base model
        specialist = router_engine
        used_adapter = False

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": query},
            ],
        }
    ]
    response = _infer(specialist, messages, max_tokens=128)
    return response, used_adapter


# ── Demo ──────────────────────────────────────────────────────────────────────

def _load_demo_example(jsonl_path: Path) -> tuple[Path, str] | None:
    """Return (image_path, query) from first entry of a test JSONL."""
    if not jsonl_path.exists():
        return None
    with open(jsonl_path) as f:
        entry = json.loads(f.readline())
    content = entry["messages"][0]["content"]
    image   = next(c["image"] for c in content if c["type"] == "image")
    query   = next(c["text"]  for c in content if c["type"] == "text")
    return Path(image), query


def run_demo(engine) -> None:
    print("=" * 60)
    print("  Demo — modality-expert pipeline")
    print("=" * 60)

    examples = [
        ("CXR",  REPO_ROOT / "data/processed/chexpert_plus/test_sft.jsonl"),
        ("ECG",  REPO_ROOT / "data/processed/ptbxl/test.jsonl"),
    ]

    for label, jsonl_path in examples:
        result = _load_demo_example(jsonl_path)
        if result is None:
            print(f"\n[{label}] Skipped — {jsonl_path} not found")
            continue

        image_path, query = result
        print(f"\n[{label}]")
        print(f"  Image:   {image_path.name}")
        print(f"  Query:   {query[:80]}{'...' if len(query) > 80 else ''}")

        modality = detect_modality(image_path, engine)
        response, used_adapter = run_specialist(image_path, query, modality, engine)
        adapter_info = ADAPTER_MAP.get(modality, "none")

        print(f"  Modality detected: {modality}")
        print(f"  Adapter loaded:    {adapter_info if used_adapter else 'none (base model)'}")
        print(f"  Response:          {response}")

    print("\n" + "=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Modality-expert inference pipeline")
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to input image")
    parser.add_argument("--query", type=str, default=None,
                        help="Question to ask the specialist. Defaults to a modality-specific prompt.")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo on one CXR and one ECG example from the test sets")
    args = parser.parse_args()

    if not args.demo and args.image is None:
        parser.error("Provide --image or --demo")

    print("Loading base model ...")
    engine = _make_engine()

    if args.demo:
        run_demo(engine)
        return

    image_path = args.image
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    modality = detect_modality(image_path, engine)
    query    = args.query or DEFAULT_QUERIES.get(modality, DEFAULT_QUERIES["other"])
    response, used_adapter = run_specialist(image_path, query, modality, engine)

    print(f"\nModality detected: {modality}")
    print(f"Adapter loaded:    {ADAPTER_MAP.get(modality, 'none') if used_adapter else 'none (base model)'}")
    print(f"Query:             {query}")
    print(f"Response:          {response}")


if __name__ == "__main__":
    main()
