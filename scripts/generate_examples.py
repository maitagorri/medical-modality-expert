"""
Generate qualitative examples demonstrating the full modality-routing pipeline.

Feeds a mixed stream of CXR and ECG test examples through the router (base model,
adapter disabled) to detect modality, then answers with the specialist SFT adapter.
Single model load. Saves outputs/results/examples.md.

Usage:
  uv run python scripts/generate_examples.py
"""

import json
import os
import re
import random
import torch
from pathlib import Path

os.environ["USE_MODELSCOPE_HUB"] = "False"

REPO = Path(__file__).parent.parent
BASE_MODEL = (
    "/home/maita/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-VL-2B-Instruct"
    "/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)
CXR_TEST_JSONL = REPO / "data/processed/chexpert_plus/test_sft.jsonl"
ECG_TEST_JSONL = REPO / "data/processed/ptbxl/test.jsonl"
CXR_ADAPTER    = REPO / "outputs/cxr_sft/v1-20260628-064230/checkpoint-20"
ECG_ADAPTER    = REPO / "outputs/ecg_sft/v0-20260628-015306/checkpoint-30"
OUT_PATH       = REPO / "outputs/results/examples.md"

N_CXR       = 5
ECG_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

ROUTER_PROMPT = (
    "What medical imaging modality does this image show? "
    "Answer with exactly one word: xray, ecg, or other."
)


# ── Sampling ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def sample_cxr(entries: list[dict], n: int) -> list[dict]:
    """One example per label, alternating Yes/No answers, tagged with true modality."""
    random.seed(42)
    by_label: dict[str, list[dict]] = {}
    for e in entries:
        text = next(c["text"] for c in e["messages"][0]["content"] if c["type"] == "text")
        m = re.search(r"Does this chest X-ray show (.+?)\?", text)
        label = m.group(1) if m else "unknown"
        by_label.setdefault(label, []).append(e)

    selected, want_yes = [], True
    for label in sorted(by_label):
        if len(selected) >= n:
            break
        bucket = by_label[label]
        yes_ = [e for e in bucket if e["messages"][-1]["content"].strip().lower().startswith("yes")]
        no_  = [e for e in bucket if e["messages"][-1]["content"].strip().lower().startswith("no")]
        pool = (yes_ if want_yes and yes_ else no_ if no_ else bucket)
        entry = random.choice(pool)
        entry = dict(entry, _true_modality="xray")
        selected.append(entry)
        want_yes = not want_yes
    return selected


def sample_ecg(entries: list[dict]) -> list[dict]:
    """One example per ECG class, tagged with true modality."""
    random.seed(42)
    by_class: dict[str, list[dict]] = {}
    for e in entries:
        gt = e["messages"][-1]["content"].strip()
        by_class.setdefault(gt, []).append(e)
    result = []
    for cls in ECG_CLASSES:
        if cls in by_class:
            entry = dict(random.choice(by_class[cls]), _true_modality="ecg")
            result.append(entry)
    return result


# ── Inference ─────────────────────────────────────────────────────────────────

def _generate(messages: list[dict], model, processor, max_new_tokens: int = 16) -> str:
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


def detect_modality(entry: dict, model, processor) -> str:
    image = next(c["image"] for c in entry["messages"][0]["content"] if c["type"] == "image")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": ROUTER_PROMPT},
    ]}]
    with model.disable_adapter():
        raw = _generate(messages, model, processor, max_new_tokens=8)
    token = raw.lower().strip().rstrip(".")
    for key in ("xray", "ecg"):
        if key in token:
            return key
    return "other"


def answer(entry: dict, model, processor) -> str:
    messages = entry["messages"][:-1]
    return _generate(messages, model, processor)


# ── Output ────────────────────────────────────────────────────────────────────

def save_markdown(results: list[dict]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_routed   = sum(1 for r in results if r["routed_modality"] == r["true_modality"])
    n_correct  = sum(1 for r in results if r["answer_correct"])

    lines = [
        "# Pipeline Examples — Mixed Modality Routing",
        "",
        "Model: Qwen3-VL-2B-Instruct. Router = base model (no adapter). "
        "Specialist answers = SFT LoRA adapter per modality.",
        "Test set: CheXpert Plus valid split (CXR) · PTB-XL held-out test (ECG).",
        f"Routing: {n_routed}/{len(results)} correct. "
        f"Answers: {n_correct}/{len(results)} correct.",
        "",
        "| # | Modality | Router | Question | Ground Truth | Prediction | Route ✓ | Answer ✓ |",
        "|---|----------|--------|----------|-------------|------------|---------|---------|",
    ]

    for i, r in enumerate(results):
        route_mark  = "✓" if r["routed_modality"] == r["true_modality"] else "✗"
        answer_mark = "✓" if r["answer_correct"] else "✗"
        question = r["question"][:60] + "…" if len(r["question"]) > 60 else r["question"]
        lines.append(
            f"| {i+1} | {r['true_modality']} | {r['routed_modality']} "
            f"| {question} | {r['ground_truth']} | {r['prediction']} "
            f"| {route_mark} | {answer_mark} |"
        )

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {OUT_PATH.relative_to(REPO)}")
    print(f"Routing accuracy:  {n_routed}/{len(results)}")
    print(f"Answer accuracy:   {n_correct}/{len(results)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    cxr_entries = sample_cxr(load_jsonl(CXR_TEST_JSONL), N_CXR)
    ecg_entries = sample_ecg(load_jsonl(ECG_TEST_JSONL))

    # Shuffle into a mixed stream
    all_entries = cxr_entries + ecg_entries
    random.seed(0)
    random.shuffle(all_entries)
    print(f"Mixed stream: {len(all_entries)} examples ({len(cxr_entries)} CXR, {len(ecg_entries)} ECG)")

    print("\nLoading base model ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    print("Loading CXR SFT adapter ...")
    model = PeftModel.from_pretrained(model, str(CXR_ADAPTER), adapter_name="xray")
    print("Loading ECG SFT adapter ...")
    model.load_adapter(str(ECG_ADAPTER), adapter_name="ecg")
    model.eval()

    results = []
    for i, entry in enumerate(all_entries):
        true_mod = entry["_true_modality"]
        gt       = entry["messages"][-1]["content"].strip()
        question = next(c["text"] for c in entry["messages"][0]["content"] if c["type"] == "text")

        print(f"\n[{i+1}/{len(all_entries)}] true={true_mod}")
        routed = detect_modality(entry, model, processor)
        print(f"  router → {routed}")

        if routed in ("xray", "ecg"):
            model.set_adapter(routed)
        pred = answer(entry, model, processor)
        print(f"  GT={gt!r}  Pred={pred!r}")

        correct = (
            pred.lower().startswith(gt.lower()[:3]) if true_mod == "xray"
            else gt.upper() in pred.upper()
        )
        results.append({
            "true_modality":    true_mod,
            "routed_modality":  routed,
            "question":         question,
            "ground_truth":     gt,
            "prediction":       pred,
            "answer_correct":   correct,
        })

    save_markdown(results)
    print("Done.")


if __name__ == "__main__":
    main()
