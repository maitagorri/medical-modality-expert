"""
Evaluation script — assembles the full metrics table for the PoC.

Computes zeroshot token_acc by running the base model (no adapter) on the
val sets. Pretrain and SFT metrics are read directly from training logs.
Single model load; no OOM risk.

Outputs: outputs/results/metrics.{json,md}

Usage:
  uv run python scripts/evaluate.py
"""

import json
import os
import re
import torch
from pathlib import Path

os.environ["USE_MODELSCOPE_HUB"] = "False"

REPO = Path(__file__).parent.parent
BASE_MODEL = (
    "/home/maita/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-VL-2B-Instruct"
    "/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)

CXR_VAL_JSONL = REPO / "data/processed/chexpert_plus/val_sft.jsonl"
ECG_VAL_JSONL = REPO / "data/processed/ptbxl/val.jsonl"

TRAINING_LOGS = {
    "cxr_pretrain": REPO / "outputs/cxr_pretrain_preserved/logging.jsonl",
    "ecg_pretrain": REPO / "outputs/ecg_pretrain_preserved/logging.jsonl",
    "cxr_sft":      REPO / "outputs/cxr_sft/v1-20260628-064230/logging.jsonl",
    "ecg_sft":      REPO / "outputs/ecg_sft/v0-20260628-015306/logging.jsonl",
}

OUT_DIR = REPO / "outputs/results"


# ── Log parsing ───────────────────────────────────────────────────────────────

def best_eval_from_log(path: Path) -> dict:
    entries = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    evals = [e for e in entries if "eval_loss" in e]
    if not evals:
        return {}
    best = min(evals, key=lambda e: e["eval_loss"])
    return {
        "eval_loss":      round(best["eval_loss"], 4),
        "eval_token_acc": round(best.get("eval_token_acc", float("nan")), 4),
        "step":           best.get("global_step/max_steps", "?"),
    }


# ── Inference ─────────────────────────────────────────────────────────────────

MAX_PIXELS = 200704  # match training configs — limits CXR images to ~250 tokens

def _generate(messages: list[dict], model, processor) -> str:
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
        out_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    generated = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def token_acc_cxr(entries: list[dict], model, processor) -> float:
    correct = 0
    for i, entry in enumerate(entries):
        messages = entry["messages"][:-1]
        gt = entry["messages"][-1]["content"].strip().lower()
        pred = _generate(messages, model, processor).lower()
        if pred.startswith(gt[:3]):
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  CXR {i+1}/{len(entries)}", flush=True)
    return correct / len(entries)


ECG_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]

def token_acc_ecg(entries: list[dict], model, processor) -> float:
    correct = 0
    for i, entry in enumerate(entries):
        messages = entry["messages"][:-1]
        gt = entry["messages"][-1]["content"].strip().upper()
        pred = _generate(messages, model, processor).upper()
        if gt in pred:
            correct += 1
        if (i + 1) % 10 == 0:
            print(f"  ECG {i+1}/{len(entries)}", flush=True)
    return correct / len(entries)


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(metrics: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    cxr = metrics["cxr"]
    ecg = metrics["ecg"]

    def pct(v) -> str:
        return f"{v:.1%}" if isinstance(v, float) else "—"

    lines = [
        "# Evaluation Metrics",
        "",
        "Val set: CheXpert Plus (CXR, 42 examples) · PTB-XL (ECG, 25 examples).  ",
        "Zeroshot = base Qwen3-VL-2B-Instruct, no adapter.  ",
        "Pretrain/SFT = best checkpoint from training logs (same val set).",
        "",
        "## CXR — Binary Label Classification (token accuracy)",
        "",
        "| Stage | Token Acc | Eval Loss |",
        "|-------|-----------|-----------|",
        f"| Zeroshot   | {pct(cxr['zeroshot']['eval_token_acc'])} | — |",
        f"| Pretrain   | {pct(cxr['pretrain']['eval_token_acc'])} | {cxr['pretrain']['eval_loss']} |",
        f"| SFT (best) | {pct(cxr['sft']['eval_token_acc'])} | {cxr['sft']['eval_loss']} |",
        "",
        "## ECG — Cardiac Diagnosis Classification (token accuracy)",
        "",
        "| Stage | Token Acc | Eval Loss |",
        "|-------|-----------|-----------|",
        f"| Zeroshot   | {pct(ecg['zeroshot']['eval_token_acc'])} | — |",
        f"| Pretrain   | {pct(ecg['pretrain']['eval_token_acc'])} | {ecg['pretrain']['eval_loss']} |",
        f"| SFT (best) | {pct(ecg['sft']['eval_token_acc'])} | {ecg['sft']['eval_loss']} |",
    ]

    md_path = OUT_DIR / "metrics.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {json_path.relative_to(REPO)}, {md_path.relative_to(REPO)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    cxr_val = load_jsonl(CXR_VAL_JSONL)
    ecg_val = load_jsonl(ECG_VAL_JSONL)
    print(f"Val sets: {len(cxr_val)} CXR, {len(ecg_val)} ECG")

    print("\nLoading base model (no adapter) ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.image_processor.max_pixels = MAX_PIXELS
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model.eval()

    print("\nZeroshot CXR ...")
    zs_cxr = token_acc_cxr(cxr_val, model, processor)
    print(f"  → {zs_cxr:.1%}")

    print("\nZeroshot ECG ...")
    zs_ecg = token_acc_ecg(ecg_val, model, processor)
    print(f"  → {zs_ecg:.1%}")

    print("\nReading pretrain/SFT metrics from training logs ...")
    cxr_pretrain = best_eval_from_log(TRAINING_LOGS["cxr_pretrain"])
    ecg_pretrain = best_eval_from_log(TRAINING_LOGS["ecg_pretrain"])
    cxr_sft      = best_eval_from_log(TRAINING_LOGS["cxr_sft"])
    ecg_sft      = best_eval_from_log(TRAINING_LOGS["ecg_sft"])

    metrics = {
        "cxr": {
            "zeroshot": {"eval_token_acc": zs_cxr},
            "pretrain": cxr_pretrain,
            "sft":      cxr_sft,
        },
        "ecg": {
            "zeroshot": {"eval_token_acc": zs_ecg},
            "pretrain": ecg_pretrain,
            "sft":      ecg_sft,
        },
    }

    save_results(metrics)
    print("Done.")


if __name__ == "__main__":
    main()
