"""
Evaluation script — assembles the full metrics table for the PoC.

Two metrics are reported, because they differ:
  - Free-gen accuracy: the model generates its answer autoregressively and is
    scored against ground truth. Computed here for the zero-shot base model
    and for the two SFT adapters, all on the val sets. This is the honest
    end-to-end measure.
  - Token accuracy (teacher-forced): the standard ms-swift training metric,
    read from each stage's best checkpoint log. Conditions on gold tokens, so
    it can overstate performance (notably for multi-token answers like ECG).

Single base-model load; SFT adapters are layered on top via PEFT. No OOM risk.

Outputs: outputs/results/metrics.{json,md}

Usage:
  uv run python scripts/eval/evaluate.py
"""

import json
import os
import re
import torch
from pathlib import Path

os.environ["USE_MODELSCOPE_HUB"] = "False"

REPO = Path(__file__).parent.parent.parent
BASE_MODEL = (
    "/home/maita/.cache/huggingface/hub"
    "/models--Qwen--Qwen3-VL-2B-Instruct"
    "/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
)

CXR_VAL_JSONL = REPO / "data/processed/chexpert_plus/val_sft.jsonl"
ECG_VAL_JSONL = REPO / "data/processed/ptbxl/val.jsonl"

CXR_SFT_ADAPTER = REPO / "outputs/cxr_sft/v1-20260628-064230/checkpoint-20"
ECG_SFT_ADAPTER = REPO / "outputs/ecg_sft/v0-20260628-015306/checkpoint-30"

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

def _add_pixel_limits(messages: list[dict]) -> list[dict]:
    """Inject max_pixels into image content dicts so fetch_image resizes correctly."""
    out = []
    for msg in messages:
        new_content = [
            dict(c, max_pixels=MAX_PIXELS) if c.get("type") == "image" else c
            for c in msg["content"]
        ]
        out.append(dict(msg, content=new_content))
    return out


def _generate(messages: list[dict], model, processor) -> str:
    from qwen_vl_utils import process_vision_info
    messages = _add_pixel_limits(messages)
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


def freegen_acc_cxr(entries: list[dict], model, processor) -> float:
    """Free-generation accuracy: generate the Yes/No answer, compare to GT."""
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

def freegen_acc_ecg(entries: list[dict], model, processor) -> float:
    """Free-generation accuracy: generate the class name, compare to GT."""
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
        return f"{v:.1%}" if isinstance(v, (int, float)) and v == v else "—"

    def block(name: str, title: str, m: dict) -> list[str]:
        return [
            f"## {name} — {title}",
            "",
            "| Stage | Token Acc (teacher-forced) | Free-gen Acc | Eval Loss |",
            "|-------|----------------------------|--------------|-----------|",
            f"| Zeroshot   | — | {pct(m['zeroshot'].get('free_gen_acc'))} | — |",
            f"| Pretrain   | {pct(m['pretrain'].get('eval_token_acc'))} | — | {m['pretrain'].get('eval_loss')} |",
            f"| SFT (best) | {pct(m['sft'].get('eval_token_acc'))} | {pct(m['sft'].get('free_gen_acc'))} | {m['sft'].get('eval_loss')} |",
            "",
        ]

    lines = [
        "# Evaluation Metrics",
        "",
        "Val set: CheXpert Plus (CXR, 42 examples) · PTB-XL (ECG, 25 examples).  ",
        "**Free-gen Acc** = answer generated autoregressively, scored vs. ground truth ",
        "(zero-shot base model, no adapter; and the SFT adapters). **Token Acc** = ",
        "teacher-forced metric from the training logs (best checkpoint, same val set).  ",
        "The two agree for CXR (single-token Yes/No) but diverge for ECG, where ",
        "teacher-forcing inflates multi-token class names.",
        "",
        *block("CXR", "Binary Label Classification", cxr),
        *block("ECG", "Cardiac Diagnosis Classification", ecg),
    ]

    md_path = OUT_DIR / "metrics.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved: {json_path.relative_to(REPO)}, {md_path.relative_to(REPO)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from peft import PeftModel

    cxr_val = load_jsonl(CXR_VAL_JSONL)
    ecg_val = load_jsonl(ECG_VAL_JSONL)
    print(f"Val sets: {len(cxr_val)} CXR, {len(ecg_val)} ECG")

    print("\nLoading base model (no adapter) ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    processor.image_processor.max_pixels = MAX_PIXELS
    model = Qwen3VLForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model.eval()

    print("\nZeroshot free-gen CXR ...")
    zs_cxr = freegen_acc_cxr(cxr_val, model, processor)
    print(f"  → {zs_cxr:.1%}")

    print("\nZeroshot free-gen ECG ...")
    zs_ecg = freegen_acc_ecg(ecg_val, model, processor)
    print(f"  → {zs_ecg:.1%}")

    # Layer the SFT adapters on top of the same base model (no second load).
    print("\nLoading SFT adapters ...")
    model = PeftModel.from_pretrained(model, str(CXR_SFT_ADAPTER), adapter_name="cxr")
    model.load_adapter(str(ECG_SFT_ADAPTER), adapter_name="ecg")
    model.eval()

    print("\nSFT free-gen CXR ...")
    model.set_adapter("cxr")
    fg_cxr_sft = freegen_acc_cxr(cxr_val, model, processor)
    print(f"  → {fg_cxr_sft:.1%}")

    print("\nSFT free-gen ECG ...")
    model.set_adapter("ecg")
    fg_ecg_sft = freegen_acc_ecg(ecg_val, model, processor)
    print(f"  → {fg_ecg_sft:.1%}")

    print("\nReading pretrain/SFT token-acc + loss from training logs ...")
    cxr_pretrain = best_eval_from_log(TRAINING_LOGS["cxr_pretrain"])
    ecg_pretrain = best_eval_from_log(TRAINING_LOGS["ecg_pretrain"])
    cxr_sft      = best_eval_from_log(TRAINING_LOGS["cxr_sft"])
    ecg_sft      = best_eval_from_log(TRAINING_LOGS["ecg_sft"])

    metrics = {
        "cxr": {
            "zeroshot": {"free_gen_acc": zs_cxr},
            "pretrain": cxr_pretrain,
            "sft":      {**cxr_sft, "free_gen_acc": fg_cxr_sft},
        },
        "ecg": {
            "zeroshot": {"free_gen_acc": zs_ecg},
            "pretrain": ecg_pretrain,
            "sft":      {**ecg_sft, "free_gen_acc": fg_ecg_sft},
        },
    }

    save_results(metrics)
    print("Done.")


if __name__ == "__main__":
    main()
