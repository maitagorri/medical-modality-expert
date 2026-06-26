"""
Evaluate Qwen3-VL-2B-Instruct (± LoRA adapter) on a held-out test JSONL.

Computes accuracy, macro F1, and per-class AUC. Saves results as JSON and
a Markdown table to outputs/results/.

Usage:
  # Zero-shot baseline (no adapter):
  python scripts/evaluate.py --modality cxr \
      --test-jsonl data/processed/chexpert_plus/test_sft.jsonl

  # Fine-tuned:
  python scripts/evaluate.py --modality cxr \
      --test-jsonl data/processed/chexpert_plus/test_sft.jsonl \
      --adapter outputs/cxr_sft

  python scripts/evaluate.py --modality ecg \
      --test-jsonl data/processed/ptbxl/test.jsonl \
      --adapter outputs/ecg_sft
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

REPO_ROOT  = Path(__file__).parent.parent
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ECG_LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
OUT_DIR    = REPO_ROOT / "outputs/results"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Parsing model output ──────────────────────────────────────────────────────

def parse_yesno(text: str) -> int:
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    # Fallback: scan for yes/no anywhere in first 20 chars
    t2 = t[:20]
    if "yes" in t2:
        return 1
    return 0


def parse_ecg_label(text: str) -> str | None:
    text = text.strip().upper()
    for label in ECG_LABELS:
        if label in text:
            return label
    return None


def extract_label_from_question(question: str) -> str:
    m = re.search(r"Does this chest X-ray show (.+?)\?", question)
    return m.group(1) if m else "unknown"


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(entries: list[dict], adapter_path: Path | None) -> list[dict]:
    from swift import InferRequest, RequestConfig, TransformersEngine

    kwargs: dict = dict(model=BASE_MODEL, torch_dtype="float32")
    if adapter_path is not None:
        kwargs["adapters"] = [str(adapter_path)]

    print(f"Loading model{' + adapter' if adapter_path else ' (zero-shot)'} ...")
    engine = TransformersEngine(**kwargs)
    config = RequestConfig(max_tokens=16, temperature=0.0)

    results = []
    for i, entry in enumerate(entries):
        messages   = entry["messages"][:-1]   # drop the assistant turn
        ground_truth = entry["messages"][-1]["content"].strip()

        response  = engine.infer([InferRequest(messages=messages)], request_config=config)[0]
        prediction = response.choices[0].message.content.strip()

        results.append({
            "ground_truth": ground_truth,
            "prediction":   prediction,
            "messages":     messages,
        })

        if (i + 1) % 50 == 0 or (i + 1) == len(entries):
            print(f"  {i+1}/{len(entries)}")

    return results


# ── Metric computation ────────────────────────────────────────────────────────

def compute_cxr_metrics(results: list[dict]) -> dict:
    by_label: dict[str, dict] = {}
    for r in results:
        user_content = r["messages"][0]["content"]
        question = next(c["text"] for c in user_content if c["type"] == "text")
        label = extract_label_from_question(question)
        gt   = 1 if r["ground_truth"].lower().startswith("yes") else 0
        pred = parse_yesno(r["prediction"])
        by_label.setdefault(label, {"gt": [], "pred": []})
        by_label[label]["gt"].append(gt)
        by_label[label]["pred"].append(pred)

    all_gt   = [v for d in by_label.values() for v in d["gt"]]
    all_pred = [v for d in by_label.values() for v in d["pred"]]

    per_class: dict[str, dict] = {}
    for label, d in by_label.items():
        gt_arr, pred_arr = np.array(d["gt"]), np.array(d["pred"])
        entry: dict = {
            "accuracy": float(accuracy_score(gt_arr, pred_arr)),
            "f1":       float(f1_score(gt_arr, pred_arr, zero_division=0)),
            "n":        int(len(gt_arr)),
        }
        if len(set(d["gt"])) > 1:
            entry["auc"] = float(roc_auc_score(gt_arr, pred_arr))
        per_class[label] = entry

    return {
        "overall_accuracy": float(accuracy_score(all_gt, all_pred)),
        "macro_f1":         float(f1_score(all_gt, all_pred, average="macro", zero_division=0)),
        "n_total":          len(all_gt),
        "per_class":        per_class,
    }


def compute_ecg_metrics(results: list[dict]) -> dict:
    gt_list, pred_list = [], []
    for r in results:
        gt_list.append(r["ground_truth"].strip().upper())
        pred = parse_ecg_label(r["prediction"])
        pred_list.append(pred if pred else "UNKNOWN")

    label_idx = {l: i for i, l in enumerate(ECG_LABELS)}
    gt_idx    = [label_idx.get(g, -1) for g in gt_list]
    pred_idx  = [label_idx.get(p, -1) for p in pred_list]
    valid     = [(g, p) for g, p in zip(gt_idx, pred_idx) if g >= 0 and p >= 0]
    gt_v      = [v[0] for v in valid]
    pred_v    = [v[1] for v in valid]

    per_class: dict[str, dict] = {}
    for label in ECG_LABELS:
        idx    = label_idx[label]
        gt_bin = [1 if g == idx else 0 for g in gt_v]
        pr_bin = [1 if p == idx else 0 for p in pred_v]
        entry: dict = {
            "accuracy":   float(accuracy_score(gt_bin, pr_bin)),
            "f1":         float(f1_score(gt_bin, pr_bin, zero_division=0)),
            "n_positive": int(sum(gt_bin)),
        }
        if len(set(gt_bin)) > 1:
            entry["auc"] = float(roc_auc_score(gt_bin, pr_bin))
        per_class[label] = entry

    return {
        "overall_accuracy":      float(accuracy_score(gt_v, pred_v)) if gt_v else 0.0,
        "macro_f1":              float(f1_score(gt_v, pred_v, average="macro",
                                               labels=list(range(len(ECG_LABELS))),
                                               zero_division=0)) if gt_v else 0.0,
        "n_total":               len(gt_list),
        "n_valid_predictions":   len(valid),
        "per_class":             per_class,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(metrics: dict, modality: str, adapter_path: Path | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "zeroshot" if adapter_path is None else "finetuned"

    json_path = OUT_DIR / f"{modality}_{tag}.json"
    json_path.write_text(json.dumps(metrics, indent=2))

    md_lines = [
        f"## {modality.upper()} — {tag}",
        "",
        f"**Overall accuracy:** {metrics['overall_accuracy']:.3f}  ",
        f"**Macro F1:** {metrics['macro_f1']:.3f}",
        f"**N:** {metrics['n_total']}",
        "",
        "| Label | N | Accuracy | F1 | AUC |",
        "|-------|---|----------|----|-----|",
    ]
    for label, m in metrics["per_class"].items():
        auc = f"{m['auc']:.3f}" if "auc" in m else "—"
        n   = m.get("n") or m.get("n_positive", "?")
        md_lines.append(f"| {label} | {n} | {m['accuracy']:.3f} | {m['f1']:.3f} | {auc} |")

    md_path = OUT_DIR / f"{modality}_{tag}.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"\nResults saved:")
    print(f"  {json_path.relative_to(REPO_ROOT)}")
    print(f"  {md_path.relative_to(REPO_ROOT)}")
    print(f"\nOverall accuracy: {metrics['overall_accuracy']:.3f}")
    print(f"Macro F1:         {metrics['macro_f1']:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL (± adapter)")
    parser.add_argument("--modality",   required=True, choices=["cxr", "ecg"])
    parser.add_argument("--test-jsonl", required=True, type=Path)
    parser.add_argument("--adapter",    default=None,  type=Path,
                        help="Path to LoRA adapter dir. Omit for zero-shot baseline.")
    args = parser.parse_args()

    print(f"Modality:   {args.modality}")
    print(f"Test JSONL: {args.test_jsonl}  ({sum(1 for _ in open(args.test_jsonl))} entries)")
    print(f"Adapter:    {args.adapter or 'None (zero-shot)'}")

    entries = load_jsonl(args.test_jsonl)
    results = run_inference(entries, args.adapter)

    metrics = (compute_cxr_metrics if args.modality == "cxr" else compute_ecg_metrics)(results)
    save_results(metrics, args.modality, args.adapter)


if __name__ == "__main__":
    main()
