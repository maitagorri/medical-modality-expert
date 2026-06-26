"""
Convert sampled datasets to ms-swift conversation-format JSONL.

All text fields are hard-truncated at 256 tokens (Qwen3 tokenizer).
Pretrain report text is additionally truncated to 200 words before
tokenisation to stay safely within the limit.

Outputs:
  data/processed/ptbxl/{train,val,test}.jsonl          — ECG classification VQA
  data/processed/chexpert_plus/{train,val,test}_pretrain.jsonl — report generation
  data/processed/chexpert_plus/{train,val,test}_sft.jsonl      — binary-label VQA
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
PTBXL_DIR   = REPO_ROOT / "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CXR_RAW     = REPO_ROOT / "data/raw/chexpert-plus"
CXR_PROC    = REPO_ROOT / "data/processed/chexpert_plus"
PTBXL_PROC  = REPO_ROOT / "data/processed/ptbxl"
PTBXL_IMGS  = PTBXL_PROC / "images"

SAMPLING_RATE    = 100
MAX_TOKENS       = 256
MAX_REPORT_WORDS = 200
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CHEXPERT_14_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]

# Global truncation counter {field_description: n_truncated}
_truncation_stats: dict[str, int] = {}


# ── Tokenizer ──────────────────────────────────────────────────────────────

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct", trust_remote_code=True
        )
    return _tokenizer


def truncate_to_words(text: str, max_words: int = MAX_REPORT_WORDS) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def truncate_to_tokens(text: str, field: str, max_tokens: int = MAX_TOKENS) -> str:
    """Hard-truncate text to max_tokens. Warns and records if truncated."""
    tok = get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    warnings.warn(
        f"[truncation] '{field}' truncated from {len(ids)} to {max_tokens} tokens",
        stacklevel=3,
    )
    _truncation_stats[field] = _truncation_stats.get(field, 0) + 1
    truncated = tok.decode(ids[:max_tokens], skip_special_tokens=True)
    return truncated


# ── ECG utilities ──────────────────────────────────────────────────────────

def ecg_to_image(signal: np.ndarray, out_path: Path) -> None:
    """Render 12-lead ECG as stacked subplot PNG. out_path parent must exist."""
    n_leads = signal.shape[1]
    t = np.arange(signal.shape[0]) / SAMPLING_RATE
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, n_leads * 1.2), sharex=True)
    for ax, name, col in zip(axes, LEAD_NAMES, range(n_leads)):
        ax.plot(t, signal[:, col], linewidth=0.6, color="black")
        ax.set_ylabel(name, fontsize=7, rotation=0, labelpad=22)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[-1].set_xlabel("Time (s)", fontsize=8)
    fig.subplots_adjust(hspace=0.1, left=0.09, right=0.98, top=0.98, bottom=0.05)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_ecg_jsonl_entry(signal: np.ndarray, label: str, img_path: Path) -> dict:
    ecg_to_image(signal, img_path)
    question = (
        "What is the primary cardiac diagnosis for this 12-lead ECG? "
        "Answer with exactly one of: NORM, MI, STTC, CD, HYP."
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path.resolve())},
                    {"type": "text", "text": truncate_to_tokens(question, "ecg_question")},
                ],
            },
            {"role": "assistant", "content": truncate_to_tokens(label, "ecg_label")},
        ]
    }


# ── CXR utilities ──────────────────────────────────────────────────────────

def resolve_cxr_image(path_to_image: str) -> Path:
    """Map path_to_image (e.g. 'train/patient.../view.jpg') to disk path."""
    parts = Path(path_to_image).parts
    split_dir = parts[0]  # "train" or "valid"
    rel = Path(*parts[1:]).with_suffix(".png")
    if split_dir == "valid":
        return (CXR_RAW / "PNG_valid" / rel).resolve()
    return (CXR_RAW / "PNG_train" / rel).resolve()


def make_pretrain_entry(row: pd.Series) -> dict | None:
    """Report-generation entry: image → findings text (200-word + 256-token limit)."""
    # Prefer section_findings; fall back to section_impression
    text = ""
    for col in ("section_findings", "section_impression"):
        candidate = str(row.get(col, "")).strip()
        if candidate and candidate.lower() not in ("nan", "none"):
            text = candidate
            break
    if not text:
        return None

    img = resolve_cxr_image(row["path_to_image"])
    if not img.exists():
        return None

    # 200-word cap first (word-level is easy to reason about)
    text = truncate_to_words(text, MAX_REPORT_WORDS)
    # Then enforce token limit
    text = truncate_to_tokens(text, "pretrain_report")

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img)},
                    {"type": "text", "text": "Write the radiology report findings for this chest X-ray:"},
                ],
            },
            {"role": "assistant", "content": text},
        ]
    }


def make_sft_entries(row: pd.Series, labels_by_path: dict) -> list[dict]:
    """Binary-label VQA entries: one per definite label (skip null / uncertain -1)."""
    label_row = labels_by_path.get(row["path_to_image"])
    if label_row is None:
        return []
    img = resolve_cxr_image(row["path_to_image"])
    if not img.exists():
        return []
    entries = []
    for lbl in CHEXPERT_14_LABELS:
        val = label_row.get(lbl)
        if val == 1.0:
            answer = "Yes"
        elif val == 0.0:
            answer = "No"
        else:
            continue
        question = truncate_to_tokens(
            f"Does this chest X-ray show {lbl}?", f"sft_question_{lbl}"
        )
        entries.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img)},
                        {"type": "text", "text": question},
                    ],
                },
                {"role": "assistant", "content": answer},
            ]
        })
    return entries


# ── Writers ────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  Wrote {len(entries):,} entries -> {path.relative_to(REPO_ROOT)}")


def print_truncation_summary() -> None:
    print("\nTruncation summary:")
    if not _truncation_stats:
        print("  No fields were truncated.")
    else:
        for field, count in sorted(_truncation_stats.items()):
            print(f"  {field}: {count} entries truncated")


# ── PTB-XL pipeline ────────────────────────────────────────────────────────

def process_ptbxl() -> None:
    print("\n=== PTB-XL ===")
    PTBXL_IMGS.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        df = pd.read_csv(PTBXL_PROC / f"{split}.csv")
        entries = []
        for _, row in df.iterrows():
            rec_path = PTBXL_DIR / row["filename_lr"]
            signal, _ = wfdb.rdsamp(str(rec_path))
            img_name = Path(row["filename_lr"]).stem + ".png"
            img_path = PTBXL_IMGS / img_name
            entries.append(make_ecg_jsonl_entry(signal, row["label"], img_path))
            if len(entries) % 50 == 0:
                print(f"  [{split}] {len(entries)}/{len(df)} ECGs converted")

        write_jsonl(PTBXL_PROC / f"{split}.jsonl", entries)

    print("\nSample PTB-XL entries (train):")
    with open(PTBXL_PROC / "train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            entry = json.loads(line)
            entry["messages"][0]["content"][0]["image"] = "<...path...>"
            print(json.dumps(entry, indent=2))


# ── CheXpert Plus pipeline ─────────────────────────────────────────────────

def process_chexpert() -> None:
    print("\n=== CheXpert Plus ===")

    labels_by_path: dict[str, dict] = {}
    label_file = CXR_RAW / "CheXpert_Labels" / "findings_fixed.json"
    with open(label_file) as f:
        for line in f:
            rec = json.loads(line)
            labels_by_path[rec["path_to_image"]] = rec
    print(f"  Loaded {len(labels_by_path):,} label rows")

    for split in ("train", "val", "test"):
        df = pd.read_csv(CXR_PROC / f"{split}.csv")

        pretrain = [e for _, row in df.iterrows() if (e := make_pretrain_entry(row)) is not None]
        write_jsonl(CXR_PROC / f"{split}_pretrain.jsonl", pretrain)

        sft = []
        for _, row in df.iterrows():
            sft.extend(make_sft_entries(row, labels_by_path))
        write_jsonl(CXR_PROC / f"{split}_sft.jsonl", sft)

    print("\nSample CheXpert pretrain entry (train):")
    with open(CXR_PROC / "train_pretrain.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            entry = json.loads(line)
            entry["messages"][0]["content"][0]["image"] = "<...path...>"
            print(json.dumps(entry, indent=2))

    print("\nSample CheXpert SFT entry (train):")
    with open(CXR_PROC / "train_sft.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            entry = json.loads(line)
            entry["messages"][0]["content"][0]["image"] = "<...path...>"
            print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    process_ptbxl()
    process_chexpert()
    print_truncation_summary()
