"""
Join df_chexpert_plus_240401.csv with findings_fixed.json binary labels.

The merged file (df_enriched.csv) becomes the single source of truth for
all downstream scripts: EDA, sampling, and JSONL conversion.

Label values in the source JSON:
  1.0  → positive   → kept as 1
  0.0  → negative   → kept as 0
  -1.0 → uncertain  → set to NaN (excluded from training)
  null → not mentioned → set to NaN

Usage:
  python scripts/join_chexpert_labels.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).parent.parent
CSV_PATH     = REPO_ROOT / "data/raw/chexpert-plus/df_chexpert_plus_240401.csv"
LABELS_PATH  = REPO_ROOT / "data/raw/chexpert-plus/CheXpert_Labels/findings_fixed.json"
OUT_PATH     = REPO_ROOT / "data/processed/chexpert_plus/df_enriched.csv"

CHEXPERT_14_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding",
]


def load_labels(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    labels_df = pd.DataFrame(rows)
    # Recode: 1.0→1, 0.0→0, -1.0→NaN, null→NaN
    for col in CHEXPERT_14_LABELS:
        if col in labels_df.columns:
            labels_df[col] = labels_df[col].replace(-1.0, np.nan)
    return labels_df


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading main CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df):,} rows, splits: {df['split'].value_counts().to_dict()}")

    print("Loading binary labels (findings_fixed.json)...")
    labels_df = load_labels(LABELS_PATH)
    print(f"  {len(labels_df):,} label rows")

    print("Merging on path_to_image...")
    enriched = df.merge(
        labels_df[["path_to_image"] + CHEXPERT_14_LABELS],
        on="path_to_image",
        how="left",
    )
    # A row "matched" if at least one label column is non-null
    any_label = enriched[CHEXPERT_14_LABELS].notna().any(axis=1)
    print(f"  Total rows:              {len(enriched):,}")
    print(f"  Rows with ≥1 label:      {any_label.sum():,}")
    print(f"  Rows with no label:      {(~any_label).sum():,}  (unlabeled in findings_fixed.json)")

    print("\nPositive rate per label (definite positives / rows with label):")
    for col in CHEXPERT_14_LABELS:
        total   = enriched[col].notna().sum()
        pos     = (enriched[col] == 1).sum()
        rate    = pos / total * 100 if total > 0 else 0
        print(f"  {col:<35} {pos:>6,} / {total:>6,}  ({rate:.1f}%)")

    enriched.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH.relative_to(REPO_ROOT)}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
