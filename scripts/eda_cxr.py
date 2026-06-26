"""
EDA for CheXpert Plus (df_chexpert_plus_240401.csv).

Works with the actual CSV contents — no assumptions about what columns exist.
Images are loaded from PNG_valid/ if present; falls back to text-only output.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHEXPERT_PLUS_CSV = Path("data/raw/chexpert-plus/df_chexpert_plus_240401.csv")
PNG_VALID_DIR = Path("data/raw/chexpert-plus/PNG_valid")
FIGURES_DIR = Path("notebooks/figures")
SEED = 42

CHEXPERT_14_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices",
]

DEMOGRAPHIC_COLS = ["sex", "age", "race", "ethnicity", "insurance_type",
                    "interpreter_needed", "deceased", "recent_bmi"]

SECTION_COLS = ["section_findings", "section_impression", "section_narrative",
                "section_clinical_history", "section_history",
                "section_comparison", "report"]


def sep(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main() -> None:
    print(f"Loading {CHEXPERT_PLUS_CSV} ...")
    df = pd.read_csv(CHEXPERT_PLUS_CSV)

    # 1. Schema
    sep("All columns and dtypes")
    print(f"{'Column':<40} {'dtype':<12} {'non-null':>10} {'null':>8}")
    print("-" * 72)
    for col in df.columns:
        n_null = df[col].isna().sum()
        n_present = len(df) - n_null
        print(f"  {col:<38} {str(df[col].dtype):<12} {n_present:>10,} {n_null:>8,}")

    # 2. Row counts / split
    sep("Row counts")
    print(f"  Total rows:   {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    if "split" in df.columns:
        print("\n  Split distribution:")
        print(df["split"].value_counts().to_string())

    # 3. Binary label check
    sep("14 CheXpert binary label columns")
    present_labels = [c for c in CHEXPERT_14_LABELS if c in df.columns]
    absent_labels  = [c for c in CHEXPERT_14_LABELS if c not in df.columns]
    if absent_labels:
        print(f"\n  [!] {len(absent_labels)} binary label columns are ABSENT from this CSV.")
        print(f"      They must be joined from the separate CheXpert labeler output")
        print(f"      (chexpert_labeler.csv / findings_fixed.json from Redivis)")
        print(f"      via path_to_image as the join key.")
    if present_labels:
        print(f"\n  Present ({len(present_labels)}):")
        for col in present_labels:
            pos = df[col].eq(1).mean()
            flag = "  [< 5% positives]" if pos < 0.05 else ""
            print(f"    {col:<35} pos={pos:.1%}{flag}")

    # 4. Report section analysis
    section_cols_present = [c for c in SECTION_COLS if c in df.columns]
    sep("Report section completeness and text length")
    print(f"\n  {'Section':<35} {'Present':>10}  {'Rate':>7}  {'mean chars':>10}  {'median':>8}")
    print("  " + "-" * 75)
    for col in section_cols_present:
        non_null = df[col].dropna()
        rate = len(non_null) / len(df)
        lengths = non_null.str.len()
        print(f"  {col:<35} {len(non_null):>10,}  {rate:>7.1%}  {lengths.mean():>10.0f}  {lengths.median():>8.0f}")

    # 5. Five random examples per key section
    for col in ["section_findings", "section_impression"]:
        if col not in df.columns:
            continue
        sep(f"5 random non-null examples — {col}")
        sample = df[col].dropna().sample(5, random_state=SEED)
        for i, text in enumerate(sample, 1):
            print(f"\n  --- Example {i} ---")
            snippet = str(text)[:500]
            print("  " + snippet.replace("\n", "\n  "))
            if len(str(text)) > 500:
                print("  [truncated]")

    # 6. Demographics
    demo_cols_present = [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    if demo_cols_present:
        sep("Demographic / metadata distributions")
        for col in demo_cols_present:
            series = df[col].dropna()
            print(f"\n  {col}  (n={len(series):,}, null={df[col].isna().sum():,})")
            if df[col].dtype == object or df[col].nunique() < 20:
                print(series.value_counts().head(10).to_string())
            else:
                desc = series.describe()
                print(f"    mean={desc['mean']:.1f}  median={desc['50%']:.1f}"
                      f"  min={desc['min']:.0f}  max={desc['max']:.0f}")

    # 7. Example images from PNG_valid
    sep("Example images")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not PNG_VALID_DIR.exists():
        print(f"\n  [!] {PNG_VALID_DIR} not found — skipping image figure.")
        print(f"      Download PNG_valid from the CheXpert Plus Redivis page to enable.")
    else:
        valid_images = list(PNG_VALID_DIR.rglob("*.jpg")) + list(PNG_VALID_DIR.rglob("*.png"))
        if not valid_images:
            print(f"\n  [!] No images found in {PNG_VALID_DIR}.")
        else:
            from PIL import Image
            sample_paths = valid_images[:4]
            fig, axes = plt.subplots(1, len(sample_paths), figsize=(5 * len(sample_paths), 5))
            if len(sample_paths) == 1:
                axes = [axes]
            for ax, img_path in zip(axes, sample_paths):
                img = Image.open(img_path).convert("L")
                ax.imshow(np.array(img), cmap="gray")
                ax.axis("off")
                ax.set_title(img_path.name, fontsize=7)
            plt.suptitle("CheXpert Plus — PNG_valid examples", fontsize=10)
            plt.tight_layout()
            out = FIGURES_DIR / "chexpert_examples.png"
            fig.savefig(out, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Saved {out}  ({len(valid_images)} images found in PNG_valid)")

    print("\nEDA complete.")


if __name__ == "__main__":
    main()
