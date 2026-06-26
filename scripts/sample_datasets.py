"""
Stratified sampling of PTB-XL into train/val/test CSV splits.

Produces 300 train / 50 val / 50 test examples stratified across the 5
diagnostic superclasses (NORM, MI, STTC, CD, HYP). No patient appears
in more than one split.

Output: data/processed/ptbxl/{train,val,test}.csv
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PTBXL_DIR = Path("data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
OUT_DIR = Path("data/processed/ptbxl")
SEED = 42

N_TRAIN = 300
N_VAL = 50
N_TEST = 50
N_TOTAL = N_TRAIN + N_VAL + N_TEST  # 400


def load_single_label(ptbxl_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(ptbxl_dir / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1.0]

    def superclasses(scp_dict: dict) -> list[str]:
        return list({scp.loc[k, "diagnostic_class"] for k in scp_dict if k in scp.index})

    df["diagnostic_superclass"] = df["scp_codes"].apply(superclasses)
    single = df[df["diagnostic_superclass"].apply(len) == 1].copy()
    single["label"] = single["diagnostic_superclass"].apply(lambda x: x[0])
    return single


def sample_stratified(df: pd.DataFrame, n_per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n_per_class records per label, one record per patient."""
    # Keep one record per patient (random) to avoid leakage within a split
    one_per_patient = df.groupby("patient_id").apply(
        lambda g: g.sample(1, random_state=int(rng.integers(1e6)))
    ).reset_index(drop=True)

    classes = one_per_patient["label"].unique()
    samples = []
    for cls in sorted(classes):
        subset = one_per_patient[one_per_patient["label"] == cls]
        if len(subset) < n_per_class:
            raise ValueError(f"Not enough records for class {cls}: have {len(subset)}, need {n_per_class}")
        samples.append(subset.sample(n_per_class, random_state=int(rng.integers(1e6))))
    return pd.concat(samples).reset_index()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("Loading PTB-XL metadata...")
    df = load_single_label(PTBXL_DIR)

    n_classes = df["label"].nunique()
    n_per_class_train = N_TRAIN // n_classes   # 60 per class
    n_per_class_val   = N_VAL  // n_classes    # 10 per class
    n_per_class_test  = N_TEST // n_classes    # 10 per class
    n_per_class_total = n_per_class_train + n_per_class_val + n_per_class_test  # 80 per class

    print(f"Sampling {n_per_class_total} records per class ({n_classes} classes)")

    # One record per patient (pick randomly), then stratify by class
    one_per_patient = (
        df.groupby("patient_id")
        .apply(lambda g: g.sample(1, random_state=int(rng.integers(1e6))))
        .reset_index(level="patient_id", drop=False)  # keep patient_id as column
        .reset_index(drop=True)
    )

    pool = []
    for cls in sorted(df["label"].unique()):
        subset = one_per_patient[one_per_patient["label"] == cls]
        if len(subset) < n_per_class_total:
            raise ValueError(f"Class {cls}: need {n_per_class_total}, have {len(subset)}")
        pool.append(subset.sample(n_per_class_total, random_state=int(rng.integers(1e6))))
    pool_df = pd.concat(pool).reset_index(drop=True)

    # Split pool into train / (val + test) without patient overlap
    # pool_df already has one row per patient, so train_test_split is safe
    train_df, valtest_df = train_test_split(
        pool_df,
        test_size=(n_per_class_val + n_per_class_test) * n_classes,
        stratify=pool_df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=n_per_class_test * n_classes,
        stratify=valtest_df["label"],
        random_state=SEED,
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = OUT_DIR / f"{name}.csv"
        split.to_csv(out, index=False)
        print(f"\n{name} ({len(split)} records) -> {out}")
        print(split["label"].value_counts().to_string())

    # Verify no patient overlap across splits
    train_pts = set(train_df["patient_id"])
    val_pts   = set(val_df["patient_id"])
    test_pts  = set(test_df["patient_id"])
    assert not (train_pts & val_pts),  "Patient overlap: train/val"
    assert not (train_pts & test_pts), "Patient overlap: train/test"
    assert not (val_pts   & test_pts), "Patient overlap: val/test"
    print("\nPatient overlap check: PASSED")


if __name__ == "__main__":
    main()
