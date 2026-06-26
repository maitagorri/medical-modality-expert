"""
Stratified sampling for PTB-XL and CheXpert Plus.

CheXpert Plus (reads from df_enriched.csv):
  - Test:  all 234 records from the 'valid' split (maps to PNG_valid)
  - Train: 75 records from 'train' split, stratified by frontal_lateral
  - Val:   25 records from 'train' split, stratified by frontal_lateral
  - Outputs image_list.txt listing path_to_image values needed from PNG_train
  -> data/processed/chexpert_plus/{train,val,test}.csv + image_list.txt

PTB-XL:
  - Train: 75 / Val: 25 / Test: 50, stratified across 5 superclasses (30/class)
  - No patient appears in more than one split
  -> data/processed/ptbxl/{train,val,test}.csv
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT      = Path(__file__).parent.parent
PTBXL_DIR      = REPO_ROOT / "data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ENRICHED_CSV   = REPO_ROOT / "data/processed/chexpert_plus/df_enriched.csv"

OUT_PTBXL      = REPO_ROOT / "data/processed/ptbxl"
OUT_CHEXPERT   = REPO_ROOT / "data/processed/chexpert_plus"

SEED = 42

# PTB-XL split sizes
PTBXL_N_TRAIN = 75
PTBXL_N_VAL   = 25
PTBXL_N_TEST  = 50

# CheXpert Plus split sizes (test is fixed = all 234 valid-split records)
CHEX_N_TRAIN = 75
CHEX_N_VAL   = 25


# ---------------------------------------------------------------------------
# CheXpert Plus
# ---------------------------------------------------------------------------

def sample_chexpert_plus() -> None:
    OUT_CHEXPERT.mkdir(parents=True, exist_ok=True)
    print("Loading CheXpert Plus enriched metadata...")
    df = pd.read_csv(ENRICHED_CSV)
    print(f"  {len(df):,} records, splits: {df['split'].value_counts().to_dict()}")

    # Test: all 234 valid-split records
    test_df = df[df["split"] == "valid"].copy()
    print(f"\nTest set: {len(test_df)} valid-split records")

    # Train+val: sample from train split
    train_pool = df[df["split"] == "train"].copy()
    n_total = CHEX_N_TRAIN + CHEX_N_VAL
    pool = train_pool.sample(n_total, random_state=SEED).reset_index(drop=True)

    strat_col = "frontal_lateral" if "frontal_lateral" in pool.columns else None
    train_df, val_df = train_test_split(
        pool,
        test_size=CHEX_N_VAL,
        stratify=pool[strat_col] if strat_col else None,
        random_state=SEED,
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = OUT_CHEXPERT / f"{name}.csv"
        split.to_csv(out, index=False)
        print(f"\n{name} ({len(split)}) -> {out.relative_to(REPO_ROOT)}")
        if strat_col and strat_col in split.columns:
            print(split[strat_col].value_counts().to_string())

    # image_list.txt: path_to_image values for selective PNG_train download
    train_val_paths = pd.concat([train_df, val_df])["path_to_image"].dropna().tolist()
    image_list_path = OUT_CHEXPERT / "image_list.txt"
    image_list_path.write_text("\n".join(train_val_paths) + "\n")
    print(f"\nimage_list.txt: {len(train_val_paths)} paths -> {image_list_path.relative_to(REPO_ROOT)}")


# ---------------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------------

def load_ptbxl_single_label() -> pd.DataFrame:
    df = pd.read_csv(PTBXL_DIR / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(PTBXL_DIR / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1.0]

    def superclasses(d: dict) -> list[str]:
        return list({scp.loc[k, "diagnostic_class"] for k in d if k in scp.index})

    df["diagnostic_superclass"] = df["scp_codes"].apply(superclasses)
    single = df[df["diagnostic_superclass"].apply(len) == 1].copy()
    single["label"] = single["diagnostic_superclass"].apply(lambda x: x[0])
    return single


def sample_ptbxl() -> None:
    OUT_PTBXL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print("\nLoading PTB-XL metadata...")
    df = load_ptbxl_single_label()
    n_classes = df["label"].nunique()
    n_per_class = (PTBXL_N_TRAIN + PTBXL_N_VAL + PTBXL_N_TEST) // n_classes  # 30
    print(f"  {len(df):,} single-label records, {n_classes} classes, {n_per_class}/class")

    # One record per patient to prevent leakage
    one_per_patient = (
        df.groupby("patient_id")
        .apply(lambda g: g.sample(1, random_state=int(rng.integers(1e6))))
        .reset_index(level="patient_id", drop=False)
        .reset_index(drop=True)
    )

    pool = []
    for cls in sorted(df["label"].unique()):
        subset = one_per_patient[one_per_patient["label"] == cls]
        if len(subset) < n_per_class:
            raise ValueError(f"Class {cls}: need {n_per_class}, have {len(subset)}")
        pool.append(subset.sample(n_per_class, random_state=int(rng.integers(1e6))))
    pool_df = pd.concat(pool).reset_index(drop=True)

    train_df, valtest_df = train_test_split(
        pool_df,
        test_size=PTBXL_N_VAL + PTBXL_N_TEST,
        stratify=pool_df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=PTBXL_N_TEST,
        stratify=valtest_df["label"],
        random_state=SEED,
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = OUT_PTBXL / f"{name}.csv"
        split.to_csv(out, index=False)
        print(f"\n{name} ({len(split)}) -> {out.relative_to(REPO_ROOT)}")
        print(split["label"].value_counts().to_string())

    assert not (set(train_df["patient_id"]) & set(val_df["patient_id"])),  "overlap: train/val"
    assert not (set(train_df["patient_id"]) & set(test_df["patient_id"])), "overlap: train/test"
    assert not (set(val_df["patient_id"])   & set(test_df["patient_id"])), "overlap: val/test"
    print("\nPatient overlap check: PASSED")


# ---------------------------------------------------------------------------

def main() -> None:
    sample_chexpert_plus()
    sample_ptbxl()


if __name__ == "__main__":
    main()
