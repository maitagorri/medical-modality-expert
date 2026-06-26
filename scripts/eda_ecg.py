"""
EDA for PTB-XL ECG dataset.

Loads PTB-XL, plots one 12-lead ECG per diagnostic superclass (NORM, MI, STTC),
prints class distribution, and saves plots to notebooks/figures/.

Also validates the ECG-to-image conversion used in the training pipeline.
"""

import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

PTBXL_DIR = Path("data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
FIGURES_DIR = Path("notebooks/figures")
SAMPLING_RATE = 100  # use 100 Hz recordings (records100/)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_metadata(ptbxl_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ptbxl_dir / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp = pd.read_csv(ptbxl_dir / "scp_statements.csv", index_col=0)
    scp = scp[scp["diagnostic"] == 1.0]

    def aggregate_superclass(scp_dict: dict) -> list[str]:
        return list({scp.loc[k, "diagnostic_class"] for k in scp_dict if k in scp.index})

    df["diagnostic_superclass"] = df["scp_codes"].apply(aggregate_superclass)
    return df


def ecg_to_image(signal: np.ndarray, lead_names: list[str], sampling_rate: int, out_path: Path) -> None:
    """Convert a (n_samples, 12) ECG array to a stacked 12-lead PNG."""
    n_leads = signal.shape[1]
    duration_sec = signal.shape[0] / sampling_rate
    t = np.linspace(0, duration_sec, signal.shape[0])

    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, signal[:, i], linewidth=0.6, color="black")
        ax.set_ylabel(lead_names[i], rotation=0, labelpad=28, fontsize=8, va="center")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", linestyle="--", linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("Time (s)", fontsize=9)
    plt.tight_layout(h_pad=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_signal(ecg_id: int, df: pd.DataFrame, ptbxl_dir: Path, sampling_rate: int) -> np.ndarray:
    fname = df.loc[ecg_id, "filename_lr" if sampling_rate == 100 else "filename_hr"]
    record = wfdb.rdsamp(str(ptbxl_dir / fname))
    return record[0]  # (n_samples, 12)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PTB-XL metadata...")
    df = load_metadata(PTBXL_DIR)

    # Class distribution (records with exactly one superclass)
    single_label = df[df["diagnostic_superclass"].apply(len) == 1].copy()
    single_label["label"] = single_label["diagnostic_superclass"].apply(lambda x: x[0])
    print(f"\nTotal records: {len(df)}")
    print(f"Single-label records: {len(single_label)}")
    print("\nClass distribution (single-label records):")
    print(single_label["label"].value_counts().to_string())

    # Plot one example per target superclass
    target_classes = ["NORM", "MI", "STTC"]
    for cls in target_classes:
        subset = single_label[single_label["label"] == cls]
        ecg_id = subset.index[0]
        signal = load_signal(ecg_id, df, PTBXL_DIR, SAMPLING_RATE)
        out_path = FIGURES_DIR / f"ecg_example_{cls.lower()}.png"
        ecg_to_image(signal, LEAD_NAMES, SAMPLING_RATE, out_path)
        print(f"  Saved {out_path}  (ecg_id={ecg_id})")

    print("\nEDA complete. Figures saved to", FIGURES_DIR)


if __name__ == "__main__":
    main()
