"""
Download CheXpert Plus files from Redivis (Stanford AIMI).

Default run downloads PNG_valid (234 images) + CheXpert Labels (binary label JSONs).
Use --train-list to selectively download only the PNG_train images you need.

Usage:
  python scripts/download_redivis.py
  python scripts/download_redivis.py --train-list data/processed/chexpert_plus/image_list.txt

Requires REDIVIS_API_TOKEN in .env
"""

import argparse
import os
import sys
from pathlib import Path

# Redivis table references (org.dataset:ref:version.table:ref)
TABLE_PNG_VALID    = "aimi.chexpert_plus:5yyj:v1_0.png_valid:41v9"
TABLE_PNG_TRAIN    = "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj"
TABLE_LABELS       = "aimi.chexpert_plus:5yyj:v1_0.chexpert_labels:pmec"
TABLE_CSV          = "aimi.chexpert_plus:5yyj:v1_0.df_chexpert_plus_240401:bavj"

OUT_ROOT           = Path("data/raw/chexpert-plus")
OUT_PNG_VALID      = OUT_ROOT / "PNG_valid"
OUT_PNG_TRAIN      = OUT_ROOT / "PNG_train"
OUT_LABELS         = OUT_ROOT / "CheXpert_Labels"
OUT_CSV            = OUT_ROOT / "df_chexpert_plus_240401.csv"


def load_env() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    if not os.environ.get("REDIVIS_API_TOKEN"):
        print("[!] REDIVIS_API_TOKEN not set. Add it to .env")
        sys.exit(1)


def download_png_valid() -> None:
    existing = list(OUT_PNG_VALID.rglob("*.jpg")) + list(OUT_PNG_VALID.rglob("*.png"))
    if existing:
        print(f"[skip] PNG_valid: {len(existing)} images already in {OUT_PNG_VALID}")
        return

    import redivis
    OUT_PNG_VALID.mkdir(parents=True, exist_ok=True)
    print(f"Downloading PNG_valid (234 images) -> {OUT_PNG_VALID} ...")
    table = redivis.table(TABLE_PNG_VALID)
    table.to_directory().download(str(OUT_PNG_VALID))
    downloaded = list(OUT_PNG_VALID.rglob("*.jpg")) + list(OUT_PNG_VALID.rglob("*.png"))
    print(f"  Done. {len(downloaded)} files.")


def download_labels() -> None:
    label_files = ["findings_fixed.json", "impression_fixed.json", "report_fixed.json"]
    if all((OUT_LABELS / f).exists() for f in label_files):
        print(f"[skip] CheXpert Labels: already in {OUT_LABELS}")
        return

    import redivis
    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CheXpert Labels (3 JSON files) -> {OUT_LABELS} ...")
    table = redivis.table(TABLE_LABELS)
    table.to_directory().download(str(OUT_LABELS))
    print(f"  Done.")


def download_csv() -> None:
    if OUT_CSV.exists():
        print(f"[skip] CSV already exists at {OUT_CSV}")
        return

    import redivis
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading df_chexpert_plus_240401 -> {OUT_CSV} ...")
    redivis.table(TABLE_CSV).download(path=str(OUT_CSV), format="csv", progress=True)
    print(f"  Done. {OUT_CSV.stat().st_size / 1e6:.1f} MB")


def download_train_selective(list_path: Path) -> None:
    """Download only the PNG_train files listed in list_path (one path_to_image per line)."""
    if not list_path.exists():
        print(f"[!] Train list not found: {list_path}")
        sys.exit(1)

    wanted = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    print(f"Selective PNG_train download: {len(wanted)} files from {list_path}")

    import redivis
    OUT_PNG_TRAIN.mkdir(parents=True, exist_ok=True)
    table = redivis.table(TABLE_PNG_TRAIN)

    skipped = 0
    for i, rel_path in enumerate(wanted, 1):
        # path_to_image looks like "train/patient00003/study1/view1_frontal.jpg"
        # Strip leading "train/" since Redivis stores files without that prefix
        file_key = rel_path.removeprefix("train/")
        dest_dir = OUT_PNG_TRAIN / Path(file_key).parent
        dest_file = dest_dir / Path(file_key).name
        if dest_file.exists():
            skipped += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{i}/{len(wanted)}] {file_key}", end=" ... ", flush=True)
        try:
            table.file(file_key).download(str(dest_dir), progress=False)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")

    downloaded = list(OUT_PNG_TRAIN.rglob("*.jpg")) + list(OUT_PNG_TRAIN.rglob("*.png"))
    print(f"\nDone. {len(downloaded)} total files in {OUT_PNG_TRAIN} ({skipped} skipped).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CheXpert Plus from Redivis")
    parser.add_argument(
        "--train-list",
        type=Path,
        default=None,
        metavar="FILE",
        help="Text file of path_to_image values to download from PNG_train",
    )
    args = parser.parse_args()

    load_env()

    if args.train_list:
        download_train_selective(args.train_list)
    else:
        download_csv()
        download_png_valid()
        download_labels()


if __name__ == "__main__":
    main()
