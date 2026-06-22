"""Convert raw medical data into model-ready format and write to data/processed/."""
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess raw medical data")
    p.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implement dataset-specific preprocessing
    print(f"Reading from {args.input_dir}")
    print(f"Writing to  {args.output_dir}")


if __name__ == "__main__":
    main()
