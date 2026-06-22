"""Evaluate a fine-tuned checkpoint on a held-out split."""
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/results"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implement evaluation loop and metric computation
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data:       {args.data}")
    print(f"Results →   {args.output_dir}")


if __name__ == "__main__":
    main()
