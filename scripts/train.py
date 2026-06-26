"""Launch ms-swift supervised fine-tuning from a YAML config."""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ms-swift SFT training")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")
    cmd = ["swift", "sft", str(args.config)]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
