"""Launch ms-swift supervised fine-tuning from a YAML config."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def load_env() -> dict:
    """Load .env from repo root; returns env dict with HF_TOKEN and USE_MODELSCOPE_HUB set."""
    env = os.environ.copy()
    env["USE_MODELSCOPE_HUB"] = "False"
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env.setdefault(key.strip(), value.strip())
    return env


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
    subprocess.run(cmd, check=True, env=load_env())


if __name__ == "__main__":
    main()
