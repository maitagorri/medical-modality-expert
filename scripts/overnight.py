"""
Overnight training runner.

Chains all 4 training stages with automatic epoch extension if val loss is
still improving at the end of a stage. Designed to be left running in a terminal.

Usage:
  uv run python scripts/overnight.py
  uv run python scripts/overnight.py --pretrain-only   # stop after ECG pretrain

Stages run in order:
  1. CXR pretrain  (waits for already-running job to finish)
  2. ECG pretrain
  3. CXR SFT
  4. ECG SFT
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
CONFIGS = {
    "cxr_pretrain": REPO / "configs/pretrain_cxr.yaml",
    "ecg_pretrain": REPO / "configs/pretrain_ecg.yaml",
    "cxr_sft":      REPO / "configs/sft_cxr.yaml",
    "ecg_sft":      REPO / "configs/sft_ecg.yaml",
}
OUTPUT_DIRS = {
    "cxr_pretrain": REPO / "outputs/cxr_pretrain",
    "ecg_pretrain": REPO / "outputs/ecg_pretrain",
    "cxr_sft":      REPO / "outputs/cxr_sft",
    "ecg_sft":      REPO / "outputs/ecg_sft",
}
# How many extra epochs to add if val loss is still dropping
EXTRA_EPOCHS = 2
# Val loss must have dropped by at least this fraction to be considered "still improving"
MIN_IMPROVEMENT_FRAC = 0.01


def env() -> dict:
    e = os.environ.copy()
    e["USE_MODELSCOPE_HUB"] = "False"
    env_file = REPO / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            e.setdefault(k.strip(), v.strip())
    return e


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def latest_run_dir(output_dir: Path) -> Path | None:
    """Return the most recent v*-<date>-<time> subdirectory, if any."""
    runs = sorted(output_dir.glob("v*-*-*"), key=lambda p: p.name)
    return runs[-1] if runs else None


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the highest-numbered checkpoint-N directory in a run dir."""
    checkpoints = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def read_logging(run_dir: Path) -> list[dict]:
    log_file = run_dir / "logging.jsonl"
    if not log_file.exists():
        return []
    entries = []
    for line in log_file.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def val_losses(entries: list[dict]) -> list[float]:
    return [e["eval_loss"] for e in entries if "eval_loss" in e]


def is_still_improving(entries: list[dict]) -> bool:
    """True if val loss dropped meaningfully from first to last eval checkpoint."""
    vl = val_losses(entries)
    if len(vl) < 2:
        # Only one (or no) eval point — use train loss trend instead
        train = [e["loss"] for e in entries if "loss" in e and "eval_loss" not in e]
        if len(train) < 2:
            return False
        drop = (train[0] - train[-1]) / train[0]
        log(f"  Only {len(vl)} eval point(s). Train loss drop: {drop:.1%}")
        return drop > MIN_IMPROVEMENT_FRAC
    drop = (vl[0] - vl[-1]) / vl[0]
    log(f"  Val loss: {vl[0]:.4f} → {vl[-1]:.4f}  ({drop:+.1%})")
    return drop > MIN_IMPROVEMENT_FRAC


def bump_epochs(config_path: Path, extra: int) -> None:
    """Increment num_train_epochs in the YAML config by `extra`."""
    text = config_path.read_text()
    for line in text.splitlines():
        if line.startswith("num_train_epochs:"):
            current = int(line.split(":")[1].strip())
            new_val = current + extra
            text = text.replace(line, f"num_train_epochs: {new_val}", 1)
            config_path.write_text(text)
            log(f"  Bumped num_train_epochs: {current} → {new_val} in {config_path.name}")
            return
    log(f"  [WARN] num_train_epochs not found in {config_path.name}")


def is_complete(run_dir: Path) -> bool:
    """Heuristic: training finished if trainer_state.json exists and contains train_runtime."""
    state = run_dir / "trainer_state.json"
    if not state.exists():
        # Check logging.jsonl for a 100%-complete marker.
        # The final JSONL entry is a metadata summary with no step info;
        # scan backwards for the last entry that has global_step/max_steps.
        entries = read_logging(run_dir)
        for entry in reversed(entries):
            step_str = entry.get("global_step/max_steps", "")
            if "/" in step_str:
                cur, total = step_str.split("/")
                return cur.strip() == total.strip()
        return False
    try:
        data = json.loads(state.read_text())
        return "train_runtime" in data
    except Exception:
        return False


def swift_run(config: Path, checkpoint: Path | None = None) -> int:
    """Run swift sft, temporarily injecting resume_from_checkpoint into the YAML if given."""
    original = config.read_text()
    try:
        if checkpoint:
            config.write_text(original + f"\nresume_from_checkpoint: {checkpoint}\n")
            log(f"  Resuming from {checkpoint.parent.name}/{checkpoint.name}")
        r = subprocess.run(["swift", "sft", str(config)], env=env(), cwd=str(REPO))
        return r.returncode
    finally:
        config.write_text(original)


def run_stage(stage: str, resume_from: Path | None = None) -> None:
    config = CONFIGS[stage]
    output_dir = OUTPUT_DIRS[stage]
    checkpoint = resume_from

    log(f"{'='*60}")
    log(f"Starting {stage}")
    log(f"{'='*60}")

    while True:
        returncode = swift_run(config, checkpoint)

        run_dir = latest_run_dir(output_dir)
        if run_dir is None:
            log(f"[ERROR] No output directory found for {stage}. Aborting.")
            sys.exit(1)

        if returncode != 0:
            log(f"[ERROR] {stage} exited with code {returncode}. Check logs in {run_dir}.")
            sys.exit(1)

        entries = read_logging(run_dir)
        log("Checking whether val loss is still improving ...")

        if is_still_improving(entries):
            checkpoint = latest_checkpoint(run_dir)
            log(f"Still improving — extending training by {EXTRA_EPOCHS} epochs.")
            if checkpoint:
                log(f"  Next run will resume from {run_dir.name}/{checkpoint.name}.")
            else:
                log("  [WARN] No checkpoint found — next run will start from scratch.")
            bump_epochs(config, EXTRA_EPOCHS)
        else:
            log(f"Val loss plateaued — {stage} done.")
            break


def wait_for_running_cxr_pretrain(poll_sec: int = 60) -> None:
    """Wait for the already-running CXR pretrain to finish before taking over."""
    output_dir = OUTPUT_DIRS["cxr_pretrain"]
    run_dir = latest_run_dir(output_dir)
    if run_dir is None:
        log("No CXR pretrain run dir found — will run from scratch.")
        return
    if is_complete(run_dir):
        log(f"CXR pretrain already complete ({run_dir.name}).")
        return
    log(f"CXR pretrain in progress ({run_dir.name}) — waiting ...")
    while not is_complete(run_dir):
        entries = read_logging(run_dir)
        if entries:
            last = entries[-1]
            loss_val = last.get("loss", last.get("eval_loss", "?"))
            try:
                loss_str = f"{float(loss_val):.4f}"
            except (TypeError, ValueError):
                loss_str = str(loss_val)
            log(f"  step {last.get('global_step/max_steps','?')} | loss {loss_str}")
        time.sleep(poll_sec)
    log("CXR pretrain finished.")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-only", action="store_true",
                    help="Stop after ECG pretrain (skip SFT stages)")
    args = ap.parse_args()

    stages = ("ecg_pretrain",) if args.pretrain_only else ("ecg_pretrain", "cxr_sft", "ecg_sft")
    log("Overnight runner started.")
    log(f"Stages: cxr_pretrain (wait) → {' → '.join(stages)}")
    print()

    # Stage 1: wait for already-running CXR pretrain, then maybe extend
    wait_for_running_cxr_pretrain()
    run_dir = latest_run_dir(OUTPUT_DIRS["cxr_pretrain"])
    entries = read_logging(run_dir)
    log("Checking CXR pretrain val loss ...")
    if is_still_improving(entries):
        ckpt = latest_checkpoint(run_dir)
        log(f"Still improving — extending CXR pretrain by {EXTRA_EPOCHS} epochs.")
        bump_epochs(CONFIGS["cxr_pretrain"], EXTRA_EPOCHS)
        run_stage("cxr_pretrain", resume_from=ckpt)
    else:
        log("CXR pretrain converged. Moving on.")

    for stage in stages:
        run_stage(stage)

    log("="*60)
    log("All stages complete. Training pipeline finished.")
    log("="*60)


if __name__ == "__main__":
    main()
