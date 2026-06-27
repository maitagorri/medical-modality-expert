"""One-shot: resume CXR pretrain from a checkpoint, then hand off to overnight.py."""
import os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).parent.parent
CONFIG = REPO / "configs/pretrain_cxr.yaml"
CHECKPOINT = "outputs/cxr_pretrain/v5-20260627-093202/checkpoint-38"
RESUME_LINE = f"resume_from_checkpoint: {CHECKPOINT}\n"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def env():
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

original = CONFIG.read_text()
try:
    CONFIG.write_text(original + RESUME_LINE)
    log(f"Added resume_from_checkpoint to config.")
    log("Starting CXR pretrain resume ...")
    r = subprocess.run(["swift", "sft", str(CONFIG)], env=env(), cwd=str(REPO))
    if r.returncode != 0:
        log(f"[ERROR] CXR pretrain exited with code {r.returncode}. Aborting.")
        sys.exit(1)
    log("CXR pretrain done.")
finally:
    CONFIG.write_text(original)
    log("Restored config (resume_from_checkpoint removed).")
