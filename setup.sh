#!/usr/bin/env bash
set -euo pipefail

# ── Install uv if not already present ─────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "uv not found — installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv $(uv --version)"

# ── Sync dependencies ──────────────────────────────────────────────────────────
uv sync

# ── Smoke test ────────────────────────────────────────────────────────────────
echo ""
echo "Running smoke test..."
uv run python - <<'EOF'
import sys
import torch
import transformers
import swift

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
RAM_WARN_GB = 16

print(f"Python:        {sys.version.split()[0]}")
print(f"PyTorch:       {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"ms-swift:      {swift.__version__}")
print()

# RAM check
import psutil
available_gb = psutil.virtual_memory().available / 1e9
total_gb     = psutil.virtual_memory().total / 1e9
print(f"RAM: {available_gb:.1f} GB available / {total_gb:.1f} GB total")
if available_gb < RAM_WARN_GB:
    print(f"  [WARNING] Less than {RAM_WARN_GB} GB available. "
          f"Model requires ~8.5 GB fp32. Close other apps before training.")
print()

# Model structure check (meta device — no download, instant)
print(f"Checking {MODEL_ID} architecture (meta device)...")
from transformers import AutoConfig, AutoModel
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
with torch.device("meta"):
    model = AutoModel.from_config(config, trust_remote_code=True)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,}  (~{total_params * 4 / 1e9:.1f} GB fp32)")
print()
print("Smoke test passed.")
print("Run  python scripts/verify_swift.py  for a full forward-pass check before training.")
EOF

echo ""
echo "Setup complete."
