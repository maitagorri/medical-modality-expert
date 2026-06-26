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

# ── Smoke test: verify packages and count Qwen3-VL-2B params (CPU-safe) ───────
# Uses PyTorch meta device — loads model structure without downloading weights,
# so this runs in seconds with no RAM cost. The real go/no-go forward-pass test
# is scripts/verify_swift.py, run separately before training.
echo ""
echo "Running smoke test..."
uv run python - <<'EOF'
import sys
import torch
import transformers
import swift

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

print(f"Python:        {sys.version.split()[0]}")
print(f"PyTorch:       {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"ms-swift:      {swift.__version__}")
print()

print(f"Loading {MODEL_ID} config (meta device — no weights downloaded)...")
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}  (~{total_params * 4 / 1e9:.1f} GB fp32)")
print()
print("Smoke test passed. Run scripts/verify_swift.py for a full forward-pass check.")
EOF

echo ""
echo "Setup complete."
