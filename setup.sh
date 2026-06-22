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

# ── Smoke test: load Qwen3-VL-2B-Instruct in 4-bit and print param count ──────
echo ""
echo "Running smoke test..."
uv run python - <<'EOF'
import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print(f"Loading {MODEL_ID} in 4-bit quantization...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model loaded successfully.")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
EOF

echo ""
echo "Setup complete."
