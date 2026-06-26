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

# ── Smoke test: full forward pass via verify_swift.py ─────────────────────────
echo ""
echo "Running smoke test (loads model in fp32, runs forward pass, prints peak RAM)..."
uv run python scripts/verify_swift.py

echo ""
echo "Setup complete."
