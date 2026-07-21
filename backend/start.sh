#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Start the SLD Inference API server.
# Run from the repo root:   bash backend/start.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
fi

echo "Starting SLD Inference API..."
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info
