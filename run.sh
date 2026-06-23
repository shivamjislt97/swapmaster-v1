#!/bin/bash
# =============================================================================
# SwapMaster V1 - Run Script (Linux/macOS)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  SwapMaster V1 - Starting..."
echo "============================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated"
else
    echo "[WARN] No virtual environment found. Run: ./setup.sh"
    exit 1
fi

# Check .env
if [ ! -f ".env" ]; then
    echo "[ERROR] .env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

# Start the application
echo "[START] Starting SwapMaster..."
python app/startup.py
