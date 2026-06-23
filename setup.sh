#!/bin/bash
# =============================================================================
# SwapMaster V1 - Native Setup Script (Linux/macOS)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  SwapMaster V1 - Native Installation"
echo "============================================"
echo ""

# 1. Check Python
echo "[1/6] Checking Python..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python3 not found. Install Python 3.10+ first."
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  macOS: brew install python@3.12"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1)
echo "  Found: $PYTHON_VERSION"

# 2. Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created: venv/"
else
    echo "  Virtual environment already exists: venv/"
fi

# 3. Activate and install dependencies
echo "[3/6] Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "  Dependencies installed"

# 4. Check system dependencies
echo "[4/6] Checking system dependencies..."

# ffmpeg
if command -v ffmpeg &>/dev/null; then
    echo "  [OK] ffmpeg: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "  [WARN] ffmpeg not found. Install it:"
    echo "    Ubuntu/Debian: sudo apt install ffmpeg"
    echo "    macOS: brew install ffmpeg"
fi

# rclone
if command -v rclone &>/dev/null; then
    echo "  [OK] rclone: $(rclone version 2>&1 | head -1)"
else
    echo "  [WARN] rclone not found. Install it:"
    echo "    curl https://rclone.org/install.sh | sudo bash"
    echo "    Or: sudo apt install rclone"
fi

# nvidia-smi (for GPU)
if command -v nvidia-smi &>/dev/null; then
    echo "  [OK] nvidia-smi found"
else
    echo "  [INFO] nvidia-smi not found (CPU-only mode)"
fi

# 5. Setup directories
echo "[5/6] Setting up directories..."
mkdir -p pipeline/logs
mkdir -p pipeline/workspace/output
mkdir -p pipeline/workspace/temp
mkdir -p pipeline/downloads/video
mkdir -p pipeline/downloads/face
mkdir -p pipeline/dashboard_sessions
mkdir -p persistent/faces
mkdir -p .config/rclone
echo "  Directories created"

# 6. Copy rclone config if exists
echo "[6/6] Checking rclone configuration..."
if [ -f ".config/rclone/rclone.conf" ]; then
    echo "  [OK] rclone.conf found"
else
    echo "  [WARN] rclone.conf not found at .config/rclone/rclone.conf"
    echo "  Run: rclone config"
    echo "  Create a remote named 'gdrive' with Google Drive"
fi

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "To start SwapMaster:"
echo "  ./run.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python app/startup.py"
echo ""
