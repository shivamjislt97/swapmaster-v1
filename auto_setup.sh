#!/bin/bash
# =============================================================================
# SwapMaster V1 - Complete Auto Setup Script
# Runs on: Ubuntu/Debian Linux (Lightning AI, Google Colab, etc.)
# Sets up: System deps, Python deps, GPU, rclone, models, bot config
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

echo "============================================"
echo "  SwapMaster V1 - Auto Setup"
echo "============================================"
echo ""

# ── 1. System Dependencies ──────────────────────────────────────────────────
info "Installing system dependencies..."
sudo apt-get update -qq 2>/dev/null || true
sudo apt-get install -y -qq \
    curl wget unzip git build-essential \
    python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libsm6 libice6 libx11-6 \
    2>/dev/null || warn "Some apt packages may have failed (non-critical)"
log "System dependencies installed"

# ── 2. Static FFmpeg (with libx264) ─────────────────────────────────────────
info "Installing static FFmpeg with libx264..."
FFMPEG_BIN="$HOME/.local/bin/ffmpeg"
mkdir -p "$HOME/.local/bin"
if [ ! -f "$FFMPEG_BIN" ]; then
    FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    cd /tmp
    curl -sL "$FFMPEG_URL" -o ffmpeg.tar.xz
    tar xf ffmpeg.tar.xz
    STATIC_DIR=$(ls -d ffmpeg-*-static 2>/dev/null | head -1)
    if [ -n "$STATIC_DIR" ]; then
        cp "$STATIC_DIR/ffmpeg" "$HOME/.local/bin/ffmpeg"
        cp "$STATIC_DIR/ffprobe" "$HOME/.local/bin/ffprobe"
        chmod +x "$HOME/.local/bin/ffmpeg" "$HOME/.local/bin/ffprobe"
        log "Static FFmpeg installed to ~/.local/bin/"
    else
        warn "FFmpeg static extraction failed, using system ffmpeg"
    fi
    rm -rf /tmp/ffmpeg* 2>/dev/null
    cd "$SCRIPT_DIR"
else
    log "Static FFmpeg already installed"
fi

# Ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc 2>/dev/null || true
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc 2>/dev/null || true
fi

# ── 3. Verify FFmpeg has libx264 ───────────────────────────────────────────
info "Verifying FFmpeg capabilities..."
if command -v ffmpeg &>/dev/null; then
    if ffmpeg -encoders 2>/dev/null | grep -q libx264; then
        log "FFmpeg has libx264 support"
    else
        warn "FFmpeg missing libx264 - output encoding may use fallback"
    fi
    FFMPEG_VER=$(ffmpeg -version 2>&1 | head -1)
    log "FFmpeg: $FFMPEG_VER"
else
    fail "FFmpeg not found in PATH"
fi

# ── 4. Python Dependencies ─────────────────────────────────────────────────
info "Installing Python dependencies..."
pip install --upgrade pip 2>/dev/null || true

# Core requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt 2>&1 | tail -5
    log "Python dependencies installed from requirements.txt"
else
    # Manual install if no requirements.txt
    pip install \
        python-telegram-bot \
        onnxruntime-gpu \
        insightface \
        opencv-python-headless \
        numpy \
        scipy \
        Pillow \
        aiohttp \
        fastapi uvicorn \
        python-dotenv \
        mega.py \
        2>&1 | tail -5
    log "Python dependencies installed (manual)"
fi

# ── 5. Verify GPU / CUDA ───────────────────────────────────────────────────
info "Checking GPU/CUDA..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    log "GPU: $GPU_NAME ($GPU_MEM)"
else
    warn "nvidia-smi not found - GPU may not be available"
fi

python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)')
else:
    print('CUDA: Not available')
" 2>/dev/null || warn "Could not check CUDA via PyTorch"

# ── 6. Verify ONNX Runtime GPU ─────────────────────────────────────────────
info "Checking ONNX Runtime..."
python3 -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'ONNX providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('ONNX GPU: OK')
else:
    print('ONNX GPU: Not available (CPU only)')
" 2>/dev/null || warn "ONNX Runtime check failed"

# ── 7. Verify FaceFusion Models ────────────────────────────────────────────
info "Checking FaceFusion models..."
MODELS_DIR="$SCRIPT_DIR/app/facefusion/.assets/models"
if [ -d "$MODELS_DIR" ]; then
    MODEL_COUNT=$(ls "$MODELS_DIR"/*.onnx 2>/dev/null | wc -l)
    if [ "$MODEL_COUNT" -ge 20 ]; then
        log "$MODEL_COUNT ONNX models found"
    else
        warn "Only $MODEL_COUNT models found (expected >=20)"
    fi
else
    warn "Models directory not found: $MODELS_DIR"
fi

# ── 8. Verify rclone ───────────────────────────────────────────────────────
info "Checking rclone..."
if command -v rclone &>/dev/null; then
    RCLONE_VER=$(rclone version 2>/dev/null | head -1)
    log "rclone: $RCLONE_VER"
else
    warn "rclone not found - GDrive upload will not work"
    info "Install: curl https://rclone.org/install.sh | sudo bash"
fi

# ── 9. Setup .env ──────────────────────────────────────────────────────────
info "Checking .env configuration..."
if [ -f ".env" ]; then
    # Fix RCLONE_CONF to absolute path if relative
    CURRENT_RCLONE_CONF=$(grep "^RCLONE_CONF=" .env | cut -d= -f2 | tr -d '"' | tr -d "'")
    if [ -n "$CURRENT_RCLONE_CONF" ] && [ ! -f "$CURRENT_RCLONE_CONF" ]; then
        ABS_RCLONE_CONF="$SCRIPT_DIR/.config/rclone/rclone.conf"
        sed -i "s|^RCLONE_CONF=.*|RCLONE_CONF=$ABS_RCLONE_CONF|" .env
        warn "Fixed RCLONE_CONF to absolute path: $ABS_RCLONE_CONF"
    fi
    # Fix GDRIVE_FOLDER to have gdrive: prefix
    CURRENT_FOLDER=$(grep "^GDRIVE_FOLDER=" .env | cut -d= -f2 | tr -d '"' | tr -d "'")
    if [ -n "$CURRENT_FOLDER" ] && [[ ! "$CURRENT_FOLDER" == *":"* ]]; then
        sed -i "s|^GDRIVE_FOLDER=.*|GDRIVE_FOLDER=gdrive:$CURRENT_FOLDER|" .env
        warn "Fixed GDRIVE_FOLDER to include gdrive: prefix"
    fi
    log ".env found and validated"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        warn ".env created from .env.example - please edit with your credentials"
    else
        fail ".env not found and no .env.example available"
    fi
fi

# ── 10. Setup rclone config ────────────────────────────────────────────────
info "Checking rclone config..."
RCLONE_CONF_DIR="$SCRIPT_DIR/.config/rclone"
RCLONE_CONF_FILE="$RCLONE_CONF_DIR/rclone.conf"
mkdir -p "$RCLONE_CONF_DIR"
if [ ! -f "$RCLONE_CONF_FILE" ]; then
    cat > "$RCLONE_CONF_FILE" << 'RCLONE_EOF'
[gdrive]
type = drive
scope = drive
token = {"access_token":"YOUR_ACCESS_TOKEN","token_type":"Bearer","refresh_token":"YOUR_REFRESH_TOKEN"}
RCLONE_EOF
    warn "rclone.conf created with placeholder tokens - update with real tokens"
else
    log "rclone.conf found"
fi

# ── 11. Create required directories ────────────────────────────────────────
info "Creating directory structure..."
mkdir -p app/pipeline/downloads/face
mkdir -p app/pipeline/downloads/video
mkdir -p app/pipeline/workspace/output
mkdir -p app/pipeline/workspace/temp
mkdir -p app/pipeline/logs
mkdir -p app/persistent/faces
mkdir -p app/persistent/config
mkdir -p app/facefusion/.assets/models
log "Directory structure created"

# ── 12. Fix permissions ────────────────────────────────────────────────────
info "Fixing permissions..."
chmod +x "$HOME/.local/bin/ffmpeg" 2>/dev/null || true
chmod +x "$HOME/.local/bin/ffprobe" 2>/dev/null || true
chmod +x app/startup.py 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
chmod +x run.sh 2>/dev/null || true
log "Permissions fixed"

# ── 13. Create symlinks ────────────────────────────────────────────────────
info "Creating symlinks..."
if [ ! -f "app/.env" ] || [ ! -L "app/.env" ]; then
    ln -sf "$SCRIPT_DIR/.env" "app/.env"
fi
if [ ! -f "app/.config" ] || [ ! -L "app/.config" ]; then
    ln -sf "$SCRIPT_DIR/.config" "app/.config"
fi
log "Symlinks created"

# ── 14. Verify complete pipeline ───────────────────────────────────────────
echo ""
echo "============================================"
echo "  SETUP VERIFICATION"
echo "============================================"
echo ""

PASS=0
FAIL=0

check() {
    if eval "$2" &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗${NC} $1"
        FAIL=$((FAIL + 1))
    fi
}

check "Python3" "command -v python3"
check "FFmpeg" "command -v ffmpeg"
check "FFprobe" "command -v ffprobe"
check "rclone" "command -v rclone"
check "libx264 in ffmpeg" "ffmpeg -encoders 2>/dev/null | grep libx264"
check "GPU detected" "nvidia-smi --query-gpu=name --format=csv,noheader"
check "CUDA in PyTorch" "python3 -c 'import torch; assert torch.cuda.is_available()'"
check "ONNX GPU" "python3 -c 'import onnxruntime as ort; assert \"CUDAExecutionProvider\" in ort.get_available_providers()'"
check ".env exists" "test -f .env"
check "rclone.conf exists" "test -f .config/rclone/rclone.conf"
check "Models directory" "test -d app/facefusion/.assets/models"
check "ONNX models (>=20)" "[ \$(ls app/facefusion/.assets/models/*.onnx 2>/dev/null | wc -l) -ge 20 ]"
check "pipeline/downloads/face" "test -d app/pipeline/downloads/face"
check "pipeline/workspace/output" "test -d app/pipeline/workspace/output"
check "persistent/faces" "test -d app/persistent/faces"
check "startup.py executable" "test -x app/startup.py"
check "app/.env symlink" "test -L app/.env"
check "app/.config symlink" "test -L app/.config"

echo ""
echo "  Results: ${GREEN}$PASS passed${NC}, ${RED}$FAIL failed${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}  SETUP COMPLETE! All checks passed.${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
    echo "  To start the bot:"
    echo "    ./run.sh"
    echo ""
    echo "  Or manually:"
    echo "    PATH=\"\$HOME/.local/bin:\$PATH\" nohup python3 app/startup.py &"
    echo ""
else
    echo -e "${YELLOW}============================================${NC}"
    echo -e "${YELLOW}  SETUP FINISHED with $FAIL warnings.${NC}"
    echo -e "${YELLOW}============================================${NC}"
    echo ""
    echo "  Fix the failed checks above before running the bot."
    echo ""
fi
