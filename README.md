# SwapMaster V1 - Telegram FaceSwap Bot

> AI-powered face swap bot for Telegram. Upload a face + video, get a face-swapped video back. Supports Google Drive and MEGA upload.

## Quick Start (One Command)

```bash
git clone https://github.com/YOUR_USERNAME/swapmaster-v1.git
cd swapmaster-v1
chmod +x auto_setup.sh && ./auto_setup.sh
# Edit .env with your credentials
./run.sh
```

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Manual Setup (Step by Step)](#manual-setup)
3. [Auto Setup Script](#auto-setup-script)
4. [Configuration](#configuration)
5. [GPU Setup & Changes](#gpu-setup--changes)
6. [Path Issues & Fixes](#path-issues--fixes)
7. [Missing Dependencies](#missing-dependencies)
8. [Testing & Verification](#testing--verification)
9. [Telegram Bot Setup](#telegram-bot-setup)
10. [Google Drive Token Setup](#google-drive-token-setup)
11. [MEGA Account Setup](#mega-account-setup)
12. [Active Bot Management](#active-bot-management)
13. [AI Setup Prompt](#ai-setup-prompt)
14. [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04+ / Debian 11+ | Ubuntu 22.04+ |
| RAM | 4 GB | 8 GB+ |
| GPU | None (CPU mode) | NVIDIA Tesla T4 / RTX 3060+ |
| VRAM | 0 (CPU) | 8 GB+ |
| Disk | 20 GB free | 50 GB+ free |
| Python | 3.10+ | 3.12 |
| CUDA | 11.8+ | 12.0+ |

---

## Manual Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/swapmaster-v1.git
cd swapmaster-v1
```

### Step 2: Install System Dependencies
```bash
sudo apt update
sudo apt install -y curl wget git build-essential python3 python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### Step 3: Install Static FFmpeg (with libx264)
```bash
mkdir -p ~/.local/bin
cd /tmp
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*-static/ffmpeg ~/.local/bin/
cp ffmpeg-*-static/ffprobe ~/.local/bin/
chmod +x ~/.local/bin/ffmpeg ~/.local/bin/ffprobe
cd -
```

**Verify:**
```bash
~/.local/bin/ffmpeg -encoders 2>&1 | grep libx264
# Should show: V....D libx264
```

### Step 4: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Install rclone (for Google Drive)
```bash
curl https://rclone.org/install.sh | sudo bash
```

### Step 6: Configure Environment
```bash
cp .env.example .env
nano .env  # Edit with your values
```

**Critical .env values:**
```bash
BOT_TOKEN=your_telegram_bot_token
ALLOWED_USER_ID=your_telegram_user_id
GDRIVE_FOLDER=gdrive:faceswap_output        # MUST have gdrive: prefix
RCLONE_CONF=/full/absolute/path/.config/rclone/rclone.conf  # MUST be absolute
OUTPUT_VIDEO_ENCODER=libx264
EXECUTION_PROVIDER=cuda
```

### Step 7: Setup rclone for Google Drive
```bash
rclone config
# Create remote named "gdrive" with Google Drive
```

Or manually edit `.config/rclone/rclone.conf`:
```ini
[gdrive]
type = drive
scope = drive
token = {"access_token":"...","refresh_token":"..."}
```

### Step 8: Create Directories
```bash
mkdir -p app/pipeline/downloads/{face,video}
mkdir -p app/pipeline/workspace/{output,temp}
mkdir -p app/pipeline/logs
mkdir -p app/persistent/faces
```

### Step 9: Create Symlinks
```bash
ln -sf $(pwd)/.env app/.env
ln -sf $(pwd)/.config app/.config
```

### Step 10: Verify & Run
```bash
chmod +x app/startup.py
PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py &
curl http://localhost:8765/healthz  # Should return: ok
```

---

## Auto Setup Script

```bash
chmod +x auto_setup.sh
./auto_setup.sh
```

The script:
- Installs all system dependencies
- Installs static FFmpeg with libx264
- Installs Python packages
- Checks GPU/CUDA
- Verifies ONNX Runtime
- Validates FaceFusion models
- Fixes .env paths (absolute RCLONE_CONF, gdrive: prefix)
- Creates directory structure
- Creates symlinks
- Runs verification checks

---

## Configuration

### All Settings Reference

See **[SETTINGS_USED.md](SETTINGS_USED.md)** for complete list of all current settings with values.

### Critical Settings That Commonly Cause Issues

| Setting | Must Be | Why |
|---------|---------|-----|
| `RCLONE_CONF` | Absolute path | Relative path breaks rclone |
| `GDRIVE_FOLDER` | `gdrive:folder_name` | Missing `gdrive:` prefix causes local path error |
| `OUTPUT_VIDEO_ENCODER` | `libx264` | Must match FFmpeg build capabilities |
| `EXECUTION_PROVIDER` | `cuda` or `cpu` | Must match available GPU |
| `PATH` | Include `~/.local/bin` | Ensures static FFmpeg is used |

---

## GPU Setup & Changes

### Check Current GPU
```bash
nvidia-smi
```

### Change GPU Provider

In `.env`:
```bash
# For NVIDIA GPU:
EXECUTION_PROVIDER=cuda
GPU_ONLY_MODE=true

# For CPU only:
EXECUTION_PROVIDER=cpu
GPU_ONLY_MODE=false
OUTPUT_VIDEO_ENCODER=libx264
```

### Different GPU Types

| GPU | Encoder | Notes |
|-----|---------|-------|
| NVIDIA (CUDA) | `h264_nvenc` or `libx264` | Best performance |
| Intel QSV | `h264_qsv` | Intel integrated graphics |
| AMD (AMF) | `h264_amf` | AMD GPUs |
| CPU only | `libx264` | Slowest, works everywhere |

### If GPU Changes After Setup
```bash
# Check new GPU
nvidia-smi

# Update .env
sed -i 's/EXECUTION_PROVIDER=.*/EXECUTION_PROVIDER=cuda/' .env

# Restart bot
pkill -f "bot.py"
PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py &
```

---

## Path Issues & Fixes

### Common Path Errors

**Error:** `rclone.conf not found at .config/rclone/rclone.conf`
```bash
# Fix: Use absolute path in .env
sed -i "s|RCLONE_CONF=.*|RCLONE_CONF=$(pwd)/.config/rclone/rclone.conf|" .env
```

**Error:** `Local file system at .../faceswap_output doesn't support public links`
```bash
# Fix: Add gdrive: prefix to GDRIVE_FOLDER
sed -i 's/GDRIVE_FOLDER=.*/GDRIVE_FOLDER=gdrive:faceswap_output/' .env
```

**Error:** `ffmpeg: command not found` or wrong ffmpeg
```bash
# Fix: Ensure ~/.local/bin is in PATH
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

**Error:** `facefusion.py headless-run: error: argument --output-video-encoder: invalid choice: 'libx264'`
```bash
# Fix: Static FFmpeg not in PATH. Ensure ~/.local/bin is BEFORE conda in PATH
export PATH="$HOME/.local/bin:$PATH"
# Or check: which ffmpeg  (should show ~/.local/bin/ffmpeg)
```

### Fix All Paths at Once
```bash
# Run this in the project directory:
ABS_PATH=$(pwd)
sed -i "s|RCLONE_CONF=.*|RCLONE_CONF=$ABS_PATH/.config/rclone/rclone.conf|" .env
grep -q "^GDRIVE_FOLDER=gdrive:" .env || sed -i 's/GDRIVE_FOLDER=.*/GDRIVE_FOLDER=gdrive:faceswap_output/' .env
ln -sf "$ABS_PATH/.env" app/.env
ln -sf "$ABS_PATH/.config" app/.config
```

---

## Missing Dependencies

### Check What's Installed
```bash
# Python packages
pip list | grep -i "onnxruntime\|insightface\|opencv\|telegram\|numpy"

# System tools
which python3 ffmpeg ffprobe rclone nvidia-smi

# FFmpeg codecs
ffmpeg -encoders 2>&1 | grep libx264

# ONNX providers
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Models
ls app/facefusion/.assets/models/*.onnx | wc -l  # Should be >=20
```

### Install Missing Packages
```bash
# Python
pip install python-telegram-bot onnxruntime-gpu insightface opencv-python-headless numpy scipy Pillow aiohttp fastapi uvicorn python-dotenv mega.py

# System
sudo apt install -y curl wget git build-essential python3 python3-pip

# rclone
curl https://rclone.org/install.sh | sudo bash
```

---

## Testing & Verification

### Quick Health Check
```bash
curl http://localhost:8765/healthz
# Expected: ok
```

### Full Verification Script
```bash
python3 -c "
import sys, os
sys.path.insert(0, 'app')

checks = []

# Python
try: import telegram; checks.append(('telegram-bot', 'OK'))
except: checks.append(('telegram-bot', 'MISSING'))

try: import onnxruntime as ort; checks.append(('onnxruntime', 'OK'))
except: checks.append(('onnxruntime', 'MISSING'))

try: import cv2; checks.append(('opencv', 'OK'))
except: checks.append(('opencv', 'MISSING'))

try: import insightface; checks.append(('insightface', 'OK'))
except: checks.append(('insightface', 'MISSING'))

# GPU
import torch
if torch.cuda.is_available():
    checks.append(('CUDA', f'OK ({torch.cuda.get_device_name(0)})'))
else:
    checks.append(('CUDA', 'NOT AVAILABLE'))

# ONNX providers
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in providers:
    checks.append(('ONNX GPU', 'OK'))
else:
    checks.append(('ONNX GPU', 'NOT AVAILABLE'))

# FFmpeg
import subprocess
r = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
if 'libx264' in r.stdout:
    checks.append(('FFmpeg libx264', 'OK'))
else:
    checks.append(('FFmpeg libx264', 'MISSING'))

# Models
models = [f for f in os.listdir('app/facefusion/.assets/models') if f.endswith('.onnx')]
checks.append(('Models', f'{len(models)} found'))

print('=== Verification ===')
for name, status in checks:
    icon = '✓' if 'OK' in status else '✗'
    print(f'  {icon} {name}: {status}')
"
```

### Test Face Swap (Telegram)
1. Send `/start` to your bot
2. Send a face photo
3. Send a video or MEGA link
4. Wait for processing
5. Check if output has GDrive link

---

## Telegram Bot Setup

### Create Bot
1. Open Telegram, search `@BotFather`
2. Send `/newbot`
3. Enter bot name: `SwapMaster Bot`
4. Enter username: `swapmaster_bot`
5. Copy the token: `123456789:ABCdef...`

### Get Your User ID
1. Search `@userinfobot` on Telegram
2. Send any message
3. Copy your user ID

### Update .env
```bash
BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
ALLOWED_USER_ID=123456789
```

---

## Google Drive Token Setup

### Method 1: rclone config (Recommended)
```bash
rclone config
# n -> New remote -> name: gdrive -> Drive -> CLIENT_ID (default) -> SCCESS (default) -> root_folder_id (blank) -> service_account_file (blank) -> y -> y
# Open URL in browser, login, paste code back
```

### Method 2: Manual Token
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create project -> Enable Google Drive API
3. Create OAuth 2.0 credentials
4. Get refresh token using OAuth flow
5. Update `.config/rclone/rclone.conf`:
```ini
[gdrive]
type = drive
scope = drive
token = {"access_token":"ya29...","refresh_token":"1//..."}
```

### Method 3: Update via Bot
Send `/change_drive_token` to your bot and follow instructions.

### Refresh Token Expiry
Tokens expire after ~1 hour. The bot auto-refreshes using refresh_token. If upload fails:
```bash
# Manual refresh test
rclone ls gdrive: --config .config/rclone/rclone.conf
# If "invalid_grant", re-run: rclone config reconnect gdrive:
```

---

## MEGA Account Setup

### Create MEGA Account
1. Go to [mega.nz](https://mega.nz)
2. Create free account (50GB storage)
3. Note email and password

### Update .env
```bash
MEGA_EMAIL=your_email@example.com
MEGA_PASSWORD=your_password
```

### Update Persistent Config (if needed)
```bash
# Edit app/persistent/config.json
# Update mega_email and mega_password fields
```

---

## Active Bot Management

### Find Running Bot Processes
```bash
# Find all bot-related processes
ps aux | grep -E "startup|process_guard|bot\.py" | grep -v grep

# Find by port
lsof -i :8765

# Find by name pattern
pgrep -fa "swapmaster"
```

### Kill Bot
```bash
# Graceful kill
pkill -f "swapmaster.*bot.py"
pkill -f "swapmaster.*process_guard"

# Force kill (if stuck)
pkill -9 -f "swapmaster.*bot.py"
pkill -9 -f "swapmaster.*process_guard"

# Kill all related
pkill -9 -f "swapmaster-v1-native"

# Clean PID files
rm -f app/pipeline/logs/*.pid
```

### Start Bot
```bash
# Method 1: Using run.sh
./run.sh

# Method 2: Direct start
PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py &

# Method 3: With logging
PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py > /tmp/native.log 2>&1 &
```

### Check Bot Status
```bash
# Health check
curl http://localhost:8765/healthz

# Current job status
cat app/pipeline/logs/current_job.json | python3 -m json.tool

# Recent logs
tail -50 app/pipeline/logs/bot_runtime.log

# Process guard logs
tail -20 /tmp/native.log
```

### Restart Bot
```bash
# Full restart
pkill -f "swapmaster.*bot.py"
pkill -f "swapmaster.*process_guard"
sleep 3
rm -f app/pipeline/logs/*.pid
PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py > /tmp/native.log 2>&1 &
echo "Started, waiting 25s..."
sleep 25
curl http://localhost:8765/healthz
```

---

## AI Setup Prompt

> Copy and paste this prompt to any AI assistant (ChatGPT, Claude, Gemini, etc.) for automatic setup:

```
I need you to set up SwapMaster V1, a Telegram FaceSwap bot, on my Linux server.

The project is at: /teamspace/studios/this_studio/swapmaster-v1-native/

Please do the following in order:

1. Clone or navigate to the project directory
2. Run: chmod +x auto_setup.sh && ./auto_setup.sh
3. Edit .env file with these values:
   - BOT_TOKEN: [GET FROM USER]
   - ALLOWED_USER_ID: [GET FROM USER]
   - GDRIVE_FOLDER=gdrive:faceswap_output
   - RCLONE_CONF=/teamspace/studios/this_studio/swapmaster-v1-native/.config/rclone/rclone.conf
   - OUTPUT_VIDEO_ENCODER=libx264
   - EXECUTION_PROVIDER=cuda
4. Setup rclone for Google Drive: rclone config
5. Create symlinks: ln -sf $(pwd)/.env app/.env && ln -sf $(pwd)/.config app/.config
6. Start bot: PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py &
7. Verify: curl http://localhost:8765/healthz (should return "ok")
8. Verify GDrive: rclone ls gdrive: --config .config/rclone/rclone.conf

IMPORTANT PATH NOTES:
- ~/.local/bin MUST be in PATH (before conda) for static ffmpeg
- RCLONE_CONF MUST be absolute path
- GDRIVE_FOLDER MUST have "gdrive:" prefix
- OUTPUT_VIDEO_ENCODER must be "libx264" (not h264_nvenc)

GPU INFO:
- Run nvidia-smi to check GPU
- If GPU changes, update EXECUTION_PROVIDER in .env

If any errors occur, check:
- ffmpeg -encoders | grep libx264
- which ffmpeg (should be ~/.local/bin/ffmpeg)
- python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
- ls app/facefusion/.assets/models/*.onnx | wc -l (should be >=20)
```

---

## Troubleshooting

### Bot Won't Start
```bash
# Check logs
cat /tmp/native.log | tail -30

# Check if port is in use
lsof -i :8765

# Kill stale processes
pkill -9 -f "swapmaster"

# Clean PID files
rm -f app/pipeline/logs/*.pid

# Restart
PATH="$HOME/.local/bin:$PATH" nohup python3 app/startup.py > /tmp/native.log 2>&1 &
```

### Face Swap Fails
```bash
# Check FaceFusion can run
cd app/facefusion
python3 facefusion.py headless-run --help

# Check models exist
ls -la .assets/models/*.onnx | wc -l

# Check GPU memory
nvidia-smi
```

### Upload Fails (GDrive)
```bash
# Test rclone directly
rclone ls gdrive: --config .config/rclone/rclone.conf

# Refresh token
rclone config reconnect gdrive:

# Check token validity
rclone about gdrive: --config .config/rclone/rclone.conf
```

### Upload Fails (MEGA)
```bash
# Check MEGA CLI
megamkdir --version

# Check credentials
cat app/persistent/config.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('mega_email','NOT SET'))"
```

### FFmpeg Encoder Error
```bash
# Error: argument --output-video-encoder: invalid choice: 'libx264'
# Fix: Ensure static ffmpeg is in PATH
which ffmpeg  # Should show ~/.local/bin/ffmpeg
~/.local/bin/ffmpeg -encoders 2>&1 | grep libx264

# If not found, reinstall static ffmpeg (Step 3 in Manual Setup)
```

### Memory Issues
```bash
# Check available memory
free -h

# Check GPU memory
nvidia-smi

# Enable low memory mode in .env
LOW_MEMORY_MODE=true
EXECUTION_THREAD_COUNT=2
VIDEO_MEMORY_STRATEGY=strict
```

---

## File Structure

```
swapmaster-v1-native/
├── .env                          # Environment configuration
├── .env.example                  # Template for .env
├── .gitignore                    # Git ignore rules
├── auto_setup.sh                 # Auto setup script
├── run.sh                        # Start bot (Linux/macOS)
├── run.bat                       # Start bot (Windows)
├── setup.sh                      # Setup script (Linux/macOS)
├── setup.bat                     # Setup script (Windows)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── SETTINGS_USED.md              # All current settings reference
├── MIGRATION_REPORT.md           # Docker→native migration docs
├── .config/
│   └── rclone/
│       └── rclone.conf           # rclone configuration
└── app/
    ├── .env -> ../.env           # Symlink to root .env
    ├── .config -> ../.config     # Symlink to root .config
    ├── bot.py                    # Main Telegram bot
    ├── startup.py                # Native startup script
    ├── config/
    │   └── credentials.py        # Credential loading
    ├── ops/
    │   ├── process_guard.py      # Process management
    │   ├── health_monitor.py     # Health checks
    │   ├── dashboard_server.py   # Web dashboard
    │   ├── safe_cleanup.py       # Cleanup engine
    │   ├── progress_writer.py    # Progress tracking
    │   └── gpu_auto_detect.py    # GPU detection
    ├── scripts/
    │   └── gdrive_upload.py      # Standalone upload
    ├── facefusion/               # FaceFusion engine
    │   ├── facefusion.py
    │   └── .assets/models/       # 25 ONNX models (~3.2GB)
    ├── pipeline/
    │   ├── downloads/
    │   │   ├── face/             # Downloaded face images
    │   │   └── video/            # Downloaded target videos
    │   ├── workspace/
    │   │   ├── output/           # Completed face swaps
    │   │   └── temp/             # Temporary processing files
    │   └── logs/                 # Runtime logs
    └── persistent/
        ├── config.json           # Persistent configuration
        └── faces/                # Saved face images
```

---

## License

Private - Internal use only.
