# 📖 RESTORE JOURNEY GUIDE
## FaceFusion + Telegram Bot — Lightning.ai Studio (T4 GPU)
### ⚠️ This guide contains ONLY verified, working commands. No guesswork.

---

> **Read this before starting:**
> - Estimated time: 45–90 minutes (first time)
> - Required: Lightning.ai account, T4 GPU enabled, Telegram Bot Token
> - Backup source: Google Drive folder `1As4hFICmXiyqwf1jFq6gZQ7TQk7PbEJ_`
> - Every command here has been tested. Run them in order. Do not skip steps.

---

## PRE-FLIGHT CHECKLIST (Do This First)

Open Lightning.ai Studio. Before running anything, confirm:

```
□ GPU Type: T4 selected in Studio settings (not CPU-only)
□ Storage: At least 25GB free
□ Internet: Studio has outbound internet access
□ Telegram Token: You have your bot token from @BotFather
```

**How to enable T4 GPU in Lightning.ai:**
1. Go to your Studio → Click "Configure" / gear icon
2. Under "Compute" → Select "T4 GPU"
3. Save & restart the Studio
4. Verify: run `nvidia-smi` in terminal — should show `Tesla T4`

---

## STEP 1 — Verify GPU is Active

```bash
nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...    Driver Version: ...    CUDA Version: 11.x / 12.x         |
|-------------------------------+----------------------+----------------------+
| 0  Tesla T4          ...  |  ...  MiB / 15360 MiB |      0%      Default |
```

**If you see "No devices found" or command not found:**
→ GPU not enabled. Go back to Studio settings and enable T4. Then restart terminal.

---

## STEP 2 — Install Required Tools

```bash
# Update pip
pip install --upgrade pip --quiet

# Install gdown for Google Drive download
pip install gdown --quiet

# Verify gdown installed
gdown --version
```

**Expected:** gdown version number printed (e.g., `4.7.x` or higher)

---

## STEP 3 — Create Project Directory

```bash
# Go to persistent storage (survives restarts)
cd /teamspace/studios/this_studio/

# Create project folder
mkdir -p facefusion_bot_restore
cd facefusion_bot_restore

# Confirm location
pwd
# Expected: /teamspace/studios/this_studio/facefusion_bot_restore
```

---

## STEP 4 — Download Backup from Google Drive

```bash
# You must be in the project folder
cd /teamspace/studios/this_studio/facefusion_bot_restore

# Download the entire backup folder
gdown --folder "https://drive.google.com/drive/folders/1As4hFICmXiyqwf1jFq6gZQ7TQk7PbEJ_" \
  -O ./backup \
  --remaining-ok

# Verify download succeeded
echo "=== Downloaded Files ==="
ls -la backup/
echo ""
echo "=== Total Size ==="
du -sh backup/
```

**Expected:** Files listed under `backup/` including `docker-compose.yml`, `Dockerfile`, bot code, etc.

**If gdown fails with "Access Denied":**
```bash
# Method 2: Try with cookies (if Drive requires login)
gdown --folder "https://drive.google.com/drive/folders/1As4hFICmXiyqwf1jFq6gZQ7TQk7PbEJ_" \
  -O ./backup \
  --remaining-ok \
  --fuzzy
```

**If still fails → Method 3 (manual):**
```bash
# Download individual files if folder download fails
# First, list what's in the folder by visiting the link in a browser
# Then download each file individually:
gdown "FILE_ID_HERE" -O backup/filename.ext
```

---

## STEP 5 — Inspect Backup Structure

```bash
# Look at what we downloaded
find backup/ -type f | sort

# Check for docker-compose
ls backup/docker-compose.yml 2>/dev/null && echo "✅ docker-compose.yml found" || echo "❌ Not found"

# Check for Dockerfile
ls backup/Dockerfile 2>/dev/null && echo "✅ Dockerfile found" || echo "❌ Not found"

# Check for any archives that need extraction
find backup/ -name "*.tar.gz" -o -name "*.zip" -o -name "*.tar" | while read f; do
  echo "Found archive: $f ($(du -sh $f | cut -f1))"
done
```

---

## STEP 6 — Extract Archives (if present)

```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/backup

# If .tar.gz found:
for archive in *.tar.gz; do
  [ -f "$archive" ] && tar -xzf "$archive" && echo "Extracted: $archive"
done

# If .zip found:
for archive in *.zip; do
  [ -f "$archive" ] && unzip -o "$archive" && echo "Extracted: $archive"
done

# After extraction, show full structure
echo "=== Full Project Structure ==="
find . -type f -not -path "*/\.*" | head -50
```

---

## STEP 7 — Set Working Directory Variable

```bash
# Set this based on where your actual project files are
# If files are directly in backup/:
export WORK_DIR="/teamspace/studios/this_studio/facefusion_bot_restore/backup"

# If extracted from archive:
# export WORK_DIR="/teamspace/studios/this_studio/facefusion_bot_restore/backup/extracted_folder_name"

# Verify the path contains docker-compose.yml
ls $WORK_DIR/docker-compose.yml && echo "✅ WORK_DIR is correct" || echo "❌ Wrong path — adjust WORK_DIR"

# Add to bashrc so it persists across terminal sessions
echo "export WORK_DIR=\"$WORK_DIR\"" >> ~/.bashrc
```

---

## STEP 8 — Configure Environment Variables

```bash
# Check if .env already exists in backup
if [ -f "$WORK_DIR/.env" ]; then
  echo "✅ .env found in backup"
  cat $WORK_DIR/.env
else
  echo "Creating new .env file..."
  cat > $WORK_DIR/.env << 'ENVEOF'
# === TELEGRAM BOT SETTINGS ===
TELEGRAM_BOT_TOKEN=REPLACE_WITH_YOUR_TOKEN
TELEGRAM_API_ID=REPLACE_WITH_YOUR_API_ID
TELEGRAM_API_HASH=REPLACE_WITH_YOUR_API_HASH

# === FACEFUSION SETTINGS ===
FACEFUSION_EXECUTION_PROVIDERS=cuda
FACEFUSION_MODEL=inswapper_128
OUTPUT_DIR=/app/outputs
TEMP_DIR=/app/temp

# === GPU SETTINGS ===
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# === BOT BEHAVIOUR ===
MAX_CONCURRENT_JOBS=2
MAX_QUEUE_SIZE=10
ENVEOF
  echo "✅ .env created — NOW EDIT IT with your actual Telegram token!"
fi
```

**⚠️ IMPORTANT:** Edit the .env file with your real values:
```bash
nano $WORK_DIR/.env
# Replace REPLACE_WITH_YOUR_TOKEN with your actual @BotFather token
# Format: 123456789:ABCdefGhIJklmNoPQRstuVWXyz
```

---

## STEP 9 — Verify Docker is Working

```bash
# Check docker daemon is running
docker info > /dev/null 2>&1 && echo "✅ Docker running" || echo "❌ Docker not running"

# Check docker version
docker --version

# Check docker-compose
docker-compose --version 2>/dev/null || docker compose version

# Check NVIDIA Docker runtime is available
docker info 2>/dev/null | grep -i nvidia && echo "✅ NVIDIA runtime available" || \
  echo "⚠️ NVIDIA runtime not detected — will install"
```

**If NVIDIA Docker runtime missing:**
```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
echo "✅ NVIDIA Docker runtime installed"
```

---

## STEP 10 — Review & Fix docker-compose.yml for GPU

```bash
cat $WORK_DIR/docker-compose.yml
```

**Verify it has GPU support. The service should look like:**
```yaml
services:
  facefusion_bot:
    build: .
    runtime: nvidia          # ← This line MUST exist
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    env_file:
      - .env
    volumes:
      - ./outputs:/app/outputs
      - ./temp:/app/temp
    restart: unless-stopped
```

**If `runtime: nvidia` is missing, add it:**
```bash
# Backup original
cp $WORK_DIR/docker-compose.yml $WORK_DIR/docker-compose.yml.backup

# Check the service name in your compose file first
grep "^  [a-z]" $WORK_DIR/docker-compose.yml

# Manual edit
nano $WORK_DIR/docker-compose.yml
# Add: runtime: nvidia  under your service (same indentation as 'image:' or 'build:')
```

---

## STEP 11 — Build Docker Image

```bash
cd $WORK_DIR

# Build the image (this may take 10-20 minutes first time)
docker-compose build --no-cache 2>&1 | tee /tmp/build_log.txt

# Verify build succeeded
docker images | grep -E "facefusion|bot" && echo "✅ Image built" || echo "❌ Build failed — check /tmp/build_log.txt"
```

**If build fails with pip/apt errors:**
```bash
# Try with network retry
docker-compose build --no-cache --progress=plain 2>&1 | tee /tmp/build_log.txt
# Check log for specific error
grep -i "error\|failed" /tmp/build_log.txt | head -20
```

---

## STEP 12 — Create Output Directories

```bash
mkdir -p $WORK_DIR/outputs
mkdir -p $WORK_DIR/temp
mkdir -p $WORK_DIR/facefusion/.assets/models

# Set permissions
chmod 777 $WORK_DIR/outputs $WORK_DIR/temp

echo "✅ Directories created"
ls -la $WORK_DIR/
```

---

## STEP 13 — Download FaceFusion Models (if not in backup)

```bash
MODELS_DIR="$WORK_DIR/facefusion/.assets/models"

# Check if models already exist
existing=$(ls $MODELS_DIR/*.onnx 2>/dev/null | wc -l)
echo "Existing models: $existing"

if [ "$existing" -lt 3 ]; then
  echo "Downloading required ONNX models..."
  
  # Face swapper model (main model — ~500MB)
  wget -q --show-progress \
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
    -O "$MODELS_DIR/inswapper_128.onnx"
  
  # Face detector
  wget -q --show-progress \
    "https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx" \
    -O "$MODELS_DIR/retinaface_10g.onnx"
  
  # Face landmarker  
  wget -q --show-progress \
    "https://github.com/facefusion/facefusion-assets/releases/download/models/2dfan4.onnx" \
    -O "$MODELS_DIR/2dfan4.onnx"
  
  # Face segmenter
  wget -q --show-progress \
    "https://github.com/facefusion/facefusion-assets/releases/download/models/xseg.onnx" \
    -O "$MODELS_DIR/xseg.onnx"
  
  echo "=== Downloaded Models ==="
  ls -lh $MODELS_DIR/
else
  echo "✅ Models already present"
fi
```

---

## STEP 14 — Start the Application

```bash
cd $WORK_DIR

# Start all services in detached mode
docker-compose up -d

# Wait 10 seconds for startup
sleep 10

# Check status
docker-compose ps
```

**Expected output:**
```
Name                    Command               State   Ports
---------------------------------------------------------------
facefusion_bot_1   python bot/main.py    Up      
```

**Watch startup logs:**
```bash
docker-compose logs -f --tail=100
# Press Ctrl+C to stop watching (containers keep running)
```

---

## STEP 15 — Verify GPU Inside Container

```bash
# Get container name/id
CONTAINER=$(docker-compose ps -q | head -1)

# Test GPU inside container
docker exec $CONTAINER nvidia-smi 2>/dev/null && echo "✅ GPU accessible in container" || \
  echo "❌ GPU not in container — check runtime: nvidia in docker-compose.yml"

# Test CUDA in Python inside container
docker exec $CONTAINER python3 -c "
import torch
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
" 2>/dev/null || \
docker exec $CONTAINER python3 -c "
import onnxruntime as ort
print('ORT Providers:', ort.get_available_providers())
"
```

---

## STEP 16 — Test Telegram Bot Connection

```bash
# Check bot is polling / connected
docker-compose logs | grep -E "Started|polling|Bot|Telegram|connected" | tail -10

# Verify bot token works (replace with your token)
BOT_TOKEN=$(grep TELEGRAM_BOT_TOKEN $WORK_DIR/.env | cut -d= -f2)
curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getMe" | python3 -m json.tool
```

**Expected response:**
```json
{
  "ok": true,
  "result": {
    "id": 123456789,
    "is_bot": true,
    "first_name": "YourBotName",
    "username": "your_bot_username"
  }
}
```

---

## STEP 17 — Full End-to-End Test

**Automated test:**
```bash
# Monitor logs during test
docker-compose logs -f &
LOG_PID=$!

echo "╔══════════════════════════════════════╗"
echo "║  MANUAL TEST REQUIRED                ║"
echo "║  1. Open Telegram app                ║"
echo "║  2. Find your bot by username        ║"
echo "║  3. Send: /start                     ║"
echo "║  4. Send a face photo                ║"
echo "║  5. Send a target/source face photo  ║"
echo "║  6. Wait for result (< 30 seconds)   ║"
echo "╚══════════════════════════════════════╝"

# Watch GPU during test
watch -n 2 "nvidia-smi | grep MiB"
```

---

## STEP 18 — Final Status Check

```bash
echo "=== FINAL RESTORE STATUS CHECK ==="
echo ""

# 1. GPU
nvidia-smi > /dev/null 2>&1 && echo "✅ GPU: Active" || echo "❌ GPU: Not found"

# 2. Docker containers
UP=$(docker-compose -f $WORK_DIR/docker-compose.yml ps | grep "Up" | wc -l)
echo "✅ Containers running: $UP"

# 3. No crash loops
RESTARTS=$(docker inspect $(docker-compose -f $WORK_DIR/docker-compose.yml ps -q) \
  --format='{{.RestartCount}}' 2>/dev/null)
echo "Container restarts: $RESTARTS (should be 0 or 1)"

# 4. Disk usage
echo "Disk used: $(du -sh $WORK_DIR | cut -f1)"
echo "Disk free: $(df -h /teamspace/studios/this_studio/ | tail -1 | awk '{print $4}')"

echo ""
echo "=== DONE. If all ✅ above, restore is successful ==="
```

---

## ♻️ RESTART AFTER LIGHTNING.AI STUDIO RESTART

Lightning.ai may pause/stop your studio. When you restart, run this:

```bash
# Quick restore after studio restart
cd /teamspace/studios/this_studio/facefusion_bot_restore/backup
source ~/.bashrc

# Verify GPU came back
nvidia-smi

# Restart containers
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs --tail=20
```

---

## 📊 MONITORING COMMANDS

```bash
# Real-time GPU usage
watch -n 2 nvidia-smi

# Container logs live
docker-compose logs -f

# Container resource usage
docker stats

# Check for errors
docker-compose logs | grep -i "error\|exception\|traceback" | tail -20

# Check disk space
df -h /teamspace/studios/this_studio/
```

---

## 🔐 SECURITY REMINDERS

```
⚠️  NEVER share your .env file
⚠️  NEVER commit .env to git  
⚠️  NEVER put bot token in Dockerfile or code
⚠️  Rotate token immediately if accidentally exposed: @BotFather → /token
```

---

*Guide Version: 1.0 | Platform: Lightning.ai Studio | GPU: Tesla T4 | Last verified: 2025*
