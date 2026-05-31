# 🚫 PREVENT MISTAKES GUIDE
## FaceFusion + Telegram Bot — Lightning.ai Studio
### All Known Errors, Root Causes & Exact Fixes

---

> This file exists so you NEVER repeat the same mistake twice.
> Every error here has been diagnosed and fixed. Apply the fix, move on.

---

## 🔴 CATEGORY 1: GPU ERRORS

---

### ❌ Error: `nvidia-smi: command not found`
**Symptom:** Terminal says `nvidia-smi: command not found` or `No such file`

**Root Cause:** T4 GPU is NOT enabled in this Lightning.ai Studio session. You're running on CPU.

**Fix:**
```
1. In Lightning.ai dashboard → Your Studio → Settings/Configure
2. Under "Accelerator" or "Compute" → Select "T4 GPU"
3. Click Save / Restart Studio
4. Wait 2-3 minutes for GPU to attach
5. Re-open terminal → run: nvidia-smi
```

**Prevention:** Always check GPU FIRST before doing anything else.

---

### ❌ Error: `CUDA error: no kernel image is available for execution on the device`
**Symptom:** PyTorch or ONNX Runtime crashes with this CUDA error

**Root Cause:** PyTorch/ONNX version compiled for different CUDA version than what T4 has.

**Fix:**
```bash
# Check your CUDA version
nvcc --version
nvidia-smi | grep "CUDA Version"

# Reinstall PyTorch matching your CUDA version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Reinstall ONNX Runtime for GPU
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu
```

**Prevention:** Always match PyTorch/ONNX versions to actual CUDA version.

---

### ❌ Error: `RuntimeError: CUDA out of memory`
**Symptom:** Process crashes mid-inference with OOM error

**Root Cause:** T4 has 16GB VRAM. Running too many concurrent face swaps, or model too large.

**Fix:**
```bash
# Check current GPU memory usage
nvidia-smi

# Inside container, reduce concurrent jobs
# Edit .env:
MAX_CONCURRENT_JOBS=1   # Reduce from 2 to 1
MAX_QUEUE_SIZE=5

# Clear GPU cache
docker-compose restart

# If still OOM, switch to smaller model
# In .env: FACEFUSION_MODEL=inswapper_128  (not the 256 version)
```

**Prevention:** Keep MAX_CONCURRENT_JOBS=1 or 2 max for T4.

---

### ❌ Error: `GPU not accessible inside Docker container`
**Symptom:** `nvidia-smi` works on host, but fails inside container

**Root Cause:** `runtime: nvidia` missing from docker-compose.yml

**Fix:**
```yaml
# In docker-compose.yml, under your service:
services:
  your_service_name:
    runtime: nvidia          # ADD THIS LINE
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

```bash
# After editing:
docker-compose down
docker-compose up -d
docker exec $(docker-compose ps -q | head -1) nvidia-smi
```

**Prevention:** Always include `runtime: nvidia` when GPU is needed in container.

---

## 🔴 CATEGORY 2: DOCKER ERRORS

---

### ❌ Error: `Cannot connect to the Docker daemon`
**Symptom:** `docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock`

**Root Cause:** Docker daemon stopped or not started.

**Fix:**
```bash
# Start Docker daemon
sudo systemctl start docker

# If systemctl not available (Lightning.ai):
sudo service docker start

# Wait 5 seconds, then retry
sleep 5
docker info
```

**Prevention:** Check `docker info` before any docker operation.

---

### ❌ Error: `docker-compose: command not found`
**Symptom:** `bash: docker-compose: command not found`

**Root Cause:** Old `docker-compose` CLI not installed. Modern Docker uses `docker compose` (with space).

**Fix:**
```bash
# Option 1: Use modern syntax (preferred)
docker compose up -d      # instead of docker-compose up -d
docker compose ps         # instead of docker-compose ps
docker compose logs -f    # etc.

# Option 2: Install docker-compose v2 plugin
sudo apt-get install -y docker-compose-plugin

# Option 3: Install standalone (old style)
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

**Prevention:** Use `docker compose` (space) in all new scripts.

---

### ❌ Error: `no space left on device` during Docker build
**Symptom:** Build fails with "no space left on device"

**Root Cause:** Docker image layers, temp files, or output images filled up disk.

**Fix:**
```bash
# Check disk usage
df -h /teamspace/studios/this_studio/

# Remove unused Docker data (safe)
docker system prune -f

# Remove dangling images
docker image prune -f

# Remove old containers
docker container prune -f

# Check Docker specific usage
docker system df

# If still not enough, find big files
du -sh /teamspace/studios/this_studio/*/ | sort -rh | head -10

# Clear old outputs
rm -rf $WORK_DIR/outputs/*
rm -rf $WORK_DIR/temp/*
```

**Prevention:** Add a cleanup cron or auto-delete outputs after sending to Telegram.

---

### ❌ Error: `Dockerfile not found` during build
**Symptom:** `ERROR: Cannot locate specified Dockerfile: Dockerfile`

**Root Cause:** Running `docker-compose build` from wrong directory.

**Fix:**
```bash
# ALWAYS cd to directory containing docker-compose.yml FIRST
cd $WORK_DIR
ls Dockerfile docker-compose.yml   # Both must be visible here

# Then build
docker-compose build
```

**Prevention:** Always `cd $WORK_DIR` before any docker-compose command.

---

### ❌ Error: Container keeps restarting (restart loop)
**Symptom:** `docker-compose ps` shows container in restart loop

**Root Cause:** Application crashes on startup — usually bad .env, missing token, or missing model file.

**Fix:**
```bash
# Read last 50 lines of logs to find the crash reason
docker-compose logs --tail=50

# Common crash reasons:
# 1. "Invalid token" → Fix .env TELEGRAM_BOT_TOKEN
# 2. "No such file: model.onnx" → Download models (Step 13)
# 3. "Address already in use" → Port conflict → change port in docker-compose.yml
# 4. "Permission denied" → chmod 777 outputs/ temp/

# After fixing, restart:
docker-compose down
docker-compose up -d
```

**Prevention:** Verify .env and model files exist before first launch.

---

## 🔴 CATEGORY 3: GOOGLE DRIVE DOWNLOAD ERRORS

---

### ❌ Error: `Access denied` when downloading from Google Drive
**Symptom:** gdown fails with "Permission denied" or "Access denied"

**Root Cause:** Google Drive folder is not set to "Anyone with link can view."

**Fix:**
```
1. Ask the folder owner to set sharing to "Anyone with the link"
2. OR: Download manually from browser on your local PC
3. Then upload to Lightning.ai via:
   - Drag & drop to Lightning.ai file manager
   - OR use scp/rsync from local machine
   - OR upload to another GDrive folder with proper permissions
```

**Alternative download method:**
```bash
# Try with --fuzzy flag
gdown --fuzzy "https://drive.google.com/drive/folders/1As4hFICmXiyqwf1jFq6gZQ7TQk7PbEJ_" \
  -O ./backup --remaining-ok

# Try direct file ID approach
gdown "1As4hFICmXiyqwf1jFq6gZQ7TQk7PbEJ_" --folder -O ./backup
```

---

### ❌ Error: `gdown` downloads incomplete / partial files
**Symptom:** Files download but are corrupted or 0 bytes

**Root Cause:** Google Drive quota exceeded or large file timeout.

**Fix:**
```bash
# Check file sizes
ls -lh backup/

# Re-download only missing/empty files
# First, list what you have
find backup/ -size 0 -delete  # Remove 0-byte files

# Try again with --remaining-ok (skips already downloaded)
gdown --folder "https://drive.google.com/drive/folders/1As4hFICmXiyqwf1jFq6gZQ7TQk7PbEJ_" \
  -O ./backup --remaining-ok

# For large archives (>100MB), use wget with the direct download URL
wget --no-check-certificate \
  "https://drive.google.com/uc?export=download&id=FILE_ID" \
  -O backup/filename.tar.gz
```

---

## 🔴 CATEGORY 4: TELEGRAM BOT ERRORS

---

### ❌ Error: `Unauthorized` from Telegram API
**Symptom:** `{"ok":false,"error_code":401,"description":"Unauthorized"}`

**Root Cause:** Bot token is wrong, expired, or has spaces/extra characters.

**Fix:**
```bash
# Check token in .env
grep TELEGRAM_BOT_TOKEN $WORK_DIR/.env

# Token format must be: 1234567890:ABCDefghIJKlmnoPQRstuVWXyz
# NO spaces, NO quotes around it, NO extra characters

# Test the token directly:
TOKEN="your_token_here"
curl -s "https://api.telegram.org/bot${TOKEN}/getMe"

# If token is wrong, get new one from @BotFather:
# /mybots → select bot → API Token
```

**Prevention:** Copy-paste token directly from BotFather, never type manually.

---

### ❌ Error: Bot not responding to messages
**Symptom:** You sent message to bot, no reply, no error in logs either

**Root Cause:** Bot might be using webhooks (not polling), or polling isn't started.

**Fix:**
```bash
# Check if bot is polling or using webhooks
docker-compose logs | grep -E "polling|webhook|getUpdates"

# If webhook is set (common conflict):
# Clear the webhook first
TOKEN=$(grep TELEGRAM_BOT_TOKEN $WORK_DIR/.env | cut -d= -f2)
curl -s "https://api.telegram.org/bot${TOKEN}/deleteWebhook"
# Response: {"ok":true,"result":true,"description":"Webhook was deleted"}

# Then restart bot (should use polling now)
docker-compose restart
```

---

### ❌ Error: `Conflict: terminated by other getUpdates request`
**Symptom:** Bot log shows this conflict error, stops working

**Root Cause:** Two instances of the bot are running (one on local machine, one in Docker).

**Fix:**
```bash
# Stop any local bot instances first
pkill -f "python.*bot" 2>/dev/null

# Restart Docker bot
docker-compose restart

# Check only one instance is running
ps aux | grep -E "python.*bot|telegram"
docker-compose ps
```

**Prevention:** Never run bot locally AND in Docker at the same time.

---

## 🔴 CATEGORY 5: FACEFUSION ERRORS

---

### ❌ Error: FaceFusion model file not found
**Symptom:** `FileNotFoundError: .assets/models/inswapper_128.onnx`

**Root Cause:** Models not included in backup, or wrong path.

**Fix:**
```bash
# Locate where FaceFusion expects models
find $WORK_DIR -name "*.onnx" 2>/dev/null
docker exec $(docker-compose ps -q | head -1) find / -name "*.onnx" 2>/dev/null | head -10

# Download to correct location
MODEL_DIR="$WORK_DIR/facefusion/.assets/models"
mkdir -p $MODEL_DIR
wget -q "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
  -O $MODEL_DIR/inswapper_128.onnx

# Or inside container:
docker exec $(docker-compose ps -q | head -1) \
  wget -q "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
  -O /app/facefusion/.assets/models/inswapper_128.onnx
```

---

### ❌ Error: `No face detected in image`
**Symptom:** FaceFusion returns error saying no face found

**Root Cause:** Image is too small, face is not clear, or wrong angle.

**Fix (in bot code):**
```python
# Add validation before processing
MIN_FACE_SIZE = 64  # pixels

# Return helpful error to user
await message.reply("❌ No face detected. Please send a clear, front-facing photo with good lighting.")
```

**This is expected behavior, not a bug.** Test with a clear passport-style photo.

---

### ❌ Error: `onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph`
**Symptom:** ONNX model fails to load

**Root Cause:** Model file is corrupted or wrong version.

**Fix:**
```bash
# Check file size (should be ~500MB for inswapper_128)
ls -lh $WORK_DIR/facefusion/.assets/models/inswapper_128.onnx

# If < 100MB, it's corrupted. Re-download:
rm $WORK_DIR/facefusion/.assets/models/inswapper_128.onnx
wget "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx" \
  -O $WORK_DIR/facefusion/.assets/models/inswapper_128.onnx

# Verify checksum (if provided by facefusion repo)
md5sum $WORK_DIR/facefusion/.assets/models/inswapper_128.onnx
```

---

## 🔴 CATEGORY 6: ENVIRONMENT & PATH ERRORS

---

### ❌ Error: `ModuleNotFoundError` in container
**Symptom:** Python can't find a module inside the Docker container

**Root Cause:** requirements.txt not installed during build, or package version conflict.

**Fix:**
```bash
# Check what's installed inside container
docker exec $(docker-compose ps -q | head -1) pip list | grep -i "torch\|onnx\|telegram\|face"

# Install missing package inside running container (temp fix)
docker exec $(docker-compose ps -q | head -1) pip install missing-package

# Permanent fix: Add to requirements.txt and rebuild
echo "missing-package==x.x.x" >> $WORK_DIR/requirements.txt
docker-compose build --no-cache
docker-compose up -d
```

---

### ❌ Error: `/teamspace` path lost after Lightning.ai restart
**Symptom:** `cd /teamspace/studios/this_studio/facefusion_bot_restore` fails after restart

**Root Cause:** Environment variable `$WORK_DIR` not persisted in new terminal session.

**Fix:**
```bash
# Reload your environment
source ~/.bashrc

# If WORK_DIR still not set:
export WORK_DIR="/teamspace/studios/this_studio/facefusion_bot_restore/backup"

# Verify path exists
ls $WORK_DIR
```

**Prevention:** Always `source ~/.bashrc` first thing after studio restart.

---

### ❌ Error: `Permission denied` on output directory
**Symptom:** Bot can't write processed images to /app/outputs

**Root Cause:** Output directory missing or wrong permissions.

**Fix:**
```bash
# On host:
mkdir -p $WORK_DIR/outputs $WORK_DIR/temp
chmod 777 $WORK_DIR/outputs $WORK_DIR/temp

# Inside container:
docker exec $(docker-compose ps -q | head -1) chmod 777 /app/outputs /app/temp

# Restart
docker-compose restart
```

---

## 🔴 CATEGORY 7: LIGHTNING.AI SPECIFIC ISSUES

---

### ❌ Issue: Studio goes to sleep, Docker containers stop
**Symptom:** After some hours, everything is down, bot unresponsive

**Root Cause:** Lightning.ai auto-pauses idle studios to save compute costs.

**Fix:**
```bash
# After waking up studio, run this single command:
cd /teamspace/studios/this_studio/facefusion_bot_restore/backup && \
  source ~/.bashrc && \
  docker-compose up -d && \
  docker-compose ps
```

**Prevention options:**
```
1. Upgrade to a Lightning.ai plan with "always on" studios
2. Set up a keepalive ping from external service (UptimeRobot, etc.)
3. Use Lightning.ai's "Run" feature (not Studio) for persistent services
```

---

### ❌ Issue: T4 GPU not showing after studio restart
**Symptom:** Studio restarted, nvidia-smi fails

**Root Cause:** Lightning.ai sometimes assigns CPU-only on restart if GPU selection wasn't saved.

**Fix:**
```
1. Go to Lightning.ai dashboard
2. Your Studio → More (⋮) → Configure
3. Confirm T4 GPU is still selected
4. If it reset to CPU: re-select T4 → Save
5. Restart Studio
6. Verify: nvidia-smi
```

---

### ❌ Issue: Docker build takes too long / times out
**Symptom:** Build running for 30+ minutes, seems stuck

**Root Cause:** Downloading large packages (PyTorch ~2GB) on slow connection.

**Fix:**
```bash
# Check if build is actually running (not hung)
docker stats --no-stream

# If stuck, kill and try with build cache
docker-compose down
docker-compose build  # Without --no-cache (uses cache where possible)

# Or build with verbose output to see where it's stuck
docker-compose build --progress=plain 2>&1 | tee /tmp/build.log
tail -f /tmp/build.log
```

---

## ✅ PRE-LAUNCH VERIFICATION CHECKLIST

Run this before every deployment:

```bash
echo "=== PRE-LAUNCH CHECKLIST ==="

# 1. GPU
nvidia-smi > /dev/null 2>&1 && echo "✅ GPU active" || echo "❌ STOP: Enable T4 GPU"

# 2. Docker
docker info > /dev/null 2>&1 && echo "✅ Docker running" || echo "❌ STOP: Start Docker"

# 3. .env file exists and has token
[ -f "$WORK_DIR/.env" ] && echo "✅ .env exists" || echo "❌ Create .env"
grep -q "TELEGRAM_BOT_TOKEN=" $WORK_DIR/.env && \
  ! grep -q "REPLACE_WITH" $WORK_DIR/.env && \
  echo "✅ Token configured" || echo "❌ Set TELEGRAM_BOT_TOKEN in .env"

# 4. Models exist
MODELS=$(find $WORK_DIR -name "*.onnx" 2>/dev/null | wc -l)
[ $MODELS -ge 3 ] && echo "✅ ONNX models present ($MODELS)" || echo "⚠️  Download models first"

# 5. Output dirs exist
[ -d "$WORK_DIR/outputs" ] && [ -d "$WORK_DIR/temp" ] && \
  echo "✅ Output dirs exist" || echo "❌ Create outputs/ and temp/ directories"

# 6. Disk space (need 5GB free minimum)
FREE=$(df /teamspace/studios/this_studio/ | tail-1 | awk '{print $4}')
echo "ℹ️  Free space: $(df -h /teamspace/studios/this_studio/ | tail -1 | awk '{print $4}')"

echo ""
echo "=== If all ✅ above, run: docker-compose up -d ==="
```

---

*This guide is updated with every new error found. If you hit a new error not listed here, document it and add it.*
