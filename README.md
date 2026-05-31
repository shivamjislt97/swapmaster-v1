# 🤖 SwapMaster V1 — FaceFusion Telegram Bot

> AI-powered face swap bot running on Lightning.ai Studio (T4 GPU) inside Docker.  
> Users send a face photo + target video/image via Telegram → bot returns the swapped result.

---

## 📋 Table of Contents

1. [Project Overview & Architecture](#architecture)
2. [Quick Start — Restore on Any Machine](#quick-start)
3. [Full Step-by-Step Restore Guide](#full-restore-guide)
4. [Environment Variables Reference](#environment-variables)
5. [Project Logic & Pipeline Flow](#pipeline-logic)
6. [Docker Backup & GDrive Storage](#docker-backup)
7. [Monitoring & Debugging](#monitoring)
8. [FAQ](#faq)
9. [Known Issues & Fixes](#known-issues--fixes)

---

## Architecture

```
User (Telegram)
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Telegram Bot (bot.py)  — Python + aiogram/pyTG     │
│  • Receives face photo + target video/image         │
│  • Manages job queue (one job at a time)            │
│  • Downloads MEGA links via rclone/megadl           │
│  • Uploads results to GDrive via rclone             │
│  • Health endpoint: http://localhost:8765/healthz   │
└──────────────────┬──────────────────────────────────┘
                   │ subprocess call
                   ▼
┌─────────────────────────────────────────────────────┐
│  FaceFusion Pipeline  (facefusion/)                 │
│  • ONNX Runtime + CUDA (T4 GPU)                     │
│  • Models: inswapper_128, retinaface, 2dfan4, xseg  │
│  • Input:  face image + target video/image          │
│  • Output: swapped video/image                      │
└─────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│  Docker Container  (facefusion-v5-pro:latest)       │
│  • runtime: nvidia  (CUDA passthrough)              │
│  • Volumes: outputs/, temp/, logs/, rclone.conf     │
│  • Managed by: docker-compose + process_guard.py    │
└─────────────────────────────────────────────────────┘
                   │
                   ▼
         Lightning.ai Studio (T4 GPU)
         /teamspace/studios/this_studio/
```

### Key Files Inside Container

| Path | Purpose |
|------|---------|
| `/app/bot.py` | Main bot — all Telegram handling, job queue, MEGA download, GDrive upload |
| `/app/facefusion/` | FaceFusion core pipeline |
| `/app/ops/process_guard.py` | Watchdog — auto-restarts bot if it crashes |
| `/app/ops/state_manager.py` | Persists job state across restarts |
| `/app/.env` | All secrets and config (never commit this) |
| `/workspace/pipeline/` | Runtime data: downloads, outputs, temp, logs |

---

## Quick Start

**Prerequisites:** Lightning.ai Studio with T4 GPU, Docker, Telegram bot token.

```bash
# 1. Clone this repo
git clone https://github.com/shivamjislt97/swapmaster-v1.git
cd swapmaster-v1

# 2. Download Docker image from GDrive (see link below)
pip install gdown
gdown "GDRIVE_FILE_ID" -O facefusion_v5_pro_latest.tar.gz

# 3. Load image into Docker
docker load < facefusion_v5_pro_latest.tar.gz

# 4. Configure environment
cp facefusion_bot_restore/run/.env.example facefusion_bot_restore/run/.env
nano facefusion_bot_restore/run/.env   # Add your BOT_TOKEN, MEGA creds, etc.

# 5. Copy your rclone.conf (GDrive auth)
cp /path/to/rclone.conf facefusion_bot_restore/run/rclone.conf

# 6. Start the bot
cd facefusion_bot_restore/run
docker-compose up -d

# 7. Verify
docker ps
docker logs facefusion_bot --tail=30
```

---

## Full Restore Guide

### STEP 0 — Pre-flight Checks

```bash
# GPU active?
nvidia-smi
# Expected: Tesla T4, 15360 MiB VRAM

# Docker running?
docker info | grep "Server Version"

# Disk space? (need 30GB+ free)
df -h /teamspace/studios/this_studio/

# Internet?
curl -s https://api.telegram.org | grep -c "ok"
```

**If GPU not found:** Go to Lightning.ai Studio → Configure → Compute → T4 GPU → Save & Restart.

---

### STEP 1 — Get the Docker Backup

```bash
cd /teamspace/studios/this_studio/
mkdir -p facefusion_bot_restore/backup
cd facefusion_bot_restore/backup

# Download from GDrive (replace FOLDER_ID with actual ID from README or GDrive link)
pip install gdown --quiet
gdown --folder "https://drive.google.com/drive/folders/FOLDER_ID" -O . --remaining-ok

# OR download single tar.gz file directly:
gdown "FILE_ID" -O facefusion_v5_pro_latest.tar.gz

# Verify file is not 0 bytes
ls -lh *.tar.gz
```

> 📌 **Current GDrive backup folder:** See [Docker Backup & GDrive Storage](#docker-backup) section below for the latest shareable link.

---

### STEP 2 — Load Docker Image

```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/backup

# Load the image (takes 5-10 min for 16GB image)
docker load < facefusion_v5_pro_*.tar.gz

# Verify image loaded
docker images | grep facefusion
# Expected: facefusion-v5-pro   latest   <id>   <size>GB
```

---

### STEP 3 — Set Up Run Directory

```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/run

# Create required directories
mkdir -p outputs temp logs

# Verify docker-compose.yml is present
cat docker-compose.yml
```

---

### STEP 4 — Configure .env

```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/run

# Create .env from example
cat > .env << 'EOF'
# === REQUIRED ===
BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN_HERE
ALLOWED_USER_ID=YOUR_TELEGRAM_USER_ID

# === MEGA (for downloading user-sent MEGA links) ===
MEGA_EMAIL=your@email.com
MEGA_PASSWORD=yourpassword

# === GDRIVE UPLOAD (rclone) ===
GDRIVE_ENABLED=true
GDRIVE_REMOTE_NAME=gdrive
GDRIVE_FOLDER=faceswap_output
RCLONE_BIN=/usr/local/bin/rclone
RCLONE_CONF=/app/.config/rclone/rclone.conf

# === GPU / FACEFUSION ===
EXECUTION_PROVIDER=cuda
GPU_ONLY_MODE=true
FACE_SWAPPER_MODEL=inswapper_128
FACE_ENHANCER_MODEL=gfpgan_1.4
FACE_ENHANCER_BLEND=80
ENABLE_FACE_ENHANCER=true
OUTPUT_VIDEO_ENCODER=libx264
EXECUTION_THREAD_COUNT=4

# === BOT BEHAVIOUR ===
AUTO_SLEEP_ENABLED=true
AUTO_SLEEP_MINUTES=30
POST_JOB_AUTO_SLEEP_SECONDS=300
DASHBOARD_ENABLED=true
DASHBOARD_PORT=8765
BYPASS_CONTENT_ANALYSER=false
EOF

echo "✅ .env created — edit with your real values"
nano .env
```

**Get your Telegram User ID:**
```bash
# Message @userinfobot on Telegram — it replies with your numeric ID
# Example: 123456789
```

---

### STEP 5 — Set Up rclone.conf (GDrive Auth)

```bash
# Option A: Copy from existing backup
cp /teamspace/studios/this_studio/facefusion_bot_restore/run/rclone.conf \
   /teamspace/studios/this_studio/facefusion_bot_restore/run/rclone.conf
# (already present if you cloned the run/ directory)

# Option B: Create new rclone config for GDrive
# Run this on a machine with a browser (not headless):
#   rclone config
#   → New remote → name: gdrive → type: drive → follow OAuth flow
# Then copy the generated rclone.conf here

# Verify rclone.conf has gdrive section
grep "\[gdrive\]" facefusion_bot_restore/run/rclone.conf && echo "✅ rclone.conf OK"
```

---

### STEP 6 — Start the Bot

```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/run

# Start
docker-compose up -d

# Watch startup (wait ~90 seconds for health check)
docker-compose logs -f --tail=50
# Press Ctrl+C when you see "Bot started" or "polling"

# Check health
docker ps
# Expected: facefusion_bot   Up X seconds (healthy)
```

---

### STEP 7 — Verify Everything Works

```bash
# 1. Container healthy?
docker ps --filter name=facefusion_bot --format "{{.Names}}\t{{.Status}}"

# 2. GPU accessible inside container?
docker exec facefusion_bot nvidia-smi | grep "Tesla T4"

# 3. Bot token valid?
BOT_TOKEN=$(grep BOT_TOKEN facefusion_bot_restore/run/.env | cut -d= -f2)
curl -s "https://api.telegram.org/bot${BOT_TOKEN}/getMe" | python3 -m json.tool

# 4. Health endpoint responding?
curl -s http://localhost:8765/healthz

# 5. Send /start to your bot on Telegram → should reply
```

---

### STEP 8 — After Lightning.ai Studio Restart

Lightning.ai pauses studios. After any restart:

```bash
# Quick resume (30 seconds)
cd /teamspace/studios/this_studio/facefusion_bot_restore/run
nvidia-smi                    # confirm GPU back
docker-compose up -d          # restart container
docker-compose logs --tail=20 # verify startup
```

---

## Environment Variables

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `BOT_TOKEN` | ✅ | Telegram bot token from @BotFather | `123456:ABCdef...` |
| `ALLOWED_USER_ID` | ✅ | Your Telegram numeric user ID | `123456789` |
| `MEGA_EMAIL` | ✅ | MEGA account email (for downloading MEGA links) | `you@email.com` |
| `MEGA_PASSWORD` | ✅ | MEGA account password | `yourpass` |
| `GDRIVE_ENABLED` | ✅ | Enable GDrive upload of results | `true` |
| `GDRIVE_REMOTE_NAME` | ✅ | rclone remote name | `gdrive` |
| `GDRIVE_FOLDER` | ✅ | GDrive folder name for outputs | `faceswap_output` |
| `RCLONE_CONF` | ✅ | Path to rclone.conf inside container | `/app/.config/rclone/rclone.conf` |
| `EXECUTION_PROVIDER` | ✅ | `cuda` for GPU, `cpu` for CPU-only | `cuda` |
| `FACE_SWAPPER_MODEL` | ✅ | Face swap model name | `inswapper_128` |
| `ENABLE_FACE_ENHANCER` | ⚪ | Enable GFPGAN face enhancement | `true` |
| `FACE_ENHANCER_BLEND` | ⚪ | Enhancement blend % (0-100) | `80` |
| `AUTO_SLEEP_ENABLED` | ⚪ | Auto-sleep after idle period | `true` |
| `AUTO_SLEEP_MINUTES` | ⚪ | Minutes idle before sleep | `30` |
| `DASHBOARD_PORT` | ⚪ | Health/status dashboard port | `8765` |
| `BYPASS_CONTENT_ANALYSER` | ⚪ | Skip NSFW content check | `false` |
| `GPU_ONLY_MODE` | ⚪ | Reject jobs if GPU unavailable | `true` |

---

## Pipeline Logic

### How a Job Flows

```
1. User sends face photo to bot
   └─ Bot saves as "source face" for this user session

2. User sends target (video/image or MEGA link)
   └─ Bot downloads target:
      ├─ Direct Telegram upload → saved to VIDEO_DIR
      ├─ MEGA link → megadl/rclone download → VIDEO_DIR
      └─ HTTP link → wget download → VIDEO_DIR

3. Filename normalization (normalize_processing_target)
   └─ Original filename preserved + datetime suffix appended
      e.g. 3278172064293.mp4 → 3278172064293_20260531_191402.mp4

4. FaceFusion pipeline runs (subprocess)
   └─ Input:  source face + target video/image
   └─ Models: retinaface (detect) → 2dfan4 (landmark) → inswapper_128 (swap)
              → xseg (mask) → gfpgan_1.4 (enhance, optional)
   └─ Output: /workspace/pipeline/workspace/output/result_*.mp4

5. Result uploaded to GDrive via rclone
   └─ Remote: gdrive:faceswap_output/

6. Bot sends result back to user via Telegram
   └─ If file > 50MB: sends GDrive link instead

7. Cleanup: temp files deleted, state reset
```

### MEGA Filename Decoding

MEGA links contain an encrypted filename in the `at` attribute. The bot decodes it using AES-CBC with the file key:

```
MEGA link: https://mega.nz/file/{file_id}#{file_key}
                                    ↓
                         AES-CBC decrypt 'at' attribute
                                    ↓
                         JSON: {"n": "3278172064293.mp4"}
                                    ↓
                         Saved as: 3278172064293.mp4
```

If decoding fails → file saved as `{file_id}.mp4` (e.g. `y9RElKiB.mp4`), then `normalize_processing_target` strips any `mega_` prefix.

### Job Queue

- Only **1 job runs at a time** per bot instance
- Additional requests are queued
- Queue state persisted to `QUEUE_STATE_FILE` (survives restarts)
- Watchdog (`process_guard.py`) monitors pipeline — kills stale jobs after timeout

---

## Docker Backup

### Current Backup Location (GDrive)

| Backup | GDrive Folder | Date | Size |
|--------|--------------|------|------|
| `facefusion_v5_pro_20260531_193402.tar.gz` | `SwapMaster V1 Backup (2026-05-31)` | 2026-05-31 | ~16GB |
| `master_docker_backup_20260527_203904.tar.gz` | `Master Docker Backup` | 2026-05-27 | ~16GB |

> 📌 **GDrive shareable link will be added here once upload completes.**

### How to Create a New Backup

```bash
# 1. Save Docker image to tar.gz
BACKUP="facefusion_v5_pro_$(date +%Y%m%d_%H%M%S).tar.gz"
docker save facefusion-v5-pro:latest | gzip > /teamspace/studios/this_studio/facefusion_bot_restore/backup/$BACKUP
echo "Saved: $BACKUP"

# 2. Upload to GDrive (from inside container)
docker exec facefusion_bot rclone \
  --config /workspace/.config/rclone/rclone.conf \
  copy /path/to/$BACKUP \
  "gdrive:SwapMaster V1 Backup ($(date +%Y-%m-%d))/" \
  --progress

# 3. Get shareable link
docker exec facefusion_bot rclone \
  --config /workspace/.config/rclone/rclone.conf \
  link "gdrive:SwapMaster V1 Backup ($(date +%Y-%m-%d))/$BACKUP"
```

### How to Restore from Backup

```bash
# Download
gdown "GDRIVE_FILE_ID" -O facefusion_v5_pro_latest.tar.gz

# Load
docker load < facefusion_v5_pro_latest.tar.gz

# Verify
docker images | grep facefusion-v5-pro
```

---

## Monitoring

```bash
# Live container logs
docker logs facefusion_bot -f --tail=50

# GPU usage (real-time)
watch -n 2 nvidia-smi

# Container resource stats
docker stats facefusion_bot

# Check for errors
docker logs facefusion_bot 2>&1 | grep -i "error\|exception\|traceback" | tail -20

# Health endpoint
curl -s http://localhost:8765/healthz

# Active job state
docker exec facefusion_bot cat /workspace/pipeline/logs/active_job_state.json 2>/dev/null | python3 -m json.tool

# Queue state
docker exec facefusion_bot cat /workspace/pipeline/logs/queue_state.json 2>/dev/null | python3 -m json.tool

# Disk usage inside container
docker exec facefusion_bot du -sh /workspace/pipeline/workspace/output/ /workspace/pipeline/workspace/temp/
```

---

## FAQ

**Q: Bot is not responding to messages**
```bash
# Check container is running
docker ps --filter name=facefusion_bot
# Check logs for errors
docker logs facefusion_bot --tail=50
# Verify bot token
curl -s "https://api.telegram.org/bot$BOT_TOKEN/getMe"
# Restart
docker restart facefusion_bot
```

**Q: "GPU not available" error**
```bash
# Verify GPU on host
nvidia-smi
# Verify GPU inside container
docker exec facefusion_bot nvidia-smi
# If container can't see GPU, check docker-compose.yml has:
#   runtime: nvidia
#   environment: NVIDIA_VISIBLE_DEVICES=all
```

**Q: MEGA download fails / "over quota"**
```bash
# Check MEGA credentials in .env
docker exec facefusion_bot env | grep MEGA
# Test MEGA login manually
docker exec facefusion_bot megadl --version
# If quota exceeded, wait 6 hours or use a different MEGA account
```

**Q: Output video not sent back / GDrive upload fails**
```bash
# Test rclone config
docker exec facefusion_bot rclone \
  --config /workspace/.config/rclone/rclone.conf \
  lsd gdrive:
# If fails: rclone.conf token may be expired — regenerate with `rclone config reconnect gdrive:`
```

**Q: Face swap quality is poor**
```bash
# Enable face enhancer in .env:
ENABLE_FACE_ENHANCER=true
FACE_ENHANCER_MODEL=gfpgan_1.4
FACE_ENHANCER_BLEND=80
# Restart container after .env change
docker-compose down && docker-compose up -d
```

**Q: Bot crashes after processing a few jobs**
```bash
# Check disk space (outputs accumulate)
df -h /teamspace/studios/this_studio/
docker exec facefusion_bot du -sh /workspace/pipeline/workspace/
# Clean up old outputs
docker exec facefusion_bot find /workspace/pipeline/workspace/output -mtime +1 -delete
docker exec facefusion_bot find /workspace/pipeline/workspace/temp -mtime +0 -delete
```

**Q: Lightning.ai studio restarted and bot is gone**
```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/run
docker-compose up -d
# The Docker image persists in Lightning.ai storage — no need to reload
```

**Q: How do I update bot.py without rebuilding the image?**
```bash
# Edit locally
nano /tmp/bot.py
# Copy into running container
docker cp /tmp/bot.py facefusion_bot:/app/bot.py
# Restart container
docker restart facefusion_bot
```

---

## Known Issues & Fixes

### Issue 1: MEGA filename decoded as `mega_XXXXXXXX` instead of real name

**Symptom:** Files saved as `mega_y9RElKiB_20260531_191906.mp4` instead of `3278172064293_20260531_191906.mp4`

**Root cause:** `_mega_decode_filename()` was stripping the `MEGA` prefix from the raw encrypted bytes *before* AES decryption, corrupting block alignment → JSON parse fails → fallback to file_id.

**Fix applied (bot.py):**
```python
# WRONG (old): strip MEGA prefix before decrypt
if raw[:4] == b'MEGA':
    raw = raw[4:]
cipher = AES.new(aes_key, AES.MODE_CBC, b'\x00' * 16)
decrypted = cipher.decrypt(raw)

# CORRECT (new): decrypt first, then strip MEGA prefix
cipher = AES.new(aes_key, AES.MODE_CBC, b'\x00' * 16)
decrypted = cipher.decrypt(raw)
if decrypted.startswith(b'MEGA'):
    decrypted = decrypted[4:]
```

---

### Issue 2: Filename renamed to `video_YYYYMMDD_HHMMSS.mp4` losing original name

**Symptom:** `normalize_processing_target` replaced any MEGA-style temp name with generic `"video"`.

**Root cause:** Regex `^mega_[A-Za-z0-9]{6,12}$` matched temp names and replaced stem with `"video"`.

**Fix applied (bot.py):** Removed the `_mega_id_only` check. Now always preserves original stem, only sanitizes unsafe characters, appends datetime suffix.

---

### Issue 3: Container won't start — "port already in use"

```bash
# Find what's using port 8765
lsof -i :8765
# Kill it or change DASHBOARD_PORT in .env
```

---

### Issue 4: `docker-compose up` fails — "image not found"

```bash
# Image was lost (studio storage cleared)
# Reload from backup:
docker load < /teamspace/studios/this_studio/facefusion_bot_restore/backup/facefusion_v5_pro_*.tar.gz
```

---

### Issue 5: rclone.conf "device or resource busy" error

**Symptom:** `ERROR: Failed to save config after 10 tries: ... device or resource busy`

**Cause:** rclone.conf is mounted read-only (`:ro`) in docker-compose.yml — rclone tries to update token and fails.

**This is harmless** — the error is logged but rclone still works. The `:ro` mount is intentional to prevent the container from modifying the host config file.

---

### Issue 6: Face swap takes too long / times out

```bash
# Check GPU utilization during processing
watch -n 1 "nvidia-smi | grep -E 'MiB|%'"

# Increase watchdog timeout in .env:
PIPELINE_WATCHDOG_PROCESSING_SEC=600   # 10 minutes
PIPELINE_WATCHDOG_MERGING_SEC=300

# Restart after .env change
docker-compose down && docker-compose up -d
```

---

### Issue 7: Bot processes job but sends no result

```bash
# Check output directory
docker exec facefusion_bot ls -lh /workspace/pipeline/workspace/output/

# Check GDrive upload logs
docker logs facefusion_bot 2>&1 | grep -i "rclone\|upload\|gdrive" | tail -20

# Check if file is too large for Telegram (50MB limit)
# Bot should auto-send GDrive link for large files
```

---

### Issue 8: `ALLOWED_USER_ID` — bot ignores messages

```bash
# Get your Telegram user ID
# Message @userinfobot on Telegram

# Update .env
ALLOWED_USER_ID=123456789

# Restart
docker restart facefusion_bot
```

---

## Project Structure

```
/teamspace/studios/this_studio/
├── README.md                          ← This file
├── RESTORE JOURNEY GUIDE.md           ← Detailed restore walkthrough
├── MASTER INSTRUCTION PROMPT.md       ← AI restore prompt for Claude
├── 🚫 PREVENT MISTAKES GUIDE.md       ← Common mistakes to avoid
├── facefusion_bot_restore/
│   ├── backup/
│   │   └── facefusion_v5_pro_*.tar.gz ← Docker image backups
│   └── run/
│       ├── docker-compose.yml         ← Main orchestration
│       ├── .env                       ← Secrets (never commit!)
│       ├── rclone.conf                ← GDrive auth (never commit!)
│       ├── outputs/                   ← Processed video outputs
│       ├── temp/                      ← Temp processing files
│       └── logs/                      ← Bot and pipeline logs
└── main.py                            ← Minimal stub
```

---

## Security Notes

```
⚠️  NEVER commit .env to git
⚠️  NEVER commit rclone.conf to git
⚠️  NEVER share your BOT_TOKEN publicly
⚠️  If token is exposed: @BotFather → /token → revoke immediately
⚠️  ALLOWED_USER_ID restricts who can use the bot — always set this
```

---

*Last updated: 2026-05-31 | Platform: Lightning.ai Studio | GPU: Tesla T4 | Docker image: facefusion-v5-pro:latest*
