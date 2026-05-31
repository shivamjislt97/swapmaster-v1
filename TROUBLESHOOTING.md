# 🔧 TROUBLESHOOTING GUIDE
## SwapMaster V1 — Missing Models, Paths, Libraries & Pipeline Failures

---

## Quick Diagnosis — Run This First

```bash
docker exec facefusion_bot python3 - << 'EOF'
import os, subprocess

checks = {
    "ONNX Runtime":     lambda: __import__('onnxruntime').__version__,
    "OpenCV":           lambda: __import__('cv2').__version__,
    "NumPy":            lambda: __import__('numpy').__version__,
    "Requests":         lambda: __import__('requests').__version__,
    "PyCryptodome":     lambda: __import__('Crypto.Cipher.AES', fromlist=['AES']) and "OK",
    "aiohttp":          lambda: __import__('aiohttp').__version__,
    "ffmpeg":           lambda: subprocess.check_output(['ffmpeg','-version']).decode().split('\n')[0],
    "rclone":           lambda: subprocess.check_output(['rclone','version']).decode().split('\n')[0],
}

models = [
    "/app/facefusion/.assets/models/inswapper_128.onnx",
    "/app/facefusion/.assets/models/retinaface_10g.onnx",
    "/app/facefusion/.assets/models/2dfan4.onnx",
    "/app/facefusion/.assets/models/gfpgan_1.4.onnx",
    "/app/facefusion/.assets/models/xseg_1.onnx",
    "/app/facefusion/.assets/models/yoloface_8n.onnx",
]

paths = [
    "/app/bot.py",
    "/app/facefusion/",
    "/app/ops/process_guard.py",
    "/workspace/pipeline/logs/",
    "/workspace/pipeline/workspace/output/",
    "/workspace/pipeline/workspace/temp/",
    "/workspace/.config/rclone/rclone.conf",
]

print("\n=== LIBRARIES ===")
for name, fn in checks.items():
    try:
        print(f"  ✅ {name}: {fn()}")
    except Exception as e:
        print(f"  ❌ {name}: MISSING — {e}")

print("\n=== MODELS ===")
for m in models:
    size = os.path.getsize(m) if os.path.exists(m) else 0
    status = f"✅ {size//1024//1024}MB" if size > 0 else "❌ MISSING"
    print(f"  {status}  {m}")

print("\n=== PATHS ===")
for p in paths:
    status = "✅" if os.path.exists(p) else "❌ MISSING"
    print(f"  {status}  {p}")

print("\n=== GPU ===")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  {'✅' if 'CUDAExecutionProvider' in providers else '❌'} CUDA: {'available' if 'CUDAExecutionProvider' in providers else 'NOT available'}")
    print(f"  Providers: {providers}")
except Exception as e:
    print(f"  ❌ ORT error: {e}")
EOF
```

---

## Issue 1: Missing ONNX Models

### Symptom
```
Error: model file not found: /app/facefusion/.assets/models/inswapper_128.onnx
FileNotFoundError: [Errno 2] No such file or directory
```

### All Required Models & Download URLs

| Model | Purpose | Size | Download |
|-------|---------|------|---------|
| `inswapper_128.onnx` | Face swapper (main) | ~500MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx) |
| `inswapper_128_fp16.onnx` | Face swapper FP16 | ~250MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx) |
| `retinaface_10g.onnx` | Face detector | ~16MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx) |
| `2dfan4.onnx` | Face landmarker | ~50MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/2dfan4.onnx) |
| `gfpgan_1.4.onnx` | Face enhancer | ~350MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx) |
| `xseg_1.onnx` | Face segmenter | ~1MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/xseg_1.onnx) |
| `yoloface_8n.onnx` | Face detector v2 | ~6MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/yoloface_8n.onnx) |
| `scrfd_2.5g.onnx` | Face detector v3 | ~3MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/scrfd_2.5g.onnx) |
| `arcface_w600k_r50.onnx` | Face recognizer | ~166MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx) |
| `codeformer.onnx` | Face enhancer v2 | ~370MB | [facefusion-assets](https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx) |

### Fix — Download All Missing Models

```bash
MODELS_DIR="/app/facefusion/.assets/models"
docker exec facefusion_bot mkdir -p "$MODELS_DIR"

# Download all models in one shot
MODELS=(
  "inswapper_128.onnx"
  "inswapper_128_fp16.onnx"
  "retinaface_10g.onnx"
  "2dfan4.onnx"
  "gfpgan_1.4.onnx"
  "xseg_1.onnx"
  "yoloface_8n.onnx"
  "scrfd_2.5g.onnx"
  "arcface_w600k_r50.onnx"
  "codeformer.onnx"
)

BASE_URL="https://github.com/facefusion/facefusion-assets/releases/download/models"

for model in "${MODELS[@]}"; do
  if docker exec facefusion_bot test -f "$MODELS_DIR/$model"; then
    echo "✅ Already exists: $model"
  else
    echo "⬇️  Downloading: $model"
    docker exec facefusion_bot wget -q --show-progress \
      "$BASE_URL/$model" -O "$MODELS_DIR/$model"
  fi
done

echo "=== Final model list ==="
docker exec facefusion_bot ls -lh "$MODELS_DIR/"
```

---

## Issue 2: Missing Python Libraries

### Symptom
```
ModuleNotFoundError: No module named 'cv2'
ModuleNotFoundError: No module named 'onnxruntime'
ModuleNotFoundError: No module named 'Crypto'
ImportError: cannot import name 'AES' from 'Crypto.Cipher'
```

### Required Libraries & Versions

| Library | Package Name | Version in Container |
|---------|-------------|---------------------|
| OpenCV | `opencv-python` | 4.13.0 |
| ONNX Runtime GPU | `onnxruntime-gpu` | 1.19.2 |
| NumPy | `numpy` | 2.4.6 |
| PyCryptodome | `pycryptodome` | latest |
| aiohttp | `aiohttp` | latest |
| aiofiles | `aiofiles` | latest |
| requests | `requests` | latest |
| Pillow | `Pillow` | latest |

### Fix

```bash
# Install inside running container
docker exec facefusion_bot pip install \
  opencv-python-headless \
  onnxruntime-gpu==1.19.2 \
  numpy \
  pycryptodome \
  aiohttp \
  aiofiles \
  requests \
  Pillow \
  --quiet

# Verify
docker exec facefusion_bot python3 -c "
import cv2, onnxruntime, numpy, Crypto, aiohttp, aiofiles, requests, PIL
print('All libraries OK')
print('ORT providers:', onnxruntime.get_available_providers())
"
```

> ⚠️ **Note:** Installing inside a running container is temporary. After `docker restart`, changes are lost. To make permanent, rebuild the image or use a volume mount.

---

## Issue 3: Path Mismatches

### Critical Paths Inside Container

| Path | What It Is | If Missing |
|------|-----------|-----------|
| `/app/bot.py` | Main bot | Container broken — reload image |
| `/app/facefusion/` | FaceFusion pipeline | Models won't run |
| `/app/ops/process_guard.py` | Watchdog | Bot won't auto-restart |
| `/app/.env` | Secrets/config | Bot won't start |
| `/workspace/pipeline/logs/` | Log files | Create manually |
| `/workspace/pipeline/workspace/output/` | Output videos | Create manually |
| `/workspace/pipeline/workspace/temp/` | Temp files | Create manually |
| `/workspace/.config/rclone/rclone.conf` | GDrive auth | Uploads will fail |

### Fix — Recreate Missing Directories

```bash
docker exec facefusion_bot bash -c "
  mkdir -p /workspace/pipeline/logs
  mkdir -p /workspace/pipeline/workspace/output
  mkdir -p /workspace/pipeline/workspace/temp
  mkdir -p /workspace/pipeline/downloads/face
  mkdir -p /workspace/pipeline/downloads/video
  mkdir -p /workspace/.config/rclone
  echo '✅ All directories created'
"
```

### Fix — Path Mismatch in .env

```bash
# Check what paths bot.py expects
docker exec facefusion_bot grep -E "^PIPELINE|^WORKSPACE|^RCLONE_CONF|^RCLONE_BIN" /app/.env

# Common mismatch: RCLONE_CONF wrong path
# Should be:
docker exec facefusion_bot ls -la /workspace/.config/rclone/rclone.conf
# If missing, copy from host:
docker cp /teamspace/studios/this_studio/facefusion_bot_restore/run/rclone.conf \
  facefusion_bot:/workspace/.config/rclone/rclone.conf
```

---

## Issue 4: GPU / CUDA Not Available

### Symptom
```
Error: CUDA provider not available
onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: CUDAExecutionProvider is not available
```

### Diagnosis

```bash
# On host
nvidia-smi

# Inside container
docker exec facefusion_bot nvidia-smi

# Check ORT providers
docker exec facefusion_bot python3 -c "
import onnxruntime as ort
print(ort.get_available_providers())
"
```

### Fix

```bash
# 1. Verify docker-compose.yml has nvidia runtime
grep "runtime\|NVIDIA" /teamspace/studios/this_studio/facefusion_bot_restore/run/docker-compose.yml

# 2. If missing, add to docker-compose.yml:
#    runtime: nvidia
#    environment:
#      - NVIDIA_VISIBLE_DEVICES=all
#      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 3. Restart
cd /teamspace/studios/this_studio/facefusion_bot_restore/run
docker-compose down && docker-compose up -d

# 4. If still failing — Lightning.ai GPU not enabled:
#    Studio → Configure → Compute → T4 GPU → Save & Restart
```

---

## Issue 5: ffmpeg Missing or Wrong Version

### Symptom
```
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
Error: ffmpeg failed with exit code 1
```

### Fix

```bash
# Check if ffmpeg exists
docker exec facefusion_bot which ffmpeg
docker exec facefusion_bot ffmpeg -version 2>&1 | head -1

# If missing, install inside container:
docker exec -u root facefusion_bot bash -c "
  apt-get update -qq && apt-get install -y ffmpeg --quiet
  ffmpeg -version | head -1
"

# Verify video encoding works
docker exec facefusion_bot ffmpeg -f lavfi -i testsrc=duration=1:size=128x128:rate=1 \
  /tmp/test_output.mp4 -y 2>&1 | tail -3
```

---

## Issue 6: rclone Not Found / GDrive Upload Fails

### Symptom
```
FileNotFoundError: rclone not found
Error: directory not found
NOTICE: Failed to copy
```

### Fix

```bash
# Check rclone
docker exec facefusion_bot which rclone
docker exec facefusion_bot rclone version 2>&1 | head -1

# If missing, install:
docker exec -u root facefusion_bot bash -c "
  curl https://rclone.org/install.sh | bash
"

# Test GDrive connection
docker exec facefusion_bot rclone \
  --config /workspace/.config/rclone/rclone.conf \
  lsd gdrive: 2>&1 | grep -v "Failed to save config"

# If token expired:
# Run on a machine WITH browser:
#   rclone config reconnect gdrive:
# Then copy updated rclone.conf into container:
docker cp /path/to/rclone.conf facefusion_bot:/workspace/.config/rclone/rclone.conf
```

---

## Issue 7: megadl Not Found

### Symptom
```
FileNotFoundError: megadl not found
MEGA download failed
```

### Fix

```bash
# Check
docker exec facefusion_bot which megadl

# If missing, install megatools:
docker exec -u root facefusion_bot bash -c "
  apt-get update -qq && apt-get install -y megatools --quiet
  megadl --version
"
```

---

## Issue 8: Container Starts But Bot Crashes Immediately

### Diagnosis

```bash
# Check logs for crash reason
docker logs facefusion_bot --tail=50 2>&1 | grep -iE "error|exception|traceback|missing|not found"

# Check process guard
docker exec facefusion_bot ps aux | grep -E "process_guard|bot.py"

# Check .env is loaded
docker exec facefusion_bot env | grep -E "BOT_TOKEN|ALLOWED_USER|MEGA_EMAIL" | sed 's/=.*/=***/'
```

### Common Crash Causes

| Error in logs | Cause | Fix |
|--------------|-------|-----|
| `BOT_TOKEN not set` | .env not loaded | Check `env_file: .env` in docker-compose.yml |
| `Unauthorized` | Wrong BOT_TOKEN | Get new token from @BotFather |
| `ModuleNotFoundError` | Missing library | See Issue 2 above |
| `Address already in use :8765` | Port conflict | Change `DASHBOARD_PORT` in .env |
| `CUDA out of memory` | GPU VRAM full | Restart container, check `GPU_ONLY_MODE=true` |
| `No such file: bot.py` | Image corrupted | Reload Docker image from backup |

---

## Issue 9: Docker Image Lost After Studio Restart

### Symptom
```
docker: Error response from daemon: No such image: facefusion-v5-pro:latest
```

### Fix

```bash
# Check if image exists
docker images | grep facefusion

# If missing, reload from backup:
docker load < /teamspace/studios/this_studio/facefusion_bot_restore/backup/facefusion_v5_pro_20260531_193402.tar.gz

# If backup not on disk, download from GDrive:
pip install gdown --quiet
gdown "1QMNTPhlQ7QL6i211iaiuakExhyoZQN7Z" -O facefusion_v5_pro_latest.tar.gz
docker load < facefusion_v5_pro_latest.tar.gz

# Then start:
cd /teamspace/studios/this_studio/facefusion_bot_restore/run
docker-compose up -d
```

---

## Issue 10: Pipeline Hangs / Job Never Completes

### Symptom
Bot accepts job but never sends result. Logs show pipeline stuck.

### Diagnosis

```bash
# Check active job state
docker exec facefusion_bot cat /workspace/pipeline/logs/active_job_state.json 2>/dev/null | python3 -m json.tool

# Check GPU is actually working
docker exec facefusion_bot nvidia-smi

# Check disk space (full disk = silent failure)
docker exec facefusion_bot df -h /workspace/
```

### Fix

```bash
# Kill stuck pipeline and reset state
docker exec facefusion_bot pkill -f facefusion 2>/dev/null || true
docker exec facefusion_bot rm -f /workspace/pipeline/logs/active_job_state.json
docker exec facefusion_bot rm -rf /workspace/pipeline/workspace/temp/*

# Increase timeouts in .env if jobs are just slow:
# PIPELINE_WATCHDOG_PROCESSING_SEC=600
# PIPELINE_WATCHDOG_MERGING_SEC=300

docker restart facefusion_bot
```

---

## Full Reset (Last Resort)

If nothing works:

```bash
cd /teamspace/studios/this_studio/facefusion_bot_restore/run

# Stop everything
docker-compose down

# Clean temp data (keeps outputs)
docker exec facefusion_bot rm -rf /workspace/pipeline/workspace/temp/* 2>/dev/null || true
docker exec facefusion_bot rm -f /workspace/pipeline/logs/active_job_state.json 2>/dev/null || true

# Restart fresh
docker-compose up -d

# Watch startup
docker-compose logs -f --tail=30
```

---

*Last updated: 2026-05-31 | Container: facefusion-v5-pro:latest | ORT: 1.19.2 | ffmpeg: 4.4.2*
