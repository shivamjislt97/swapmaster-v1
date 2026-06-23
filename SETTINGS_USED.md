# SwapMaster V1 - Current Settings Used

> Copy this file and paste values into `.env` to restore exact same configuration.

## Environment Variables (.env)

```bash
# === TELEGRAM ===
BOT_TOKEN=YOUR_BOT_TOKEN_HERE
ALLOWED_USER_ID=YOUR_TELEGRAM_USER_ID

# === MEGA ===
MEGA_EMAIL=your_mega_email@example.com
MEGA_PASSWORD=your_mega_password

# === GDRIVE UPLOAD ===
GDRIVE_ENABLED=true
GDRIVE_REMOTE_NAME=gdrive
GDRIVE_FOLDER=gdrive:faceswap_output
RCLONE_BIN=rclone
RCLONE_CONF=/full/path/to/swapmaster-v1-native/.config/rclone/rclone.conf

# === GPU / EXECUTION ===
EXECUTION_PROVIDER=cuda
GPU_ONLY_MODE=true
OUTPUT_VIDEO_ENCODER=libx264
EXECUTION_THREAD_COUNT=4
THREAD_COUNT=4

# === FACE SWAP MODELS ===
FACE_SWAPPER_MODEL=inswapper_128
FACE_ENHANCER_MODEL=gfpgan_1.4
FACE_ENHANCER_BLEND=80
ENABLE_FACE_ENHANCER=true

# === AUTO SLEEP ===
AUTO_SLEEP_ENABLED=true
AUTO_SLEEP_MINUTES=30
POST_JOB_AUTO_SLEEP_SECONDS=300

# === DASHBOARD ===
DASHBOARD_ENABLED=true
DASHBOARD_PORT=8765

# === CONTENT ANALYSIS ===
BYPASS_CONTENT_ANALYSER=false

# === WATCHDOG TIMEOUTS (seconds) ===
PIPELINE_WATCHDOG_PROCESSING_SEC=600
PIPELINE_WATCHDOG_MERGING_SEC=300
PIPELINE_WATCHDOG_UPLOADING_SEC=300
```

## FaceFusion Face Swap Settings (hardcoded in bot.py)

These are the exact values currently used. To change, modify bot.py or set env vars:

| Setting | Value | Location |
|---------|-------|----------|
| Face Swapper Model | `inswapper_128` | .env `FACE_SWAPPER_MODEL` |
| Face Enhancer Model | `gfpgan_1.4` | .env `FACE_ENHANCER_MODEL` |
| Face Enhancer Blend | `80` | .env `FACE_ENHANCER_BLEND` |
| Face Swapper Pixel Boost | `512x512` | bot.py default |
| Face Swapper Weight | `0.85` | bot.py default |
| Face Enhancer Weight | `0.70` | bot.py default |
| Face Detector Size | `640x640` | bot.py default |
| Face Detector Score | `0.5` | bot.py default |
| Face Landmarker Score | `0.35` | bot.py default |
| Face Mask Blur | `0.3` | bot.py default |
| Face Selector Mode | `reference` | hardcoded in pipeline |
| Reference Face Distance | `0.3` | hardcoded in pipeline |
| Expression Restorer Factor | `90` | bot.py default |
| Expression Restorer Enabled | `true` | bot.py default |
| Video Memory Strategy | `tolerant` | bot.py default |
| Video Encoder | `libx264` | .env `OUTPUT_VIDEO_ENCODER` |
| Video Encoder Preset | `fast` | bot.py default |
| Video Quality | `80` | bot.py default |
| Video Scale | `1.0` | bot.py default (full resolution) |
| Output Video Quality | `95` (image) | hardcoded in pipeline |
| Execution Provider | `cuda` | .env `EXECUTION_PROVIDER` |
| GPU Only Mode | `true` | .env `GPU_ONLY_MODE` |
| CPU Threads | `4` | .env `EXECUTION_THREAD_COUNT` |
| Low Memory Mode | `true` | bot.py default |
| CPU Fast Mode | `true` | bot.py default |

## Upload Settings

| Setting | Value | Location |
|---------|-------|----------|
| GDrive Upload Timeout | `1200` sec (20 min) | bot.py default |
| GDrive Upload Retries | `2` | bot.py default |
| MEGA Upload Timeout | `0` (disabled) | bot.py default |
| MEGA Download Timeout | `120` sec | bot.py default |
| MEGA Min Operation Gap | `25` sec | bot.py default |
| MEGA Auth Cooldown Base | `600` sec (10 min) | bot.py default |
| MEGA Auth Cooldown Max | `7200` sec (2 hr) | bot.py default |
| MEGA Test Fallback on 509 | `true` | bot.py default |

## Download Settings

| Setting | Value | Location |
|---------|-------|----------|
| Download Attempt Timeout | `420` sec (7 min) | bot.py default |
| Download Stall Timeout | `90` sec | bot.py default |
| Download Retry Count | `3` | bot.py default |
| Face Download Attempt Timeout | `120` sec | bot.py default |
| Face Download Stall Timeout | `35` sec | bot.py default |
| Face Download Retry Count | `2` | bot.py default |
| Face Download Total Timeout | `300` sec (5 min) | bot.py default |
| Download Progress Poll | `2` sec | bot.py default |
| Direct Link Probe Timeout | `15` sec | bot.py default |

## Cleanup Settings

| Setting | Value | Location |
|---------|-------|----------|
| Min Free Space | `20` GB | bot.py default |
| Max Storage Usage | `150` GB | bot.py default |
| Keep Latest Outputs | `3` | bot.py default |
| Temp Job Retention | `6` hours | bot.py default |
| Temp Job Keep Latest | `2` | bot.py default |
| Safe Cleanup Min Age | `900` sec (15 min) | bot.py default |
| Safe Cleanup Disk Trigger | `80%` | bot.py default |
| Safe Cleanup Periodic | `1800` sec (30 min) | bot.py default |

## Dashboard Settings

| Setting | Value | Location |
|---------|-------|----------|
| Dashboard Host | `0.0.0.0` | bot.py default |
| Dashboard Port | `8765` | .env `DASHBOARD_PORT` |
| Dashboard Log Level | `warning` | bot.py default |

## GPU Settings

| Setting | Value | Location |
|---------|-------|----------|
| GPU Startup Balanced Mode | `true` | bot.py default |
| GPU Startup Thread Count | `2` | bot.py default |
| GPU Startup Face Detector Size | `640x640` | bot.py default |
| GPU Startup Pixel Boost | `512x512` | bot.py default |
| GPU Startup Video Memory Strategy | `tolerant` | bot.py default |
| GPU OOM Max Levels | `3` | bot.py default |
| GPU Retry Chunk Seconds L2 | `8` | bot.py default |
| GPU Retry Chunk Seconds L3 | `4` | bot.py default |
| Auto CPU Fallback on OOM | `false` | bot.py default |

## Primary/Fallback Face Detector

| Setting | Value | Location |
|---------|-------|----------|
| Primary Face Detector Model | `yolo_face` | bot.py default |
| Fallback Face Detector Model | `yolo_face` | bot.py default |
| Frame Debug Detection Stride | `2` | bot.py default |
| Frame Debug Analysis Timeout | `90` sec | bot.py default |
| Swap Validation Timeout | `90` sec | bot.py default |
| Frame Debug Tracking TTL | `6` | bot.py default |
| Frame Debug Min Detection Ratio | `0.50` | bot.py default |
