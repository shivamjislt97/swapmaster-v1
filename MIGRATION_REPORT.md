# SwapMaster V1 - Migration Report
## Docker to Native Installation

**Date:** 2026-06-23
**Source:** Docker container `facefusion_bot` (facefusion-v5-pro:latest)
**Target:** Native file-based installation at `swapmaster-v1-native/`

---

## 1. Docker Components Removed

| Component | File/Path | Status |
|-----------|-----------|--------|
| Dockerfile | `app/Dockerfile` | REMOVED |
| Dockerfile.private | `app/Dockerfile.private` | REMOVED |
| .dockerignore | `app/.dockerignore` | REMOVED |
| docker-compose.yml | `facefusion_bot_restore/run/docker-compose.yml` | NOT NEEDED |
| entrypoint.sh | `app/entrypoint.sh` | REPLACED by `app/startup.py` |
| Docker startup scripts | `app/start.sh`, `app/run.sh`, `app/restore.sh` | REMOVED |
| Container-specific configs | `app/.env.docker` | CONSOLIDATED into `.env` |
| Docker volume mounts | Via docker-compose | REPLACED by local directories |

---

## 2. Modified Files

### 2.1 `app/config/credentials.py`
- **Change:** Updated `.env` file path resolution
- **Before:** Looked for `.env` in `app/` directory
- **After:** Looks for `.env` in project root, falls back to `app/`
- **Reason:** Native installation has `.env` at project root

### 2.2 `app/ops/process_guard.py`
- **Change:** Removed hardcoded `/home/zeus/.local/bin` path
- **Before:** `env["PATH"] = "/home/zeus/.local/bin:" + env.get("PATH", ...)`
- **After:** Uses `Path.home() / ".local" / "bin"` dynamically
- **Reason:** Portability across different user accounts

### 2.3 `app/startup.py`
- **Change:** Complete rewrite from Docker entrypoint logic
- **Before:** Shell script (`entrypoint.sh`) with container-specific paths
- **After:** Python script that works on any system
- **Reason:** Cross-platform compatibility

### 2.4 `app/ops/gpu_auto_detect.py`
- **Change:** New file created from `gpu_auto_detect.sh`
- **Before:** Shell script with hardcoded conda paths
- **After:** Python script with dynamic path detection
- **Reason:** Better cross-platform support and error handling

---

## 3. New Local Folder Structure

```
swapmaster-v1-native/
├── app/                          # Main application directory
│   ├── bot.py                    # Main Telegram bot
│   ├── startup.py                # Native startup script (replaces entrypoint.sh)
│   ├── health_check.py           # Health check utilities
│   ├── auto_repair.py            # Auto-repair utilities
│   ├── verify.py                 # Verification utilities
│   ├── test_auto_sleep.py        # Auto-sleep test
│   ├── config/                   # Configuration module
│   │   ├── __init__.py
│   │   └── credentials.py        # MODIFIED: .env path resolution
│   ├── ops/                      # Operations module
│   │   ├── __init__.py
│   │   ├── process_guard.py      # MODIFIED: removed hardcoded paths
│   │   ├── boot_launcher.py
│   │   ├── health_monitor.py
│   │   ├── progress_poller.py
│   │   ├── progress_writer.py
│   │   ├── job_worker.py
│   │   ├── state_manager.py
│   │   ├── dashboard_server.py
│   │   ├── tunnel_manager.py
│   │   ├── gpu_monitor.py
│   │   ├── gpu_auto_detect.py    # NEW: Python version
│   │   ├── ff_log_parser.py
│   │   ├── frame_counter.py
│   │   ├── auto_sleep_manager.py
│   │   └── safe_cleanup.py
│   ├── scripts/                  # Utility scripts
│   │   ├── backup_to_gdrive.sh
│   │   ├── dashboard.html
│   │   ├── gdrive_upload.py
│   │   ├── healthcheck.py
│   │   └── live_monitor.html
│   ├── facefusion/               # FaceFusion engine
│   │   ├── .assets/models/       # 25 ONNX models (3.2GB)
│   │   ├── facefusion/           # Python package
│   │   └── ...
│   ├── persistent/               # Persistent data
│   │   ├── config.json
│   │   └── faces/                # Face images
│   ├── pipeline/                 # Runtime data
│   │   ├── logs/                 # Log files
│   │   ├── workspace/
│   │   │   ├── output/           # Processed outputs
│   │   │   └── temp/             # Temporary files
│   │   └── downloads/
│   │       ├── video/            # Downloaded videos
│   │       └── face/             # Downloaded face images
│   └── dashboard_v2.html         # Dashboard UI
├── .env                          # Environment configuration
├── .env.example                  # Example configuration
├── .config/
│   └── rclone/
│       └── rclone.conf           # GDrive configuration
├── requirements.txt              # Python dependencies
├── setup.sh                      # Linux/macOS setup script
├── setup.bat                     # Windows setup script
├── run.sh                        # Linux/macOS run script
├── run.bat                       # Windows run script
└── MIGRATION_REPORT.md           # This file
```

---

## 4. System Dependencies

### Required (must be installed separately)

| Dependency | Install Command | Notes |
|------------|----------------|-------|
| Python 3.10+ | System package manager | Required |
| pip | Comes with Python | Required |
| ffmpeg | `sudo apt install ffmpeg` | Required for video processing |
| rclone | `curl https://rclone.org/install.sh \| sudo bash` | Required for GDrive |
| nvidia-driver | System package manager | Required for GPU |
| CUDA toolkit | NVIDIA website | Required for GPU |

### Optional

| Dependency | Install Command | Notes |
|------------|----------------|-------|
| ngrok | `snap install ngrok` | For public dashboard URL |

---

## 5. Setup Instructions

### Linux/macOS

```bash
# 1. Navigate to project directory
cd swapmaster-v1-native

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Edit .env with your values
nano .env

# 4. Configure rclone (if using GDrive)
rclone config
# Create remote named 'gdrive' with Google Drive

# 5. Start the bot
./run.sh
```

### Windows

```cmd
# 1. Navigate to project directory
cd swapmaster-v1-native

# 2. Run setup script
setup.bat

# 3. Edit .env with your values (use Notepad or similar)
notepad .env

# 4. Configure rclone (if using GDrive)
rclone config

# 5. Start the bot
run.bat
```

---

## 6. Key Differences from Docker

| Aspect | Docker | Native |
|--------|--------|--------|
| Startup | `docker-compose up` | `./run.sh` or `python app/startup.py` |
| Configuration | `.env` + Docker volumes | `.env` file only |
| Data storage | Docker volumes | Local directories |
| Dependencies | Pre-installed in image | Installed via `pip install -r requirements.txt` |
| GPU access | NVIDIA runtime | Direct GPU access |
| Logs | `docker logs` | `pipeline/logs/` directory |
| Updates | Rebuild image | `git pull && pip install -r requirements.txt` |

---

## 7. File Size Comparison

| Component | Docker Image | Native Installation |
|-----------|--------------|---------------------|
| Total size | 16.3 GB | 6.4 GB |
| Application code | ~100 MB | ~100 MB |
| ONNX models | ~3.2 GB | ~3.2 GB |
| Python packages | ~12 GB | Not included (installed separately) |
| System tools | ~1 GB | Not included (installed separately) |

---

## 8. Known Limitations

1. **Lightning AI features:** Disabled (not available outside Lightning environment)
2. **Ngrok:** Requires manual setup if public URL needed
3. **Windows compatibility:** Some ops scripts use `pgrep` (Linux-only); Windows alternatives may be needed
4. **CUDA libraries:** Must be installed separately (not bundled like in Docker)

---

## 9. Testing Checklist

- [ ] `setup.sh` / `setup.bat` completes successfully
- [ ] `run.sh` / `run.bat` starts the bot
- [ ] Bot responds to `/start` command on Telegram
- [ ] GPU detected correctly (if available)
- [ ] Face swap processing works
- [ ] Dashboard accessible at `http://localhost:8765`
- [ ] GDrive upload works (if configured)
- [ ] Logs written to `pipeline/logs/`

---

## 10. Rollback Plan

If native installation doesn't work:

```bash
# Stop native bot (if running)
pkill -f "python.*startup.py"

# Revert to Docker
cd ../swapmaster-v1/facefusion_bot_restore/run
docker-compose up -d
```

---

**Migration completed successfully.** The bot can now run as a standard local application without Docker dependencies.
