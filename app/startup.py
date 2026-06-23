#!/usr/bin/env python3
"""
SwapMaster V1 - Native Startup Script
Replaces Docker entrypoint.sh for local installation.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def log(msg):
    print(f"[STARTUP] {msg}", flush=True)


def ok(msg):
    print(f"[OK] {msg}", flush=True)


def warn(msg):
    print(f"[WARN] {msg}", flush=True)


def fail(msg):
    print(f"[FAIL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_env_file(env_path):
    """Load .env file into os.environ (always override)."""
    if not env_path.exists():
        return False
    for line in env_path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            os.environ[k] = v
    return True


def check_binary(name, path=None):
    """Check if a binary is available."""
    if path and Path(path).is_file():
        return True
    return shutil.which(name) is not None


def main():
    log("=== SwapMaster V1 - Native Startup ===")

    # 1. Create runtime directories
    app_dir = ROOT_DIR / "app"
    dirs = [
        app_dir / "pipeline" / "logs",
        app_dir / "pipeline" / "workspace" / "temp",
        app_dir / "pipeline" / "workspace" / "output",
        app_dir / "pipeline" / "downloads" / "video",
        app_dir / "pipeline" / "downloads" / "face",
        app_dir / "pipeline" / "dashboard_sessions",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 2. Load .env
    env_file = ROOT_DIR / ".env"
    if load_env_file(env_file):
        ok(".env loaded")
    else:
        warn(".env not found - using defaults")

    # 3. Validate credentials
    bot_token = os.environ.get("BOT_TOKEN", "")
    mega_email = os.environ.get("MEGA_EMAIL", "")
    mega_password = os.environ.get("MEGA_PASSWORD", "")
    allowed_user_id = os.environ.get("ALLOWED_USER_ID", "")

    if not bot_token:
        fail("BOT_TOKEN missing in .env")
    if not mega_email:
        warn("MEGA_EMAIL missing - MEGA downloads will not work")
    if not mega_password:
        warn("MEGA_PASSWORD missing - MEGA downloads will not work")
    if not allowed_user_id:
        warn("ALLOWED_USER_ID missing - bot may not respond to messages")

    ok("Credentials validated")

    # 4. GPU auto-detection
    gpu_detect_script = ROOT_DIR / "app" / "ops" / "gpu_auto_detect.py"
    if gpu_detect_script.exists():
        result = subprocess.run(
            [sys.executable, str(gpu_detect_script)],
            cwd=str(ROOT_DIR),
            capture_output=False
        )
        if result.returncode != 0:
            warn("GPU detection failed - using CPU mode")
    else:
        warn("gpu_auto_detect.py not found - using CPU fallback")
        os.environ["EXECUTION_PROVIDER"] = "cpu"
        os.environ["GPU_ONLY_MODE"] = "0"
        os.environ["OUTPUT_VIDEO_ENCODER"] = "libx264"

    ok(f"GPU: {os.environ.get('GPU_DETECTED_NAME', 'None')} "
       f"({os.environ.get('GPU_DETECTED_VRAM_MB', '0')}MB) -> "
       f"{os.environ.get('EXECUTION_PROVIDER', 'cpu')}")

    # 5. Verify binaries
    for binary in ["python3", "ffmpeg", "rclone"]:
        if not check_binary(binary):
            fail(f"Binary missing: {binary}")
    ok("Binaries OK (python3, ffmpeg, rclone)")

    # 6. Verify ONNX models
    models_dir = ROOT_DIR / "app" / "facefusion" / ".assets" / "models"
    if models_dir.exists():
        model_count = len(list(models_dir.glob("*.onnx")))
        if model_count < 20:
            fail(f"Models missing: {model_count} (need >=20)")
        ok(f"{model_count} ONNX models found")
    else:
        warn("Models directory not found - FaceFusion may not work")

    # 7. Validate CUDA
    execution_provider = os.environ.get("EXECUTION_PROVIDER", "cpu")
    if execution_provider == "cuda":
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "import torch, onnxruntime as ort; "
                 "assert torch.cuda.is_available(), 'torch CUDA unavailable'; "
                 "assert 'CUDAExecutionProvider' in ort.get_available_providers(), 'ONNX CUDA unavailable'; "
                 "print('CUDA OK:', torch.cuda.get_device_name(0))"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                ok(result.stdout.strip())
            else:
                warn(f"CUDA validation failed: {result.stderr}")
                os.environ["EXECUTION_PROVIDER"] = "cpu"
                os.environ["GPU_ONLY_MODE"] = "0"
        except Exception as e:
            warn(f"CUDA validation error: {e}")

    # 8. rclone config
    rclone_conf = os.environ.get("RCLONE_CONF", "")
    if not rclone_conf:
        rclone_conf = str(ROOT_DIR / ".config" / "rclone" / "rclone.conf")
    if Path(rclone_conf).exists():
        ok(f"rclone.conf found: {rclone_conf}")
    else:
        warn(f"rclone.conf not found at {rclone_conf} - GDrive disabled")
        os.environ["GDRIVE_ENABLED"] = "0"

    # 9. Print summary
    log("=== STARTUP SUMMARY ===")
    log(f"  GPU:      {os.environ.get('GPU_DETECTED_NAME', 'None')} "
        f"({os.environ.get('GPU_DETECTED_VRAM_MB', '0')}MB)")
    log(f"  Provider: {os.environ.get('EXECUTION_PROVIDER', 'cpu')}")
    log(f"  Model:    {os.environ.get('FACE_SWAPPER_MODEL', 'inswapper_128_fp16')}")
    log(f"  GDrive:   {os.environ.get('GDRIVE_ENABLED', '1')}")
    log(f"  Port:     {os.environ.get('DASHBOARD_PORT', '8765')}")

    # 10. Start the application
    log("Starting SwapMaster...")
    process_guard = ROOT_DIR / "app" / "ops" / "process_guard.py"
    if process_guard.exists():
        os.execv(sys.executable, [sys.executable, str(process_guard), "--max-backoff", "120"])
    else:
        fail("process_guard.py not found")


if __name__ == "__main__":
    main()
