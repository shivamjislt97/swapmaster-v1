#!/usr/bin/env python3
"""Standalone health check — prints color-coded PASS/FAIL for every critical dependency."""
import sys, os, subprocess

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; E = "\033[0m"
passed = failed = 0

def check(name, fn, fix=""):
    global passed, failed
    try:
        fn()
        print(f"{G}PASS{E}  {name}")
        passed += 1
    except Exception as e:
        print(f"{R}FAIL{E}  {name}: {e}")
        if fix:
            print(f"      {Y}Fix:{E} {fix}")
        failed += 1

# Python version
check("Python >= 3.10",
      lambda: __import__('sys') and sys.version_info >= (3, 10) or (_ for _ in ()).throw(Exception(f"Got {sys.version}")),
      "Use Python 3.10+")

# Core packages
for pkg, imp, fix in [
    ("onnxruntime-gpu + CUDA provider", "import onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()", "pip install onnxruntime-gpu==1.19.2"),
    ("python-telegram-bot", "import telegram", "pip install python-telegram-bot==20.7"),
    ("opencv", "import cv2", "pip install opencv-python-headless"),
    ("numpy", "import numpy", "pip install numpy"),
    ("fastapi", "import fastapi", "pip install fastapi==0.135.1"),
    ("mega.py", "from mega import Mega", "pip install mega.py==1.0.8 tenacity>=8.0"),
]:
    check(pkg, lambda i=imp: exec(i), fix)

# System tools
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, timeout=10)
    assert r.returncode == 0, r.stderr.decode()[:100]

check("nvidia-smi", lambda: run(["nvidia-smi"]), "Install NVIDIA drivers")
check("ffmpeg", lambda: run(["ffmpeg", "-version"]), "apt-get install ffmpeg")
check("ffmpeg h264_nvenc", lambda: subprocess.run("ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc", shell=True, check=True), "GPU must support NVENC")
check("rclone", lambda: run(["rclone", "version"]), "apt-get install rclone")

# Environment variables
for var in ["BOT_TOKEN", "ALLOWED_USER_ID"]:
    check(f"env {var}", lambda v=var: os.environ.get(v) or (_ for _ in ()).throw(Exception("not set")), f"Set {var} in .env")

# Project files
check("facefusion dir", lambda: os.path.isdir("facefusion") or (_ for _ in ()).throw(Exception("missing")), "git clone the repo")
check("inswapper model", lambda: os.path.isfile("facefusion/.assets/models/inswapper_128_fp16.onnx") or (_ for _ in ()).throw(Exception("missing")), "Models must be present in facefusion/.assets/models/")

print(f"\n{'='*40}")
print(f"Results: {G}{passed} passed{E}, {R}{failed} failed{E}")
sys.exit(0 if failed == 0 else 1)
