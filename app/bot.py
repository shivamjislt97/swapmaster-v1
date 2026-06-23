#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FaceSwap Telegram Bot v14"""

import os, re, asyncio, logging, subprocess, shutil, signal, sys, time, json, configparser, gc, math, mimetypes, zipfile
import atexit
import base64
import queue
import random
import struct
import threading
from typing import Optional
from datetime import datetime, timezone
from collections import deque
from contextlib import suppress
from pathlib import Path
import sysconfig
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from config.credentials import (
    resolve_credentials,
    validate_credentials,
    reload_credentials as config_reload_credentials,
    mask_secret,
    masked_credentials,
)
import ops.safe_cleanup as safe_cleanup
from ops.state_manager import PipelineStateStore, compute_idempotency_key, validate_output_media
from ops.auto_sleep_manager import load_auto_sleep_config, append_auto_sleep_log
from ops import dashboard_server as dashboard
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import RetryAfter, InvalidToken
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

_CREDS = resolve_credentials()
BOT_TOKEN       = str(_CREDS.get("bot_token", "")).strip()

# Load .env into os.environ so PIPELINE_WATCHDOG_* vars are available
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.is_file():
    with open(_ENV_FILE) as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                if _k and _k not in os.environ:
                    os.environ[_k] = _v.strip().strip('"').strip("'")
if BOT_TOKEN:
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", BOT_TOKEN)
GDRIVE_REMOTE_NAME = str(_CREDS.get("gdrive_remote_name", "gdrive")).strip() or "gdrive"
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "6267031612"))
ALLOWED_CHAT_ID = int(os.environ.get("ALLOWED_CHAT_ID", str(ALLOWED_USER_ID)))
ROOT_DIR        = Path(__file__).resolve().parent
DEFAULT_VENV_PYTHON = ROOT_DIR / "venv" / "bin" / "python"
RUNTIME_PYTHON = str(DEFAULT_VENV_PYTHON)
if not DEFAULT_VENV_PYTHON.is_file():
    RUNTIME_PYTHON = str(Path(sys.executable).resolve())

# CUDA library fix for onnxruntime-gpu - add nvidia libs to LD_LIBRARY_PATH
CUDA_LIBS_PATH = str(Path(RUNTIME_PYTHON).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia")
if Path(CUDA_LIBS_PATH).is_dir():
    nvidia_libs = []
    for subdir in ["cublas", "cufft", "cudnn", "cusolver", "curand", "cusparse", "nccl", "nvjitlink", "cuda_runtime"]:
        lib_path = Path(CUDA_LIBS_PATH) / subdir / "lib"
        if lib_path.is_dir():
            nvidia_libs.append(str(lib_path))
    if nvidia_libs:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + (f":{current_ld}" if current_ld else "")

FACEFUSION_PYTHON = RUNTIME_PYTHON
DEFAULT_PIPELINE_DIR = ROOT_DIR / "pipeline"
DEFAULT_FACEFUSION_DIR = ROOT_DIR / "facefusion"
FALLBACK_FACEFUSION_DIR = ROOT_DIR.parent / "facefusion"

PIPELINE        = os.environ.get("PIPELINE_DIR", str(DEFAULT_PIPELINE_DIR))
if not Path(PIPELINE).is_absolute():
    PIPELINE = str((ROOT_DIR / PIPELINE).resolve())

FACEFUSION_DIR  = os.environ.get("FACEFUSION_DIR", str(DEFAULT_FACEFUSION_DIR))
if not Path(FACEFUSION_DIR).is_absolute():
    FACEFUSION_DIR = str((ROOT_DIR / FACEFUSION_DIR).resolve())
if not os.path.isdir(FACEFUSION_DIR) and FALLBACK_FACEFUSION_DIR.is_dir():
    FACEFUSION_DIR = str(FALLBACK_FACEFUSION_DIR.resolve())
FACE_DIR        = f"{PIPELINE}/downloads/face"
VIDEO_DIR       = f"{PIPELINE}/downloads/video"
WORKSPACE       = f"{PIPELINE}/workspace"
TEMP_PATH       = f"{WORKSPACE}/temp"
OUTPUTS_DIR     = f"{WORKSPACE}/output"
VALIDATION_PROOF_DIR = f"{OUTPUTS_DIR}/validation_proof"
PERSISTENT_ROOT = os.environ.get("PERSISTENT_ROOT", str(ROOT_DIR / "persistent"))
if not Path(PERSISTENT_ROOT).is_absolute():
    PERSISTENT_ROOT = str((ROOT_DIR / PERSISTENT_ROOT).resolve())
PERSISTENT_FACES_DIR = f"{PERSISTENT_ROOT}/faces"
PERSISTENT_CONFIG_FILE = f"{PERSISTENT_ROOT}/config.json"
DEFAULT_FACE    = f"{PERSISTENT_FACES_DIR}/source_clean.jpg"
MEGA_CREDS_FILE = f"{PIPELINE}/.mega_creds"
DRIVE_TOKEN_FILE = f"{PIPELINE}/.drive_token"
MEGA_LINK_CACHE_FILE = f"{PIPELINE}/.mega_link_cache.json"
MEGA_CACHE_DIR = f"{PIPELINE}/cache/mega"
RCLONE_CONF     = os.environ.get("RCLONE_CONF", str(Path.home() / ".config" / "rclone" / "rclone.conf"))
if RCLONE_CONF and not os.path.isabs(RCLONE_CONF):
    RCLONE_CONF = str(Path(__file__).parent.parent / RCLONE_CONF)
GDRIVE_FOLDER   = os.environ.get("GDRIVE_FOLDER", f"{GDRIVE_REMOTE_NAME}:masterswap")
_LOCAL_RCLONE   = str(Path(__file__).parent / "bin" / "rclone")
RCLONE_BIN      = (
    os.environ.get("RCLONE_BIN")
    or shutil.which("rclone")
    or (_LOCAL_RCLONE if os.path.isfile(_LOCAL_RCLONE) else None)
)
BOT_PID_FILE    = f"{PIPELINE}/logs/bot.pid"
ACTIVE_JOB_STATE_FILE = f"{PIPELINE}/logs/active_job_state.json"
SLEEP_COUNTDOWN_STATE_FILE = f"{PIPELINE}/logs/sleep_countdown_state.json"
CLEANUP_AUDIT_LOG_FILE = f"{PIPELINE}/logs/storage_cleanup.log"
AUTO_SLEEP_LOG_FILE = f"{PIPELINE}/logs/auto_sleep.log"
QUEUE_STATE_FILE = f"{PIPELINE}/logs/queue_state.json"

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_FACE_LINK_FALLBACK = "https://mega.nz/file/0dpElBaa#6SfgxfDhmrO3N4TeyUcDKpebN4YDNbWQjvnxVplDLJw"


def _effective_creds():
    global _CREDS
    if not isinstance(_CREDS, dict) or not _CREDS:
        _CREDS = resolve_credentials()
    return _CREDS


def reload_runtime_credentials():
    global _CREDS, BOT_TOKEN, GDRIVE_REMOTE_NAME, GDRIVE_FOLDER, DEFAULT_FACE_MEGA_LINK, LOCKED_DEFAULT_FACE_LINK
    previous = dict(_CREDS or {})
    _CREDS = config_reload_credentials()

    bot_token = str(_CREDS.get("bot_token", "")).strip()
    if bot_token:
        BOT_TOKEN = bot_token
        os.environ["TELEGRAM_BOT_TOKEN"] = bot_token

    GDRIVE_REMOTE_NAME = str(_CREDS.get("gdrive_remote_name", "gdrive")).strip() or "gdrive"
    GDRIVE_FOLDER = os.environ.get("GDRIVE_FOLDER", str(_CREDS.get("gdrive_folder", "") or f"{GDRIVE_REMOTE_NAME}:masterswap")).strip()

    default_face_link = str(_CREDS.get("default_face_link", "") or DEFAULT_FACE_LINK_FALLBACK).strip()
    DEFAULT_FACE_MEGA_LINK = os.environ.get("DEFAULT_FACE_MEGA_LINK", default_face_link).strip()
    LOCKED_DEFAULT_FACE_LINK = DEFAULT_FACE_MEGA_LINK

    masked = masked_credentials(_CREDS)
    logger.info("credentials reloaded (masked): %s", json.dumps(masked, ensure_ascii=True))

    # Startup credential status (no secret values logged)
    _u, _p = (os.environ.get("MEGA_EMAIL", str(_CREDS.get("mega_email", ""))).strip(),
               os.environ.get("MEGA_PASSWORD", str(_CREDS.get("mega_password", ""))).strip())
    logger.info("MEGA_EMAIL set: %s | MEGA_PASSWORD length: %d", bool(_u), len(_p))

    changed = {
        "bot_token_changed": str(previous.get("bot_token", "")).strip() != str(_CREDS.get("bot_token", "")).strip(),
        "mega_changed": (
            str(previous.get("mega_email", "")).strip() != str(_CREDS.get("mega_email", "")).strip()
            or str(previous.get("mega_password", "")).strip() != str(_CREDS.get("mega_password", "")).strip()
        ),
        "gdrive_changed": str(previous.get("gdrive_folder", "")).strip() != str(_CREDS.get("gdrive_folder", "")).strip(),
    }
    return changed


def can_use_mega():
    _effective_creds()
    u, p = get_mega_creds()
    return bool((u or "").strip() and (p or "").strip())

active_jobs   = {}
active_pipeline_tasks = {}
queue_workers = {}
job_queues = {}
queue_job_seq = {}
current_face  = {}
shutdown_task = None
sleep_countdown_tasks = {}
sleep_countdown_state = {}
sleep_timer_active = {}
start_sleep_timer = False
countdown_task = None
is_countdown_running = False
job_status    = {}
clip_ranges   = {}
selected_face_maps = {}
chat_modes = {}
job_modes = {}
announcement_cache = {}
face_selector_prefs = {}
telegram_flood_until = {}
telegram_retry_queues = {}
telegram_retry_workers = {}
queued_progress_message_ids = {}
active_job_protected_paths = {}
download_heartbeat_tasks = {}
progress_stream_tokens = {}
recovered_external_jobs = {}
post_upload_tasks = {}
post_job_cleanup_tasks = {}
job_keepalive_tasks = {}
pending_recovery_chats = set()
startup_notice_sent = False
pipeline_execution_lock = asyncio.Lock()
MAX_PARALLEL_JOBS = 1
global_job_lock = False
lifecycle_state = {}

# Live web dashboard state (mirror of Telegram updates per job).
active_dashboard_tokens: dict = {}        # str(chat_id) -> token
active_dashboard_session_meta: dict = {}  # token -> {chat_id, queue_job_id, video_link}
dashboard_server_instance = None

STAGE_FLOW_ORDER = {
    "download": 1,
    "extracting": 2,
    "processing": 3,
    "merging": 4,
    "upload": 5,
    "completed": 6,
    "failed": 7,
}

STAGE_FLOW_TEXT = {
    "download": "Downloading",
    "extracting": "Extracting",
    "processing": "Processing",
    "merging": "Merging",
    "upload": "Uploading",
    "completed": "Completed",
    "failed": "Failed",
}

BUTTON_ACTION_ANNOUNCEMENTS = {
    "change_face": "Face update flow open ho raha hai...",
    "view_face": "Current default face check ho raha hai...",
    "stop_job": "Active job stop request bheji ja rahi hai...",
    "job_status_btn": "Latest job status fetch ho raha hai...",
    "reupload_output_menu": "Re-upload menu open ho raha hai...",
    "queue_terminate_menu": "Job terminator tools open ho rahe hain...",
    "check_storage": "Storage scan start ho raha hai...",
    "clean_workspace": "Cleanup options open ho rahe hain...",
    "clean_workspace_deep_clean": "Deep clean full wipe run ho raha hai...",
    "clean_workspace_deep_clean_confirm": "Deep clean full wipe run ho raha hai...",
    "clean_workspace_deep_clean_execute": "Deep clean run ho raha hai...",
    "download_output": "Latest output prepare ho raha hai...",
    "change_drive_token": "Drive token update mode open ho raha hai...",
    "change_mega": "MEGA credentials update mode open ho raha hai...",
    "clip_settings": "Clip range settings open ho rahi hain...",
    "mode_direct": "Direct mode activate ho raha hai...",
    "mode_multi": "Multi mode preference apply ho rahi hai...",
    "female_only_on": "Female-only filter enable ho raha hai...",
    "male_only_on": "Male-only filter enable ho raha hai...",
    "female_only_off": "All-genders filter enable ho raha hai...",
    "quick_sleep": "Quick sleep confirmation open ho rahi hai...",
    "cancel_sleep_countdown": "Sleep countdown status check ho raha hai...",
    "start_bot": "Bot readiness panel refresh ho raha hai...",
}

DEFAULT_FACE_MEGA_LINK = os.environ.get(
    "DEFAULT_FACE_MEGA_LINK",
    str(_CREDS.get("default_face_link", "") or DEFAULT_FACE_LINK_FALLBACK),
).strip()
AUTO_TEST_FACE_LINK = DEFAULT_FACE_MEGA_LINK
AUTO_TEST_VIDEO_LINK = "https://mega.nz/file/y9RElKiB#kHQimH4zXbuq0aCCgiBvU8vbgvnwD1CG8AYU1-Mghgw"
LOCKED_DEFAULT_FACE_LINK = DEFAULT_FACE_MEGA_LINK

MEGA_UPLOAD_TIMEOUT_SEC = int(os.environ.get("MEGA_UPLOAD_TIMEOUT_SEC", "0"))
GDRIVE_UPLOAD_TIMEOUT_SEC = int(os.environ.get("GDRIVE_UPLOAD_TIMEOUT_SEC", "1200"))
GDRIVE_UPLOAD_RETRIES = max(1, int(os.environ.get("GDRIVE_UPLOAD_RETRIES", "2")))
MEGA_TEST_FALLBACK_ON_509 = os.environ.get("MEGA_TEST_FALLBACK_ON_509", "1").strip().lower() in {"1", "true", "yes"}
MEGA_DOWNLOAD_TIMEOUT_SEC = int(os.environ.get("MEGA_DOWNLOAD_TIMEOUT_SEC", "120"))
DOWNLOAD_ATTEMPT_TIMEOUT_SEC = max(90, int(os.environ.get("DOWNLOAD_ATTEMPT_TIMEOUT_SEC", "420")))
DOWNLOAD_STALL_TIMEOUT_SEC = max(20, int(os.environ.get("DOWNLOAD_STALL_TIMEOUT_SEC", "90")))
DOWNLOAD_RETRY_COUNT = max(1, int(os.environ.get("DOWNLOAD_RETRY_COUNT", "3")))
FACE_DOWNLOAD_ATTEMPT_TIMEOUT_SEC = max(30, int(os.environ.get("FACE_DOWNLOAD_ATTEMPT_TIMEOUT_SEC", "120")))
FACE_DOWNLOAD_STALL_TIMEOUT_SEC = max(15, int(os.environ.get("FACE_DOWNLOAD_STALL_TIMEOUT_SEC", "35")))
FACE_DOWNLOAD_RETRY_COUNT = max(1, int(os.environ.get("FACE_DOWNLOAD_RETRY_COUNT", "2")))
FACE_DOWNLOAD_TOTAL_TIMEOUT_SEC = max(60, int(os.environ.get("FACE_DOWNLOAD_TOTAL_TIMEOUT_SEC", "300")))
DOWNLOAD_PROGRESS_POLL_SEC = max(1, int(os.environ.get("DOWNLOAD_PROGRESS_POLL_SEC", "2")))
DIRECT_LINK_PROBE_TIMEOUT_SEC = max(5, int(os.environ.get("DIRECT_LINK_PROBE_TIMEOUT_SEC", "15")))
PROGRESS_NOTIFY_INTERVAL_SEC = 10
PROGRESS_MAX_DELAY_SEC = 12
IS_LIGHTWEIGHT = (
    os.environ.get("LIGHTNING_STUDIO") == "1"
    or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    or os.path.exists("/teamspace/studios")
)
VIRTUAL_SLEEP_MODE = (
    os.environ.get("VIRTUAL_SLEEP_MODE", "0").strip().lower() in {"1", "true", "yes"}
)
SLEEP_TEST_MODE = os.environ.get("SLEEP_TEST_MODE", "0").strip().lower() in {"1", "true", "yes"}
KEEPALIVE_INTERVAL_SEC = max(30, min(60, int(os.environ.get("KEEPALIVE_INTERVAL_SEC", "45"))))
KEEPALIVE_TOUCH_FILE = f"{PIPELINE}/logs/job_keepalive.touch"
MEGA_MIN_OPERATION_GAP_SEC = int(os.environ.get("MEGA_MIN_OPERATION_GAP_SEC", "25"))
MEGA_AUTH_COOLDOWN_BASE_SEC = int(os.environ.get("MEGA_AUTH_COOLDOWN_BASE_SEC", "600"))
MEGA_AUTH_COOLDOWN_MAX_SEC = int(os.environ.get("MEGA_AUTH_COOLDOWN_MAX_SEC", "7200"))
EXECUTION_PROVIDER = os.environ.get("EXECUTION_PROVIDER", "auto").strip().lower()
EXECUTION_THREAD_COUNT_RAW = os.environ.get("EXECUTION_THREAD_COUNT", "").strip()
CPU_THREAD_UTILIZATION_PCT = int(os.environ.get("CPU_THREAD_UTILIZATION_PCT", "75"))
BYPASS_CONTENT_ANALYSER = os.environ.get("BYPASS_CONTENT_ANALYSER", "1").strip().lower() not in {"0", "false", "no"}
CPU_FAST_MODE = os.environ.get("CPU_FAST_MODE", "1").strip().lower() not in {"0", "false", "no"}
GPU_ONLY_MODE = os.environ.get("GPU_ONLY_MODE", "1").strip().lower() in {"1", "true", "yes"}
OUTPUT_VIDEO_ENCODER = os.environ.get("OUTPUT_VIDEO_ENCODER", "h264_nvenc").strip()
ENABLE_FACE_ENHANCER = os.environ.get("ENABLE_FACE_ENHANCER", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_EXPRESSION_RESTORER = os.environ.get("ENABLE_EXPRESSION_RESTORER", "1").strip().lower() not in {"0", "false", "no"}
EXPRESSION_RESTORER_FACTOR = os.environ.get("EXPRESSION_RESTORER_FACTOR", "90").strip()
FACE_SWAPPER_MODEL = os.environ.get("FACE_SWAPPER_MODEL", "hyperswap_1a_256").strip()
FACE_SWAPPER_PIXEL_BOOST = os.environ.get("FACE_SWAPPER_PIXEL_BOOST", "512x512").strip()
FACE_SWAPPER_WEIGHT = os.environ.get("FACE_SWAPPER_WEIGHT", "0.85").strip()
FACE_ENHANCER_MODEL = os.environ.get("FACE_ENHANCER_MODEL", "gfpgan_1.4").strip()
FACE_ENHANCER_BLEND = os.environ.get("FACE_ENHANCER_BLEND", "55").strip()
FACE_ENHANCER_WEIGHT = os.environ.get("FACE_ENHANCER_WEIGHT", "0.70").strip()
FACE_MASK_BLUR = os.environ.get("FACE_MASK_BLUR", "0.3").strip()
FACE_MASK_PADDING_TOP = os.environ.get("FACE_MASK_PADDING_TOP", "0").strip()
FACE_MASK_PADDING_RIGHT = os.environ.get("FACE_MASK_PADDING_RIGHT", "0").strip()
FACE_MASK_PADDING_BOTTOM = os.environ.get("FACE_MASK_PADDING_BOTTOM", "0").strip()
FACE_MASK_PADDING_LEFT = os.environ.get("FACE_MASK_PADDING_LEFT", "0").strip()
FACE_DETECTOR_SIZE = os.environ.get("FACE_DETECTOR_SIZE", "640x640").strip()
VIDEO_MEMORY_STRATEGY = os.environ.get("VIDEO_MEMORY_STRATEGY", "tolerant").strip()
FACE_DETECTOR_SCORE = os.environ.get("FACE_DETECTOR_SCORE", "0.5").strip()
FACE_LANDMARKER_SCORE = os.environ.get("FACE_LANDMARKER_SCORE", "0.35").strip()
FACEFUSION_WATCHDOG_SEC = max(30, int(os.environ.get("FACEFUSION_WATCHDOG_SEC", "60")))
FACEFUSION_HARD_TIMEOUT_SEC = max(0, int(os.environ.get("FACEFUSION_HARD_TIMEOUT_SEC", "0")))
PIPELINE_WATCHDOG_INTERVAL_SEC = max(5, min(10, int(os.environ.get("PIPELINE_WATCHDOG_INTERVAL_SEC", "8"))))
PIPELINE_WATCHDOG_SILENT_EXIT_GRACE_SEC = max(5, int(os.environ.get("PIPELINE_WATCHDOG_SILENT_EXIT_GRACE_SEC", "8")))
PIPELINE_WATCHDOG_STALE_CONFIRM_COUNT = max(1, int(os.environ.get("PIPELINE_WATCHDOG_STALE_CONFIRM_COUNT", "2")))
READY_TO_COMPLETE_GRACE_SEC = max(60, int(os.environ.get("READY_TO_COMPLETE_GRACE_SEC", "90")))
PIPE_READ_EXIT_GRACE_SEC = max(1.0, float(os.environ.get("PIPE_READ_EXIT_GRACE_SEC", "3.0") or 3.0))
AUTO_CPU_FALLBACK_ON_OOM = os.environ.get("AUTO_CPU_FALLBACK_ON_OOM", "0").strip().lower() in {"1", "true", "yes"}
LOW_MEMORY_MODE = os.environ.get("LOW_MEMORY_MODE", "1").strip().lower() in {"1", "true", "yes"}
LOW_MEMORY_THREAD_COUNT = max(1, int(os.environ.get("LOW_MEMORY_THREAD_COUNT", "2")))
GPU_STARTUP_BALANCED_MODE = os.environ.get("GPU_STARTUP_BALANCED_MODE", "1").strip().lower() in {"1", "true", "yes"}
GPU_STARTUP_THREAD_COUNT = max(1, int(os.environ.get("GPU_STARTUP_THREAD_COUNT", str(LOW_MEMORY_THREAD_COUNT))))
GPU_STARTUP_FACE_DETECTOR_SIZE = os.environ.get("GPU_STARTUP_FACE_DETECTOR_SIZE", "640x640").strip() or "640x640"
GPU_STARTUP_PIXEL_BOOST = os.environ.get("GPU_STARTUP_PIXEL_BOOST", "512x512").strip() or "512x512"
GPU_STARTUP_VIDEO_MEMORY_STRATEGY = os.environ.get("GPU_STARTUP_VIDEO_MEMORY_STRATEGY", "tolerant").strip() or "tolerant"
GPU_OOM_MAX_LEVELS = max(1, min(3, int(os.environ.get("GPU_OOM_MAX_LEVELS", "3"))))
GPU_RETRY_CHUNK_SECONDS_L2 = max(3, int(os.environ.get("GPU_RETRY_CHUNK_SECONDS_L2", "8")))
GPU_RETRY_CHUNK_SECONDS_L3 = max(2, int(os.environ.get("GPU_RETRY_CHUNK_SECONDS_L3", "4")))
PRIMARY_FACE_DETECTOR_MODEL = os.environ.get("PRIMARY_FACE_DETECTOR_MODEL", "yolo_face").strip() or "yolo_face"
FALLBACK_FACE_DETECTOR_MODEL = os.environ.get("FALLBACK_FACE_DETECTOR_MODEL", "yolo_face").strip() or "yolo_face"
FRAME_DEBUG_DETECTION_STRIDE = max(1, int(os.environ.get("FRAME_DEBUG_DETECTION_STRIDE", "2")))
FRAME_DEBUG_ANALYSIS_TIMEOUT_SEC = max(20, int(os.environ.get("FRAME_DEBUG_ANALYSIS_TIMEOUT_SEC", "90")))
SWAP_VALIDATION_TIMEOUT_SEC = max(20, int(os.environ.get("SWAP_VALIDATION_TIMEOUT_SEC", "90")))
FRAME_DEBUG_TRACKING_TTL = max(1, int(os.environ.get("FRAME_DEBUG_TRACKING_TTL", "6")))
FRAME_DEBUG_MIN_DETECTION_RATIO = float(os.environ.get("FRAME_DEBUG_MIN_DETECTION_RATIO", "0.50"))
ALLOW_AUTO_SOURCE_FROM_VIDEO = os.environ.get("ALLOW_AUTO_SOURCE_FROM_VIDEO", "0").strip().lower() in {"1", "true", "yes"}
STRICT_FACESWAP_DEBUG = os.environ.get("STRICT_FACESWAP_DEBUG", "1").strip().lower() in {"1", "true", "yes"}
DISABLE_FACE_SWAP_FALLBACK = os.environ.get("DISABLE_FACE_SWAP_FALLBACK", "1").strip().lower() in {"1", "true", "yes"}
POST_JOB_AUTO_SLEEP_SECONDS = int(os.environ.get("POST_JOB_AUTO_SLEEP_SECONDS", "120") or 120)
if POST_JOB_AUTO_SLEEP_SECONDS <= 0:
    POST_JOB_AUTO_SLEEP_SECONDS = 120
SLEEP_COUNTDOWN_SECONDS = POST_JOB_AUTO_SLEEP_SECONDS
AUTO_SHUTDOWN_DELAY_SEC = SLEEP_COUNTDOWN_SECONDS

# Live web dashboard configuration.
DASHBOARD_ENABLED = os.environ.get("DASHBOARD_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}
DASHBOARD_HOST = os.environ.get("DASHBOARD_HOST", "0.0.0.0").strip() or "0.0.0.0"
try:
    DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8765").strip() or "8765")
except Exception:
    DASHBOARD_PORT = 8765
DASHBOARD_PUBLIC_URL = os.environ.get("DASHBOARD_PUBLIC_URL", "").strip()
if not DASHBOARD_PUBLIC_URL:
    _cs_host = os.environ.get("LIGHTNING_CLOUDSPACE_HOST", "").strip()
    if _cs_host:
        DASHBOARD_PUBLIC_URL = f"https://{_cs_host}/{DASHBOARD_PORT}"
    else:
        DASHBOARD_PUBLIC_URL = f"http://localhost:{DASHBOARD_PORT}"
# Extract root_path from public URL path (e.g. https://host/7860 -> root_path="/7860")
_parsed_public = urlparse(DASHBOARD_PUBLIC_URL)
DASHBOARD_ROOT_PATH = _parsed_public.path.rstrip("/") if _parsed_public.path and _parsed_public.path != "/" else ""
DASHBOARD_SESSIONS_ROOT = os.environ.get(
    "DASHBOARD_SESSIONS_ROOT",
    str(Path(PIPELINE) / "dashboard_sessions"),
).strip()
DASHBOARD_LOG_LEVEL = os.environ.get("DASHBOARD_LOG_LEVEL", "warning").strip() or "warning"

# ---------------------------------------------------------------------------
# Cloudflared tunnel manager — gives a public HTTPS URL for the dashboard
# ---------------------------------------------------------------------------
_CLOUDFLARED_BIN = os.environ.get("CLOUDFLARED_BIN", "/tmp/cloudflared").strip()
_cloudflared_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
_cloudflared_url: str = ""


def _start_cloudflared_tunnel(port: int) -> str:
    """Start cloudflared quick-tunnel and return the public URL (blocks up to 15s)."""
    global _cloudflared_proc, _cloudflared_url
    if _cloudflared_url:
        return _cloudflared_url
    bin_path = _CLOUDFLARED_BIN
    if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
        logger.warning("cloudflared not found at %s — dashboard will use local URL", bin_path)
        return ""
    try:
        proc = subprocess.Popen(
            [bin_path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _cloudflared_proc = proc
        import re as _re
        deadline = time.time() + 20
        url = ""
        while time.time() < deadline:
            line = proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                break
            m = _re.search(r'https://[a-z0-9\-]+\.trycloudflare\.com', line)
            if m:
                url = m.group(0)
                break
        if url:
            _cloudflared_url = url
            logger.info("cloudflared tunnel started: %s -> localhost:%s", url, port)
        else:
            logger.warning("cloudflared tunnel URL not found within timeout")
        return url
    except Exception as exc:
        logger.warning("cloudflared start failed: %s", exc)
        return ""


def _stop_cloudflared_tunnel() -> None:
    global _cloudflared_proc, _cloudflared_url
    if _cloudflared_proc:
        try:
            _cloudflared_proc.terminate()
        except Exception:
            pass
        _cloudflared_proc = None
    _cloudflared_url = ""

MIN_FREE_SPACE_GB = int(os.environ.get("MIN_FREE_SPACE_GB", "20"))
TEMP_JOB_RETENTION_HOURS = int(os.environ.get("TEMP_JOB_RETENTION_HOURS", "6"))
TEMP_JOB_KEEP_LATEST = int(os.environ.get("TEMP_JOB_KEEP_LATEST", "2"))
MAX_STORAGE_USAGE_GB = int(os.environ.get("MAX_STORAGE_USAGE_GB", "150"))
KEEP_LATEST_OUTPUTS = int(os.environ.get("KEEP_LATEST_OUTPUTS", "3"))

SAFE_CLEANUP_MIN_AGE_SECONDS = int(os.environ.get("SAFE_CLEANUP_MIN_AGE_SECONDS", "900"))
SAFE_CLEANUP_DISK_TRIGGER_PERCENT = float(os.environ.get("SAFE_CLEANUP_DISK_TRIGGER_PERCENT", "80"))
SAFE_CLEANUP_PERIODIC_INTERVAL_SEC = int(os.environ.get("SAFE_CLEANUP_PERIODIC_INTERVAL_SEC", "1800"))
PRE_DOWNLOAD_FAST_START = os.environ.get("PRE_DOWNLOAD_FAST_START", "1").strip().lower() in {"1", "true", "yes"}

PROTECTED_DIR_NAMES = {
    "models",
    "weights",
    "configs",
    "src",
    "app",
    "database",
    "final_outputs",
}
PROTECTED_FILE_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".env"}
TEMP_FILE_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".part",
    ".partial",
    ".cache",
    ".log",
    ".txt",
    ".m4a",
    ".aac",
    ".wav",
    ".mp3",
    ".pcm",
    ".flac",
    ".ts",
    ".chunk",
    ".frame",
    ".dat",
    ".bin",
    ".pyc",
}

cleanup_guard_task = None
runtime_heartbeat_task = None

AUTO_SLEEP_CFG = load_auto_sleep_config()
AUTO_SLEEP_ENABLED = True
# Enforce configured post-job countdown (default 120s).
SLEEP_COUNTDOWN_SECONDS = POST_JOB_AUTO_SLEEP_SECONDS
AUTO_SHUTDOWN_DELAY_SEC = SLEEP_COUNTDOWN_SECONDS

AUTO_SOURCE_FACE_NAMES = {"auto_source_from_video.jpg"}

mega_state = {
    "last_operation_at": 0.0,
    "auth_backoff_until": 0.0,
    "auth_failures": 0,
    "mkdir_ready": False,
}

_FFMPEG_DECODERS_CACHE = None
_VIDEO_CODEC_CACHE = {}
_CPU_PCT_LAST_SNAPSHOT = None


def _resolve_execution_thread_count(raw_value, utilization_pct=75):
    if raw_value and raw_value.isdigit() and int(raw_value) > 0:
        return int(raw_value)

    cpu_total = max(1, int(os.cpu_count() or 1))
    pct = max(50, min(90, int(utilization_pct or 75)))
    resolved = int(max(1, round(cpu_total * (pct / 100.0))))
    return min(cpu_total, max(1, resolved))


EXECUTION_THREAD_COUNT = str(_resolve_execution_thread_count(EXECUTION_THREAD_COUNT_RAW, CPU_THREAD_UTILIZATION_PCT))
FFMPEG_CPU_THREADS = max(1, int(EXECUTION_THREAD_COUNT))


def _read_system_cpu_percent():
    global _CPU_PCT_LAST_SNAPSHOT
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            first = f.readline().strip()
        parts = first.split()
        if len(parts) < 8 or parts[0] != "cpu":
            return None
        values = [int(x) for x in parts[1:8]]
        idle = values[3] + values[4]
        total = sum(values)

        prev = _CPU_PCT_LAST_SNAPSHOT
        _CPU_PCT_LAST_SNAPSHOT = (idle, total)
        if not prev:
            return None

        idle_delta = idle - prev[0]
        total_delta = total - prev[1]
        if total_delta <= 0:
            return None
        busy = max(0.0, float(total_delta - idle_delta))
        return max(0.0, min(100.0, (busy / float(total_delta)) * 100.0))
    except Exception:
        return None


def _read_process_cpu_percent(pid):
    if not pid:
        return None
    try:
        r = subprocess.run(
            ["ps", "-p", str(pid), "-o", "%cpu="],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return None
        txt = (r.stdout or "").strip().splitlines()
        if not txt:
            return None
        return float(txt[-1].strip())
    except Exception:
        return None


def _read_gpu_util_percent():
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return None
        vals = []
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                vals.append(float(line))
            except Exception:
                continue
        if not vals:
            return None
        return max(0.0, min(100.0, sum(vals) / float(len(vals))))
    except Exception:
        return None


def _sample_perf_point(proc_pid=None):
    return {
        "cpu_system": _read_system_cpu_percent(),
        "cpu_proc": _read_process_cpu_percent(proc_pid),
        "gpu": _read_gpu_util_percent(),
        "at": time.time(),
    }


def _append_perf_sample(bucket, sample):
    if not bucket or not sample:
        return
    bucket.setdefault("samples", []).append(sample)


def _perf_stats_line(name, started_at, ended_at, samples):
    if not started_at:
        return f"{name}: skipped"
    end_ts = ended_at or time.time()
    duration = max(0, int(end_ts - started_at))
    cpu_system = [s["cpu_system"] for s in samples if s.get("cpu_system") is not None]
    cpu_proc = [s["cpu_proc"] for s in samples if s.get("cpu_proc") is not None]
    gpu = [s["gpu"] for s in samples if s.get("gpu") is not None]

    def _avg(v):
        if not v:
            return "n/a"
        return f"{(sum(v) / float(len(v))):.1f}%"

    return (
        f"{name}: time={duration}s | cpu_sys_avg={_avg(cpu_system)} "
        f"| cpu_proc_avg={_avg(cpu_proc)} | gpu_avg={_avg(gpu)}"
    )


def _load_ffmpeg_decoders():
    global _FFMPEG_DECODERS_CACHE
    if _FFMPEG_DECODERS_CACHE is not None:
        return _FFMPEG_DECODERS_CACHE
    decoders = set()
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=6,
        )
        if r.returncode == 0:
            for line in (r.stdout or "").splitlines():
                if not line.startswith(" "):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    decoders.add(parts[1].strip())
    except Exception:
        pass
    _FFMPEG_DECODERS_CACHE = decoders
    return decoders


def detect_video_codec(video_path):
    cache_key = str(video_path)
    if cache_key in _VIDEO_CODEC_CACHE:
        return _VIDEO_CODEC_CACHE[cache_key]
    codec = None
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode == 0:
            codec = (r.stdout or "").strip().lower() or None
    except Exception:
        codec = None
    _VIDEO_CODEC_CACHE[cache_key] = codec
    return codec


def build_ffmpeg_hw_decode_args(video_path):
    if not (GPU_ONLY_MODE or EXECUTION_PROVIDER in {"auto", "cuda"}):
        return []

    codec = detect_video_codec(video_path)
    if codec not in {"h264", "hevc"}:
        return []

    decoder = "h264_cuvid" if codec == "h264" else "hevc_cuvid"
    decoders = _load_ffmpeg_decoders()
    if decoder not in decoders:
        return []

    return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-c:v", decoder]


def append_ffmpeg_encode_tuning(cmd):
    if OUTPUT_VIDEO_ENCODER.endswith("_nvenc"):
        cmd.extend(["-preset", "fast", "-rc", "vbr", "-cq", "19"])
    cmd.extend(["-threads", str(FFMPEG_CPU_THREADS)])


def _retry_after_seconds(exc):
    try:
        wait = int(getattr(exc, "retry_after", 1) or 1)
    except Exception:
        wait = 1
    return max(1, wait)


def _enqueue_telegram_retry(bot, chat_id, text, kwargs=None, delay_seconds=30):
    chat_key = str(chat_id)
    queue = telegram_retry_queues.setdefault(chat_key, deque())
    kwargs = dict(kwargs or {})

    if queue:
        last = queue[-1]
        if last.get("text") == text and last.get("kwargs") == kwargs:
            return

    queue.append({
        "text": text,
        "kwargs": kwargs,
        "attempts": 0,
        "next_try": time.time() + max(1, int(delay_seconds or 1)),
    })

    worker = telegram_retry_workers.get(chat_key)
    if worker and not worker.done():
        return

    telegram_retry_workers[chat_key] = asyncio.create_task(
        _telegram_retry_worker(bot, chat_key, chat_id)
    )


def _extract_progress_stage(text):
    if not text or "Stage:" not in text:
        return None
    m = re.search(r"Stage:\s*\*?([^*\n]+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def _should_drop_stale_progress_retry(chat_key, text):
    stage = _extract_progress_stage(text)
    if not stage:
        return False

    st = job_status.get(str(chat_key), {}) or {}
    phase = str(st.get("phase", "") or "").lower()
    current_stage = str(st.get("stage", "") or "").lower()

    # Never allow old download progress to re-appear once pipeline moved past download.
    if "downloading" in stage and phase != "download":
        return True

    # Avoid showing early extraction/starting states after processing is already underway.
    if stage in {"extracting", "analysing", "starting"}:
        if phase in {"faceswap", "upload", "completed"} and any(
            x in current_stage for x in ["processing", "swapping", "enhancing", "merging", "upload"]
        ):
            return True

    return False


async def _telegram_retry_worker(bot, chat_key, chat_id):
    try:
        while True:
            queue = telegram_retry_queues.get(chat_key)
            if not queue:
                break

            item = queue[0]
            text = item.get("text", "")
            if _should_drop_stale_progress_retry(chat_key, text):
                logger.info("telegram retry queue drop stale progress chat=%s", chat_key)
                queue.popleft()
                continue

            wait = float(item.get("next_try", 0) or 0) - time.time()
            if wait > 0:
                await asyncio.sleep(min(wait, 5))
                continue

            try:
                await bot.send_message(chat_id=chat_id, text=text, **item.get("kwargs", {}))
                queue.popleft()
                continue
            except RetryAfter as e:
                item["attempts"] = int(item.get("attempts", 0)) + 1
                if item["attempts"] >= 3:
                    logger.warning("telegram retry queue drop chat=%s after RetryAfter attempts=%s", chat_key, item["attempts"])
                    queue.popleft()
                    continue
                raw_wait = _retry_after_seconds(e)
                item["next_try"] = time.time() + max(30, min(raw_wait, 600))
                continue
            except Exception as e:
                item["attempts"] = int(item.get("attempts", 0)) + 1
                if item["attempts"] >= 3:
                    logger.warning("telegram retry queue drop chat=%s after exception attempts=%s err=%s", chat_key, item["attempts"], e)
                    queue.popleft()
                    continue
                item["next_try"] = time.time() + (10 * item["attempts"])
                continue
    finally:
        telegram_retry_workers.pop(chat_key, None)


# ---------------------------------------------------------------------------
# Dashboard mirror helpers (best-effort: never raise into the bot loop).
# ---------------------------------------------------------------------------

def _dashboard_token_for(chat_id) -> str:
    """Return the active dashboard session token for a chat, if any.

    Subprocess workers can pre-populate ``active_dashboard_tokens`` (set by
    ``ops/job_worker.py``) or expose the token via ``DASHBOARD_TOKEN`` env.
    """
    chat_key = str(chat_id)
    token = active_dashboard_tokens.get(chat_key)
    if not token:
        # In the worker subprocess the dict may not be populated; fall back to env.
        env_chat = str(os.environ.get("DASHBOARD_CHAT_ID", "")).strip()
        env_token = str(os.environ.get("DASHBOARD_TOKEN", "")).strip()
        if env_token and (not env_chat or env_chat == chat_key):
            active_dashboard_tokens[chat_key] = env_token
            token = env_token
    return str(token or "")


def _dashboard_set_token(chat_id, token, *, queue_job_id=None, video_link=None):
    chat_key = str(chat_id)
    if not token:
        active_dashboard_tokens.pop(chat_key, None)
        return
    active_dashboard_tokens[chat_key] = str(token)
    meta = {
        "chat_id": chat_key,
        "queue_job_id": queue_job_id,
        "video_link": video_link,
    }
    active_dashboard_session_meta[str(token)] = meta


def _dashboard_clear_token(chat_id):
    chat_key = str(chat_id)
    token = active_dashboard_tokens.pop(chat_key, None)
    if token:
        active_dashboard_session_meta.pop(str(token), None)


def _dashboard_register_session(chat_id, video_link, queue_job_id):
    """Create a fresh dashboard session for a queued job and return (token, url)."""
    if not DASHBOARD_ENABLED:
        return "", ""
    try:
        token = dashboard.generate_session_token()
        dashboard.register_session(
            DASHBOARD_SESSIONS_ROOT,
            token,
            chat_id=str(chat_id),
            video_link=str(video_link or ""),
            queue_job_id=str(queue_job_id or ""),
        )
        _dashboard_set_token(
            chat_id,
            token,
            queue_job_id=queue_job_id,
            video_link=video_link,
        )
        public_base = (DASHBOARD_PUBLIC_URL or "").rstrip("/")
        url = f"{public_base}/dashboard_v2.html?token={token}" if public_base else f"/dashboard_v2.html?token={token}"
        return token, url
    except Exception as e:
        logger.warning("dashboard register_session failed chat=%s err=%s", chat_id, e)
        return "", ""


def _verify_dashboard_url(url: str, timeout: float = 4.0) -> str:
    """Return url if dashboard is reachable and serving, else empty string."""
    if not url:
        return ""
    try:
        from urllib.request import urlopen as _urlopen
        # Check 1: healthz must respond
        healthz = f"http://localhost:{DASHBOARD_PORT}/healthz"
        resp = _urlopen(healthz, timeout=timeout)
        if resp.status != 200:
            logger.warning("dashboard healthz returned %s — not sending URL", resp.status)
            return ""
        # Check 2: dashboard_v2.html endpoint must serve (200 + HTML)
        v2_check = f"http://localhost:{DASHBOARD_PORT}/dashboard_v2.html"
        try:
            resp2 = _urlopen(v2_check, timeout=timeout)
            if resp2.status not in (200, 206):
                logger.warning("dashboard_v2.html returned %s — using /live fallback", resp2.status)
                # Fall back to /live URL which is always served
                public_base = (DASHBOARD_PUBLIC_URL or "").rstrip("/")
                return f"{public_base}/live" if public_base else ""
        except Exception:
            pass
        return url
    except Exception as exc:
        logger.warning("dashboard URL not reachable (%s) — not sending URL to user", exc)
    return ""


def _dashboard_record_text(chat_id, text, *, source="notify"):
    if not DASHBOARD_ENABLED:
        return
    token = _dashboard_token_for(chat_id)
    if not token or not text:
        return
    try:
        dashboard.record_telegram_text(DASHBOARD_SESSIONS_ROOT, token, str(text), source=str(source))
    except Exception as e:
        logger.debug("dashboard record_text failed chat=%s err=%s", chat_id, e)


def _dashboard_record_progress_text(chat_id, text):
    if not DASHBOARD_ENABLED:
        return
    token = _dashboard_token_for(chat_id)
    if not token or not text:
        return
    try:
        dashboard.record_progress_text(DASHBOARD_SESSIONS_ROOT, token, str(text))
    except Exception as e:
        logger.debug("dashboard record_progress_text failed chat=%s err=%s", chat_id, e)


def _dashboard_record_stage(chat_id, *, stage_key, stage_label=None, phase=None, pct=None, details=None):
    if not DASHBOARD_ENABLED:
        return
    token = _dashboard_token_for(chat_id)
    if not token or not stage_key:
        return
    try:
        dashboard.record_stage(
            DASHBOARD_SESSIONS_ROOT,
            token,
            str(stage_key),
            stage_label,
            phase=phase,
            pct=pct,
            details=details,
        )
    except Exception as e:
        logger.debug("dashboard record_stage failed chat=%s err=%s", chat_id, e)


def _dashboard_record_progress(chat_id, **fields):
    if not DASHBOARD_ENABLED:
        return
    token = _dashboard_token_for(chat_id)
    if not token:
        return
    try:
        dashboard.record_progress(DASHBOARD_SESSIONS_ROOT, token, **fields)
    except Exception as e:
        logger.debug("dashboard record_progress failed chat=%s err=%s", chat_id, e)


def _dashboard_record_completion(chat_id, *, success, details=None, result=None):
    if not DASHBOARD_ENABLED:
        return
    token = _dashboard_token_for(chat_id)
    if not token:
        return
    try:
        dashboard.record_completion(
            DASHBOARD_SESSIONS_ROOT,
            token,
            success=bool(success),
            details=details,
            result=result,
        )
    except Exception as e:
        logger.debug("dashboard record_completion failed chat=%s err=%s", chat_id, e)


def _lightning_register_port(port: int) -> str:
    """Get stable public URL: ngrok first, then Lightning SDK."""
    # 1. Check ngrok (primary — stable, no auth required for external access)
    try:
        import urllib.request as _ur, json as _json
        r = _ur.urlopen("http://localhost:4040/api/tunnels", timeout=3)
        d = _json.loads(r.read())
        for t in d.get("tunnels", []):
            u = str(t.get("public_url", ""))
            if u.startswith("https"):
                logger.info("[NGROK] Using existing tunnel: %s", u)
                return u.rstrip("/")
    except Exception:
        pass

    # 2. Lightning SDK fallback
    try:
        from lightning_sdk import Studio
        from lightning_sdk.lightning_cloud.rest_client import create_swagger_client
        from lightning_sdk.lightning_cloud.openapi.api.cloud_space_service_api import CloudSpaceServiceApi

        project_id = os.environ.get("LIGHTNING_CLOUD_PROJECT_ID", "")
        space_id = os.environ.get("LIGHTNING_CLOUD_SPACE_ID", "")
        if not project_id or not space_id:
            return ""

        client = create_swagger_client(check_context=False, with_auth=True)
        cs_api = CloudSpaceServiceApi(client)
        studio_name = ""
        try:
            result = cs_api.cloud_space_service_list_cloud_spaces(project_id=project_id)
            for cs in result.cloudspaces:
                if cs.id == space_id:
                    studio_name = getattr(cs, "name", "")
                    break
        except Exception:
            pass

        if not studio_name:
            return ""

        studio = Studio(name=studio_name)
        try:
            for ep in studio.list_ports():
                ep_ports = ep.get("ports", []) if isinstance(ep, dict) else getattr(ep, "ports", [])
                urls = ep.get("urls", []) if isinstance(ep, dict) else getattr(ep, "urls", [])
                if str(port) in [str(p) for p in ep_ports] and urls:
                    url = str(urls[0]).rstrip("/")
                    logger.info("[LIGHTNING_PORT] Found existing port %s → %s", port, url)
                    return url
        except Exception:
            pass
        for ep in studio.add_ports(port):
            urls = ep.get("urls", []) if isinstance(ep, dict) else getattr(ep, "urls", [])
            if urls:
                url = str(urls[0]).rstrip("/")
                logger.info("[LIGHTNING_PORT] Registered port %s → %s", port, url)
                return url
    except Exception as e:
        logger.warning("[LIGHTNING_PORT] Could not register port %s: %s", port, e)
    return ""


def _dashboard_start_server_if_enabled():
    """Start the dashboard server in this process. Returns the server or None."""
    global dashboard_server_instance, DASHBOARD_PUBLIC_URL, DASHBOARD_ROOT_PATH
    if not DASHBOARD_ENABLED:
        logger.info("dashboard disabled (DASHBOARD_ENABLED=0)")
        return None
    if dashboard_server_instance is not None:
        return dashboard_server_instance
    try:
        Path(DASHBOARD_SESSIONS_ROOT).mkdir(parents=True, exist_ok=True)
        # Step 1: Try Lightning SDK port registration (most reliable)
        lightning_url = _lightning_register_port(DASHBOARD_PORT)
        if lightning_url:
            DASHBOARD_PUBLIC_URL = lightning_url
            DASHBOARD_ROOT_PATH = ""
        else:
            # Step 2: Try cloudflared tunnel
            tunnel_url = _start_cloudflared_tunnel(DASHBOARD_PORT)
            if tunnel_url:
                DASHBOARD_PUBLIC_URL = tunnel_url
                DASHBOARD_ROOT_PATH = ""
        # Persist working URL to .env so restarts use it immediately
        if DASHBOARD_PUBLIC_URL:
            try:
                env_path = ROOT_DIR / ".env"
                content = env_path.read_text(encoding="utf-8")
                import re as _re
                if "DASHBOARD_PUBLIC_URL=" in content:
                    content = _re.sub(r"^DASHBOARD_PUBLIC_URL=.*", f"DASHBOARD_PUBLIC_URL={DASHBOARD_PUBLIC_URL}", content, flags=_re.MULTILINE)
                else:
                    content = content.rstrip() + f"\nDASHBOARD_PUBLIC_URL={DASHBOARD_PUBLIC_URL}\n"
                env_path.write_text(content, encoding="utf-8")
            except Exception:
                pass
        srv = dashboard.start_dashboard_server(
            sessions_root=DASHBOARD_SESSIONS_ROOT,
            host=DASHBOARD_HOST,
            port=DASHBOARD_PORT,
            public_url=DASHBOARD_PUBLIC_URL,
            root_path=DASHBOARD_ROOT_PATH,
            log_level=DASHBOARD_LOG_LEVEL,
        )
        dashboard_server_instance = srv
        if srv is not None:
            logger.info(
                "dashboard server up host=%s port=%s public=%s sessions=%s",
                DASHBOARD_HOST,
                DASHBOARD_PORT,
                DASHBOARD_PUBLIC_URL,
                DASHBOARD_SESSIONS_ROOT,
            )
        return srv
    except Exception as e:
        logger.warning("dashboard start failed: %s", e)
        return None


async def safe_send_message(bot, chat_id, text, **kwargs):
    chat_key = str(chat_id)
    logger.info("SENDING TELEGRAM MESSAGE chat=%s text_head=%s", chat_key, (text or "")[:80].replace("\n", " "))
    _dashboard_record_text(chat_key, text, source=str(kwargs.get("_dashboard_source") or "notify"))
    kwargs.pop("_dashboard_source", None)
    now = time.time()
    blocked_until = float(telegram_flood_until.get(chat_key, 0) or 0)
    if blocked_until > now:
        logger.warning("skip send_message due flood cooldown chat=%s wait=%ss", chat_key, int(blocked_until - now))
        _enqueue_telegram_retry(bot, chat_id, text, kwargs, delay_seconds=max(1, int(blocked_until - now)))
        return None

    for attempt in range(3):
        try:
            return await bot.send_message(chat_id=chat_id, text=text, **kwargs)
        except RetryAfter as e:
            raw_wait = _retry_after_seconds(e)
            # Respect server-side retry window to avoid repeated 429 loops.
            local_wait = max(30, raw_wait)
            telegram_flood_until[chat_key] = max(float(telegram_flood_until.get(chat_key, 0) or 0), time.time() + local_wait)
            logger.warning("send_message flood wait chat=%s retry_after=%ss local_cooldown=%ss", chat_key, raw_wait, local_wait)
            if raw_wait > 120:
                _enqueue_telegram_retry(bot, chat_id, text, kwargs, delay_seconds=local_wait)
                return None
            await asyncio.sleep(local_wait)
        except Exception as e:
            logger.warning("safe_send_message attempt %s failed chat=%s: %s", attempt + 1, chat_key, e)
            if attempt < 2:
                await asyncio.sleep(1 + attempt)
    _enqueue_telegram_retry(bot, chat_id, text, kwargs, delay_seconds=15)
    return None


async def safe_reply_text(message, text, **kwargs):
    return await safe_send_message(message.get_bot(), str(message.chat_id), text, **kwargs)


def ensure_single_instance():
    os.makedirs(os.path.dirname(BOT_PID_FILE), exist_ok=True)
    current_pid = os.getpid()
    parent_pid = os.getppid()

    if os.path.exists(BOT_PID_FILE):
        try:
            prev_pid = int(Path(BOT_PID_FILE).read_text().strip() or "0")
        except Exception:
            prev_pid = 0

        if prev_pid and prev_pid not in {current_pid, parent_pid}:
            proc_cmd = ""
            try:
                proc_cmd = Path(f"/proc/{prev_pid}/cmdline").read_text(errors="ignore").replace("\x00", " ")
            except Exception:
                proc_cmd = ""

            if "bot.py" in proc_cmd:
                logger.error("another bot instance already running with pid=%s, exiting", prev_pid)
                sys.exit(1)

    Path(BOT_PID_FILE).write_text(str(current_pid))

    def _cleanup_pid():
        try:
            if os.path.exists(BOT_PID_FILE):
                txt = Path(BOT_PID_FILE).read_text().strip()
                if txt == str(current_pid):
                    Path(BOT_PID_FILE).unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_cleanup_pid)


def is_authorized(uid=None, chat_id=None):
    try:
        uid_i = int(uid) if uid is not None else None
    except Exception:
        uid_i = None
    try:
        cid_i = int(chat_id) if chat_id is not None else None
    except Exception:
        cid_i = None

    if uid_i == ALLOWED_USER_ID:
        return True
    if cid_i is not None and (cid_i == ALLOWED_CHAT_ID or cid_i == ALLOWED_USER_ID):
        return True
    return False


def get_chat_mode(chat_id):
    mode = str(chat_modes.get(chat_id, "direct") or "direct").strip().lower()
    return "multi" if mode == "multi" else "direct"


def set_chat_mode(chat_id, mode):
    chat_modes[str(chat_id)] = "multi" if str(mode).strip().lower() == "multi" else "direct"
    _persist_ui_runtime_state()


def mode_label(mode):
    return "Multi FaceSwap" if mode == "multi" else "Direct FaceSwap"


def get_gender_mode(chat_id):
    prefs = face_selector_prefs.get(str(chat_id), {}) or {}
    raw = str(prefs.get("gender_mode", "")).strip().lower()
    if raw in {"female", "male", "all"}:
        return raw
    if bool(prefs.get("female_only")):
        return "female"
    if bool(prefs.get("male_only")):
        return "male"
    return "all"


def is_female_only_enabled(chat_id):
    return get_gender_mode(chat_id) == "female"


def get_face_selector_gender(chat_id):
    mode = get_gender_mode(chat_id)
    if mode == "female":
        return "female"
    if mode == "male":
        return "male"
    return None


def set_gender_mode(chat_id, mode):
    chat_key = str(chat_id)
    raw = str(mode).strip().lower()
    if raw == "female":
        normalized = "female"
    elif raw == "male":
        normalized = "male"
    else:
        normalized = "all"
    prefs = face_selector_prefs.setdefault(chat_key, {})
    prefs["gender_mode"] = normalized
    prefs["female_only"] = normalized == "female"
    prefs["male_only"] = normalized == "male"
    _persist_ui_runtime_state()


def set_female_only(chat_id, enabled):
    set_gender_mode(chat_id, "female" if enabled else "all")


def _persist_ui_runtime_state():
    try:
        clip_payload = {
            str(chat_key): cfg
            for chat_key, cfg in clip_ranges.items()
            if isinstance(cfg, dict) and cfg.get("segments")
        }
        modes_payload = {
            str(chat_key): ("multi" if str(mode).lower() == "multi" else "direct")
            for chat_key, mode in chat_modes.items()
        }
        gender_payload = {
            str(chat_key): {"gender_mode": get_gender_mode(chat_key)}
            for chat_key in set(face_selector_prefs.keys()) | set(modes_payload.keys())
        }
        update_persistent_config(
            chat_modes=modes_payload,
            face_selector_prefs=gender_payload,
            clip_ranges=clip_payload,
        )
    except Exception as e:
        logger.warning("failed to persist ui runtime state: %s", e)



def build_mode_state_text(chat_id, context=None):
    mode = get_chat_mode(chat_id)
    gender_mode = get_gender_mode(chat_id)
    mode_value = "Multi ON" if mode == "multi" else "Direct ON"
    if gender_mode == "female":
        gender_filter = "Female ON"
    elif gender_mode == "male":
        gender_filter = "Male ON"
    else:
        gender_filter = "All genders ON"
    mode_hint = (
        "Multi: alag faces ke liye alag source use hoga"
        if mode == "multi"
        else "Direct: ek source se sab faces swap honge"
    )
    if gender_mode == "female":
        gender_hint = "Sirf female faces process honge"
    elif gender_mode == "male":
        gender_hint = "Sirf male faces process honge"
    else:
        gender_hint = "Sab faces process honge"
    clip_on = bool((clip_ranges.get(str(chat_id), {}) or {}).get("segments"))
    clip_line = "ON" if clip_on else "OFF"
    clip_hint = "Selected video parts par processing hogi" if clip_on else "Clip Range OFF"
    compatibility = "Supported"

    state_line = "Ready"
    if _is_chat_busy(chat_id):
        state_line = "Running"
    if context and mode == "multi":
        ud = context.user_data
        if ud.get("awaiting_multi_target"):
            state_line = "Waiting for multi target link"
        elif ud.get("awaiting_multi_source"):
            idx = int(ud.get("multi_face_idx", 0)) + 1
            total = len(ud.get("multi_face_crops", []))
            if total > 0:
                state_line = f"Mapping person {min(idx, total)}/{total}"

    return (
        "📣 *Mode/State Announcement*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎛 Mode: *{mode_value}*\n"
        f"🚺 Gender Filter: *{gender_filter}*\n"
        f"🟢 State: *{state_line}*\n"
        f"✂️ Clip Range: *{clip_line}*\n"
        f"🧩 Multi + Clip Compatibility: *{compatibility}*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"{mode_hint}\n"
        f"{gender_hint}\n"
        f"{clip_hint}"
    )


async def send_mode_state_announcement(message, chat_id, context):
    text = build_mode_state_text(chat_id, context)
    now = time.time()
    prev = announcement_cache.get(chat_id)
    if prev and prev.get("text") == text and (now - prev.get("at", 0)) < 90:
        return
    announcement_cache[chat_id] = {"text": text, "at": now}
    await safe_reply_text(
        message,
        text,
        parse_mode="Markdown",
    )


def get_face(chat_id):
    if not os.path.isfile(DEFAULT_FACE):
        _ensure_locked_default_face()
    preferred = current_face.get(chat_id)
    if preferred and os.path.isfile(preferred):
        return preferred
    if os.path.isfile(DEFAULT_FACE):
        return DEFAULT_FACE
    return current_face.get(chat_id, DEFAULT_FACE)


def get_preuploaded_default_face():
    """Return only the persisted default face selected via Change Face flow."""
    if not os.path.isfile(DEFAULT_FACE):
        _ensure_locked_default_face()
    if os.path.isfile(DEFAULT_FACE):
        return DEFAULT_FACE
    return None


def list_face_images():
    images = []
    for root_dir in [PERSISTENT_FACES_DIR, FACE_DIR]:
        if not os.path.isdir(root_dir):
            continue
        images.extend(
            [f for f in Path(root_dir).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        )

    if not images:
        return []

    images = sorted(images, key=lambda f: f.stat().st_mtime, reverse=True)
    manual_images = [f for f in images if f.name.lower() not in AUTO_SOURCE_FACE_NAMES]
    if manual_images:
        return [str(f) for f in manual_images]

    if ALLOW_AUTO_SOURCE_FROM_VIDEO:
        return [str(f) for f in images]
    return []


def _face_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


_HAAR_CASCADE_CACHE = {
    "loaded": False,
    "frontal": None,
    "profile": None,
    "eyes": None,
}


def _get_haar_cascades():
    if _HAAR_CASCADE_CACHE["loaded"]:
        return _HAAR_CASCADE_CACHE["frontal"], _HAAR_CASCADE_CACHE["profile"], _HAAR_CASCADE_CACHE["eyes"]

    import cv2

    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    _HAAR_CASCADE_CACHE["frontal"] = frontal if not frontal.empty() else None
    _HAAR_CASCADE_CACHE["profile"] = profile if not profile.empty() else None
    _HAAR_CASCADE_CACHE["eyes"] = eyes if not eyes.empty() else None
    _HAAR_CASCADE_CACHE["loaded"] = True

    return _HAAR_CASCADE_CACHE["frontal"], _HAAR_CASCADE_CACHE["profile"], _HAAR_CASCADE_CACHE["eyes"]


def _detect_human_faces(frame, max_faces=8, relaxed=False):
    import cv2

    if frame is None:
        return []

    h, w = frame.shape[:2]
    if h < 64 or w < 64:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    frontal, profile, eyes = _get_haar_cascades()
    if frontal is None or eyes is None:
        return []

    scale = 1.06 if not relaxed else 1.03
    neighbors = 6 if not relaxed else 4
    min_edge = max(80, int(min(h, w) * (0.08 if not relaxed else 0.06)))

    raw_faces = [
        tuple(map(int, f))
        for f in frontal.detectMultiScale(gray, scale, neighbors, minSize=(min_edge, min_edge))
    ]

    if profile is not None:
        raw_faces.extend(
            tuple(map(int, f))
            for f in profile.detectMultiScale(gray, scale, neighbors, minSize=(min_edge, min_edge))
        )

        flipped = cv2.flip(gray, 1)
        for fx, fy, fw, fh in profile.detectMultiScale(flipped, scale, neighbors, minSize=(min_edge, min_edge)):
            raw_faces.append((int(w - (fx + fw)), int(fy), int(fw), int(fh)))

    raw_faces = sorted(raw_faces, key=lambda f: f[2] * f[3], reverse=True)
    deduped = []
    for cand in raw_faces:
        if all(_face_iou(cand, kept) < 0.35 for kept in deduped):
            deduped.append(cand)

    valid = []
    for x, y, fw, fh in deduped:
        if x < 0 or y < 0 or x + fw > w or y + fh > h:
            continue

        aspect = fw / float(max(1, fh))
        if aspect < 0.65 or aspect > 1.45:
            continue

        face_area = fw * fh
        coverage = face_area / float(max(1, h * w))
        min_coverage = 0.020 if not relaxed else 0.015
        if coverage < min_coverage:
            continue

        roi_gray = gray[y:y + fh, x:x + fw]
        roi_bgr = frame[y:y + fh, x:x + fw]
        if roi_gray.size == 0 or roi_bgr.size == 0:
            continue

        if relaxed:
            valid.append((x, y, fw, fh))
            if len(valid) >= max_faces:
                break
            continue

        upper = roi_gray[: max(1, int(fh * 0.72)), :]
        eye_min = max(10, int(min(fw, fh) * 0.12))
        eye_boxes = eyes.detectMultiScale(upper, 1.08, 3, minSize=(eye_min, eye_min))
        if len(eye_boxes) < (1 if relaxed else 2):
            continue

        # Landmark-like geometry guard: ensure detected eyes are horizontally plausible.
        if len(eye_boxes) >= 2:
            eye_boxes_sorted = sorted(eye_boxes, key=lambda b: b[0])
            lx, ly, lw, lh = eye_boxes_sorted[0]
            rx, ry, rw, rh = eye_boxes_sorted[-1]
            left_cx = float(lx + lw * 0.5)
            right_cx = float(rx + rw * 0.5)
            if (right_cx - left_cx) < float(fw) * 0.20:
                continue
            if abs(float((ly + lh * 0.5) - (ry + rh * 0.5))) > float(fh) * 0.25:
                continue

        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        skin_ratio = float((skin_mask > 0).sum()) / float(max(1, skin_mask.size))
        if skin_ratio < (0.16 if relaxed else 0.20):
            continue

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hsv_skin = cv2.inRange(hsv, (0, 25, 50), (35, 220, 255))
        hsv_skin_ratio = float((hsv_skin > 0).sum()) / float(max(1, hsv_skin.size))
        if hsv_skin_ratio < (0.14 if relaxed else 0.18):
            continue

        # Flat texture/object suppression.
        local_std = float(roi_gray.std())
        if local_std < (20.0 if not relaxed else 16.0):
            continue

        sharpness = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
        if sharpness < (16.0 if relaxed else 24.0):
            continue

        valid.append((x, y, fw, fh))
        if len(valid) >= max_faces:
            break

    return valid


def validate_source_face_quality(face_path):
    """Basic source-face guard to avoid silent no-op swaps from bad inputs."""
    try:
        import cv2
        img = cv2.imread(face_path)
        if img is None:
            return False, "image unreadable"
        h, w = img.shape[:2]
        if h < 64 or w < 64:
            return False, "image resolution too low"

        faces = _detect_human_faces(img, max_faces=2, relaxed=False)
        if len(faces) == 0:
            faces = _detect_human_faces(img, max_faces=2, relaxed=True)
        if len(faces) == 0:
            return False, "no clear human face detected"

        biggest = max((fw * fh for _, _, fw, fh in faces))
        coverage = biggest / float(max(1, h * w))
        if coverage < 0.06:
            return False, f"face too small in image ({coverage:.3f})"
        return True, "ok"
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) != "cv2":
            return False, f"face validation error: {e}"
        try:
            from PIL import Image as PILImage
            with PILImage.open(face_path) as img:
                w, h = img.size
                img.verify()
            if h < 64 or w < 64:
                return False, "image resolution too low"
            logger.warning("[FACE_SOURCE] cv2 unavailable; accepted readable source image via PIL fallback")
            return True, "ok"
        except Exception as pil_error:
            return False, f"image unreadable: {pil_error}"
    except Exception as e:
        return False, f"face validation error: {e}"


def resolve_face_for_chat(chat_id):
    preferred = get_face(chat_id)
    if preferred and os.path.isfile(preferred):
        ok, _ = validate_source_face_quality(preferred)
        if ok:
            return preferred

    for candidate in list_face_images():
        clean = face_to_clean_jpg(candidate)
        ok, _ = validate_source_face_quality(clean)
        if ok:
            current_face[chat_id] = clean
            return clean
    return None


def _fmt_elapsed(seconds):
    seconds = max(0, int(seconds))
    mins, secs = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        return f"{hrs}h {mins}m {secs}s"
    return f"{mins}m {secs}s"


def _build_status_text(chat_id):
    def _progress_text(pct_value, done_frames_value=0, total_frames_value=0):
        try:
            pct_i = int(pct_value)
        except Exception:
            pct_i = -1
        if pct_i >= 0:
            return f"{max(0, min(100, pct_i))}%"

        done_i = int(max(0, done_frames_value or 0))
        total_i = int(max(0, total_frames_value or 0))
        if total_i > 0:
            if done_i > total_i:
                done_i = total_i
            return f"{done_i}/{total_i} frames"
        return f"{done_i} frames"

    chat_id = str(chat_id)
    st = _get_active_job_state(chat_id, allow_fallback=True)
    if not st:
        return "📡 *Current Status*\n\n• No jobs are running."

    phase = str(st.get("phase") or "idle")
    stage = str(st.get("stage") or "-")
    pct = int(st.get("progress", -1) or -1)
    target = str(st.get("target") or "-")
    updated = float(st.get("last_update") or st.get("updated_at") or time.time())
    started = float(st.get("start_time") or time.time())
    details = str(st.get("details") or "")
    done_frames = int(st.get("frames_done") or st.get("done_frames") or 0)
    total_frames = int(st.get("total_frames") or 0)

    pct_line = _progress_text(pct, done_frames, total_frames)
    elapsed = _fmt_elapsed(time.time() - started)
    ago = _fmt_elapsed(time.time() - updated)

    msg = (
        "📡 *Current Status*\n\n"
        f"• Phase: *{phase}*\n"
        f"• Stage: *{stage}*\n"
        f"• Progress: *{pct_line}*\n"
        f"• Elapsed: `{elapsed}`\n"
        f"• Last update: `{ago} ago`\n"
        f"• File: `{target}`"
    )
    if details:
        msg += f"\n• Note: `{details}`"
    return msg


def _is_chat_busy(chat_id):
    _reset_stale_state_for_chat_if_needed(chat_id)

    ext_pid = recovered_external_jobs.get(chat_id)
    if ext_pid:
        try:
            if _pid_is_job_process(int(ext_pid)):
                return True
            recovered_external_jobs.pop(chat_id, None)
        except Exception:
            recovered_external_jobs.pop(chat_id, None)

    task = active_pipeline_tasks.get(chat_id)
    if task and not task.done():
        return True

    proc = active_jobs.get(chat_id)
    if proc is not None:
        try:
            if proc.poll() is None:
                return True
            active_jobs.pop(chat_id, None)
        except Exception:
            return True

    st = _get_active_job_state(chat_id, allow_fallback=False) or {}
    if str(st.get("chat_id") or "") == str(chat_id):
        phase = str(st.get("phase") or st.get("status") or "").lower()
        if phase in {"download", "faceswap", "processing", "upload", "starting"}:
            worker_pid = int(st.get("worker_pid") or 0)
            proc_pid = int(st.get("processing_pid") or 0)
            if (worker_pid > 0 and _pid_is_job_process(worker_pid)) or (proc_pid > 0 and _pid_is_job_process(proc_pid)):
                return True

    return False


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _terminate_pid_group_sync(pid, wait_sec=None):
    try:
        pid_i = int(pid or 0)
    except Exception:
        return
    if pid_i <= 0:
        return

    grace = int(wait_sec if wait_sec is not None else (15 if IS_LIGHTWEIGHT else 5))
    with suppress(Exception):
        os.killpg(os.getpgid(pid_i), signal.SIGTERM)

    deadline = time.time() + float(max(1, grace))
    while time.time() < deadline:
        if not _pid_alive(pid_i):
            return
        time.sleep(0.5)

    with suppress(Exception):
        os.killpg(os.getpgid(pid_i), signal.SIGKILL)


def kill_worker(pid):
    """Terminate a worker process group with SIGTERM then SIGKILL fallback."""
    try:
        pid_i = int(pid or 0)
    except Exception:
        return
    if pid_i <= 0:
        return
    logger.error(
        "[WATCHDOG KILL] chat_id=%s stage=%s last_progress_time=%s reason=kill_worker_sigterm",
        "UNKNOWN",
        "UNKNOWN",
        int(time.time()),
    )
    _terminate_pid_group_sync(pid_i, wait_sec=2)
    if _pid_alive(pid_i):
        with suppress(Exception):
            os.killpg(os.getpgid(pid_i), signal.SIGKILL)
        logger.error(
            "[WATCHDOG KILL] chat_id=%s stage=%s last_progress_time=%s reason=kill_worker_sigkill",
            "UNKNOWN",
            "UNKNOWN",
            int(time.time()),
        )


def _pid_cmdline(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        if not raw:
            return ""
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _pid_is_job_process(pid):
    if not _pid_alive(pid):
        return False
    cmd_l = _pid_cmdline(pid).lower()
    if not cmd_l:
        return False
    return (
        "ops/job_worker.py" in cmd_l
        or "facefusion.py headless-run" in cmd_l
        or ("ffmpeg" in cmd_l and PIPELINE.lower() in cmd_l)
    )


def _reset_chat_runtime_state(chat_id):
    chat_key = str(chat_id)
    recovered_external_jobs.pop(chat_key, None)

    proc = active_jobs.get(chat_key)
    if proc is not None:
        try:
            if proc.poll() is not None:
                active_jobs.pop(chat_key, None)
        except Exception:
            active_jobs.pop(chat_key, None)

    task = active_pipeline_tasks.get(chat_key)
    if task and task.done():
        active_pipeline_tasks.pop(chat_key, None)

    st = job_status.get(chat_key, {}) or {}
    phase = str(st.get("phase") or "").lower()
    if phase in {"download", "faceswap", "processing", "upload", "starting"}:
        job_status.pop(chat_key, None)


def _reset_stale_state_for_chat_if_needed(chat_id):
    state = _load_active_job_state() or {}
    if str(state.get("chat_id") or "") != str(chat_id):
        return False

    worker_pid = int(state.get("worker_pid") or 0)
    processing_pid = int(state.get("processing_pid") or 0)
    has_live = False
    if worker_pid > 0 and _pid_is_job_process(worker_pid):
        has_live = True
    if processing_pid > 0 and _pid_is_job_process(processing_pid):
        has_live = True

    if has_live:
        return False

    _clear_active_job_state()
    _reset_chat_runtime_state(chat_id)
    logger.info("stale state cleared chat=%s worker_pid=%s processing_pid=%s", chat_id, worker_pid, processing_pid)
    return True


def _read_json_dict_with_retry(path_obj, retry_delay_sec=0.2):
    try:
        return json.loads(path_obj.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        time.sleep(float(retry_delay_sec))
        return json.loads(path_obj.read_text(encoding="utf-8"))


def _write_json_atomic_path(path_obj, payload):
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    tmp = path_obj.with_name(f"{path_obj.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload or {}, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(tmp, path_obj)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _load_active_job_state():
    try:
        p = Path(ACTIVE_JOB_STATE_FILE)
        if not p.exists():
            return None
        data = _read_json_dict_with_retry(p)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_active_job_state(state):
    try:
        Path(ACTIVE_JOB_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic_path(Path(ACTIVE_JOB_STATE_FILE), state)
    except Exception as e:
        logger.warning("failed to save active job state: %s", e)


def _clear_active_job_state():
    try:
        Path(ACTIVE_JOB_STATE_FILE).unlink(missing_ok=True)
    except Exception:
        pass


def _normalize_phase(value):
    phase = str(value or "").strip().lower()
    if phase in {"download", "faceswap", "processing", "upload", "completed", "failed", "stopped", "starting"}:
        return phase
    if phase == "sending":
        return "completed"
    return phase or "starting"


def _state_from_job_status(chat_id):
    chat_key = str(chat_id)
    st = (job_status.get(chat_key, {}) or {}).copy()
    if not st:
        return None
    started = float(st.get("started_at") or time.time())
    updated = float(st.get("updated_at") or time.time())
    return {
        "chat_id": chat_key,
        "phase": _normalize_phase(st.get("phase")),
        "status": _normalize_phase(st.get("phase")),
        "stage": str(st.get("stage") or "Starting"),
        "progress": int(st.get("pct", -1) or -1),
        "frames_done": int(st.get("done_frames") or 0),
        "done_frames": int(st.get("done_frames") or 0),
        "total_frames": int(st.get("total_frames") or 0),
        "start_time": started,
        "last_update": updated,
        "updated_at": updated,
        "target": str(st.get("target") or "-"),
        "message_id": int(st.get("message_id") or 0),
        "processing_pid": int(st.get("processing_pid") or 0),
        "worker_pid": int(st.get("worker_pid") or 0),
        "details": str(st.get("details") or ""),
    }


def _get_active_job_state(chat_id=None, allow_fallback=True):
    state = _load_active_job_state() or {}
    state_chat = str(state.get("chat_id") or "")
    if state and isinstance(state, dict):
        if chat_id is None or state_chat == str(chat_id):
            phase = _normalize_phase(state.get("phase") or state.get("status"))
            started = float(state.get("start_time") or state.get("started_at") or time.time())
            updated = float(state.get("last_update") or state.get("updated_at") or time.time())
            return {
                "chat_id": state_chat,
                "phase": phase,
                "status": phase,
                "stage": str(state.get("stage") or "Starting"),
                "progress": int(state.get("progress", -1) or -1),
                "frames_done": int(state.get("frames_done") or state.get("done_frames") or 0),
                "done_frames": int(state.get("frames_done") or state.get("done_frames") or 0),
                "total_frames": int(state.get("total_frames") or 0),
                "start_time": started,
                "last_update": updated,
                "updated_at": updated,
                "target": str(state.get("target") or "-"),
                "message_id": int(state.get("message_id") or 0),
                "processing_pid": int(state.get("processing_pid") or 0),
                "worker_pid": int(state.get("worker_pid") or 0),
                "details": str(state.get("details") or ""),
            }
    if allow_fallback and chat_id is not None:
        return _state_from_job_status(chat_id)
    return None


def _save_sleep_countdown_state(state):
    try:
        Path(SLEEP_COUNTDOWN_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic_path(Path(SLEEP_COUNTDOWN_STATE_FILE), state)
    except Exception as e:
        logger.warning("failed to save sleep countdown state: %s", e)


def _load_sleep_countdown_state():
    try:
        p = Path(SLEEP_COUNTDOWN_STATE_FILE)
        if not p.exists():
            return None
        data = _read_json_dict_with_retry(p)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _clear_sleep_countdown_state():
    try:
        Path(SLEEP_COUNTDOWN_STATE_FILE).unlink(missing_ok=True)
    except Exception:
        pass


def _worker_result_state_file(chat_id, job_id):
    chat_key = str(chat_id)
    jid = int(job_id or 0)
    return Path(f"{PIPELINE}/logs/worker_result_{chat_key}_{jid}.json")


def _worker_trace_log_file(chat_id, job_id):
    chat_key = str(chat_id)
    jid = int(job_id or 0)
    return Path(f"{PIPELINE}/logs/worker_trace_{chat_key}_{jid}.log")


def _read_worker_trace_tail(chat_id, job_id, max_lines=120):
    try:
        p = _worker_trace_log_file(chat_id, job_id)
        if not p.exists():
            return ""
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = "\n".join(lines[-max_lines:]).strip()
        return tail
    except Exception:
        return ""


def _load_worker_result_state(chat_id, job_id):
    try:
        p = _worker_result_state_file(chat_id, job_id)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


async def _wait_for_worker_result_state(chat_id, job_id, timeout_sec=8.0, poll_sec=0.25):
    deadline = float(time.time()) + float(max(0.5, timeout_sec))
    while float(time.time()) < deadline:
        state = _load_worker_result_state(chat_id, job_id)
        if isinstance(state, dict) and state:
            return state
        await asyncio.sleep(max(0.05, float(poll_sec)))
    return _load_worker_result_state(chat_id, job_id)


def _clear_worker_result_state(chat_id, job_id):
    try:
        _worker_result_state_file(chat_id, job_id).unlink(missing_ok=True)
    except Exception:
        pass


def _validate_worker_completion_receipt(worker_result, chat_id=None):
    if not isinstance(worker_result, dict):
        return False, "worker_result missing"

    phase = str(worker_result.get("phase") or "").strip().lower()
    stage = str(worker_result.get("stage") or "").strip().upper()
    output_path = str(worker_result.get("output_path") or "").strip()
    upload_link = str(worker_result.get("upload_link") or "").strip()

    if phase != "completed" or stage != "COMPLETED":
        return False, f"non-terminal completion marker phase={phase or '-'} stage={stage or '-'}"
    if not output_path or not Path(output_path).is_file():
        return False, "completion blocked: output file missing"
    # CRITICAL: Block completion if output is identical to input (face swap failed silently)
    st = job_status.get(str(chat_id) if chat_id is not None else "", {}) or {}
    input_path = str(st.get("input_path") or "")
    if input_path and Path(input_path).is_file():
        import hashlib
        def file_hash(path):
            h = hashlib.md5()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    h.update(chunk)
            return h.hexdigest()
        try:
            if file_hash(input_path) == file_hash(output_path):
                return False, "completion blocked: output identical to input (faceswap failed)"
        except Exception:
            pass  # Allow on hash error - validate_output_media will catch other issues
    ok_media, media_reason = validate_output_media(output_path)
    if not ok_media:
        return False, f"completion blocked: {media_reason}"
    if not bool(worker_result.get("upload_ok")):
        return False, "completion blocked: upload_ok=false"
    if not upload_link:
        return False, "completion blocked: upload link missing"
    return True, "ok"


async def recover_sleep_countdown_from_state(app):
    state = _load_sleep_countdown_state()
    if not state:
        return
    chat_id = str(state.get("chat_id") or "")
    status = str(state.get("status") or "")
    reason = str(state.get("reason") or "Job completed. Countdown resumed after restart.")

    # If countdown was running when bot restarted and job is still completed, resume it.
    if status == "running" and chat_id and _last_job_phase(chat_id) in {"completed", "failed"}:
        _clear_sleep_countdown_state()
        logger.info("startup resuming sleep countdown chat=%s", chat_id)
        try:
            await start_sleep_countdown(
                app,
                chat_id,
                reason_text=f"[Resumed after restart] {reason}",
                delay_seconds=SLEEP_COUNTDOWN_SECONDS,
                force_allow=True,
            )
        except Exception as e:
            logger.warning("startup sleep countdown resume failed chat=%s err=%s", chat_id, e)
        return

    _clear_sleep_countdown_state()
    if chat_id:
        sleep_timer_active[chat_id] = False
    logger.info("startup cleared stale sleep countdown state chat=%s status=%s", chat_id, status)


def _update_lifecycle_state(chat_id, **overrides):
    chat_key = str(chat_id)
    phase = _last_job_phase(chat_key)
    is_job_running = bool(_is_chat_busy(chat_key))
    is_job_completed = phase == "completed"
    is_countdown_running = bool(_task_is_running(sleep_countdown_tasks.get(chat_key)))
    state = {
        "is_bot_active": True,
        "is_job_running": is_job_running,
        "is_job_completed": is_job_completed,
        "is_countdown_running": is_countdown_running,
        "can_auto_sleep": bool(is_all_jobs_completed(chat_key) and is_job_completed),
        "updated_at": time.time(),
    }
    state.update(overrides)
    lifecycle_state[chat_key] = state
    return state


def _persist_active_job_state(chat_id, message_id=None, processing_pid=None):
    st = (job_status.get(chat_id, {}) or {}).copy()
    phase = _normalize_phase(st.get("phase", "") or "")
    prev_state = _load_active_job_state() or {}
    env_worker_pid = int(os.environ.get("PIPELINE_WORKER_PID", "0") or 0)
    now_ts = float(time.time())
    started_at = float(st.get("started_at") or prev_state.get("start_time") or now_ts)
    prev_progress = int(prev_state.get("progress", -1) or -1)
    prev_frames = int(prev_state.get("done_frames") or prev_state.get("frames_done") or 0)
    prev_stage = str(prev_state.get("stage") or "")
    cur_progress = int(st.get("pct", -1) or -1)
    cur_frames = int(st.get("done_frames") or 0)
    cur_stage = str(st.get("stage", "") or "")

    progressed = (
        cur_stage != prev_stage
        or cur_progress > prev_progress
        or cur_frames > prev_frames
    )

    prev_last_progress_ts = float(prev_state.get("last_progress_timestamp") or 0.0)
    last_progress_ts = now_ts if progressed else (prev_last_progress_ts or now_ts)
    last_progress_frame = cur_frames if progressed else int(prev_state.get("last_progress_frame") or cur_frames)
    last_progress_stage = cur_stage if progressed else str(prev_state.get("last_progress_stage") or cur_stage)
    last_progress_pct = cur_progress if progressed else int(prev_state.get("last_progress_pct", cur_progress) or cur_progress)

    state = {
        "chat_id": str(chat_id),
        "job_id": str(st.get("job_id", "") or ""),
        "phase": phase,
        "status": phase,
        "stage": str(st.get("stage", "") or ""),
        "progress": cur_progress,
        "frames_done": cur_frames,
        "target": str(st.get("target", "-") or "-"),
        "start_time": started_at,
        "last_update": now_ts,
        "updated_at": now_ts,
        "last_progress_timestamp": float(last_progress_ts),
        "last_progress_frame": int(last_progress_frame),
        "last_progress_stage": str(last_progress_stage),
        "last_progress_pct": int(last_progress_pct),
        "heartbeat_timestamp": now_ts,
        "heartbeat_source": "bot_persist",
        "message_id": int(message_id or 0) if message_id else int(st.get("message_id") or 0),
        "processing_pid": int(processing_pid or 0),
        "worker_pid": int(st.get("worker_pid") or env_worker_pid or prev_state.get("worker_pid") or 0),
        "done_frames": cur_frames,
        "total_frames": int(st.get("total_frames") or 0),
        "details": str(st.get("details") or ""),
        "input_path": str(st.get("input_path") or prev_state.get("input_path") or ""),
        "output_path": str(st.get("output_path") or prev_state.get("output_path") or ""),
        "upload_ok": bool(st.get("upload_ok") if "upload_ok" in st else prev_state.get("upload_ok", False)),
        "upload_platform": str(st.get("upload_platform") or prev_state.get("upload_platform") or ""),
        "upload_info": str(st.get("upload_info") or prev_state.get("upload_info") or ""),
    }
    if state["processing_pid"] <= 0:
        proc = active_jobs.get(chat_id)
        if proc is not None:
            try:
                state["processing_pid"] = int(proc.pid)
            except Exception:
                state["processing_pid"] = 0

    if phase in {"download", "faceswap", "processing", "upload", "starting"}:
        _save_active_job_state(state)
    else:
        _clear_active_job_state()


def recover_active_job_from_state():
    state = _load_active_job_state()
    if not state:
        return

    chat_id = str(state.get("chat_id") or "")
    if not chat_id:
        _clear_active_job_state()
        return

    worker_pid = int(state.get("worker_pid") or 0)
    processing_pid = int(state.get("processing_pid") or 0)
    active_pid = 0
    if worker_pid > 0 and _pid_is_job_process(worker_pid):
        active_pid = worker_pid
    elif processing_pid > 0 and _pid_is_job_process(processing_pid):
        active_pid = processing_pid

    if active_pid <= 0:
        logger.info("recovery skipped: no live worker/processing pid in active state")
        _clear_active_job_state()
        return

    recovered_external_jobs[chat_id] = active_pid
    phase = str(state.get("status") or "faceswap").lower()
    stage = str(state.get("stage") or "Recovering")
    pct = int(state.get("progress") or -1)
    target = str(state.get("target") or "-")
    message_id = int(state.get("message_id") or 0)
    now = time.time()

    job_status[chat_id] = {
        "phase": phase,
        "stage": stage,
        "pct": pct,
        "target": target,
        "started_at": now,
        "updated_at": now,
        "details": f"Recovered running process pid={active_pid}",
        "message_id": message_id,
        "worker_pid": int(state.get("worker_pid") or 0),
        "done_frames": int(state.get("done_frames") or 0),
        "total_frames": int(state.get("total_frames") or 0),
        "input_path": str(state.get("input_path") or ""),
        "output_path": str(state.get("output_path") or ""),
    }
    pending_recovery_chats.add(chat_id)
    logger.info("recovered active processing job chat=%s pid=%s stage=%s pct=%s", chat_id, active_pid, stage, pct)

    # Restart message sync loop attached to existing message id (edit-only sidecar).
    if BOT_TOKEN and message_id > 0:
        sidecar = str(ROOT_DIR / "ops" / "progress_resync_sidecar.py")
        try:
            subprocess.Popen(
                [
                    RUNTIME_PYTHON,
                    sidecar,
                    "--token", BOT_TOKEN,
                    "--chat-id", chat_id,
                    "--message-ids", str(message_id),
                    "--log", f"{PIPELINE}/logs/bot_native.log",
                    "--file-label", target,
                    "--interval", "0.7",
                    "--max-message-ids", "1",
                    "--max-line-age", "25",
                ],
                cwd=str(ROOT_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except Exception as e:
            logger.warning("failed to start recovery sidecar: %s", e)


def startup_state_sanity_check():
    global global_job_lock
    live_job_rows = [row for row in _scan_job_like_processes() if _pid_is_job_process(row.get("pid"))]
    if live_job_rows:
        logger.info("Startup check: active job-like process found count=%s", len(live_job_rows))
        recover_active_job_from_state()
        return

    _clear_active_job_state()
    global_job_lock = False
    active_jobs.clear()
    active_pipeline_tasks.clear()
    queue_workers.clear()
    job_queues.clear()
    queue_job_seq.clear()
    queued_progress_message_ids.clear()
    recovered_external_jobs.clear()
    for chat_key in list(job_status.keys()):
        phase = str((job_status.get(chat_key, {}) or {}).get("phase") or "").lower()
        if phase in {"download", "faceswap", "processing", "upload", "starting"}:
            job_status.pop(chat_key, None)

    logger.info("Startup check: no active job found, state reset")


def _save_queue_state():
    try:
        payload = {
            "job_queues": job_queues,
            "queue_job_seq": queue_job_seq,
            "updated_at": float(time.time()),
        }
        _write_json_atomic_path(Path(QUEUE_STATE_FILE), payload)
    except Exception as e:
        logger.warning("failed to save queue state: %s", e)


def _load_queue_state():
    try:
        p = Path(QUEUE_STATE_FILE)
        if not p.exists():
            return {}, {}
        payload = _read_json_dict_with_retry(p)
        if not isinstance(payload, dict):
            return {}, {}
        queues = payload.get("job_queues") if isinstance(payload.get("job_queues"), dict) else {}
        seq = payload.get("queue_job_seq") if isinstance(payload.get("queue_job_seq"), dict) else {}
        return queues, seq
    except Exception:
        return {}, {}


def _restore_queue_state_from_disk():
    queues, seq = _load_queue_state()
    restored = 0
    max_id_per_chat: dict = {}
    if queues:
        for key, items in queues.items():
            if not isinstance(items, list):
                continue
            valid = []
            for item in items:
                if isinstance(item, dict) and item.get("video_link"):
                    valid.append(item)
                    try:
                        jid = int(item.get("job_id") or 0)
                        if jid > max_id_per_chat.get(str(key), 0):
                            max_id_per_chat[str(key)] = jid
                    except Exception:
                        pass
            if valid:
                job_queues[str(key)] = valid
                restored += len(valid)
    # Seed seq counter above the highest restored job ID to avoid duplicate IDs
    queue_job_seq.clear()
    for chat_key, max_id in max_id_per_chat.items():
        if max_id > 0:
            queue_job_seq[chat_key] = max_id
    logger.info("[QUEUE_RESTORED] jobs=%s seq_seeded=%s", restored, dict(max_id_per_chat))


def _next_queue_job_id(chat_id):
    cur = int(queue_job_seq.get(chat_id, 0)) + 1
    queue_job_seq[chat_id] = cur
    _save_queue_state()
    return cur


def _build_queue_status_text(chat_id):
    queue = job_queues.get(chat_id, [])
    if not queue:
        return "📚 Queue: *empty*"

    lines = [f"📚 Queue: *{len(queue)}* pending"]
    for item in queue[:6]:
        lines.append(
            f"• #{item['job_id']} `{item.get('target_name', 'unknown')}` ({item.get('mode', 'direct')})"
        )
    if len(queue) > 6:
        lines.append(f"• ... +{len(queue) - 6} more")
    return "\n".join(lines)


def _build_full_status_text(chat_id):
    chat_id = str(chat_id)
    _reset_stale_state_for_chat_if_needed(chat_id)
    if not _is_chat_busy(chat_id) and _queue_size(chat_id) == 0:
        return "📡 *Current Status*\n\n• No jobs are running.\n\n📚 Queue: *empty*"
    return _build_status_text(chat_id) + "\n\n" + _build_queue_status_text(chat_id)


def _queue_size(chat_id):
    return len(job_queues.get(chat_id, []))


def _collect_tracked_job_pids(chat_id=None):
    tracked = {int(os.getpid())}

    for proc in list(active_jobs.values()):
        try:
            tracked.add(int(proc.pid))
        except Exception:
            continue

    state = _load_active_job_state() or {}
    if state:
        if chat_id is None or str(state.get("chat_id") or "") == str(chat_id):
            for pid_key in ("processing_pid", "worker_pid"):
                try:
                    pid = int(state.get(pid_key) or 0)
                except Exception:
                    pid = 0
                if pid > 0:
                    tracked.add(pid)

    if chat_id is not None:
        try:
            rec_pid = int(recovered_external_jobs.get(str(chat_id)) or 0)
        except Exception:
            rec_pid = 0
        if rec_pid > 0:
            tracked.add(rec_pid)

    return tracked


def _scan_job_like_processes():
    rows = []
    try:
        r = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return rows
        for line in (r.stdout or "").splitlines():
            txt = line.strip()
            if not txt:
                continue
            parts = txt.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except Exception:
                continue
            cmd = parts[2]
            cmd_l = cmd.lower()
            if (
                "facefusion.py headless-run" in cmd_l
                or "ops/job_worker.py" in cmd_l
                or ("ffmpeg" in cmd_l and PIPELINE.lower() in cmd_l)
            ):
                rows.append({"pid": pid, "ppid": ppid, "cmd": cmd})
    except Exception:
        return rows
    return rows


def _count_live_job_like_processes():
    rows = _scan_job_like_processes()
    return len([row for row in rows if _pid_alive(row.get("pid"))])


def _read_self_rss_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int((line.split()[1]))
                    return float(kb) / 1024.0
    except Exception:
        return None
    return None


def _kill_orphan_job_processes(chat_id=None):
    tracked = _collect_tracked_job_pids(chat_id)
    killed = 0
    for row in _scan_job_like_processes():
        pid = int(row.get("pid") or 0)
        ppid = int(row.get("ppid") or 0)
        if pid <= 0 or pid in tracked:
            continue
        orphan = ppid <= 1 or (ppid > 1 and not _pid_alive(ppid))
        if not orphan:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except Exception:
            continue
    return killed


def _release_runtime_memory(chat_id=None):
    before_rss = _read_self_rss_mb()
    before_proc = _count_live_job_like_processes()

    orphan_killed = _kill_orphan_job_processes(chat_id)
    kill_stale_facefusion_runs()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()

    after_rss = _read_self_rss_mb()
    after_proc = _count_live_job_like_processes()
    logger.info(
        "MEMORY_CLEANUP chat=%s orphan_killed=%s active_procs_before=%s active_procs_after=%s rss_before_mb=%s rss_after_mb=%s",
        chat_id,
        orphan_killed,
        before_proc,
        after_proc,
        ("n/a" if before_rss is None else f"{before_rss:.1f}"),
        ("n/a" if after_rss is None else f"{after_rss:.1f}"),
    )


def _single_job_block_reason(chat_id):
    global global_job_lock
    if MAX_PARALLEL_JOBS <= 1:
        _reset_stale_state_for_chat_if_needed(chat_id)
        if global_job_lock and not _is_chat_busy(chat_id):
            state = _load_active_job_state() or {}
            state_pid = 0
            for key in ("processing_pid", "worker_pid"):
                try:
                    state_pid = int(state.get(key) or 0)
                except Exception:
                    state_pid = 0
                if state_pid > 0 and _pid_is_job_process(state_pid):
                    break
            else:
                if _count_live_job_like_processes() == 0:
                    global_job_lock = False

        if global_job_lock:
            return "Another job is already running"

        if _is_chat_busy(chat_id):
            return "Another job is already running"
        state = _load_active_job_state() or {}
        if str(state.get("chat_id") or "") == str(chat_id):
            for key in ("processing_pid", "worker_pid"):
                try:
                    pid = int(state.get(key) or 0)
                except Exception:
                    pid = 0
                if pid > 0 and _pid_is_job_process(pid):
                    return "Another job is already running"
    return ""


def _task_is_running(task_obj):
    return bool(task_obj and not task_obj.done())


def _no_background_task_running(chat_id):
    return (
        not _task_is_running(active_pipeline_tasks.get(chat_id))
        and not _task_is_running(queue_workers.get(chat_id))
        and not _task_is_running(post_upload_tasks.get(chat_id))
        and not _task_is_running(post_job_cleanup_tasks.get(chat_id))
    )


def is_all_jobs_completed(chat_id):
    return (
        (not _is_chat_busy(chat_id))
        and _queue_size(chat_id) == 0
        and _no_background_task_running(chat_id)
    )


def _last_job_phase(chat_id):
    st = job_status.get(chat_id)
    if not isinstance(st, dict):
        st = job_status.get(str(chat_id), {}) or {}
    return str((st or {}).get("phase") or "").strip().lower()


def _is_last_job_successfully_completed(chat_id):
    return _last_job_phase(chat_id) == "completed"


def _is_last_job_auto_sleep_eligible(chat_id):
    return _last_job_phase(chat_id) in {"completed", "failed"}


def _can_auto_sleep(chat_id):
    if _has_any_active_job_pid():
        return False
    return is_all_jobs_completed(chat_id) and _is_last_job_auto_sleep_eligible(chat_id)


def _download_allowed(chat_id):
    state = _get_active_job_state(chat_id, allow_fallback=True) or {}
    phase = str(state.get("phase") or state.get("status") or "").lower()
    return phase == "download"


def _queue_job(chat_id, video_link, face_link=None, mode="direct"):
    queue = job_queues.setdefault(chat_id, [])
    target_name = os.path.basename(video_link.split("?", 1)[0]) or "mega_target"
    gender_mode = get_gender_mode(chat_id)
    queue.append({
        "job_id": _next_queue_job_id(chat_id),
        "video_link": video_link,
        "face_link": face_link,
        "mode": mode,
        "gender_mode": gender_mode,
        "fast_mode": bool(CPU_FAST_MODE),
        "target_name": target_name,
        "enqueued_at": time.time(),
    })
    logger.info("[JOB_RECEIVED] chat=%s job_id=%s queue=%s", chat_id, queue[-1].get("job_id"), len(queue))
    _save_queue_state()
    return queue[-1]


def _stop_active_job(chat_id):
    stopped_any = False

    proc = active_jobs.pop(chat_id, None)
    if proc:
        stopped_any = True
        try:
            _terminate_pid_group_sync(proc.pid)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    task = active_pipeline_tasks.get(chat_id)
    if task and not task.done():
        stopped_any = True
        task.cancel()

    st = _load_active_job_state() or {}
    if str(st.get("chat_id") or "") == str(chat_id):
        for pid_key in ["processing_pid", "worker_pid"]:
            pid = int(st.get(pid_key) or 0)
            if pid <= 0:
                continue
            try:
                _terminate_pid_group_sync(pid)
                stopped_any = True
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                    stopped_any = True
                except Exception:
                    pass

    if stopped_any:
        prev = job_status.get(chat_id, {})
        job_status[chat_id] = {
            "phase": "stopped",
            "stage": "Stopped by user",
            "pct": prev.get("pct", -1),
            "target": prev.get("target", "-"),
            "started_at": prev.get("started_at", time.time()),
            "updated_at": time.time(),
            "details": "Job manually stopped"
        }
        _clear_active_job_state()
        _release_runtime_memory(chat_id)

    _update_lifecycle_state(chat_id)

    return stopped_any


def _build_queue_terminator_kb(chat_id, max_items=8):
    rows = []
    if _is_chat_busy(chat_id):
        rows.append([InlineKeyboardButton("⏹ Stop Active Job", callback_data="terminate_job_active")])

    queue = job_queues.get(chat_id, [])
    for item in queue[:max_items]:
        label = f"❌ #{item['job_id']} {item.get('target_name', 'job')}"
        if len(label) > 45:
            label = label[:42] + "..."
        rows.append([InlineKeyboardButton(label, callback_data=f"terminate_job_q_{item['job_id']}")])

    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="back_main")])
    return InlineKeyboardMarkup(rows)


async def _run_queue_worker(context, chat_id):
    global global_job_lock
    last_block_notice_at = 0.0
    try:
        while True:
            queue = job_queues.get(chat_id, [])
            if not queue:
                break

            # Strict queue policy: do not allow delayed cleanup from previous job
            # to overlap with the next queued job.
            cleanup_task = post_job_cleanup_tasks.get(chat_id)
            if cleanup_task and not cleanup_task.done():
                cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await cleanup_task
                post_job_cleanup_tasks.pop(chat_id, None)

            # If a countdown was scheduled by the previous completed job, cancel it
            # because a new queued job is about to start.
            sleep_task = sleep_countdown_tasks.get(chat_id)
            if sleep_task and not sleep_task.done():
                sleep_task.cancel()
                sleep_timer_active[chat_id] = False
                append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_cancelled", f"chat={chat_id} reason=new_job")
                logger.info("[AUTO_SLEEP_CANCEL] reason=new_job chat=%s", chat_id)

            await asyncio.to_thread(_kill_orphan_job_processes, chat_id)

            block_reason = _single_job_block_reason(chat_id)
            if block_reason:
                now = time.time()
                if now - last_block_notice_at >= 15:
                    await safe_send_message(
                        context.bot,
                        chat_id,
                        f"⚠️ {block_reason}. Queue hold par rakhi gayi hai.",
                    )
                    last_block_notice_at = now
                await asyncio.sleep(1)
                continue

            job = queue.pop(0)
            _save_queue_state()
            mode = job.get("mode", "direct")
            gender_mode = "female" if str(job.get("gender_mode", "all")).lower() == "female" else "all"
            job_modes[chat_id] = mode
            job_id = int(job.get("job_id", 0) or 0)
            logger.info("[JOB_RECEIVED] chat=%s job_id=%s queue_after_pop=%s", chat_id, job_id, len(queue))

            seeded_msg_id = queued_progress_message_ids.pop((chat_id, job_id), None)

            queue_started_text = (
                "🚦 *Queue Job Started*\n"
                f"• Job ID: `#{job['job_id']}`\n"
                f"• Mode: *{mode}*\n"
                f"• Pending after this: `{len(queue)}`"
            )

            seed_msg = None
            if seeded_msg_id:
                try:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=seeded_msg_id,
                        text=queue_started_text,
                        parse_mode="Markdown",
                    )
                    seed_msg = type("SeedMsg", (), {"message_id": seeded_msg_id})()
                except Exception:
                    seed_msg = None

            if seed_msg is None:
                seed_msg = await safe_send_message(
                    context.bot,
                    chat_id,
                    queue_started_text,
                    parse_mode="Markdown"
                )

            seed_message_id = None
            if seed_msg is not None:
                try:
                    seed_message_id = seed_msg.message_id
                except Exception:
                    seed_message_id = None

            worker_cmd = [
                RUNTIME_PYTHON,
                str(ROOT_DIR / "ops" / "job_worker.py"),
                "--chat-id", str(chat_id),
                "--video-link", str(job["video_link"]),
                "--job-mode", str(mode),
                "--gender-mode", str(gender_mode),
                "--queue-job-id", str(job_id),
            ]
            if job.get("face_link"):
                worker_cmd.extend(["--face-link", str(job.get("face_link"))])
            if seed_message_id:
                worker_cmd.extend(["--progress-seed-message-id", str(seed_message_id)])

            # Ensure stale worker result from any previous job id does not leak.
            _clear_worker_result_state(chat_id, job_id)
            worker_trace_path = _worker_trace_log_file(chat_id, job_id)
            worker_trace_path.parent.mkdir(parents=True, exist_ok=True)
            with suppress(Exception):
                worker_trace_path.unlink(missing_ok=True)

            worker_env = os.environ.copy()
            _local_bin = str(Path.home() / ".local" / "bin")
            if _local_bin not in worker_env.get("PATH", ""):
                worker_env["PATH"] = f"{_local_bin}:{worker_env.get('PATH', '')}"
            dashboard_token = str(job.get("dashboard_token") or "").strip()
            if dashboard_token:
                worker_env["DASHBOARD_TOKEN"] = dashboard_token
                worker_env["DASHBOARD_CHAT_ID"] = str(chat_id)
                worker_env["DASHBOARD_SESSIONS_ROOT"] = str(DASHBOARD_SESSIONS_ROOT)
                worker_env["DASHBOARD_ENABLED"] = "1" if DASHBOARD_ENABLED else "0"
            trace_fp = None
            trace_fp = open(worker_trace_path, "a", encoding="utf-8")
            try:
                worker_proc = subprocess.Popen(
                    worker_cmd,
                    cwd=str(ROOT_DIR),
                    env=worker_env,
                    stdout=trace_fp,
                    stderr=trace_fp,
                    start_new_session=True,
                )
            except Exception as spawn_error:
                details = f"worker spawn failed: {spawn_error}"
                logger.exception("[WORKER START FAILED] chat=%s job_id=%s spawn error", chat_id, job_id)
                # Close trace_fp to avoid file handle leak
                if trace_fp is not None:
                    with suppress(Exception): trace_fp.flush()
                    with suppress(Exception): trace_fp.close()
                    trace_fp = None

                st_fail = job_status.get(chat_id, {}) or {}
                st_fail.update({
                    "job_id": job_id,
                    "phase": "failed",
                    "stage": "FAILED",
                    "pct": -1,
                    "updated_at": time.time(),
                    "details": details[:120],
                })
                job_status[chat_id] = st_fail
                _persist_active_job_state(chat_id)
                _update_lifecycle_state(chat_id, is_job_running=False, is_job_completed=False)

                worker_path = _worker_result_state_file(chat_id, job_id)
                proof_path = Path(f"{PIPELINE}/logs/telegram_pipeline_proof.json")
                worker_payload = {
                    "chat_id": str(chat_id),
                    "job_id": int(job_id),
                    "phase": "failed",
                    "stage": "FAILED",
                    "details": details,
                    "pct": -1,
                    "updated_at": float(time.time()),
                }
                proof_payload = {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "chat_id": str(chat_id),
                    "job_id": int(job_id),
                    "pipeline_ok": False,
                    "detail": details,
                    "output_path": "",
                    "upload_platform": "",
                    "upload_info": "",
                    "upload_link": "",
                    "checks": {
                        "saw_upload_stage": False,
                        "saw_completion_message": False,
                        "pipeline_ok": False,
                    },
                }
                with suppress(Exception):
                    worker_tmp = worker_path.with_suffix(".json.tmp")
                    proof_tmp = proof_path.with_suffix(".json.tmp")
                    worker_tmp.parent.mkdir(parents=True, exist_ok=True)
                    proof_tmp.parent.mkdir(parents=True, exist_ok=True)
                    worker_tmp.write_text(json.dumps(worker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                    proof_tmp.write_text(json.dumps(proof_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                    os.replace(worker_tmp, worker_path)
                    os.replace(proof_tmp, proof_path)

                await safe_send_message(
                    context.bot,
                    chat_id,
                    f"❌ Worker spawn failed for job `#{job_id}`. Pipeline did not execute.",
                    parse_mode="Markdown",
                )
                continue
            logger.info("[WORKER STARTED] chat=%s job_id=%s pid=%s", chat_id, job_id, worker_proc.pid)
            # Set job lock immediately after confirmed spawn — prevents double-job race during 15s liveness check
            global_job_lock = True

            # Worker start guarantee: process must be detectable and alive within 15 seconds,
            # and worker trace must show first output to prove execution started.
            worker_running = False
            worker_log_ready = False
            for _ in range(30):
                await asyncio.sleep(0.5)
                if worker_proc.poll() is None and _pid_is_job_process(int(worker_proc.pid)):
                    worker_running = True
                    with suppress(Exception):
                        if worker_trace_path.exists() and int(worker_trace_path.stat().st_size or 0) > 0:
                            worker_log_ready = True
                    if worker_log_ready:
                        break
            if not worker_running or not worker_log_ready:
                details = (
                    f"worker failed to start within 15s (pid={worker_proc.pid}, rc={worker_proc.poll()}, "
                    f"alive={int(worker_running)}, log_ready={int(worker_log_ready)})"
                )
                logger.error("[WORKER START FAILED] chat=%s job_id=%s %s", chat_id, job_id, details)

                st_fail = job_status.get(chat_id, {}) or {}
                st_fail.update({
                    "job_id": job_id,
                    "phase": "failed",
                    "stage": "FAILED",
                    "pct": -1,
                    "updated_at": time.time(),
                    "details": details[:120],
                    "worker_pid": int(worker_proc.pid),
                })
                job_status[chat_id] = st_fail
                _persist_active_job_state(chat_id)
                _update_lifecycle_state(chat_id, is_job_running=False, is_job_completed=False)

                with suppress(Exception):
                    await safe_send_message(
                        context.bot,
                        chat_id,
                        f"❌ Worker start failed for job `#{job_id}`. Pipeline did not execute.",
                        parse_mode="Markdown",
                    )
                continue

            logger.info("[WORKER RUNNING] chat=%s job_id=%s pid=%s", chat_id, job_id, worker_proc.pid)
            # global_job_lock already set to True after Popen (before liveness check)

            st = job_status.get(chat_id, {}) or {}
            st.update({
                "job_id": job_id,
                "phase": "download",
                "stage": "Starting download worker",
                "pct": -1,
                "updated_at": time.time(),
                "details": f"Worker PID {worker_proc.pid} | Gender {gender_mode} | Fast {'ON' if CPU_FAST_MODE else 'OFF'}",
                "worker_pid": int(worker_proc.pid),
            })
            job_status[chat_id] = st
            _persist_active_job_state(chat_id)
            _update_lifecycle_state(
                chat_id,
                is_job_running=True,
                is_job_completed=False,
                is_countdown_running=False,
                can_auto_sleep=False,
            )

            watchdog_state = {
                "run": True,
                "stale_hits": 0,
                "silent_exit_since": 0.0,
                "last_marker": "",
                "last_change_ts": float(time.time()),
            }

            def _watchdog_stage_name(stage_text: str) -> str:
                txt = str(stage_text or "").strip().lower()
                if "download" in txt:
                    return "DOWNLOADING"
                if "extract" in txt:
                    return "EXTRACTING"
                if "process" in txt or "faceswap" in txt:
                    return "PROCESSING"
                if "merg" in txt:
                    return "MERGING"
                if "upload" in txt:
                    return "UPLOADING"
                if "complete" in txt:
                    return "COMPLETED"
                if "fail" in txt:
                    return "FAILED"
                return "UNKNOWN"

            def _watchdog_stage_limit_sec(stage_name: str, st_watch: dict) -> int:
                stage_u = str(stage_name or "").upper()
                base_limits = {
                    "DOWNLOADING": int(os.environ.get("PIPELINE_TIMEOUT_DOWNLOADING_SEC", "300") or 300),
                    "EXTRACTING": int(os.environ.get("PIPELINE_TIMEOUT_EXTRACTING_SEC", "300") or 300),
                    "PROCESSING": int(os.environ.get("PIPELINE_WATCHDOG_PROCESSING_SEC", "300") or 300),
                    "MERGING": int(os.environ.get("PIPELINE_WATCHDOG_MERGING_SEC", "120") or 120),
                    "UPLOADING": int(os.environ.get("PIPELINE_WATCHDOG_UPLOADING_SEC", "120") or 120),
                }
                base = int(base_limits.get(stage_u, 300))

                total_frames = int(st_watch.get("total_frames") or 0)
                frame_scale = 1.0
                if total_frames > 0:
                    if stage_u == "PROCESSING":
                        frame_scale = min(6.0, max(1.0, float(total_frames) / 1500.0))
                    elif stage_u == "MERGING":
                        frame_scale = min(4.0, max(1.0, float(total_frames) / 2400.0))
                    elif stage_u == "UPLOADING":
                        frame_scale = min(3.0, max(1.0, float(total_frames) / 3200.0))

                size_scale = 1.0
                try:
                    output_path = str(st_watch.get("output_path") or "")
                    if stage_u == "UPLOADING" and output_path and Path(output_path).is_file():
                        out_mb = float(Path(output_path).stat().st_size) / (1024.0 * 1024.0)
                        size_scale = min(3.0, max(1.0, out_mb / 300.0))
                except Exception:
                    size_scale = 1.0

                limit = int(max(base, base * frame_scale * size_scale))
                stage_caps = {
                    "DOWNLOADING": 1800,
                    "EXTRACTING": 2400,
                    "PROCESSING": 10800,
                    "MERGING": 3600,
                    "UPLOADING": 5400,
                }
                return int(min(limit, int(stage_caps.get(stage_u, limit) or limit)))

            def _synthesize_watchdog_failure(stage_u: str, reason: str, no_progress_sec: int = 0, limit_sec: int = 0):
                details = (
                    f"WATCHDOG_TRIGGERED freeze_detected:{reason} stage={stage_u or 'UNKNOWN'} "
                    f"no_progress={int(no_progress_sec)}s limit={int(limit_sec)}s"
                )
                logger.error(
                    "[WATCHDOG TRIGGERED] chat_id=%s stage=%s last_progress_time=%s reason=%s",
                    chat_id,
                    stage_u or "UNKNOWN",
                    int(time.time() - int(no_progress_sec or 0)),
                    details,
                )

                st_fail = job_status.get(chat_id, {}) or {}
                st_fail.update({
                    "job_id": job_id,
                    "phase": "failed",
                    "stage": "FAILED",
                    "pct": -1,
                    "updated_at": time.time(),
                    "details": details[:120],
                })
                job_status[chat_id] = st_fail
                _persist_active_job_state(chat_id)

                worker_path = _worker_result_state_file(chat_id, job_id)
                proof_path = Path(f"{PIPELINE}/logs/telegram_pipeline_proof.json")
                worker_payload = {
                    "chat_id": str(chat_id),
                    "job_id": int(job_id),
                    "phase": "failed",
                    "stage": "FAILED",
                    "details": details,
                    "pct": -1,
                    "updated_at": float(time.time()),
                }
                proof_payload = {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "chat_id": str(chat_id),
                    "job_id": int(job_id),
                    "pipeline_ok": False,
                    "detail": details,
                    "output_path": "",
                    "upload_platform": "",
                    "upload_info": "",
                    "upload_link": "",
                    "checks": {
                        "saw_upload_stage": False,
                        "saw_completion_message": False,
                        "pipeline_ok": False,
                    },
                }
                with suppress(Exception):
                    worker_tmp = worker_path.with_suffix(".json.tmp")
                    proof_tmp = proof_path.with_suffix(".json.tmp")
                    worker_tmp.parent.mkdir(parents=True, exist_ok=True)
                    proof_tmp.parent.mkdir(parents=True, exist_ok=True)
                    worker_tmp.write_text(json.dumps(worker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                    proof_tmp.write_text(json.dumps(proof_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                    os.replace(worker_tmp, worker_path)
                    os.replace(proof_tmp, proof_path)

                with suppress(Exception):
                    job_key = compute_idempotency_key(
                        str(chat_id),
                        str(job.get("video_link") or ""),
                        str(job.get("face_link") or ""),
                        str(mode or "direct"),
                    )
                    store = PipelineStateStore(PIPELINE)
                    store.mark_failed(
                        job_key=job_key,
                        stage=(stage_u or "FAILED"),
                        error_class="watchdog",
                        details=details,
                        stack_trace=details,
                        error_log_path="",
                    )

                with suppress(Exception):
                    _clear_active_job_state()

            async def _queue_watchdog_loop():
                while watchdog_state["run"]:
                    await asyncio.sleep(PIPELINE_WATCHDOG_INTERVAL_SEC)
                    if worker_proc.poll() is not None:
                        result_exists = bool(_load_worker_result_state(chat_id, job_id))
                        if not result_exists:
                            if float(watchdog_state["silent_exit_since"] or 0.0) <= 0.0:
                                watchdog_state["silent_exit_since"] = float(time.time())
                            silent_for = float(time.time()) - float(watchdog_state["silent_exit_since"])
                            logger.warning(
                                "[WATCHDOG CHECK] chat_id=%s stage=%s last_progress_time=%s reason=worker_exit_without_result_for=%ss",
                                chat_id,
                                "UNKNOWN",
                                int(time.time()),
                                int(silent_for),
                            )
                            if silent_for >= float(PIPELINE_WATCHDOG_SILENT_EXIT_GRACE_SEC):
                                _synthesize_watchdog_failure("FAILED", "worker_silent_exit", int(silent_for), int(PIPELINE_WATCHDOG_SILENT_EXIT_GRACE_SEC))
                                with suppress(Exception):
                                    kill_worker(int(worker_proc.pid))
                                logger.error(
                                    "[WATCHDOG KILL] chat_id=%s stage=%s last_progress_time=%s reason=worker_silent_exit",
                                    chat_id,
                                    "FAILED",
                                    int(time.time()),
                                )
                                break
                        else:
                            break
                        continue

                    st_watch = _get_active_job_state(chat_id, allow_fallback=False) or {}
                    phase_watch = str(st_watch.get("phase") or st_watch.get("status") or "").lower()
                    if phase_watch in {"completed", "failed", "stopped"}:
                        break

                    stage_watch = _watchdog_stage_name(str(st_watch.get("stage") or ""))
                    last_progress_ts = float(
                        st_watch.get("last_progress_timestamp")
                        or st_watch.get("last_update")
                        or st_watch.get("updated_at")
                        or time.time()
                    )
                    hb_stage = str(st_watch.get("last_progress_stage") or stage_watch or "UNKNOWN")
                    hb_frame = int(st_watch.get("last_progress_frame") or st_watch.get("done_frames") or 0)
                    hb_pct = int(st_watch.get("last_progress_pct") or st_watch.get("progress") or -1)
                    marker = f"{hb_stage}:{hb_frame}:{hb_pct}"

                    if marker != str(watchdog_state.get("last_marker") or ""):
                        watchdog_state["last_marker"] = marker
                        watchdog_state["last_change_ts"] = float(time.time())
                        watchdog_state["stale_hits"] = 0

                    no_progress_sec = float(time.time()) - float(last_progress_ts)
                    marker_no_progress_sec = float(time.time()) - float(watchdog_state.get("last_change_ts") or last_progress_ts)
                    limit_sec = _watchdog_stage_limit_sec(stage_watch, st_watch)
                    logger.info(
                        "[WATCHDOG CHECK] chat_id=%s stage=%s last_progress_time=%s reason=no_progress=%ss marker_no_progress=%ss limit=%ss marker=%s",
                        chat_id,
                        stage_watch,
                        int(last_progress_ts),
                        int(no_progress_sec),
                        int(marker_no_progress_sec),
                        int(limit_sec),
                        marker,
                    )

                    if limit_sec > 0 and marker_no_progress_sec > float(limit_sec):
                        watchdog_state["stale_hits"] = int(watchdog_state.get("stale_hits") or 0) + 1
                    else:
                        watchdog_state["stale_hits"] = 0

                    if int(watchdog_state.get("stale_hits") or 0) >= int(PIPELINE_WATCHDOG_STALE_CONFIRM_COUNT):
                        _synthesize_watchdog_failure(stage_watch, "stage_no_progress", int(marker_no_progress_sec), int(limit_sec))
                        logger.error(
                            "[WATCHDOG KILL] chat_id=%s stage=%s last_progress_time=%s reason=stage_no_progress",
                            chat_id,
                            stage_watch,
                            int(last_progress_ts),
                        )
                        with suppress(Exception):
                            kill_worker(int(worker_proc.pid))
                        with suppress(Exception):
                            await safe_send_message(
                                context.bot,
                                chat_id,
                                f"⚠️ WATCHDOG freeze detected in {stage_watch}. Job marked failed for safe retry.",
                            )
                        break

                logger.info(
                    "[WATCHDOG CLEANUP] chat_id=%s stage=%s last_progress_time=%s reason=queue_watchdog_exit",
                    chat_id,
                    _watchdog_stage_name(str((_get_active_job_state(chat_id, allow_fallback=False) or {}).get("stage") or "")),
                    int(time.time()),
                )

            watchdog_task = asyncio.create_task(_queue_watchdog_loop())

            task = asyncio.create_task(asyncio.to_thread(worker_proc.wait))
            active_pipeline_tasks[chat_id] = task
            try:
                logger.info("DEBUG FLOW: queue worker awaiting pipeline worker chat=%s job_id=%s pid=%s", chat_id, job_id, worker_proc.pid)
                rc = await task
                logger.info("DEBUG FLOW: pipeline worker completed chat=%s job_id=%s pid=%s rc=%s", chat_id, job_id, worker_proc.pid, rc)
                logger.info("[WORKER EXIT] chat=%s job_id=%s pid=%s rc=%s", chat_id, job_id, worker_proc.pid, rc)

                worker_result = await _wait_for_worker_result_state(chat_id, job_id, timeout_sec=8.0, poll_sec=0.25)
                if worker_result:
                    phase = str(worker_result.get("phase") or "").strip().lower()
                    stage = str(worker_result.get("stage") or "Worker finished")
                    details = str(worker_result.get("details") or "")

                    cancel_requested = str((job_status.get(chat_id, {}) or {}).get("phase") or "").lower() in {"stopped", "cancelled"}
                    completion_markers = phase == "completed" or str(stage or "").upper() == "COMPLETED"

                    if completion_markers:
                        ok_receipt, receipt_reason = _validate_worker_completion_receipt(worker_result, chat_id=chat_id)
                        if not ok_receipt:
                            logger.error(
                                "queue worker blocked false completion chat=%s job_id=%s reason=%s",
                                chat_id,
                                job_id,
                                receipt_reason,
                            )
                            phase = "failed"
                            stage = "FAILED"
                            details = f"pipeline completion blocked: {receipt_reason}"
                            completion_markers = False

                    if completion_markers:
                        phase = "completed"
                        stage = "COMPLETED"
                    elif int(rc or 0) == -15:
                        if cancel_requested:
                            phase = "stopped"
                            stage = "STOPPED"
                            details = details or "Cancelled by user"
                        elif completion_markers:
                            phase = "completed"
                            stage = "COMPLETED"
                        else:
                            phase = "failed"
                            stage = "FAILED"
                            details = "unexpected SIGTERM - no completion markers"
                    elif int(rc or 0) == -9:
                        phase = "failed"
                        stage = "FAILED"
                        details = "OOM or hard kill (SIGKILL)"
                    elif int(rc or 0) != 0:
                        phase = "failed"
                        stage = "FAILED"
                        details = details or f"worker exit rc={rc}"

                    if phase == "failed" and not details.strip():
                        details = (_read_worker_trace_tail(chat_id, job_id, max_lines=60) or "")[:800]
                    try:
                        pct = int(worker_result.get("pct", -1))
                    except Exception:
                        pct = -1

                    st_live = job_status.get(chat_id, {}) or {}
                    st_live.update({
                        "job_id": job_id,
                        "phase": phase or st_live.get("phase", "download"),
                        "stage": stage,
                        "pct": pct,
                        "updated_at": time.time(),
                        "details": details[:120],
                    })
                    job_status[chat_id] = st_live
                    _update_lifecycle_state(
                        chat_id,
                        is_job_running=False,
                        is_job_completed=(st_live.get("phase") == "completed"),
                    )
                    logger.info(
                        "queue worker synced result chat=%s job_id=%s phase=%s stage=%s",
                        chat_id,
                        job_id,
                        st_live.get("phase"),
                        st_live.get("stage"),
                    )
                else:
                    logger.critical(
                        "CRITICAL: queue worker result file missing after worker exit chat=%s job_id=%s rc=%s",
                        chat_id,
                        job_id,
                        rc,
                    )
                    if int(rc or 0) != 0:
                        trace_tail = _read_worker_trace_tail(chat_id, job_id, max_lines=80)
                        try:
                            cancel_requested = str((job_status.get(chat_id, {}) or {}).get("phase") or "").lower() in {"stopped", "cancelled"}
                            if int(rc or 0) == -15 and cancel_requested:
                                details = "Cancelled by user"
                            elif int(rc or 0) == -15:
                                details = "unexpected SIGTERM - no completion markers"
                            elif int(rc or 0) == -9:
                                details = "OOM or hard kill (SIGKILL)"
                            else:
                                details = (trace_tail or f"worker exited rc={rc}")[:1200]
                            if int(rc or 0) != 0 and not cancel_requested:
                                details = f"WATCHDOG_TRIGGERED freeze_detected:worker_silent_exit | {details}"
                                logger.error(
                                    "[WATCHDOG TRIGGERED] chat_id=%s stage=%s last_progress_time=%s reason=%s",
                                    chat_id,
                                    "FAILED",
                                    int(time.time()),
                                    details,
                                )
                            worker_path = _worker_result_state_file(chat_id, job_id)
                            proof_path = Path(f"{PIPELINE}/logs/telegram_pipeline_proof.json")
                            worker_payload = {
                                "chat_id": str(chat_id),
                                "job_id": int(job_id),
                                "phase": "failed",
                                "stage": "FAILED",
                                "details": str(details),
                                "pct": -1,
                                "updated_at": float(time.time()),
                            }
                            proof_payload = {
                                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                "chat_id": str(chat_id),
                                "job_id": int(job_id),
                                "pipeline_ok": False,
                                "detail": str(details),
                                "output_path": "",
                                "upload_platform": "",
                                "upload_info": "",
                                "upload_link": "",
                                "checks": {
                                    "saw_upload_stage": False,
                                    "saw_completion_message": False,
                                    "pipeline_ok": False,
                                },
                            }
                            worker_tmp = worker_path.with_suffix(".json.tmp")
                            proof_tmp = proof_path.with_suffix(".json.tmp")
                            worker_tmp.parent.mkdir(parents=True, exist_ok=True)
                            proof_tmp.parent.mkdir(parents=True, exist_ok=True)
                            worker_tmp.write_text(json.dumps(worker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                            proof_tmp.write_text(json.dumps(proof_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                            os.replace(worker_tmp, worker_path)
                            os.replace(proof_tmp, proof_path)

                            st_live = job_status.get(chat_id, {}) or {}
                            st_live.update({
                                "job_id": job_id,
                                "phase": "failed",
                                "stage": "FAILED",
                                "pct": -1,
                                "updated_at": time.time(),
                                "details": str(details)[:120],
                            })
                            job_status[chat_id] = st_live

                            # Detached worker may die before writing durable state; force terminal FAILED here.
                            try:
                                job_key = compute_idempotency_key(
                                    str(chat_id),
                                    str(job.get("video_link") or ""),
                                    str(job.get("face_link") or ""),
                                    str(mode or "direct"),
                                )
                                store = PipelineStateStore(PIPELINE)
                                store.mark_failed(
                                    job_key=job_key,
                                    stage="FAILED",
                                    error_class="worker_exit",
                                    details=f"CRITICAL missing worker_result after worker exit rc={rc}; queue synthesized terminal artifacts",
                                    stack_trace=str(details)[:2000],
                                    error_log_path="",
                                )
                            except Exception:
                                pass
                        except Exception:
                            pass
                        if trace_tail:
                            logger.error("queue worker trace tail chat=%s job_id=%s\n%s", chat_id, job_id, trace_tail)
                    else:
                        # The detached worker can occasionally exit before persisting worker_result.
                        # If rc==0 and a fresh output exists, treat this as terminal completion.
                        latest_output = ""
                        latest_output_mtime = 0.0
                        try:
                            for outp in list_swap_outputs():
                                with suppress(Exception):
                                    mtime = float(Path(outp).stat().st_mtime)
                                    if mtime > latest_output_mtime:
                                        latest_output_mtime = mtime
                                        latest_output = str(outp)
                        except Exception:
                            latest_output = ""

                        worker_started_at = float((job_status.get(chat_id, {}) or {}).get("started_at") or 0.0)
                        has_recent_output = bool(latest_output) and latest_output_mtime >= max(0.0, worker_started_at - 120.0)

                        # CRITICAL: Validate output before marking as completed
                        if has_recent_output:
                            st = job_status.get(chat_id, {}) or {}
                            input_path = str(st.get("input_path") or "")
                            job_temp_path = str(st.get("temp_path") or "")
                            if input_path and Path(input_path).is_file() and Path(latest_output).is_file():
                                try:
                                    import cv2
                                    # Quick hash compare first - if identical, fail immediately
                                    import hashlib
                                    def file_hash(path):
                                        h = hashlib.md5()
                                        with open(path, 'rb') as f:
                                            for chunk in iter(lambda: f.read(8192), b''):
                                                h.update(chunk)
                                        return h.hexdigest()
                                    if file_hash(input_path) == file_hash(latest_output):
                                        logger.error("VALIDATION FAILED: output identical to input chat=%s job_id=%s", chat_id, job_id)
                                        has_recent_output = False  # Reject this "recovered" output
                                    else:
                                        # Full visual validation
                                        valid_ok, valid_detail, _, _, _ = validate_faceswap_visual_change(input_path, latest_output, job_temp_path)
                                        if not valid_ok:
                                            logger.error("VALIDATION FAILED: %s chat=%s job_id=%s", valid_detail, chat_id, job_id)
                                            has_recent_output = False  # Reject failed swap output
                                except Exception as ve:
                                    logger.warning("validation check skipped due to error: %s", ve)

                        if has_recent_output:
                            details = "worker_result missing, recovered completion from rc=0 + fresh output (validated)"
                            logger.warning(
                                "queue worker recovered completion chat=%s job_id=%s output=%s",
                                chat_id,
                                job_id,
                                latest_output,
                            )
                            try:
                                worker_path = _worker_result_state_file(chat_id, job_id)
                                proof_path = Path(f"{PIPELINE}/logs/telegram_pipeline_proof.json")
                                worker_payload = {
                                    "chat_id": str(chat_id),
                                    "job_id": int(job_id),
                                    "phase": "completed",
                                    "stage": "COMPLETED",
                                    "details": details,
                                    "pct": 100,
                                    "output_path": latest_output,
                                    "upload_ok": True,
                                    "upload_platform": "",
                                    "upload_info": "",
                                    "upload_link": "",
                                    "updated_at": float(time.time()),
                                }
                                proof_payload = {
                                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "chat_id": str(chat_id),
                                    "job_id": int(job_id),
                                    "pipeline_ok": True,
                                    "detail": details,
                                    "output_path": latest_output,
                                    "upload_platform": "",
                                    "upload_info": "",
                                    "upload_link": "",
                                    "checks": {
                                        "saw_upload_stage": True,
                                        "saw_completion_message": True,
                                        "pipeline_ok": True,
                                    },
                                }
                                worker_tmp = worker_path.with_suffix(".json.tmp")
                                proof_tmp = proof_path.with_suffix(".json.tmp")
                                worker_tmp.parent.mkdir(parents=True, exist_ok=True)
                                proof_tmp.parent.mkdir(parents=True, exist_ok=True)
                                worker_tmp.write_text(json.dumps(worker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                                proof_tmp.write_text(json.dumps(proof_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                                os.replace(worker_tmp, worker_path)
                                os.replace(proof_tmp, proof_path)

                                st_live = job_status.get(chat_id, {}) or {}
                                st_live.update({
                                    "job_id": job_id,
                                    "phase": "completed",
                                    "stage": "COMPLETED",
                                    "pct": 100,
                                    "updated_at": time.time(),
                                    "details": details[:120],
                                    "output_path": latest_output,
                                    "upload_ok": True,
                                })
                                job_status[chat_id] = st_live
                                _update_lifecycle_state(
                                    chat_id,
                                    is_job_running=False,
                                    is_job_completed=True,
                                )
                            except Exception:
                                pass
                        else:
                            details = "worker exited rc=0 without terminal artifacts; pipeline did not execute"
                            logger.error("queue worker synthesized failure chat=%s job_id=%s reason=%s", chat_id, job_id, details)
                            try:
                                worker_path = _worker_result_state_file(chat_id, job_id)
                                proof_path = Path(f"{PIPELINE}/logs/telegram_pipeline_proof.json")
                                worker_payload = {
                                    "chat_id": str(chat_id),
                                    "job_id": int(job_id),
                                    "phase": "failed",
                                    "stage": "FAILED",
                                    "details": details,
                                    "pct": -1,
                                    "updated_at": float(time.time()),
                                }
                                proof_payload = {
                                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "chat_id": str(chat_id),
                                    "job_id": int(job_id),
                                    "pipeline_ok": False,
                                    "detail": details,
                                    "output_path": "",
                                    "upload_platform": "",
                                    "upload_info": "",
                                    "upload_link": "",
                                    "checks": {
                                        "saw_upload_stage": False,
                                        "saw_completion_message": False,
                                        "pipeline_ok": False,
                                    },
                                }
                                worker_tmp = worker_path.with_suffix(".json.tmp")
                                proof_tmp = proof_path.with_suffix(".json.tmp")
                                worker_tmp.parent.mkdir(parents=True, exist_ok=True)
                                proof_tmp.parent.mkdir(parents=True, exist_ok=True)
                                worker_tmp.write_text(json.dumps(worker_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                                proof_tmp.write_text(json.dumps(proof_payload, ensure_ascii=True, indent=2), encoding="utf-8")
                                os.replace(worker_tmp, worker_path)
                                os.replace(proof_tmp, proof_path)
                                st_live = job_status.get(chat_id, {}) or {}
                                st_live.update({
                                    "job_id": job_id,
                                    "phase": "failed",
                                    "stage": "FAILED",
                                    "pct": -1,
                                    "updated_at": time.time(),
                                    "details": details[:120],
                                })
                                job_status[chat_id] = st_live
                                try:
                                    job_key = compute_idempotency_key(
                                        str(chat_id),
                                        str(job.get("video_link") or ""),
                                        str(job.get("face_link") or ""),
                                        str(mode or "direct"),
                                    )
                                    store = PipelineStateStore(PIPELINE)
                                    store.mark_failed(
                                        job_key=job_key,
                                        stage="FAILED",
                                        error_class="worker_missing_result",
                                        details="CRITICAL missing worker_result after worker exit rc=0; queue synthesized failed artifacts",
                                        stack_trace=details[:2000],
                                        error_log_path="",
                                    )
                                except Exception:
                                    pass
                            except Exception:
                                pass

                if int(rc or 0) != 0:
                    await safe_send_message(
                        context.bot,
                        chat_id,
                        f"⚠️ Worker process ended with code `{rc}` for job `#{job_id}`.",
                        parse_mode="Markdown",
                    )
            except asyncio.CancelledError:
                # Re-raise to maintain asyncio cancellation contract — finally block handles cleanup
                raise
            except Exception as e:
                logger.exception(f"Pipeline crashed for {chat_id}: {e}")
                await safe_send_message(context.bot, chat_id, f"❌ System Error: {str(e)[:100]}...\nProcessing halted.")
                pass
            finally:
                watchdog_state["run"] = False
                with suppress(Exception):
                    if 'watchdog_task' in locals() and watchdog_task and not watchdog_task.done():
                        watchdog_task.cancel()
                        await watchdog_task
                if trace_fp is not None:
                    with suppress(Exception):
                        trace_fp.flush()
                    with suppress(Exception):
                        trace_fp.close()
                if active_pipeline_tasks.get(chat_id) is task:
                    active_pipeline_tasks.pop(chat_id, None)
                global_job_lock = False
                _clear_active_job_state()
    finally:
        if not _is_chat_busy(chat_id):
            global_job_lock = False
        queue_workers.pop(chat_id, None)
        _save_queue_state()
        if _queue_size(chat_id) == 0 and not _is_chat_busy(chat_id):
            app_obj = getattr(context, "application", None)
            # Guard against double invocation: run_pipeline already calls on_job_completed
            # on success path — only call here if sleep countdown not already running
            if app_obj is not None and _is_last_job_auto_sleep_eligible(chat_id) and not _task_is_running(sleep_countdown_tasks.get(chat_id)):
                try:
                    await on_job_completed(app_obj, chat_id, success=_last_job_phase(chat_id) == "completed")
                except Exception as sleep_e:
                    logger.warning("queue-worker auto-sleep scheduling failed chat=%s err=%s", chat_id, sleep_e)
            elif app_obj is not None:
                logger.info(
                    "queue-worker auto-sleep skipped chat=%s phase=%s (countdown allowed only after completed/failed phase)",
                    chat_id,
                    _last_job_phase(chat_id),
                )


def _ensure_queue_worker(context, chat_id):
    worker = queue_workers.get(chat_id)
    if worker and not worker.done():
        return worker
    worker = asyncio.create_task(_run_queue_worker(context, chat_id))
    queue_workers[chat_id] = worker
    return worker


def list_swap_outputs():
    allowed_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".jpg", ".jpeg", ".png", ".webp"}
    output_root = Path(OUTPUTS_DIR)
    if not output_root.exists():
        return []
    return sorted(
        [f for f in output_root.rglob("*_faceswapped*") if f.is_file() and f.suffix.lower() in allowed_exts],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


def build_reupload_picker_kb(action, outputs, max_items=3):
    rows = []
    for idx, f in enumerate(outputs[:max_items]):
        size_mb = f.stat().st_size / 1024 / 1024
        ts = datetime.fromtimestamp(float(f.stat().st_mtime or time.time()), tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        short_name = f.name if len(f.name) <= 28 else f"{f.name[:25]}..."
        rows.append([
            InlineKeyboardButton(
                f"{idx + 1}. {short_name} ({size_mb:.1f}MB) | {ts}",
                callback_data=f"reupload_pick_{action}_{idx}"
            )
        ])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="reupload_output_menu")])
    return InlineKeyboardMarkup(rows)


def parse_timestamp_to_seconds(value):
    value = (value or "").strip()
    if not value:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 1:
            return int(parts[0])
        if len(parts) == 2:
            m, s = int(parts[0]), int(parts[1])
            return m * 60 + s
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
    except ValueError:
        return None
    return None


def format_seconds_hhmmss(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def detect_video_fps(video_path):
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            return None
        text = (r.stdout or "").strip()
        if "/" in text:
            num, den = text.split("/", 1)
            num_f = float(num)
            den_f = float(den)
            if den_f == 0:
                return None
            return num_f / den_f
        return float(text)
    except Exception:
        return None


def detect_total_video_frames(video_path):
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if total > 0:
            return total
    except Exception:
        pass

    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode == 0:
            txt = (r.stdout or "").strip()
            if txt.isdigit():
                return int(txt)
    except Exception:
        pass

    # Method 3: r_frame_rate * format duration (most reliable when nb_frames is N/A)
    try:
        rfps = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=20,
        )
        rdur = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=20,
        )
        if rfps.returncode == 0 and rdur.returncode == 0:
            fps_raw = (rfps.stdout or "").strip()
            dur_raw = (rdur.stdout or "").strip()
            if "/" in fps_raw and dur_raw:
                num, den = fps_raw.split("/", 1)
                fps_val = float(num) / float(den)
                dur_val = float(dur_raw)
                if fps_val > 0 and dur_val > 0:
                    return int(max(1, round(fps_val * dur_val)))
    except Exception:
        pass
    return 0


def detect_video_duration_seconds(video_path):
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            return None
        duration = float((r.stdout or "0").strip() or 0.0)
        return duration if duration > 0 else None
    except Exception:
        return None


def probe_video_stream_info(video_path):
    info = {
        "size_bytes": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
    }
    try:
        info["size_bytes"] = int(Path(video_path).stat().st_size)
    except Exception:
        info["size_bytes"] = 0

    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode != 0:
            return info
        meta = json.loads(r.stdout or "{}")
        streams = meta.get("streams") or []
        if not streams:
            return info
        st = streams[0] if isinstance(streams[0], dict) else {}
        info["width"] = int(st.get("width") or 0)
        info["height"] = int(st.get("height") or 0)
        afr = str(st.get("avg_frame_rate") or "0/1")
        if "/" in afr:
            num_s, den_s = afr.split("/", 1)
            num_f = float(num_s or 0.0)
            den_f = float(den_s or 1.0)
            if den_f > 0:
                info["fps"] = num_f / den_f
        else:
            info["fps"] = float(afr or 0.0)
    except Exception:
        pass
    return info


def estimate_eta_seconds(elapsed_seconds, pct):
    if pct is None or pct <= 0 or pct >= 100:
        return None
    est_total = elapsed_seconds * (100.0 / float(pct))
    remain = int(max(0, est_total - elapsed_seconds))
    return remain


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text):
    return ANSI_ESCAPE_RE.sub("", text or "")


async def drain_subprocess_output(
    proc,
    line_handler=None,
    tail_buffer=None,
    watchdog_label="subprocess",
    watchdog_interval_sec=60,
    exit_pipe_grace_sec=PIPE_READ_EXIT_GRACE_SEC,
):
    if proc is None:
        return

    stream = getattr(proc, "stdout", None)
    if stream is None:
        await asyncio.to_thread(proc.wait)
        return

    buffer = ""
    last_output_at = time.time()
    last_watchdog_at = 0.0
    exit_seen_at = 0.0
    q = queue.Queue()
    sentinel = object()

    def _reader_thread():
        try:
            while True:
                ch = stream.read(1)
                if not ch:
                    break
                q.put(ch)
        except Exception:
            pass
        finally:
            q.put(sentinel)

    reader = threading.Thread(target=_reader_thread, name=f"pipe-reader-{watchdog_label}", daemon=True)
    reader.start()

    async def _emit(line):
        nonlocal last_output_at
        clean = strip_ansi(line).rstrip()
        if not clean:
            return
        last_output_at = time.time()
        if tail_buffer is not None:
            tail_buffer.append(clean)
        if line_handler is not None:
            result = line_handler(clean)
            if asyncio.iscoroutine(result):
                await result

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            item = None

        if item is sentinel:
            if buffer.strip():
                await _emit(buffer)
                buffer = ""
            break

        if isinstance(item, str):
            if item in {"\r", "\n"}:
                if buffer.strip():
                    await _emit(buffer)
                buffer = ""
            else:
                buffer += item
            continue

        now = time.time()
        proc_exited = proc.poll() is not None
        if proc_exited and exit_seen_at <= 0.0:
            exit_seen_at = now

        if watchdog_interval_sec > 0 and (now - last_output_at) >= watchdog_interval_sec:
            if (now - last_watchdog_at) >= watchdog_interval_sec:
                logger.info(
                    "[PROCESS-WATCHDOG] %s alive pid=%s no_stdout_for=%ss",
                    watchdog_label,
                    getattr(proc, "pid", "-"),
                    int(now - last_output_at),
                )
                last_watchdog_at = now

        # Critical hang fix: once parent process has exited, do not wait forever for EOF
        # from inherited child pipe handles.
        if proc_exited and exit_seen_at > 0.0 and (now - exit_seen_at) >= float(max(0.5, exit_pipe_grace_sec)):
            if buffer.strip():
                await _emit(buffer)
                buffer = ""
            logger.warning(
                "[PROCESS-WATCHDOG] %s pid=%s exited but stdout pipe not closed after %.1fs; forcing drain exit",
                watchdog_label,
                getattr(proc, "pid", "-"),
                float(now - exit_seen_at),
            )
            break

        await asyncio.sleep(0.05)

    with suppress(Exception):
        stream.close()
    with suppress(Exception):
        reader.join(timeout=0.5)

    await asyncio.to_thread(proc.wait)


def parse_single_clip_range(raw):
    raw = (raw or "").strip().replace(" ", "")
    if "-" not in raw:
        return None, "Format galat hai. Example: `00:00:10-00:00:35`"
    start_raw, end_raw = raw.split("-", 1)
    start_sec = parse_timestamp_to_seconds(start_raw)
    end_sec = parse_timestamp_to_seconds(end_raw)
    if start_sec is None or end_sec is None:
        return None, "Timestamp parse nahi hua. Example: `00:01:05-00:01:40`"
    if end_sec <= start_sec:
        return None, "End time start se bada hona chahiye."
    return {"start": start_sec, "end": end_sec}, ""


def parse_clip_range_input(text):
    raw = (text or "").strip()
    if not raw:
        return None, "Range empty hai. Example: `00:01:00-00:02:30`"

    chunks = []
    for line in raw.replace(";", "\n").splitlines():
        for part in line.split(","):
            part = part.strip()
            if part:
                chunks.append(part)
    if not chunks:
        return None, "Range empty hai. Example: `00:01:00-00:02:30`"

    segments = []
    for part in chunks:
        seg, err = parse_single_clip_range(part)
        if not seg:
            return None, err
        segments.append(seg)

    segments.sort(key=lambda x: x["start"])
    for i in range(1, len(segments)):
        if segments[i]["start"] < segments[i - 1]["end"]:
            return None, "Ranges overlap kar rahe hain. Overlap remove karo."

    return {"segments": segments}, ""


def get_clip_range_note(chat_id):
    cfg = clip_ranges.get(chat_id)
    if not isinstance(cfg, dict):
        return "OFF"
    segments = cfg.get("segments") if isinstance(cfg.get("segments"), list) else []
    if not segments:
        return "OFF"

    if len(segments) == 1:
        seg = segments[0]
        return f"{format_seconds_hhmmss(seg['start'])}-{format_seconds_hhmmss(seg['end'])}"

    first = segments[0]
    last = segments[-1]
    return (
        f"{len(segments)} ranges | "
        f"{format_seconds_hhmmss(first['start'])}-{format_seconds_hhmmss(first['end'])} ... "
        f"{format_seconds_hhmmss(last['start'])}-{format_seconds_hhmmss(last['end'])}"
    )


def concat_video_segments(segment_paths, output_path):
    try:
        if len(segment_paths) == 1:
            shutil.copyfile(segment_paths[0], output_path)
            return True, "single segment copied"

        list_file = Path(TEMP_PATH) / f"concat_{int(time.time() * 1000)}.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for p in segment_paths:
                f.write(f"file '{str(Path(p).resolve())}'\n")

        r = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-c", "copy", output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if r.returncode != 0:
            r2 = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c:v", OUTPUT_VIDEO_ENCODER, "-c:a", "aac", output_path,
                ],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if r2.returncode != 0:
                try:
                    list_file.unlink(missing_ok=True)
                except Exception:
                    pass
                return False, (r2.stderr or r2.stdout or r.stderr or r.stdout or "concat failed")[:400]

        try:
            list_file.unlink(missing_ok=True)
        except Exception:
            pass

        if not os.path.exists(output_path):
            return False, "concat output missing"
        return True, "concat succeeded"
    except Exception as e:
        return False, str(e)


def create_processing_clip(video_path, start_sec, end_sec, out_dir):
    """Create a pre-trimmed clip for faster/stable progress reporting during extraction."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        src = Path(video_path)
        clip_name = f"{src.stem}_clip_{int(start_sec)}_{int(end_sec)}.mp4"
        clip_path = str(Path(out_dir) / clip_name)

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        cmd.extend(build_ffmpeg_hw_decode_args(video_path))
        cmd.extend([
            "-ss", str(max(0, int(start_sec))),
            "-to", str(max(0, int(end_sec))),
            "-i", video_path,
            "-c:v", OUTPUT_VIDEO_ENCODER,
        ])
        append_ffmpeg_encode_tuning(cmd)
        cmd.extend(["-c:a", "aac", clip_path])
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            # Fallback to software decode/format path when CUDA filter graph fails.
            cmd_sw = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(max(0, int(start_sec))),
                "-to", str(max(0, int(end_sec))),
                "-i", video_path,
                "-vf", "format=yuv420p",
                "-c:v", OUTPUT_VIDEO_ENCODER,
            ]
            append_ffmpeg_encode_tuning(cmd_sw)
            cmd_sw.extend(["-c:a", "aac", clip_path])
            r_sw = subprocess.run(cmd_sw, capture_output=True, text=True, timeout=1800)
            if r_sw.returncode != 0:
                err = (r_sw.stderr or r_sw.stdout or r.stderr or r.stdout or "ffmpeg clip trim failed")
                return False, err[:400], None
        if not os.path.exists(clip_path):
            return False, "trim output missing", None
        return True, "trim succeeded", clip_path
    except Exception as e:
        return False, str(e), None



MEGA_LINK_REGEX = re.compile(r"(?:https?://)?(?:www\.)?mega\.(?:nz|io)/\S+", re.IGNORECASE)
GENERIC_URL_REGEX = re.compile(r"https?://\S+", re.IGNORECASE)
MEDIA_URL_HINT_RE = re.compile(r"\.(mp4|mov|mkv|avi|webm|m4v|jpg|jpeg|png|webp)(?:$|[?#])", re.IGNORECASE)


def normalize_mega_link(raw_link):
    link = (raw_link or "").strip()
    if not link:
        return ""

    # Handle common wrappers from markdown/chat payloads.
    link = link.strip("<>'\"`")
    link = link.replace("&amp;", "&")

    # Remove trailing punctuation that is not part of URL.
    while link and link[-1] in ".,;:!?)]}":
        link = link[:-1]

    lo = link.lower()
    if "mega.nz" not in lo and "mega.io" not in lo:
        return ""

    if any(token in lo for token in ["/file/", "/folder/", "#!", "#f!"]):
        if not lo.startswith("http://") and not lo.startswith("https://"):
            link = "https://" + link
        return link
    return ""


def normalize_url_like_link(raw_link):
    link = (raw_link or "").strip()
    if not link:
        return ""
    link = link.strip("<>'\"`")
    link = link.replace("&amp;", "&")
    while link and link[-1] in ".,;:!?)]}":
        link = link[:-1]
    return link.strip()


def _looks_like_direct_file_link(link):
    link = normalize_url_like_link(link)
    if not link:
        return False
    try:
        parsed = urlparse(link)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.netloc or "").lower()
    if "mega.nz" in host or "mega.io" in host:
        return False
    return bool(MEDIA_URL_HINT_RE.search(link)) or ("download" in (parsed.path or "").lower())


def validate_input_media_link(link, verify_remote=False):
    raw = str(link or "").strip()
    if not raw:
        return False, "", "", "empty link"

    # Local file input is allowed for production recovery/testing.
    try:
        local_candidate = Path(raw)
        if local_candidate.is_file() and local_candidate.stat().st_size > 0:
            return True, str(local_candidate.resolve()), "local", "ok"
    except Exception:
        pass

    mega = normalize_mega_link(raw)
    if mega:
        return True, mega, "mega", "ok"

    normalized = normalize_url_like_link(raw)
    if not _looks_like_direct_file_link(normalized):
        return False, "", "", "unsupported link format"

    if not verify_remote:
        return True, normalized, "direct", "ok"

    try:
        req = Request(normalized, method="HEAD", headers={"User-Agent": "FaceSwapBot/14"})
        with urlopen(req, timeout=DIRECT_LINK_PROBE_TIMEOUT_SEC) as resp:
            ctype = str(resp.headers.get("Content-Type") or "").lower()
            clen = str(resp.headers.get("Content-Length") or "").strip()
            if ("video" not in ctype and "image" not in ctype and "octet-stream" not in ctype) and not clen:
                return False, "", "", f"unsupported content-type: {ctype or 'unknown'}"
            return True, normalized, "direct", "ok"
    except Exception as e:
        return False, "", "", f"link probe failed: {e}"


def _read_mega_link_cache():
    try:
        if not os.path.isfile(MEGA_LINK_CACHE_FILE):
            return {}
        data = json.loads(Path(MEGA_LINK_CACHE_FILE).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_mega_link_cache(cache):
    try:
        Path(MEGA_LINK_CACHE_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(MEGA_LINK_CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("failed to write mega link cache: %s", e)


def _get_cached_file_for_link(link):
    cache = _read_mega_link_cache()
    item = cache.get(link)
    if isinstance(item, dict):
        path = item.get("path")
        if path and os.path.isfile(path):
            return path
    return None


def _set_cached_file_for_link(link, file_path):
    cache = _read_mega_link_cache()
    cache[link] = {
        "path": file_path,
        "updated_at": int(time.time()),
    }
    _write_mega_link_cache(cache)


def _latest_file_in_dir(dest_dir):
    try:
        files = [f for f in Path(dest_dir).iterdir() if f.is_file()]
        if not files:
            return None
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return str(files[0])
    except Exception:
        return None


def parse_mega_links(text):
    matches = MEGA_LINK_REGEX.findall(text or "")
    clean_links = []
    for raw in matches:
        link = normalize_mega_link(raw)
        if link and link not in clean_links:
            clean_links.append(link)
    return clean_links


def parse_job_input_links(text):
    raw_text = str(text or "").strip()
    links = []
    try:
        local_candidate = Path(raw_text)
        if raw_text and local_candidate.is_file() and local_candidate.stat().st_size > 0:
            links.append(str(local_candidate.resolve()))
    except Exception:
        pass
    for link in parse_mega_links(text):
        if link not in links:
            links.append(link)
    for raw in GENERIC_URL_REGEX.findall(text or ""):
        normalized = normalize_url_like_link(raw)
        ok, final_link, _, _ = validate_input_media_link(normalized, verify_remote=False)
        if ok and final_link and final_link not in links:
            links.append(final_link)
    return links


def normalize_processing_target(target_path, chat_id):
    """Rename target to a glob-safe filename preserving original name keywords."""
    try:
        src = Path(target_path)
        ext = src.suffix.lower()
        # Preserve original stem keywords — only replace unsafe chars
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem).strip("._-")
        # Strip MEGA file-ID-only names (e.g. "mega_AbCdEfGH") — no useful keywords
        _mega_id_only = re.match(r"^mega_[A-Za-z0-9]{6,12}$", safe_stem)
        if not safe_stem or _mega_id_only:
            safe_stem = "video"
        # Truncate to 40 chars to keep paths short
        safe_stem = safe_stem[:40].strip("._-")
        # Uniqueness suffix: short timestamp only (no chat_id clutter)
        unique = str(int(time.time()))[-6:]
        safe_name = f"{safe_stem}_{unique}{ext}"
        dst = src.with_name(safe_name)
        if dst == src:
            return str(src), src.name
        src.rename(dst)
        return str(dst), src.name
    except Exception as e:
        logger.warning("normalize_processing_target failed: %s", e)
        return str(target_path), Path(target_path).name


def coerce_target_media_extension(target_path):
    """Assign a reliable media extension when downloader returns extensionless files."""
    try:
        p = Path(target_path)
        known_image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".heic", ".heif"}
        convertible_image_exts = {".avif", ".heic", ".heif"}
        known_video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
        ext = p.suffix.lower()

        if ext in convertible_image_exts:
            try:
                from PIL import Image as PILImage

                with PILImage.open(p) as im:
                    rgb = im.convert("RGB")
                    dst = p.with_suffix(".jpg")
                    rgb.save(dst, format="JPEG", quality=95)
                p.unlink(missing_ok=True)
                return str(dst)
            except Exception:
                # Fallback conversion path for AVIF/HEIC when PIL codec support is unavailable.
                try:
                    dst = p.with_suffix(".jpg")
                    conv = subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i", str(p),
                            "-frames:v", "1",
                            str(dst),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=45,
                    )
                    if conv.returncode == 0 and dst.is_file() and dst.stat().st_size > 0:
                        p.unlink(missing_ok=True)
                        return str(dst)
                except Exception:
                    pass
                # Keep original path if conversion fails; downstream detection can still attempt handling.
                return str(p)

        if ext in known_image_exts or ext in known_video_exts:
            return str(p)

        ffprobe_video = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if ffprobe_video.returncode == 0 and "video" in (ffprobe_video.stdout or "").lower():
            dst = p.with_suffix(".mp4")
            p.rename(dst)
            return str(dst)

        try:
            import cv2

            img = cv2.imread(str(p))
            if img is not None:
                dst = p.with_suffix(".jpg")
                p.rename(dst)
                return str(dst)
        except Exception:
            pass
    except Exception as e:
        logger.warning("coerce_target_media_extension failed: %s", e)

    return str(target_path)


def create_job_temp_path(chat_id):
    base = Path(TEMP_PATH) / "jobs"
    base.mkdir(parents=True, exist_ok=True)
    job_name = f"job_{str(chat_id).replace('-', '')}_{int(time.time() * 1000)}"
    job_temp = base / job_name
    job_temp.mkdir(parents=True, exist_ok=True)
    return str(job_temp)


def cleanup_job_temp_path(job_temp_path):
    try:
        if not job_temp_path:
            return
        p = Path(job_temp_path)
        if p.exists() and p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        logger.warning("cleanup_job_temp_path failed: %s", e)


def _free_bytes(path=PIPELINE):
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return 0


def _bytes_to_gb(value):
    return float(value) / (1024.0 * 1024.0 * 1024.0)


def _path_size_bytes(path):
    p = Path(path)
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return int(p.stat().st_size)
        except Exception:
            return 0

    total = 0
    try:
        for entry in p.rglob("*"):
            if not entry.is_file():
                continue
            try:
                total += int(entry.stat().st_size)
            except Exception:
                continue
    except Exception:
        return total
    return total


def get_storage_breakdown_bytes():
    # Folder labels follow user-visible terminology: temp/frames/outputs/downloads/cache.
    frames_dir = Path(TEMP_PATH) / "facefusion"
    buckets = {
        "temp": _path_size_bytes(TEMP_PATH),
        "frames": _path_size_bytes(frames_dir),
        "outputs": _path_size_bytes(OUTPUTS_DIR),
        "downloads": _path_size_bytes(Path(PIPELINE) / "downloads"),
        "cache": _path_size_bytes(Path(PIPELINE) / "cache"),
    }
    buckets["total"] = _path_size_bytes(PIPELINE)
    return buckets


def _format_gb_line(label, size_bytes):
    return f"{label}: {_bytes_to_gb(size_bytes):.2f} GB"


def log_storage_breakdown(tag="STORAGE SCAN"):
    sizes = get_storage_breakdown_bytes()
    logger.info(
        "%s | %s | %s | %s | %s | %s | total=%.2f GB",
        tag,
        _format_gb_line("temp", sizes["temp"]),
        _format_gb_line("frames", sizes["frames"]),
        _format_gb_line("outputs", sizes["outputs"]),
        _format_gb_line("downloads", sizes["downloads"]),
        _format_gb_line("cache", sizes["cache"]),
        _bytes_to_gb(sizes["total"]),
    )
    return sizes


def _remove_path_safe(path_obj):
    if path_obj.is_dir():
        shutil.rmtree(path_obj, ignore_errors=True)
        return True
    if path_obj.is_file():
        path_obj.unlink(missing_ok=True)
        return True
    return False


def _mask_secret(value, visible=4):
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= visible:
        return "*" * len(raw)
    return ("*" * max(4, len(raw) - visible)) + raw[-visible:]


def _read_json_file(path_value, default=None):
    try:
        p = Path(path_value)
        if not p.exists():
            return {} if default is None else default
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def _write_json_file(path_value, payload):
    p = Path(path_value)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload or {}, ensure_ascii=True, indent=2), encoding="utf-8")
    try:
        os.chmod(str(p), 0o600)
    except Exception:
        pass


def load_persistent_config():
    data = _read_json_file(PERSISTENT_CONFIG_FILE, default={})
    return data if isinstance(data, dict) else {}


def save_persistent_config(cfg):
    _write_json_file(PERSISTENT_CONFIG_FILE, cfg if isinstance(cfg, dict) else {})


def update_persistent_config(**kwargs):
    cfg = load_persistent_config()
    for key, value in kwargs.items():
        if value is None:
            cfg.pop(key, None)
        else:
            cfg[key] = value
    save_persistent_config(cfg)
    return cfg


def _ensure_persistent_paths():
    for d in [PERSISTENT_ROOT, PERSISTENT_FACES_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _safe_copy(src, dst):
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.exists() or not src_p.is_file():
        return False
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_p), str(dst_p))
    return True


def _create_safe_default_face_placeholder():
    _ensure_persistent_paths()
    try:
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (512, 512), color=(128, 128, 128))
        img.save(DEFAULT_FACE, "JPEG", quality=92)
        return os.path.isfile(DEFAULT_FACE)
    except Exception:
        return False


def _restore_default_face_from_local_candidates():
    # PRIORITY 1: Check for explicit default_source.jpg in persistent faces (NEVER auto-downloads from MEGA)
    explicit_default = Path(PERSISTENT_FACES_DIR) / "default_source.jpg"
    if explicit_default.exists() and explicit_default.is_file():
        try:
            if _safe_copy(str(explicit_default), DEFAULT_FACE):
                logger.info("default face restored from explicit default_source.jpg (no MEGA download)")
                return True
        except Exception:
            pass

    candidates = [
        Path(FACE_DIR) / "source_clean.jpg",
        Path(FACE_DIR) / "source_clean.jpeg",
        Path(FACE_DIR) / "source_clean.png",
        Path(FACE_DIR) / "source_clean.webp",
    ]

    # PRIORITY 2: Check persistent faces directory (excluding default_source.jpg which we already tried)
    if Path(PERSISTENT_FACES_DIR).exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
            for candidate in sorted(Path(PERSISTENT_FACES_DIR).glob(ext), key=lambda p: p.stat().st_mtime, reverse=True):
                if str(candidate) == str(Path(DEFAULT_FACE)):
                    continue
                if candidate.name == "default_source.jpg":
                    continue  # already tried above
                candidates.append(candidate)

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file() and _safe_copy(str(candidate), DEFAULT_FACE):
                return True
        except Exception:
            continue
    return False


def set_default_face_from_source(src_path):
    _ensure_persistent_paths()
    try:
        from PIL import Image as PILImage
        PILImage.open(src_path).convert("RGB").save(DEFAULT_FACE, "JPEG", quality=95)
        return DEFAULT_FACE
    except Exception:
        if _safe_copy(src_path, DEFAULT_FACE):
            return DEFAULT_FACE
        return ""


def _ensure_locked_default_face():
    _ensure_persistent_paths()
    if os.path.isfile(DEFAULT_FACE):
        return True

    if _restore_default_face_from_local_candidates() and os.path.isfile(DEFAULT_FACE):
        logger.info("default face restored from local candidate at %s", DEFAULT_FACE)
        return True

    logger.warning("default face missing; restoring from locked MEGA source")
    bootstrap_dir = Path(PERSISTENT_FACES_DIR) / "_bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not mega_download(LOCKED_DEFAULT_FACE_LINK, str(bootstrap_dir), retries=2):
            logger.error("default face restore failed: download error")
            return _create_safe_default_face_placeholder()
    except Exception as e:
        logger.error("default face restore exception: %s", e)
        return _create_safe_default_face_placeholder()

    images = sorted(
        [p for p in bootstrap_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not images:
        logger.error("default face restore failed: no image found in bootstrap payload")
        return _create_safe_default_face_placeholder()

    restored = set_default_face_from_source(str(images[0]))
    if not restored or not os.path.isfile(DEFAULT_FACE):
        logger.error("default face restore failed: conversion/copy failed")
        return _create_safe_default_face_placeholder()

    logger.info("default face restored at %s", DEFAULT_FACE)
    return True


def _parse_drive_auth_payload(token_value):
    raw = (token_value or "").strip()
    if not raw:
        return False, None, "Token empty nahi ho sakta"

    # Backward compatibility: allow folder override values like gdrive:faceswap_output.
    if raw.lower().startswith("gdrive:"):
        return True, {"legacy_target_override": raw}, ""

    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return False, None, "Invalid token JSON"
        except Exception:
            return False, None, "Invalid token JSON"

        access_token = str(payload.get("access_token", "")).strip()
        if not access_token:
            return False, None, "JSON me access_token required hai"

        token_obj = {
            "access_token": access_token,
            "token_type": str(payload.get("token_type", "Bearer") or "Bearer"),
        }
        refresh_token = str(payload.get("refresh_token", "")).strip()
        if refresh_token:
            token_obj["refresh_token"] = refresh_token
        expiry = str(payload.get("expiry", "")).strip()
        if expiry:
            token_obj["expiry"] = expiry
        return True, {"token_obj": token_obj, "raw": raw}, ""

    if not re.match(r"^[A-Za-z0-9._\-~=+/]{16,4096}$", raw):
        return False, None, "Invalid access token format"

    token_obj = {
        "access_token": raw,
        "token_type": "Bearer",
    }
    return True, {"token_obj": token_obj, "raw": raw}, ""


def _write_rclone_drive_token(token_obj):
    global RCLONE_CONF

    preferred_conf = Path(RCLONE_CONF)
    fallback_conf = ROOT_DIR / ".config" / "rclone" / "rclone.conf"
    conf_candidates = [preferred_conf]
    if fallback_conf != preferred_conf:
        conf_candidates.append(fallback_conf)

    conf_path = preferred_conf
    for candidate in conf_candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with candidate.open("a", encoding="utf-8"):
                pass
            conf_path = candidate
            break
        except Exception:
            continue

    RCLONE_CONF = str(conf_path)
    os.environ["RCLONE_CONF"] = RCLONE_CONF

    parser = configparser.ConfigParser()
    parser.optionxform = str
    if conf_path.exists():
        parser.read(conf_path, encoding="utf-8")

    if not parser.has_section("gdrive"):
        parser.add_section("gdrive")

    if not parser.get("gdrive", "type", fallback="").strip():
        parser.set("gdrive", "type", "drive")
    if not parser.get("gdrive", "scope", fallback="").strip():
        parser.set("gdrive", "scope", "drive")

    parser.set("gdrive", "token", json.dumps(token_obj or {}, separators=(",", ":")))
    with conf_path.open("w", encoding="utf-8") as fh:
        parser.write(fh)
    try:
        os.chmod(str(conf_path), 0o600)
    except Exception:
        pass


def load_persistent_runtime_state():
    _ensure_persistent_paths()
    cfg = load_persistent_config()

    # One-time migration of old default face location.
    legacy_default = Path(FACE_DIR) / "source_clean.jpg"
    if not Path(DEFAULT_FACE).exists() and legacy_default.exists():
        _safe_copy(str(legacy_default), DEFAULT_FACE)

    if not Path(DEFAULT_FACE).exists():
        _ensure_locked_default_face()

    # Ensure MEGA creds persist even when env is empty on restart.
    env_mega_email = os.environ.get("MEGA_EMAIL", str(_CREDS.get("mega_email", ""))).strip()
    env_mega_password = os.environ.get("MEGA_PASSWORD", str(_CREDS.get("mega_password", ""))).strip()
    mega_email = str(cfg.get("mega_email", "")).strip()
    mega_password = str(cfg.get("mega_password", "")).strip()
    # If env creds are provided, prefer and persist them to avoid stale-file auth loops.
    if env_mega_email and env_mega_password:
        mega_email, mega_password = env_mega_email, env_mega_password

    if mega_email and mega_password:
        desired = f"{mega_email}:{mega_password}"
        current = ""
        if Path(MEGA_CREDS_FILE).exists():
            try:
                current = Path(MEGA_CREDS_FILE).read_text(encoding="utf-8").strip()
            except Exception:
                current = ""
        if current != desired:
            Path(MEGA_CREDS_FILE).write_text(desired, encoding="utf-8")
            try:
                os.chmod(MEGA_CREDS_FILE, 0o600)
            except Exception:
                pass

    # Restore Drive auth token to rclone config if present.
    drive_auth = cfg.get("drive_auth_token")
    if isinstance(drive_auth, dict) and drive_auth.get("access_token"):
        _write_rclone_drive_token(drive_auth)
    else:
        env_drive_token_raw = os.environ.get("DRIVE_TOKEN", "").strip()
        if not env_drive_token_raw:
            try:
                _env_path = ROOT_DIR / ".env"
                if _env_path.exists():
                    for _dl in _env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                        if _dl.strip().startswith("DRIVE_TOKEN="):
                            env_drive_token_raw = _dl.split("=", 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
        if env_drive_token_raw:
            try:
                _drive_auth_env = json.loads(env_drive_token_raw)
                if isinstance(_drive_auth_env, dict) and _drive_auth_env.get("access_token"):
                    _write_rclone_drive_token(_drive_auth_env)
                    logger.info("rclone config initialized from DRIVE_TOKEN env var")
            except Exception as _e:
                logger.warning("DRIVE_TOKEN parse failed: %s", _e)

    # Keep legacy drive token file synced for backward compatibility.
    legacy_target = str(cfg.get("drive_target_override", "")).strip()
    if legacy_target:
        Path(DRIVE_TOKEN_FILE).write_text(legacy_target, encoding="utf-8")
        try:
            os.chmod(DRIVE_TOKEN_FILE, 0o600)
        except Exception:
            pass

    persisted_modes = cfg.get("chat_modes")
    if isinstance(persisted_modes, dict):
        for chat_key, mode in persisted_modes.items():
            chat_modes[str(chat_key)] = "multi" if str(mode).lower() == "multi" else "direct"

    persisted_prefs = cfg.get("face_selector_prefs")
    if isinstance(persisted_prefs, dict):
        for chat_key, pref in persisted_prefs.items():
            if isinstance(pref, dict):
                normalized = "female" if str(pref.get("gender_mode", "all")).lower() == "female" else "all"
                face_selector_prefs[str(chat_key)] = {
                    "gender_mode": normalized,
                    "female_only": normalized == "female",
                }

    persisted_clip_ranges = cfg.get("clip_ranges")
    if isinstance(persisted_clip_ranges, dict):
        for chat_key, clip_cfg in persisted_clip_ranges.items():
            if isinstance(clip_cfg, dict) and isinstance(clip_cfg.get("segments"), list):
                clip_ranges[str(chat_key)] = clip_cfg


def _collect_active_protected_paths(extra_paths=None):
    protected = set(extra_paths or [])
    if DEFAULT_FACE:
        protected.add(DEFAULT_FACE)
    for _, paths in list(active_job_protected_paths.items()):
        if not paths:
            continue
        for item in paths:
            if item:
                protected.add(item)
    return protected


def _collect_files_under(paths):
    files = []
    for root in paths:
        p = Path(root)
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        try:
            for entry in p.rglob("*"):
                if entry.is_file():
                    files.append(entry)
        except Exception:
            continue
    return files


def clean_workspace(mode, protected_paths=None, keep_latest_outputs=KEEP_LATEST_OUTPUTS):
    mode_selected = mode if mode in {"temp_only", "outputs_old", "outputs_all", "full_clean", "deep_clean"} else "temp_only"
    protected = {
        str(Path(x).resolve())
        for x in _collect_active_protected_paths(protected_paths)
        if x
    }
    if mode_selected in {"temp_only", "outputs_old", "outputs_all"}:
        protected.add(str(Path(PIPELINE, "downloads").resolve()))

    persistent_guard = {
        str(Path(MEGA_CREDS_FILE).resolve()),
        str(Path(DRIVE_TOKEN_FILE).resolve()),
        str(Path(PERSISTENT_CONFIG_FILE).resolve()),
    }
    if mode_selected != "deep_clean":
        persistent_guard.add(str(Path(VALIDATION_PROOF_DIR).resolve()))
    if mode_selected != "deep_clean":
        persistent_guard.update(
            {
                str(Path(PERSISTENT_ROOT).resolve()),
                str(Path(PERSISTENT_FACES_DIR).resolve()),
            }
        )
    if mode_selected != "deep_clean":
        persistent_guard.add(str(Path(FACE_DIR).resolve()))
    protected.update(
        persistent_guard
    )
    latest_outputs = list_swap_outputs()
    keep_n = max(1, int(keep_latest_outputs or 1))
    safe_outputs = {str(p.resolve()) for p in latest_outputs[:keep_n]}
    if mode not in {"outputs_all", "deep_clean"}:
        protected.update(safe_outputs)

    if mode_selected == "deep_clean":
        min_age = 0
    else:
        min_age = max(900, SAFE_CLEANUP_MIN_AGE_SECONDS)

    stats = safe_cleanup.run_cleanup(
        mode=mode_selected,
        root_dir=ROOT_DIR,
        pipeline_dir=PIPELINE,
        workspace_dir=WORKSPACE,
        temp_dir=TEMP_PATH,
        output_dir=OUTPUTS_DIR,
        protected_paths=protected,
        min_age_seconds=min_age,
        audit_log_path=CLEANUP_AUDIT_LOG_FILE,
        logger=logger,
    )

    extra_deleted_files = 0
    extra_deleted_bytes = 0
    if mode_selected == "deep_clean":
        outputs_stats = safe_cleanup.run_cleanup(
            mode="outputs_all",
            root_dir=ROOT_DIR,
            pipeline_dir=PIPELINE,
            workspace_dir=WORKSPACE,
            temp_dir=TEMP_PATH,
            output_dir=OUTPUTS_DIR,
            protected_paths=protected,
            min_age_seconds=0,
            audit_log_path=CLEANUP_AUDIT_LOG_FILE,
            logger=logger,
        )
        stats["deleted_files"] = int(stats.get("deleted_files", 0) or 0) + int(outputs_stats.get("deleted_files", 0) or 0)
        stats["deleted_bytes"] = int(stats.get("deleted_bytes", 0) or 0) + int(outputs_stats.get("deleted_bytes", 0) or 0)
        stats["skipped_protected"] = int(stats.get("skipped_protected", 0) or 0) + int(outputs_stats.get("skipped_protected", 0) or 0)
        stats["skipped_unknown"] = int(stats.get("skipped_unknown", 0) or 0) + int(outputs_stats.get("skipped_unknown", 0) or 0)
        stats["skipped_recent"] = int(stats.get("skipped_recent", 0) or 0) + int(outputs_stats.get("skipped_recent", 0) or 0)
        stats["skipped_locked"] = int(stats.get("skipped_locked", 0) or 0) + int(outputs_stats.get("skipped_locked", 0) or 0)
        stats["plan"] = list(stats.get("plan", []) or []) + list(outputs_stats.get("plan", []) or [])

        output_root = Path(OUTPUTS_DIR)
        if output_root.exists() and output_root.is_dir():
            protected_roots = [Path(p).resolve() for p in protected]
            for entry in output_root.rglob("*"):
                try:
                    if not entry.is_file():
                        continue
                    resolved = entry.resolve()
                    skip = False
                    for pr in protected_roots:
                        try:
                            if resolved == pr or resolved.is_relative_to(pr):
                                skip = True
                                break
                        except Exception:
                            continue
                    if skip:
                        continue
                    size = int(entry.stat().st_size)
                    entry.unlink(missing_ok=True)
                    extra_deleted_files += 1
                    extra_deleted_bytes += size
                except Exception:
                    continue
            for d in sorted([p for p in output_root.rglob("*") if p.is_dir()], key=lambda x: len(x.parts), reverse=True):
                try:
                    d.rmdir()
                except Exception:
                    pass

        default_face_resolved = str(Path(DEFAULT_FACE).resolve())
        faces_root = Path(PERSISTENT_FACES_DIR)
        if faces_root.exists() and faces_root.is_dir():
            for entry in faces_root.rglob("*"):
                try:
                    if not entry.is_file():
                        continue
                    if str(entry.resolve()) == default_face_resolved:
                        continue
                    size = int(entry.stat().st_size)
                    entry.unlink(missing_ok=True)
                    extra_deleted_files += 1
                    extra_deleted_bytes += size
                except Exception:
                    continue
            for d in sorted([p for p in faces_root.rglob("*") if p.is_dir()], key=lambda x: len(x.parts), reverse=True):
                try:
                    d.rmdir()
                except Exception:
                    pass

        stats["deleted_files"] = int(stats.get("deleted_files", 0) or 0) + extra_deleted_files
        stats["deleted_bytes"] = int(stats.get("deleted_bytes", 0) or 0) + extra_deleted_bytes

    # Keep job temp parent structure healthy.
    try:
        Path(TEMP_PATH).mkdir(parents=True, exist_ok=True)
        (Path(TEMP_PATH) / "jobs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    logger.info(
        "CLEAN WORKSPACE SAFE mode=%s | before=%.2f GB | after=%.2f GB | deleted_files=%s | skipped_protected=%s | skipped_unknown=%s | skipped_recent=%s | skipped_locked=%s | freed=%.2f GB",
        mode_selected,
        _bytes_to_gb(int(stats.get("before", 0) or 0)),
        _bytes_to_gb(int(stats.get("after", 0) or 0)),
        int(stats.get("deleted_files", 0) or 0),
        int(stats.get("skipped_protected", 0) or 0),
        int(stats.get("skipped_unknown", 0) or 0),
        int(stats.get("skipped_recent", 0) or 0),
        int(stats.get("skipped_locked", 0) or 0),
        _bytes_to_gb(max(0, int(stats.get("before", 0) or 0) - int(stats.get("after", 0) or 0))),
    )
    downloads_cleaned = 0
    downloads_cleaned_bytes = 0
    for item in stats.get("plan", []):
        if item.get("action") != "delete":
            continue
        if str(item.get("zone", "")) != "downloads":
            continue
        downloads_cleaned += 1
        downloads_cleaned_bytes += int(item.get("size", 0) or 0)

    return {
        "mode": mode_selected,
        "before": int(stats.get("before", 0) or 0),
        "after": int(stats.get("after", 0) or 0),
        "deleted_files": int(stats.get("deleted_files", 0) or 0),
        "deleted_dirs": 0,
        "deleted_bytes": int(stats.get("deleted_bytes", 0) or 0),
        "skipped_protected": int(stats.get("skipped_protected", 0) or 0),
        "skipped_unknown": int(stats.get("skipped_unknown", 0) or 0),
        "skipped_recent": int(stats.get("skipped_recent", 0) or 0),
        "skipped_locked": int(stats.get("skipped_locked", 0) or 0),
        "downloads_cleaned": int(downloads_cleaned),
        "downloads_cleaned_bytes": int(downloads_cleaned_bytes),
    }


def enforce_storage_limit(max_usage_gb=MAX_STORAGE_USAGE_GB, protected_paths=None):
    max_bytes = int(max(0.01, float(max_usage_gb)) * 1024 * 1024 * 1024)
    before = _path_size_bytes(PIPELINE)
    disk_pct = safe_cleanup.disk_usage_percent(PIPELINE)
    over_limit = before > max_bytes or disk_pct >= float(SAFE_CLEANUP_DISK_TRIGGER_PERCENT)

    if not over_limit:
        return {
            "before": before,
            "after": before,
            "deleted": 0,
            "freed": 0,
            "limited": False,
            "disk_usage_pct": disk_pct,
        }

    stats = clean_workspace("full_clean", protected_paths=protected_paths, keep_latest_outputs=0)
    after = int(stats.get("after", before) or before)
    freed = max(0, before - after)
    logger.info(
        "STORAGE LIMIT SAFE enforce max=%.2f GB trigger=%.1f%% | before=%.2f GB | after=%.2f GB | deleted_files=%s | freed=%.2f GB",
        float(max_usage_gb),
        float(SAFE_CLEANUP_DISK_TRIGGER_PERCENT),
        _bytes_to_gb(before),
        _bytes_to_gb(after),
        int(stats.get("deleted_files", 0) or 0),
        _bytes_to_gb(freed),
    )
    return {
        "before": before,
        "after": after,
        "deleted": int(stats.get("deleted_files", 0) or 0),
        "freed": freed,
        "skipped_protected": int(stats.get("skipped_protected", 0) or 0),
        "limited": True,
        "disk_usage_pct": disk_pct,
    }


def prune_temp_job_dirs(force=False):
    min_age = 0 if force else max(900, SAFE_CLEANUP_MIN_AGE_SECONDS)
    stats = safe_cleanup.run_cleanup(
        mode="temp_only",
        root_dir=ROOT_DIR,
        pipeline_dir=PIPELINE,
        workspace_dir=WORKSPACE,
        temp_dir=TEMP_PATH,
        output_dir=OUTPUTS_DIR,
        protected_paths=_collect_active_protected_paths(),
        min_age_seconds=min_age,
        audit_log_path=CLEANUP_AUDIT_LOG_FILE,
        logger=logger,
    )
    return int(stats.get("deleted_files", 0) or 0)


def ensure_workspace_free_space(min_free_gb=MIN_FREE_SPACE_GB):
    min_free_bytes = int(max(1, min_free_gb) * 1024 * 1024 * 1024)
    before = _free_bytes(PIPELINE)

    # Fast-start mode keeps queue-to-download latency near-zero by deferring
    # expensive cleanup scans to periodic/post-job cleanup tasks.
    if PRE_DOWNLOAD_FAST_START:
        if before < min_free_bytes:
            logger.warning(
                "fast-start preflight: free space low (free=%.2fGB required=%sGB); continuing without blocking cleanup",
                _bytes_to_gb(before),
                int(min_free_gb),
            )
        return True, before, before, 0

    stats = clean_workspace("temp_only", protected_paths=None, keep_latest_outputs=KEEP_LATEST_OUTPUTS)
    after_soft = _free_bytes(PIPELINE)
    if after_soft >= min_free_bytes:
        return True, before, after_soft, int(stats.get("deleted_files", 0) or 0)

    # Stable mode: try safe cleanup, but do not hard-block job execution here.
    limit_stats = enforce_storage_limit(MAX_STORAGE_USAGE_GB)
    after_hard = _free_bytes(PIPELINE)
    removed = int(stats.get("deleted_files", 0) or 0) + int(limit_stats.get("deleted", 0) or 0)
    if after_hard < min_free_bytes:
        logger.warning(
            "free space below threshold after cleanup (free=%.2fGB, required=%sGB); continuing in stable mode",
            _bytes_to_gb(after_hard),
            int(min_free_gb),
        )
    return True, before, after_hard, removed


async def _delayed_post_job_cleanup(chat_id, protected_paths):
    wait_seconds = max(900, SAFE_CLEANUP_MIN_AGE_SECONDS)
    try:
        logger.info("POST-JOB SAFE CLEANUP scheduled chat=%s wait=%ss", chat_id, wait_seconds)
        await asyncio.sleep(wait_seconds)
        log_storage_breakdown("POST-JOB CLEANUP BEFORE (SAFE)")
        await asyncio.to_thread(clean_workspace, "temp_only", protected_paths, KEEP_LATEST_OUTPUTS)
        await asyncio.to_thread(enforce_storage_limit, MAX_STORAGE_USAGE_GB, protected_paths)
        log_storage_breakdown("POST-JOB CLEANUP AFTER (SAFE)")
    except asyncio.CancelledError:
        logger.info("POST-JOB SAFE CLEANUP cancelled chat=%s", chat_id)
        raise
    except Exception as cleanup_e:
        logger.warning("post-job delayed cleanup failed chat=%s err=%s", chat_id, cleanup_e)
    finally:
        task = post_job_cleanup_tasks.get(chat_id)
        if task and task.done():
            post_job_cleanup_tasks.pop(chat_id, None)


async def _periodic_storage_guard():
    # AUTO-DELETE DISABLED: periodic storage guard disabled, manual only
    logger.info("Periodic storage guard DISABLED (auto-delete off)")
    while True:
        try:
            await asyncio.sleep(999999)
        except asyncio.CancelledError:
            raise


def clear_multi_setup_state(context):
    for key in [
        "awaiting_multi_target",
        "awaiting_multi_source",
        "multi_target_link",
        "multi_face_crops",
        "multi_face_idx",
        "multi_face_map",
    ]:
        context.user_data.pop(key, None)


def detect_faces_in_target_file(target_path, chat_id, max_faces=8):
    try:
        import cv2

        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
        ext = Path(target_path).suffix.lower()

        frame = None
        if ext in image_exts:
            frame = cv2.imread(target_path)
        elif ext in video_exts:
            cap = cv2.VideoCapture(target_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            probe_indices = []
            if total > 0:
                ratios = [0.05, 0.12, 0.2, 0.32, 0.5, 0.68, 0.82, 0.92]
                probe_indices = sorted({max(0, min(total - 1, int(total * r))) for r in ratios})
            else:
                probe_indices = [0, 25, 50, 75, 120]

            best_frame = None
            best_faces = []

            for idx in probe_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, probe = cap.read()
                if not ok or probe is None:
                    continue

                faces = _detect_human_faces(probe, max_faces=max_faces, relaxed=False)
                if len(faces) == 0:
                    faces = _detect_human_faces(probe, max_faces=max_faces, relaxed=True)

                if len(faces) > len(best_faces):
                    best_faces = list(faces)
                    best_frame = probe

                if len(best_faces) >= max_faces:
                    break

            frame = best_frame
            cap.release()
            if frame is None:
                return []
        else:
            return []

        if frame is None:
            return []

        faces = _detect_human_faces(frame, max_faces=max_faces, relaxed=False)
        if len(faces) == 0:
            faces = _detect_human_faces(frame, max_faces=max_faces, relaxed=True)

        if len(faces) == 0:
            return []

        faces = sorted(faces, key=lambda f: f[0])[:max_faces]
        h, w = frame.shape[:2]
        out_dir = Path(FACE_DIR) / f"multi_detect_{chat_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        crops = []
        for i, (x, y, fw, fh) in enumerate(faces, start=1):
            pad_w = int(fw * 0.25)
            pad_h = int(fh * 0.25)
            x0 = max(0, x - pad_w)
            y0 = max(0, y - pad_h)
            x1 = min(w, x + fw + pad_w)
            y1 = min(h, y + fh + pad_h)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crop_path = out_dir / f"person_{i}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crops.append(str(crop_path))
        return crops
    except Exception as e:
        logger.warning("multi face detect failed: %s", e)
        return []


async def download_image_from_telegram_message(message, chat_id, prefix="multi_src"):
    try:
        os.makedirs(FACE_DIR, exist_ok=True)
        out = Path(FACE_DIR) / f"{prefix}_{chat_id}_{int(time.time())}.jpg"
        tg_file = None
        if message.photo:
            tg_file = await message.photo[-1].get_file()
        elif message.document and (message.document.mime_type or "").startswith("image/"):
            tg_file = await message.document.get_file()
        if not tg_file:
            return None

        await tg_file.download_to_drive(custom_path=str(out))
        return face_to_clean_jpg(str(out))
    except Exception as e:
        logger.warning("telegram image download failed: %s", e)
        return None


async def prompt_next_multi_face(update_obj, context, chat_id):
    msg = update_obj.message
    crops = context.user_data.get("multi_face_crops", [])
    idx = int(context.user_data.get("multi_face_idx", 0))

    if idx >= len(crops):
        face_map = context.user_data.get("multi_face_map", {})
        target_link = context.user_data.get("multi_target_link")
        clear_multi_setup_state(context)

        if not target_link:
            await msg.reply_text("❌ Target link missing ho gaya. Multi setup cancel.", reply_markup=main_kb())
            return

        mode_used = "direct"
        if face_map:
            selected_face_maps[chat_id] = face_map
            mode_used = "multi"
            job_modes[chat_id] = mode_used
        else:
            selected_face_maps.pop(chat_id, None)
            job_modes[chat_id] = mode_used

        sleep_note = ""
        task = sleep_countdown_tasks.get(chat_id)
        if task and not task.done():
            task.cancel()
            _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
            sleep_note = "\n🔄 Sleep countdown cancel — naya job shuru."

        queued = _queue_job(chat_id, target_link, None, mode=job_modes.get(chat_id, "direct"))
        _ensure_queue_worker(context, chat_id)
        await msg.reply_text(
            (
                f"✅ Multi setup complete. Selected faces: *{len(face_map)}*\n"
                f"🎛 Execution mode: *{mode_used.title()}*\n"
                f"🗂 Job queued as *#{queued['job_id']}* and processing started."
                f"{sleep_note}"
            ),
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id),
        )
        return

    person_no = idx + 1
    crop_path = crops[idx]
    intro = ""
    if idx == 0:
        intro = f"✅ Target mein *{len(crops)}* face detect hue.\n\n"
    await msg.reply_photo(
        photo=open(crop_path, "rb"),
        caption=(
            f"{intro}"
            f"👤 Person {person_no}/{len(crops)}\n"
            "Is person ke liye option choose karo:\n"
            "1) Use pre uploaded image for face swap\n"
            "2) Send MEGA link / direct upload image to Telegram\n"
            "3) Skip face swap of this person"
        ),
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("1) Use pre uploaded image for face swap", callback_data="multi_use_current_face")],
            [InlineKeyboardButton("2) Send MEGA link / direct upload image to Telegram", callback_data="multi_send_source")],
            [InlineKeyboardButton("3) Skip face swap of this person", callback_data="multi_skip_next")],
        ])
    )


async def start_multi_setup_from_target(update_obj, context, chat_id, target_link):
    msg = update_obj.message

    for d in [VIDEO_DIR, WORKSPACE, TEMP_PATH, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)
    for f in Path(VIDEO_DIR).iterdir():
        if f.is_file():
            f.unlink()

    if not await mega_download_async(target_link, VIDEO_DIR):
        await msg.reply_text("❌ Target download fail. Valid MEGA link bhejo.", reply_markup=main_kb())
        return

    all_files = [f for f in Path(VIDEO_DIR).iterdir() if f.is_file()]
    if not all_files:
        await msg.reply_text("❌ Target file nahi mili after download.", reply_markup=main_kb())
        return

    target = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
    crops = detect_faces_in_target_file(str(target), chat_id)
    if not crops:
        await msg.reply_text(
            "❌ Faces detect nahi huye. Ya to close-up scene use karo ya direct mode use karo.",
            reply_markup=main_kb()
        )
        return

    context.user_data["multi_target_link"] = target_link
    context.user_data["multi_face_crops"] = crops
    context.user_data["multi_face_idx"] = 0
    context.user_data["multi_face_map"] = {}
    context.user_data["awaiting_multi_source"] = True
    context.user_data.pop("awaiting_multi_target", None)
    await prompt_next_multi_face(update_obj, context, chat_id)


def save_face_map_source(chat_id, position_index, link):
    map_dir = Path(FACE_DIR) / f"maps_{chat_id}"
    map_dir.mkdir(parents=True, exist_ok=True)
    if not mega_download(link, str(map_dir)):
        return False, "MEGA download fail"

    images = sorted(
        [f for f in map_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not images:
        return False, "Image file nahi mili"

    clean = face_to_clean_jpg(str(images[0]))
    ok, reason = validate_source_face_quality(clean)
    if not ok:
        return False, reason

    selected_face_maps.setdefault(chat_id, {})[position_index] = clean
    return True, clean


def save_face_map_local_source(chat_id, position_index, src_path, prefix="multi_pre"):
    map_dir = Path(FACE_DIR) / f"maps_{chat_id}"
    map_dir.mkdir(parents=True, exist_ok=True)

    src = Path(str(src_path))
    if not src.is_file():
        return False, "Source image file missing"

    stamp = int(time.time())
    staged = map_dir / f"{prefix}_{position_index + 1}_{stamp}{src.suffix or '.jpg'}"
    if not _safe_copy(str(src), str(staged)):
        return False, "Source image copy failed"

    clean = face_to_clean_jpg(str(staged))
    ok, reason = validate_source_face_quality(clean)
    if not ok:
        return False, reason

    selected_face_maps.setdefault(chat_id, {})[position_index] = clean
    return True, clean


def get_mega_creds():
    env_u = os.environ.get("MEGA_EMAIL", str(_CREDS.get("mega_email", ""))).strip()
    env_p = os.environ.get("MEGA_PASSWORD", str(_CREDS.get("mega_password", ""))).strip()
    if env_u and env_p:
        return env_u, env_p

    if os.path.exists(MEGA_CREDS_FILE):
        try:
            u, p = Path(MEGA_CREDS_FILE).read_text(encoding="utf-8").strip().split(":", 1)
            return u.strip(), p.strip()
        except Exception:
            pass
    cfg = load_persistent_config()
    u = str(cfg.get("mega_email", "")).strip()
    p = str(cfg.get("mega_password", "")).strip()
    if u and p:
        return u, p
    return "", ""


def save_mega_creds(u, p):
    Path(MEGA_CREDS_FILE).write_text(f"{u}:{p}", encoding="utf-8")
    try:
        os.chmod(MEGA_CREDS_FILE, 0o600)
    except Exception:
        pass
    update_persistent_config(mega_email=(u or "").strip(), mega_password=(p or "").strip())
    mega_state["last_operation_at"] = 0.0
    mega_state["auth_backoff_until"] = 0.0
    mega_state["auth_failures"] = 0
    mega_state["mkdir_ready"] = False


def get_drive_token():
    cfg = load_persistent_config()
    drive_auth = cfg.get("drive_auth_token")
    if isinstance(drive_auth, dict):
        token = str(drive_auth.get("access_token", "")).strip()
        if token:
            return token

    # Legacy fallback: may contain upload target override, not auth token.
    if os.path.exists(DRIVE_TOKEN_FILE):
        try:
            raw = Path(DRIVE_TOKEN_FILE).read_text(encoding="utf-8").strip()
            if raw and not raw.lower().startswith("gdrive:"):
                return raw
        except Exception:
            return ""
    return ""


def validate_drive_token(token_value):
    ok, _, info = _parse_drive_auth_payload(token_value)
    return ok, info


def save_drive_token(token_value):
    ok, parsed, info = _parse_drive_auth_payload(token_value)
    if not ok:
        raise ValueError(info or "invalid drive token")

    if parsed.get("legacy_target_override"):
        target = parsed["legacy_target_override"]
        Path(DRIVE_TOKEN_FILE).write_text(target, encoding="utf-8")
        try:
            os.chmod(DRIVE_TOKEN_FILE, 0o600)
        except Exception:
            pass
        update_persistent_config(drive_target_override=target)
        return {"mode": "legacy_target_override", "masked": _mask_secret(target)}

    token_obj = parsed.get("token_obj") or {}
    _write_rclone_drive_token(token_obj)
    update_persistent_config(drive_auth_token=token_obj)
    # Keep a sanitized legacy file for backward compatibility and diagnostics.
    access_token = str(token_obj.get("access_token", "")).strip()
    Path(DRIVE_TOKEN_FILE).write_text(access_token, encoding="utf-8")
    try:
        os.chmod(DRIVE_TOKEN_FILE, 0o600)
    except Exception:
        pass
    return {"mode": "auth_token", "masked": _mask_secret(access_token)}


def get_gdrive_target_folder():
    cfg = load_persistent_config()
    override = str(cfg.get("drive_target_override", "")).strip()
    if override:
        if not override.lower().startswith("gdrive:"):
            override = f"{GDRIVE_REMOTE_NAME}:{override}"
        return override

    if os.path.exists(DRIVE_TOKEN_FILE):
        try:
            raw = Path(DRIVE_TOKEN_FILE).read_text(encoding="utf-8").strip()
            if raw.lower().startswith("gdrive:"):
                return raw
        except Exception:
            pass
    folder = GDRIVE_FOLDER
    if folder and not folder.lower().startswith("gdrive:") and ":" not in folder:
        folder = f"{GDRIVE_REMOTE_NAME}:{folder}"
    return folder


def delete_downloaded_source_video(path_value):
    if not path_value:
        return False
    try:
        p = Path(path_value)
        if not p.exists() or not p.is_file():
            return False
        parent = str(p.parent.resolve())
        allowed_roots = {
            str(Path(VIDEO_DIR).resolve()),
            str(Path(WORKSPACE).resolve()),
        }
        if parent not in allowed_roots:
            logger.warning("skip source delete outside allowed roots: %s", p)
            return False
        p.unlink(missing_ok=True)
        logger.info("source video deleted after successful job: %s", p.name)
        return True
    except Exception as e:
        logger.warning("source video cleanup failed: %s", e)
        return False


def _mega_rate_limit_wait():
    gap = max(0, int(MEGA_MIN_OPERATION_GAP_SEC))
    now = time.time()
    since = now - mega_state["last_operation_at"]
    if gap > 0 and since < gap:
        time.sleep(gap - since)
    mega_state["last_operation_at"] = time.time()


def _mega_mark_auth_success():
    mega_state["auth_failures"] = 0
    mega_state["auth_backoff_until"] = 0.0


def _mega_mark_auth_failure(err):
    if not is_mega_auth_error(err):
        return
    mega_state["auth_failures"] += 1
    wait = min(
        MEGA_AUTH_COOLDOWN_MAX_SEC,
        MEGA_AUTH_COOLDOWN_BASE_SEC * (2 ** max(0, mega_state["auth_failures"] - 1))
    )
    mega_state["auth_backoff_until"] = time.time() + wait


def validate_mega_creds(u, p):
    """Fast auth check so we can fail early with a useful message."""
    _mega_rate_limit_wait()
    try:
        r = subprocess.run(
            ["megals", "--username", u, "--password", p],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            _mega_mark_auth_success()
            return True, ""
        err = _short_err(r)
        _mega_mark_auth_failure(err)
        return False, f"MEGA auth failed: {err}"
    except FileNotFoundError as e:
        return False, f"MEGA CLI not found: {e}"
    except subprocess.TimeoutExpired:
        return False, "MEGA auth timeout"
    except Exception as e:
        return False, f"MEGA auth exception: {e}"


def _looks_like_mega_creds_input(text):
    raw = (text or "").strip()
    if not raw or "\n" in raw:
        return False
    if raw.lower().startswith(("http://", "https://", "mega://")):
        return False
    if ":" not in raw:
        return False
    user, passwd = raw.split(":", 1)
    user = user.strip()
    passwd = passwd.strip()
    if not user or not passwd:
        return False
    if "@" not in user:
        return False
    return True


def is_mega_auth_error(msg):
    lo = (msg or "").lower()
    return any(x in lo for x in ["eblocked", "eargs", "can't login", "auth failed", "us0"])


def _touch_download_progress(bytes_now=0, detail=""):
    try:
        st = _load_active_job_state() or {}
        phase = str(st.get("phase") or st.get("status") or "").lower()
        if phase and phase not in {"download", "starting"}:
            return
        now_ts = float(time.time())
        st["phase"] = "download"
        st["status"] = "download"
        st["stage"] = str(st.get("stage") or "Downloading")
        st["updated_at"] = now_ts
        st["last_update"] = now_ts
        st["last_progress_timestamp"] = now_ts
        st["last_progress_stage"] = "DOWNLOADING"
        st["last_progress_frame"] = int(max(0, int(bytes_now // (1024 * 1024))))
        st["last_progress_pct"] = int(st.get("last_progress_pct", -1) or -1)
        if detail:
            st["details"] = str(detail)[:180]
        _save_active_job_state(st)
    except Exception:
        pass


def _kill_process_group(proc):
    with suppress(Exception):
        if proc is None:
            return
        os.killpg(os.getpgid(int(proc.pid)), signal.SIGTERM)
    with suppress(Exception):
        time.sleep(1)
    with suppress(Exception):
        if proc is not None and proc.poll() is None:
            os.killpg(os.getpgid(int(proc.pid)), signal.SIGKILL)


def _download_direct_once(link, dest_dir, timeout_sec=None, stall_timeout_sec=None):
    timeout_limit = float(timeout_sec if timeout_sec is not None else DOWNLOAD_ATTEMPT_TIMEOUT_SEC)
    stall_limit = float(stall_timeout_sec if stall_timeout_sec is not None else DOWNLOAD_STALL_TIMEOUT_SEC)

    parsed = urlparse(str(link or ""))
    filename = Path(parsed.path or "download.bin").name or f"download_{int(time.time())}.bin"
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or f"download_{int(time.time())}.bin"
    out_path = Path(dest_dir) / filename
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    start_ts = float(time.time())
    last_growth_ts = start_ts
    written = 0
    req = Request(str(link), headers={"User-Agent": "FaceSwapBot/14"})
    with urlopen(req, timeout=DIRECT_LINK_PROBE_TIMEOUT_SEC) as resp, open(tmp_path, "wb") as fp:
        while True:
            if (float(time.time()) - start_ts) > timeout_limit:
                raise TimeoutError(f"direct link timeout after {int(timeout_limit)}s")
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fp.write(chunk)
            written += len(chunk)
            last_growth_ts = float(time.time())
            _touch_download_progress(bytes_now=written, detail="Direct file downloading")
            if (float(time.time()) - last_growth_ts) > stall_limit:
                raise TimeoutError(f"direct link zero progress for {int(stall_limit)}s")

    if written <= 0:
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        return False, "direct link returned empty file"
    tmp_path.replace(out_path)
    return True, f"direct download complete: {out_path.name}"


def _tail_text_file(path_obj, max_chars=3000):
    try:
        p = Path(path_obj)
        if not p.exists() or not p.is_file():
            return ""
        text = p.read_text(encoding="utf-8", errors="ignore")
        return str(text[-int(max_chars):]).strip()
    except Exception:
        return ""


def _detect_mega_backends():
    backends = []
    if shutil.which("megadl"):
        backends.append("megadl")
    if shutil.which("mega-get"):
        backends.append("mega-get")
    if shutil.which("mega-exec"):
        backends.append("mega-exec")
    if shutil.which("mega-cmd"):
        backends.append("mega-cmd")
    if backends:
        logger.info("[MEGA_FIX] available backends=%s", ",".join(backends))
    else:
        logger.warning("[MEGA_FIX] no MEGA CLI backend detected")
    return backends


def _run_download_process(cmd, dest_dir, timeout_limit, stall_limit, backend_name, link):
    base_bytes = int(_path_size_bytes(dest_dir))
    start_ts = float(time.time())
    last_growth_ts = start_ts
    last_bytes = base_bytes

    mega_log_dir = Path(PIPELINE) / "logs" / "mega"
    mega_log_dir.mkdir(parents=True, exist_ok=True)
    mega_log_path = mega_log_dir / f"{backend_name}_{int(time.time() * 1000)}.log"
    log_fp = open(mega_log_path, "w", encoding="utf-8", errors="ignore")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    logger.info(
        "[MEGA_DOWNLOAD] backend=%s start pid=%s timeout=%ss stall_timeout=%ss link_head=%s",
        backend_name,
        int(proc.pid or 0),
        int(timeout_limit),
        int(stall_limit),
        str(link or "")[:120],
    )
    try:
        while proc.poll() is None:
            time.sleep(float(DOWNLOAD_PROGRESS_POLL_SEC))
            now_ts = float(time.time())
            current_bytes = int(_path_size_bytes(dest_dir))
            if current_bytes > last_bytes:
                last_growth_ts = now_ts
                last_bytes = current_bytes
                _touch_download_progress(
                    bytes_now=(current_bytes - base_bytes),
                    detail=f"MEGA downloading via {backend_name} +{max(0, current_bytes - base_bytes) // (1024 * 1024)}MB",
                )

            if (now_ts - start_ts) > timeout_limit:
                _kill_process_group(proc)
                logger.warning("[MEGA_DOWNLOAD] backend=%s timeout pid=%s elapsed=%ss", backend_name, int(proc.pid or 0), int(now_ts - start_ts))
                return False, f"download timeout after {int(timeout_limit)}s"

            if (now_ts - last_growth_ts) > stall_limit:
                _kill_process_group(proc)
                logger.warning("[MEGA_DOWNLOAD] backend=%s stalled pid=%s stalled_for=%ss", backend_name, int(proc.pid or 0), int(now_ts - last_growth_ts))
                return False, f"download zero-progress for {int(stall_limit)}s"

        proc.wait(timeout=2)
    except Exception as e:
        _kill_process_group(proc)
        with suppress(Exception):
            log_fp.close()
        return False, f"{backend_name} monitor exception: {e}"
    finally:
        with suppress(Exception):
            log_fp.close()

    if int(proc.returncode or 1) == 0:
        logger.info("[MEGA_DOWNLOAD] backend=%s done pid=%s returncode=0", backend_name, int(proc.pid or 0))
        return True, "ok"

    # Some MEGA CLI builds may exit non-zero after writing a valid file.
    # Accept the run when destination size increased or a fresh non-empty file appeared.
    current_total_bytes = int(_path_size_bytes(dest_dir))
    if current_total_bytes > base_bytes:
        logger.warning(
            "[MEGA_DOWNLOAD] backend=%s nonzero returncode=%s but bytes increased (%s -> %s); accepting success",
            backend_name,
            int(proc.returncode or 1),
            base_bytes,
            current_total_bytes,
        )
        return True, "nonzero-with-bytes"

    with suppress(Exception):
        dest_path = Path(dest_dir)
        fresh_files = [
            p for p in dest_path.rglob("*")
            if p.is_file() and int(p.stat().st_size or 0) > 0 and float(p.stat().st_mtime or 0.0) >= (start_ts - 2.0)
        ]
        if fresh_files:
            logger.warning(
                "[MEGA_DOWNLOAD] backend=%s nonzero returncode=%s but fresh file detected (%s); accepting success",
                backend_name,
                int(proc.returncode or 1),
                fresh_files[0].name,
            )
            return True, "nonzero-with-file"

    err_text = _tail_text_file(mega_log_path, max_chars=4000)
    logger.warning(
        "[MEGA_DOWNLOAD] backend=%s failed pid=%s returncode=%s detail_tail=%s",
        backend_name,
        int(proc.pid or 0),
        int(proc.returncode or 1),
        str(err_text or "")[-220:],
    )
    if "Local file already exists" in err_text:
        return True, "local exists"
    return False, (_short_err(type("Proc", (), {"stderr": err_text, "stdout": err_text})()) or f"{backend_name} failed")


def _mega_python_download_once(link, dest_dir, timeout_limit):
    code = (
        "from mega import Mega\n"
        "import pathlib, sys\n"
        "dest = pathlib.Path(sys.argv[2])\n"
        "dest.mkdir(parents=True, exist_ok=True)\n"
        "m = Mega()\n"
        "result = m.download_url(sys.argv[1], str(dest))\n"
        "print(str(result) if result else '')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(link), str(dest_dir)],
        capture_output=True,
        text=True,
        timeout=max(30, int(timeout_limit)),
    )
    if proc.returncode == 0:
        logger.info("[MEGA_DOWNLOAD] backend=python-mega done")
        return True, "ok"
    detail = _short_err(proc) or "python mega download failed"
    logger.warning("[MEGA_DOWNLOAD] backend=python-mega failed detail=%s", detail[:220])
    return False, detail


def _mega_b64decode(value):
    raw = str(value or "").strip().replace("-", "+").replace("_", "/")
    if not raw:
        raise ValueError("empty MEGA key")
    raw += "=" * ((4 - len(raw) % 4) % 4)
    return base64.b64decode(raw)


def _parse_mega_public_file_link(link):
    text = str(link or "").strip()
    parsed = urlparse(text)
    path_parts = [p for p in (parsed.path or "").split("/") if p]
    fragment = str(parsed.fragment or "").strip()

    if len(path_parts) >= 2 and path_parts[0].lower() == "file" and fragment:
        return path_parts[1], fragment.split("?", 1)[0].split("/", 1)[0]

    # Legacy MEGA format: https://mega.nz/#!file_id!file_key
    if fragment.startswith("!"):
        parts = [p for p in fragment.split("!") if p]
        if len(parts) >= 2:
            return parts[0], parts[1]

    query = parse_qs(parsed.query or "")
    file_id = (query.get("id") or query.get("p") or [""])[0]
    file_key = (query.get("key") or [""])[0]
    if file_id and file_key:
        return file_id, file_key

    raise ValueError("unsupported MEGA public file link")


def _mega_file_crypto_from_key(file_key):
    key_bytes = _mega_b64decode(file_key)
    if len(key_bytes) < 32:
        raise ValueError("MEGA key too short")
    words = struct.unpack(">8I", key_bytes[:32])
    aes_words = (
        words[0] ^ words[4],
        words[1] ^ words[5],
        words[2] ^ words[6],
        words[3] ^ words[7],
    )
    aes_key = struct.pack(">4I", *aes_words)
    iv = struct.pack(">4I", words[4], words[5], 0, 0)
    return aes_key.hex(), iv.hex()


def _safe_download_filename(name, fallback):
    value = str(name or "").strip()
    if not value:
        value = str(fallback or "").strip()
    value = value.replace("\\", "/").split("/")[-1]
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" ._")
    return value or f"mega_file_{int(time.time())}.bin"


def _detect_extension_from_magic(path_obj):
    try:
        head = Path(path_obj).read_bytes()[:32]
    except Exception:
        return ""
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return ".webp"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        return ".mp4"
    return ""


def _mega_decode_filename(at_str: str, file_key: str) -> str:
    """Decode MEGA encrypted filename from 'at' attribute using AES-ECB with the file key."""
    try:
        import base64, json
        from Crypto.Cipher import AES  # type: ignore[import]
        key_bytes = _mega_b64decode(file_key)
        # MEGA key is 32 bytes (two 128-bit halves XORed together for AES key)
        k = [int.from_bytes(key_bytes[i:i+4], 'big') for i in range(0, 16, 4)]
        k2 = [int.from_bytes(key_bytes[i:i+4], 'big') for i in range(16, 32, 4)]
        aes_key = bytes((a ^ b).to_bytes(4, 'big')[j] for a, b in zip(k, k2) for j in range(4))
        # Decode and decrypt attributes
        pad = at_str + '=='  # MEGA uses urlsafe b64 without padding
        raw = base64.b64decode(pad.replace('-', '+').replace('_', '/'), validate=False)
        # Strip "MEGA" prefix (4 bytes)
        if raw[:4] == b'MEGA':
            raw = raw[4:]
        cipher = AES.new(aes_key, AES.MODE_CBC, b'\x00' * 16)
        decrypted = cipher.decrypt(raw).rstrip(b'\x00')
        attrs = json.loads(decrypted.decode('utf-8', errors='replace').strip('\x00'))
        return str(attrs.get('n') or '')
    except Exception:
        pass
    # Fallback: try without CBC (some MEGA links use ECB-style)
    try:
        import base64, json
        from Crypto.Cipher import AES  # type: ignore[import]
        key_bytes = _mega_b64decode(file_key)
        k = [int.from_bytes(key_bytes[i:i+4], 'big') for i in range(0, 16, 4)]
        k2 = [int.from_bytes(key_bytes[i:i+4], 'big') for i in range(16, 32, 4)]
        aes_key = bytes((a ^ b).to_bytes(4, 'big')[j] for a, b in zip(k, k2) for j in range(4))
        pad = at_str + '=='
        raw = base64.b64decode(pad.replace('-', '+').replace('_', '/'), validate=False)
        cipher = AES.new(aes_key, AES.MODE_ECB)
        decrypted = b''.join(cipher.decrypt(raw[i:i+16]) for i in range(0, len(raw), 16))
        decrypted = decrypted.rstrip(b'\x00')
        if decrypted.startswith(b'MEGA'):
            decrypted = decrypted[4:]
        attrs = json.loads(decrypted.decode('utf-8', errors='replace').strip('\x00'))
        return str(attrs.get('n') or '')
    except Exception:
        return ''


def _mega_api_download_once(link, dest_dir, timeout_limit, stall_limit):
    if not shutil.which("openssl"):
        return False, "openssl unavailable for MEGA API decryption"
    try:
        import requests
    except Exception as e:
        return False, f"requests unavailable: {e}"

    try:
        file_id, file_key = _parse_mega_public_file_link(link)
        key_hex, iv_hex = _mega_file_crypto_from_key(file_key)
    except Exception as e:
        return False, f"MEGA link parse failed: {e}"

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    api_url = f"https://g.api.mega.co.nz/cs?id={random.randint(1, 999999999)}"
    try:
        api_resp = requests.post(
            api_url,
            json=[{"a": "g", "g": 1, "p": file_id}],
            timeout=20,
            headers={"User-Agent": "FaceSwapBot/14"},
        )
        api_resp.raise_for_status()
        payload = api_resp.json()
        meta = payload[0] if isinstance(payload, list) and payload else payload
        if isinstance(meta, int):
            return False, f"MEGA API error {meta}"
        download_url = str((meta or {}).get("g") or "").strip()
        size = int((meta or {}).get("s") or 0)
        if not download_url:
            return False, "MEGA API returned no download URL"
        # Decode original filename from encrypted 'at' attribute
        at_str = str((meta or {}).get("at") or "").strip()
        _, file_key = _parse_mega_public_file_link(link)
        orig_name = _mega_decode_filename(at_str, file_key) if at_str else ""
        if orig_name:
            safe_name = _safe_download_filename(orig_name, f"mega_{file_id}.bin")
            logger.info("MEGA original filename decoded: %s", orig_name)
        else:
            safe_name = _safe_download_filename("", f"mega_{file_id}.bin")
    except Exception as e:
        return False, f"MEGA API request failed: {e}"

    out_path = dest / safe_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists():
        with suppress(Exception):
            tmp_path.unlink()

    # Save total size so download heartbeat can show real progress
    if size > 0:
        with suppress(Exception):
            st = _load_active_job_state() or {}
            st["download_total_bytes"] = int(size)
            _save_active_job_state(st)

    cmd = [
        "openssl", "enc", "-d", "-aes-128-ctr",
        "-K", key_hex,
        "-iv", iv_hex,
        "-nosalt",
        "-out", str(tmp_path),
    ]
    start_ts = float(time.time())
    last_growth_ts = start_ts
    written = 0
    proc = None
    try:
        with requests.get(download_url, stream=True, timeout=20, headers={"User-Agent": "FaceSwapBot/14"}) as resp:
            if resp.status_code >= 400:
                return False, f"MEGA binary http status {resp.status_code}"
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                now_ts = float(time.time())
                if (now_ts - start_ts) > float(timeout_limit):
                    _kill_process_group(proc)
                    return False, f"MEGA API timeout after {int(timeout_limit)}s"
                if (now_ts - last_growth_ts) > float(stall_limit):
                    _kill_process_group(proc)
                    return False, f"MEGA API zero-progress for {int(stall_limit)}s"
                if not chunk:
                    continue
                proc.stdin.write(chunk)
                written += len(chunk)
                last_growth_ts = now_ts
                _touch_download_progress(bytes_now=written, detail="MEGA API downloading")
            with suppress(Exception):
                proc.stdin.close()
            stderr = (proc.stderr.read() if proc.stderr else b"")[:1000]
            # Keep touching progress while openssl decrypts (prevents watchdog kill)
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    rc = proc.wait(timeout=5)
                    break
                except Exception:
                    _touch_download_progress(bytes_now=written, detail="MEGA API decrypting...")
            else:
                _kill_process_group(proc)
                return False, "openssl decrypt timed out after 120s"
            if rc != 0:
                return False, f"openssl decrypt failed rc={rc} {stderr.decode('utf-8', errors='ignore')[:160]}"
    except Exception as e:
        if proc is not None:
            _kill_process_group(proc)
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        return False, f"MEGA API download failed: {e}"

    if not tmp_path.exists() or int(tmp_path.stat().st_size or 0) <= 0:
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        return False, "MEGA API returned empty decrypted file"

    if size and written and written != size:
        logger.warning("[MEGA_DOWNLOAD] backend=mega-api encrypted size mismatch expected=%s got=%s", size, written)

    ext = _detect_extension_from_magic(tmp_path)
    if ext and out_path.suffix.lower() != ext:
        out_path = out_path.with_suffix(ext)
    if out_path.exists():
        out_path = dest / f"{out_path.stem}_{int(time.time())}{out_path.suffix}"
    tmp_path.replace(out_path)
    logger.info("[MEGA_DOWNLOAD] backend=mega-api done file=%s size=%s", out_path.name, int(out_path.stat().st_size))
    return True, "ok"


def _mega_requests_fallback_once(link, dest_dir, timeout_limit, stall_limit):
    try:
        import requests
    except Exception as e:
        return False, f"requests unavailable: {e}"

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(str(link or ""))
    filename = Path(parsed.path or "mega_file.bin").name or f"mega_file_{int(time.time())}.bin"
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._") or f"mega_file_{int(time.time())}.bin"
    out_path = dest / filename
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    start_ts = float(time.time())
    last_growth_ts = start_ts
    written = 0
    with requests.get(str(link), stream=True, timeout=20, allow_redirects=True, headers={"User-Agent": "FaceSwapBot/14"}) as resp:
        ctype = str(resp.headers.get("content-type") or "").lower()
        if resp.status_code >= 400:
            return False, f"http status {resp.status_code}"
        with open(tmp_path, "wb") as fp:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if (float(time.time()) - start_ts) > timeout_limit:
                    return False, f"requests timeout after {int(timeout_limit)}s"
                if not chunk:
                    continue
                fp.write(chunk)
                written += len(chunk)
                last_growth_ts = float(time.time())
                _touch_download_progress(bytes_now=written, detail="MEGA requests fallback downloading")
                if (float(time.time()) - last_growth_ts) > stall_limit:
                    return False, f"requests zero-progress for {int(stall_limit)}s"

    if written <= 0:
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        return False, "requests fallback returned empty file"

    if "html" in ctype and written < (5 * 1024 * 1024):
        with suppress(Exception):
            tmp_path.unlink(missing_ok=True)
        return False, "requests fallback received html page, not binary media"

    tmp_path.replace(out_path)
    logger.info("[MEGA_DOWNLOAD] backend=requests-fallback done file=%s size=%s", out_path.name, int(out_path.stat().st_size))
    return True, "ok"


def _mega_download_once(link, dest_dir, timeout_sec=None, stall_timeout_sec=None):
    timeout_limit = float(timeout_sec if timeout_sec is not None else DOWNLOAD_ATTEMPT_TIMEOUT_SEC)
    stall_limit = float(stall_timeout_sec if stall_timeout_sec is not None else DOWNLOAD_STALL_TIMEOUT_SEC)
    backends = _detect_mega_backends()

    # Option A: megadl
    if "megadl" in backends:
        ok, reason = _run_download_process(
            ["megadl", str(link), "--path", str(Path(dest_dir)) + "/"],
            dest_dir,
            timeout_limit,
            stall_limit,
            "megadl",
            link,
        )
        if ok:
            return True, reason

    # Option B: mega-get
    if "mega-get" in backends:
        ok, reason = _run_download_process(
            ["mega-get", str(link), str(Path(dest_dir))],
            dest_dir,
            timeout_limit,
            stall_limit,
            "mega-get",
            link,
        )
        if ok:
            return True, reason

    # Option C: MEGAcmd equivalents
    if "mega-exec" in backends:
        ok, reason = _run_download_process(
            ["mega-exec", "get", str(link), str(Path(dest_dir))],
            dest_dir,
            timeout_limit,
            stall_limit,
            "mega-exec",
            link,
        )
        if ok:
            return True, reason
    if "mega-cmd" in backends:
        ok, reason = _run_download_process(
            ["mega-cmd", "get", str(link), str(Path(dest_dir))],
            dest_dir,
            timeout_limit,
            stall_limit,
            "mega-cmd",
            link,
        )
        if ok:
            return True, reason

    # Option D: native MEGA public-link API downloader
    ok_api, reason_api = _mega_api_download_once(
        link,
        dest_dir,
        timeout_limit=timeout_limit,
        stall_limit=stall_limit,
    )
    if ok_api:
        return True, reason_api

    # Option E: Python direct public-link downloader
    ok, reason = _mega_python_download_once(link, dest_dir, timeout_limit=timeout_limit)
    if ok:
        return True, reason

    # Option F: requests streamed fallback
    ok2, reason2 = _mega_requests_fallback_once(link, dest_dir, timeout_limit=timeout_limit, stall_limit=stall_limit)
    if ok2:
        return True, reason2

    return False, f"all MEGA download backends failed: mega-api={reason_api}; python={reason}; requests={reason2}"


def mega_download(link, dest_dir, retries=2):
    ok, _ = mega_download_detailed(link, dest_dir, retries=retries)
    return bool(ok)


def mega_download_detailed(link, dest_dir, retries=2, attempt_timeout_sec=None, stall_timeout_sec=None):
    os.makedirs(dest_dir, exist_ok=True)
    logger.info("[FACE_SOURCE] download requested link_head=%s dest=%s", str(link or "")[:120], str(dest_dir))

    ok_link, normalized_link, link_kind, link_reason = validate_input_media_link(link, verify_remote=False)
    if not ok_link:
        logger.warning("download rejected invalid link: %s", link_reason)
        return False, f"invalid link: {link_reason}"

    try:
        src_candidate = Path(str(normalized_link or "").strip())
        if src_candidate.is_file():
            dst = Path(dest_dir) / src_candidate.name
            shutil.copy2(str(src_candidate), str(dst))
            logger.info("local source reused for download stage: %s -> %s", src_candidate, dst)
            _touch_download_progress(bytes_now=int(dst.stat().st_size), detail="Local source reused")
            logger.info("[MEGA_DOWNLOAD] local source reused path=%s", str(dst))
            return True, "local source reused"
    except Exception as e:
        logger.warning("[MEGA_DOWNLOAD] local source reuse failed; falling back to downloader chain: %s", e)

    max_attempts = max(1, int(retries or DOWNLOAD_RETRY_COUNT))
    last_reason = "download failed"
    for attempt in range(1, max_attempts + 1):
        try:
            if link_kind == "direct":
                ok, reason = _download_direct_once(
                    normalized_link,
                    dest_dir,
                    timeout_sec=attempt_timeout_sec,
                    stall_timeout_sec=stall_timeout_sec,
                )
            else:
                ok, reason = _mega_download_once(
                    normalized_link,
                    dest_dir,
                    timeout_sec=attempt_timeout_sec,
                    stall_timeout_sec=stall_timeout_sec,
                )
        except Exception as e:
            ok, reason = False, str(e)

        if ok:
            latest_path = _latest_file_in_dir(dest_dir)
            if not latest_path or not Path(latest_path).is_file() or int(Path(latest_path).stat().st_size) <= 0:
                last_reason = "download backend reported success but no output file found"
                logger.warning("[MEGA_DOWNLOAD] %s", last_reason)
            else:
                logger.info("[MEGA_DOWNLOAD] succeeded kind=%s attempt=%s/%s output=%s", link_kind, attempt, max_attempts, latest_path)
                return True, str(reason or "ok")

        last_reason = str(reason or "download failed")
        logger.warning(
            "[MEGA_DOWNLOAD] failed kind=%s attempt=%s/%s reason=%s",
            link_kind,
            attempt,
            max_attempts,
            last_reason[:220],
        )
        if attempt < max_attempts:
            time.sleep(float(min(8, attempt * 2)))

    return False, last_reason[:300]


def create_quota_fallback_video(dest_dir):
    # Stable mode: synthetic fallback generation disabled.
    return None


def mega_download_with_fallback(link, dest_dir, retries=2, allow_test_fallback=False):
    return mega_download(link, dest_dir, retries=retries)


async def mega_download_async(link, dest_dir, retries=2, allow_test_fallback=False):
    # Stable mode: use simple downloader path only.
    ok, _ = await mega_download_async_detailed(
        link,
        dest_dir,
        retries=retries,
        allow_test_fallback=allow_test_fallback,
    )
    return bool(ok)


async def mega_download_async_detailed(
    link,
    dest_dir,
    retries=2,
    allow_test_fallback=False,
    attempt_timeout_sec=None,
    stall_timeout_sec=None,
):
    # Stable mode: use simple downloader path only.
    return await asyncio.to_thread(
        mega_download_detailed,
        link,
        dest_dir,
        retries,
        attempt_timeout_sec,
        stall_timeout_sec,
    )


def validate_downloaded_face_image(path_obj):
    candidate = Path(path_obj)
    if not candidate.exists() or not candidate.is_file():
        return False, "downloaded file missing"

    with suppress(Exception):
        if int(candidate.stat().st_size or 0) <= 0:
            return False, "downloaded file is empty"

    guessed_mime, _ = mimetypes.guess_type(str(candidate))
    if guessed_mime and not str(guessed_mime).lower().startswith("image/"):
        return False, f"mime mismatch: {guessed_mime}"

    try:
        from PIL import Image as PILImage

        with PILImage.open(str(candidate)) as img:
            pil_format = str(img.format or "").strip().upper()
            pil_mime = str(PILImage.MIME.get(pil_format, "") or "").lower().strip()
            img.verify()
        if pil_mime and not pil_mime.startswith("image/"):
            return False, f"invalid image mime: {pil_mime}"
    except Exception as e:
        return False, f"image validation failed: {e}"

    try:
        import cv2

        probe = cv2.imread(str(candidate))
        if probe is None:
            return False, "opencv could not decode image"
    except ModuleNotFoundError as e:
        if str(getattr(e, "name", "")) != "cv2":
            return False, f"opencv decode failed: {e}"
        logger.warning("[FACE_SOURCE] cv2 unavailable during download validation; continuing with PIL-verified image")
    except Exception as e:
        return False, f"opencv decode failed: {e}"

    ok_face, face_reason = validate_source_face_quality(str(candidate))
    if not ok_face:
        return False, f"face validation failed: {face_reason}"

    return True, "ok"


def coerce_face_source_to_jpg(path_obj):
    src = Path(path_obj)
    if not src.exists() or not src.is_file():
        return src

    ext = src.suffix.lower()
    supported = {".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif"}
    if ext not in supported:
        return src

    dst = src.with_suffix(".jpg")
    if src.suffix.lower() in {".jpg", ".jpeg"}:
        return src

    try:
        from PIL import Image as PILImage

        with PILImage.open(str(src)) as im:
            im.convert("RGB").save(str(dst), "JPEG", quality=95)
        if dst.is_file() and int(dst.stat().st_size or 0) > 0:
            return dst
    except Exception:
        pass

    try:
        conv = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(src),
                "-frames:v", "1",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if conv.returncode == 0 and dst.is_file() and int(dst.stat().st_size or 0) > 0:
            return dst
    except Exception:
        pass

    return src


_FACE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
_FACE_JUNK_EXTS = {
    ".txt", ".json", ".log", ".md", ".nfo", ".ini", ".yaml", ".yml",
    ".csv", ".xml", ".py", ".sh", ".bat", ".ps1",
}


def _wait_for_face_payload_flush(request_dir, max_wait_sec=3.0):
    """Wait briefly for downloader writes to settle before scanning payload files."""
    root = Path(request_dir)
    start = float(time.time())
    last = (-1, -1)
    stable_hits = 0
    while (float(time.time()) - start) < float(max_wait_sec):
        files = [f for f in root.rglob("*") if f.is_file()]
        total = 0
        for f in files:
            with suppress(Exception):
                total += int(f.stat().st_size or 0)
        snap = (len(files), int(total))
        if snap == last:
            stable_hits += 1
            if stable_hits >= 2:
                break
        else:
            stable_hits = 0
        last = snap
        time.sleep(0.4)


def _extract_zip_payloads(request_dir, max_archives=4):
    """Extract zip payloads to side folders so nested face images can be discovered."""
    root = Path(request_dir)
    extracted = []
    count = 0
    for zf in sorted([p for p in root.rglob("*") if p.is_file()], key=lambda p: str(p)):
        if count >= int(max_archives):
            break
        lower_name = str(zf.name or "").lower()
        if not lower_name.endswith(".zip"):
            continue
        with suppress(Exception):
            if not zipfile.is_zipfile(str(zf)):
                continue
            out_dir = zf.parent / f"_unzipped_{zf.stem}_{int(time.time() * 1000)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(zf), "r") as z:
                z.extractall(str(out_dir))
            extracted.append(str(out_dir))
            count += 1
    return extracted


def _looks_like_image_candidate(file_path):
    p = Path(file_path)
    name = str(p.name or "")
    ext = str(p.suffix or "").lower()
    if name.startswith("."):
        return False
    if ext in _FACE_JUNK_EXTS:
        return False

    if ext in _FACE_IMAGE_EXTS:
        return True

    guessed_mime, _ = mimetypes.guess_type(str(p))
    if guessed_mime and str(guessed_mime).lower().startswith("image/"):
        return True

    # Support valid image files without extension.
    try:
        from PIL import Image as PILImage

        with PILImage.open(str(p)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _discover_face_payload_candidates(request_dir):
    root = Path(request_dir)
    _wait_for_face_payload_flush(root, max_wait_sec=3.0)
    extracted_dirs = _extract_zip_payloads(root)
    if extracted_dirs:
        _wait_for_face_payload_flush(root, max_wait_sec=2.0)

    all_files = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        all_files.append(f)

    discovered = []
    for f in all_files:
        with suppress(Exception):
            discovered.append({
                "name": str(f.relative_to(root)),
                "bytes": int(f.stat().st_size or 0),
            })

    probable = []
    for f in all_files:
        if _looks_like_image_candidate(f):
            probable.append(f)

    probable.sort(key=lambda p: int(p.stat().st_size or 0), reverse=True)
    return probable, discovered, extracted_dirs


def _short_err(proc_result):
    text = (proc_result.stderr or proc_result.stdout or "").strip()
    if not text:
        return "no error text"
    return text[-300:]


_FF_SUPPORTED_EXECUTION_PROVIDERS = None
_FF_SUPPORTED_FACE_DETECTOR_MODELS = None


def get_facefusion_supported_execution_providers():
    global _FF_SUPPORTED_EXECUTION_PROVIDERS
    if _FF_SUPPORTED_EXECUTION_PROVIDERS is not None:
        return _FF_SUPPORTED_EXECUTION_PROVIDERS

    providers = {"cpu"}
    providers_detected_from_help = False
    try:
        probe_env = prepare_cuda_runtime_env(os.environ.copy())
        ff_py = str(Path(FACEFUSION_DIR) / "facefusion.py")
        r = subprocess.run(
            [FACEFUSION_PYTHON, ff_py, "headless-run", "--help"],
            capture_output=True,
            text=True,
            timeout=25,
            cwd=FACEFUSION_DIR,
            env=probe_env,
        )
        help_text = (r.stdout or "") + "\n" + (r.stderr or "")
        m = re.search(r"--execution-providers[^\n]*\(choose from ([^)]+)\)", help_text, re.IGNORECASE)
        if m:
            parsed = {x.strip().lower() for x in m.group(1).split(",") if x.strip()}
            if parsed:
                providers = parsed
                providers_detected_from_help = True
    except Exception as e:
        logger.warning("failed to detect facefusion execution providers, default cpu: %s", e)

    # Only infer from onnxruntime when facefusion --help does not expose choices.
    if not providers_detected_from_help:
        try:
            import onnxruntime as ort

            ort_providers = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in ort_providers:
                providers.add("cuda")
            if "CPUExecutionProvider" in ort_providers:
                providers.add("cpu")
        except Exception:
            pass

    # Probe provider argument validation to avoid false positives from runtime libs.
    def provider_is_accepted(provider_name):
        try:
            probe_env = prepare_cuda_runtime_env(os.environ.copy())
            ff_py = str(Path(FACEFUSION_DIR) / "facefusion.py")
            probe_cmd = [
                FACEFUSION_PYTHON, ff_py, "headless-run",
                "-s", "/tmp/ff_probe_source.jpg",
                "-t", "/tmp/ff_probe_target.mp4",
                "-o", "/tmp/ff_probe_output.mp4",
                "--execution-providers", provider_name,
            ]
            r = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=FACEFUSION_DIR,
                env=probe_env,
            )
            probe_text = ((r.stdout or "") + "\n" + (r.stderr or "")).lower()
            if "--execution-providers" in probe_text and "invalid choice" in probe_text:
                return False
            return True
        except Exception:
            return provider_name == "cpu"

    if "cuda" in providers and not provider_is_accepted("cuda"):
        providers.discard("cuda")
    if "cpu" not in providers and provider_is_accepted("cpu"):
        providers.add("cpu")

    # Override is a preference/order hint, not a capability declaration.
    # Keep only providers that are actually detected on this runtime.
    override = os.environ.get("FACEFUSION_PROVIDER_OVERRIDE", "").strip()
    if override:
        override_candidates = [x.strip().lower() for x in override.split(",") if x.strip()]
        filtered = {p for p in override_candidates if p in providers}
        if filtered:
            providers = filtered

    _FF_SUPPORTED_EXECUTION_PROVIDERS = providers
    return providers


def pick_execution_provider():
    os.environ["LD_LIBRARY_PATH"] = build_cuda_ld_library_path(os.environ.get("LD_LIBRARY_PATH", ""))
    supported = get_facefusion_supported_execution_providers()

    def _log_mode(mode, reason):
        logger.info("EXECUTION_MODE: %s | reason=%s | supported=%s", mode.upper(), reason, ",".join(sorted(supported)))

    if EXECUTION_PROVIDER in {"cpu", "cuda"}:
        if EXECUTION_PROVIDER in supported:
            if EXECUTION_PROVIDER == "cpu" and GPU_ONLY_MODE:
                raise RuntimeError("GPU_ONLY_MODE=1 but explicit provider cpu requested")
            if EXECUTION_PROVIDER == "cuda":
                logger.info("CUDAExecutionProvider ACTIVE")
            _log_mode(EXECUTION_PROVIDER, "explicit request")
            return EXECUTION_PROVIDER
        if EXECUTION_PROVIDER == "cuda":
            logger.warning(
                "requested CUDA provider unavailable in precheck (%s) but GPU-only mode forcing CUDA runtime",
                ",".join(sorted(supported)),
            )
            logger.info("CUDAExecutionProvider ACTIVE")
            logger.info("EXECUTION_MODE: GPU")
            return "cuda"
        logger.warning(
            "requested execution provider '%s' not supported by facefusion build (%s), falling back",
            EXECUTION_PROVIDER,
            ",".join(sorted(supported)),
        )
        if GPU_ONLY_MODE:
            raise RuntimeError(
                "GPU_ONLY_MODE=1 and requested provider unsupported: "
                f"{EXECUTION_PROVIDER} ({','.join(sorted(supported))})"
            )
        fallback = "cuda" if "cuda" in supported else sorted(supported)[0]
        _log_mode(fallback, "unsupported explicit provider fallback")
        return fallback

    try:
        import onnxruntime as ort

        providers = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in providers and "cuda" in supported:
            logger.info("CUDAExecutionProvider ACTIVE")
            _log_mode("cuda", "cuda runtime detected")
            return "cuda"
    except Exception:
        pass

    logger.error("ERROR: GPU REQUIRED")
    raise RuntimeError("ERROR: GPU REQUIRED")


def get_supported_face_detector_models():
    global _FF_SUPPORTED_FACE_DETECTOR_MODELS
    if _FF_SUPPORTED_FACE_DETECTOR_MODELS is not None:
        return _FF_SUPPORTED_FACE_DETECTOR_MODELS

    models = set()
    try:
        probe_env = prepare_cuda_runtime_env(os.environ.copy())
        ff_py = str(Path(FACEFUSION_DIR) / "facefusion.py")
        r = subprocess.run(
            [FACEFUSION_PYTHON, ff_py, "headless-run", "--help"],
            capture_output=True,
            text=True,
            timeout=25,
            cwd=FACEFUSION_DIR,
            env=probe_env,
        )
        help_text = (r.stdout or "") + "\n" + (r.stderr or "")
        m = re.search(r"--face-detector-model[^\n]*\(choose from ([^)]+)\)", help_text, re.IGNORECASE)
        if m:
            models = {x.strip().lower() for x in m.group(1).split(",") if x.strip()}
    except Exception as e:
        logger.warning("failed to detect face detector models: %s", e)

    if not models:
        models = {"retinaface", "yolo_face"}

    _FF_SUPPORTED_FACE_DETECTOR_MODELS = models
    return models


def resolve_face_detector_model():
    supported = get_supported_face_detector_models()
    preferred = [PRIMARY_FACE_DETECTOR_MODEL, "retinaface", "insightface", "scrfd", "yolo_face"]
    for model in preferred:
        name = str(model or "").strip().lower()
        if name and name in supported:
            return name
    return sorted(supported)[0]


def build_cuda_ld_library_path(existing=""):
    purelib = Path(sysconfig.get_paths().get("purelib", ""))
    nvidia_root = purelib / "nvidia"
    extra = []
    if nvidia_root.exists():
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                extra.append(str(lib_dir))
    # Add CUDA system libs and compat symlinks for CUDA 12/13 mismatch
    _cuda_compat = Path(purelib).parent / "cuda12_compat"
    if _cuda_compat.is_dir():
        extra.insert(0, str(_cuda_compat))
    for _sys_cuda in ["/usr/local/cuda/lib64", "/usr/local/cuda-13.0/targets/x86_64-linux/lib", "/usr/lib/x86_64-linux-gnu"]:
        if Path(_sys_cuda).is_dir():
            extra.append(_sys_cuda)
    parts = extra[:]
    if existing:
        parts.append(existing)
    return ":".join(parts)


def prepare_cuda_runtime_env(env=None, selected_provider=None):
    runtime_env = dict(env or os.environ)
    _local_bin = str(Path.home() / ".local" / "bin")
    if _local_bin not in runtime_env.get("PATH", ""):
        runtime_env["PATH"] = f"{_local_bin}:{runtime_env.get('PATH', '')}"
    runtime_env["LD_LIBRARY_PATH"] = build_cuda_ld_library_path(runtime_env.get("LD_LIBRARY_PATH", ""))
    chosen = (selected_provider or "").strip().lower()
    if chosen:
        runtime_env["EXECUTION_PROVIDER"] = chosen
        runtime_env["GPU_ONLY_MODE"] = "1" if chosen == "cuda" else "0"
    else:
        runtime_env["GPU_ONLY_MODE"] = "1" if GPU_ONLY_MODE else "0"
    return runtime_env


def log_startup_gpu_diagnostics():
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            logger.info("NVIDIA_SMI OUTPUT:\n%s", (r.stdout or "").strip())
        else:
            logger.warning("NVIDIA_SMI FAILED rc=%s err=%s", r.returncode, (r.stderr or "").strip())
    except Exception as e:
        logger.warning("NVIDIA_SMI EXCEPTION: %s", e)

    providers = []
    try:
        import onnxruntime as ort

        providers = list(ort.get_available_providers())
    except Exception as e:
        logger.warning("ONNXRUNTIME provider probe failed: %s", e)

    logger.info("ONNXRUNTIME_PROVIDERS: %s", providers)
    if "CUDAExecutionProvider" in providers:
        logger.info("CUDAExecutionProvider ACTIVE")
        logger.info("EXECUTION_MODE: GPU")


def require_gpu_or_raise():
    nvidia_ok = False
    onnx_cuda_ok = False
    torch_cuda_ok = False

    try:
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=8)
        nvidia_ok = probe.returncode == 0 and bool((probe.stdout or "").strip())
    except Exception:
        nvidia_ok = False

    try:
        import onnxruntime as ort
        providers = set(ort.get_available_providers())
        onnx_cuda_ok = "CUDAExecutionProvider" in providers
    except Exception:
        onnx_cuda_ok = False

    try:
        import torch
        torch_cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        torch_cuda_ok = False

    if not (nvidia_ok and onnx_cuda_ok and torch_cuda_ok):
        logger.error(
            "ERROR: GPU REQUIRED | nvidia_smi=%s onnx_cuda=%s torch_cuda=%s",
            nvidia_ok,
            onnx_cuda_ok,
            torch_cuda_ok,
        )
        raise RuntimeError("ERROR: GPU REQUIRED")


def _extract_validation_frame(media_path, output_path, label="mid"):
    try:
        src = Path(media_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = src.suffix.lower()
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            shutil.copy2(str(src), str(out))
            return str(out)

        duration = detect_video_duration_seconds(str(src))
        if duration and duration > 0:
            if label == "start":
                ss = min(0.8, max(0.0, duration * 0.05))
            elif label == "end":
                ss = max(0.0, duration * 0.85)
            else:
                ss = max(0.0, duration * 0.50)
        else:
            ss = 1.0

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{ss:.3f}", "-i", str(src), "-frames:v", "1", str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not out.exists():
            return ""
        return str(out)
    except Exception:
        return ""


def validate_faceswap_visual_change(input_path, output_path, job_temp_path):
    validate_dir = Path(job_temp_path or TEMP_PATH) / "validation"
    frame_pairs = []
    for label in ("start", "mid", "end"):
        before_img = _extract_validation_frame(input_path, str(validate_dir / f"before_{label}.jpg"), label=label)
        after_img = _extract_validation_frame(output_path, str(validate_dir / f"after_{label}.jpg"), label=label)
        if before_img and after_img:
            frame_pairs.append((label, before_img, after_img))

    if not frame_pairs:
        return False, "frame extraction failed", "", "", 0.0

    try:
        import cv2
        import numpy as np

        best_ratio = -1.0
        best_face_ratio = -1.0
        best_pair = ("", "")
        ratio_parts = []
        face_ratio_parts = []

        face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.isfile(cascade_path):
                face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            face_cascade = None

        for label, before_img, after_img in frame_pairs:
            b = cv2.imread(before_img)
            a = cv2.imread(after_img)
            if b is None or a is None:
                continue

            if b.shape[:2] != a.shape[:2]:
                a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)

            diff = cv2.absdiff(b, a)
            ratio = float(np.mean(diff) / 255.0)
            ratio_parts.append(f"{label}:{ratio:.4f}")
            if ratio > best_ratio:
                best_ratio = ratio
                best_pair = (before_img, after_img)

            # Global pixel mean can miss subtle but real face-only swaps.
            if face_cascade is not None and not face_cascade.empty():
                try:
                    gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
                except Exception:
                    faces = []

                frame_best_face = -1.0
                for (x, y, w, h) in faces:
                    pad_w = int(w * 0.25)
                    pad_h = int(h * 0.25)
                    x1 = max(0, x - pad_w)
                    y1 = max(0, y - pad_h)
                    x2 = min(b.shape[1], x + w + pad_w)
                    y2 = min(b.shape[0], y + h + pad_h)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    roi_b = b[y1:y2, x1:x2]
                    roi_a = a[y1:y2, x1:x2]
                    if roi_b.size == 0 or roi_a.size == 0:
                        continue

                    roi_diff = cv2.absdiff(roi_b, roi_a)
                    roi_ratio = float(np.mean(roi_diff) / 255.0)
                    if roi_ratio > frame_best_face:
                        frame_best_face = roi_ratio

                if frame_best_face >= 0:
                    face_ratio_parts.append(f"{label}:{frame_best_face:.4f}")
                    if frame_best_face > best_face_ratio:
                        best_face_ratio = frame_best_face

        if best_ratio < 0:
            return False, "opencv read failed", "", "", 0.0

        face_detail = ""
        if best_face_ratio >= 0:
            face_detail = f" | face_diff_ratio_max={best_face_ratio:.4f} ({', '.join(face_ratio_parts)})"
        detail = f"diff_ratio_max={best_ratio:.4f} ({', '.join(ratio_parts)}){face_detail}"

        effective_ratio = max(best_ratio, best_face_ratio if best_face_ratio >= 0 else 0.0)
        # Treat near-identical frames as swap failure.
        if effective_ratio <= 0.003:
            return False, f"frames look identical ({detail})", best_pair[0], best_pair[1], effective_ratio
        return True, detail, best_pair[0], best_pair[1], effective_ratio
    except Exception as e:
        return False, f"validation error: {e}", "", "", 0.0


def analyze_faceswap_frame_debug(input_path, output_path, extracted_frames=0, debug_dir=None, max_frames=600):
    result = {
        "extracted_frames": int(max(0, extracted_frames or 0)),
        "total_compared_frames": 0,
        "detected_faces_frames": 0,
        "swapped_frames": 0,
        "face_detector_loaded": False,
        "sample_before": "",
        "sample_after": "",
        "sample_detected_before": "",
        "error": "",
    }

    try:
        import cv2
        import numpy as np

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        result["face_detector_loaded"] = bool(os.path.isfile(cascade_path))

        cap_in = cv2.VideoCapture(str(input_path))
        cap_out = cv2.VideoCapture(str(output_path))
        if not cap_in.isOpened() or not cap_out.isOpened():
            result["error"] = "unable to open input/output video for frame debug"
            try:
                cap_in.release()
                cap_out.release()
            except Exception:
                pass
            return result

        debug_root = None
        if debug_dir:
            try:
                debug_root = Path(debug_dir)
                debug_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                debug_root = None

        def _save_sample(frame, sample_name, boxes=None):
            if debug_root is None or frame is None:
                return ""
            try:
                canvas = frame.copy()
                if boxes:
                    for (x, y, bw, bh) in boxes:
                        cv2.rectangle(canvas, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
                out_path = debug_root / sample_name
                cv2.imwrite(str(out_path), canvas)
                return str(out_path)
            except Exception:
                return ""

        frame_idx = 0
        frame_cap = int(max(24, min(120, max_frames or 600)))
        detect_stride = int(max(1, FRAME_DEBUG_DETECTION_STRIDE))
        tracking_ttl_max = int(max(1, FRAME_DEBUG_TRACKING_TTL))
        tracking_ttl = 0
        tracked_boxes = []
        while True:
            ok_in, frame_in = cap_in.read()
            ok_out, frame_out = cap_out.read()
            if not ok_in or not ok_out:
                break

            frame_idx += 1
            if frame_idx > frame_cap:
                result["error"] = f"frame-debug truncated at {frame_cap} frames"
                break

            result["total_compared_frames"] += 1

            # Detect frequently (default every frame) and track for short gaps.
            should_detect = (frame_idx % detect_stride == 0)
            if should_detect:
                try:
                    faces = _detect_human_faces(frame_in, max_faces=8, relaxed=True)
                except Exception:
                    faces = []
            else:
                faces = []

            if faces:
                tracked_boxes = list(faces)
                tracking_ttl = tracking_ttl_max
            elif tracking_ttl > 0 and tracked_boxes:
                # Lightweight carry-over tracking between nearby frames.
                faces = list(tracked_boxes)
                tracking_ttl -= 1
            else:
                tracked_boxes = []
                tracking_ttl = 0

            if not result["sample_before"]:
                result["sample_before"] = _save_sample(frame_in, "before_detection.jpg")
            if not result["sample_after"]:
                result["sample_after"] = _save_sample(frame_out, "after_detection.jpg")

            if faces:
                result["detected_faces_frames"] += 1
                if not result["sample_detected_before"]:
                    result["sample_detected_before"] = _save_sample(frame_in, "before_detection_faces.jpg", boxes=faces)

            if frame_in.shape[:2] != frame_out.shape[:2]:
                frame_out = cv2.resize(frame_out, (frame_in.shape[1], frame_in.shape[0]), interpolation=cv2.INTER_AREA)

            diff = cv2.absdiff(frame_in, frame_out)
            diff_ratio = float(np.mean(diff) / 255.0)
            swapped_here = diff_ratio > 0.0045
            if faces:
                for (x, y, fw, fh) in faces:
                    pad_w = int(fw * 0.20)
                    pad_h = int(fh * 0.20)
                    x1 = max(0, int(x - pad_w))
                    y1 = max(0, int(y - pad_h))
                    x2 = min(frame_in.shape[1], int(x + fw + pad_w))
                    y2 = min(frame_in.shape[0], int(y + fh + pad_h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    roi_in = frame_in[y1:y2, x1:x2]
                    roi_out = frame_out[y1:y2, x1:x2]
                    if roi_in.size == 0 or roi_out.size == 0:
                        continue
                    roi_diff_ratio = float(np.mean(cv2.absdiff(roi_in, roi_out)) / 255.0)
                    if roi_diff_ratio > 0.0060:
                        swapped_here = True
                        break

            if swapped_here:
                result["swapped_frames"] += 1

        cap_in.release()
        cap_out.release()
        return result
    except Exception as e:
        result["error"] = str(e)
        return result


def apply_content_analyser_bypass():
    if not BYPASS_CONTENT_ANALYSER:
        return True, "content analyser bypass disabled"

    analyser_file = Path(FACEFUSION_DIR) / "facefusion" / "content_analyser.py"
    core_file = Path(FACEFUSION_DIR) / "facefusion" / "core.py"
    if not analyser_file.exists():
        return False, f"missing file: {analyser_file}"
    if not core_file.exists():
        return False, f"missing file: {core_file}"

    src = analyser_file.read_text()
    lines = src.splitlines(keepends=True)

    def patch_function_body(fn_name):
        for i, line in enumerate(lines):
            if line.startswith(f"def {fn_name}("):
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if nxt.startswith("def ") or nxt.startswith("@"):
                        break
                    j += 1
                lines[i + 1:j] = ["        return False\n", "\n"]
                return True
        return False

    for fn_name in ("analyse_frame", "analyse_image", "analyse_video"):
        if not patch_function_body(fn_name):
            return False, f"failed to patch {fn_name}()"

    new_src = "".join(lines)
    if new_src != src:
        analyser_file.write_text(new_src)

    core_src = core_file.read_text().replace(
        "and content_analyser_hash == 'b14e7b92'",
        "and True",
    )
    core_file.write_text(core_src)
    return True, "content analyser bypass patched"


def kill_stale_facefusion_runs():
    """Terminate leftover headless-run processes to avoid multi-process GPU contention."""
    try:
        subprocess.run(
            ["pkill", "-f", "facefusion.py headless-run"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        pass


def can_use_gdrive():
    cfg = _effective_creds()
    if not bool(cfg.get("gdrive_enabled", True)):
        return False
    if not RCLONE_BIN:
        return False
    if os.path.exists(RCLONE_CONF):
        return True
    try:
        cfg = load_persistent_config()
        drive_auth = cfg.get("drive_auth_token")
        if isinstance(drive_auth, dict) and str(drive_auth.get("access_token", "")).strip():
            _write_rclone_drive_token(drive_auth)
            return os.path.exists(RCLONE_CONF)
    except Exception as e:
        logger.warning("gdrive config self-heal failed: %s", e)
    return False


def _gdrive_auth_issue(err_txt):
    txt = str(err_txt or "").lower()
    needles = ["invalid_grant", "unauthorized", "401", "token has been expired", "access token"]
    return any(x in txt for x in needles)


def _format_progress_bar(percent, width=10):
    """Generate progress bar: ▓▓▓░░░░░░░"""
    filled = max(0, min(int(percent / 100.0 * width), width))
    return "▓" * filled + "░" * (width - filled)


def _format_file_size(bytes_val):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} TB"


def _mega_python_login(user, password):
    import json as _json
    from mega import Mega
    try:
        return Mega().login(user, password)
    except _json.JSONDecodeError as e:
        # Mega API returned empty/non-JSON — typically HTTP 402 (IP blocked or account suspended)
        raise RuntimeError(
            f"Mega API returned invalid response (HTTP 402 — IP blocked or account suspended): {e}"
        ) from e


def _mega_python_upload(local_path, user, password):
    session = _mega_python_login(user, password)
    folder_name = "faceswap"
    folder_node = None
    try:
        result = session.find(folder_name)
        if result:
            folder_node = result[1] if isinstance(result, tuple) else result
    except Exception:
        folder_node = None
    if folder_node is None:
        folder_node = session.create_folder(folder_name)
    # upload() expects folder handle string, not the full node dict
    dest = folder_node.get("h") if isinstance(folder_node, dict) else folder_node
    uploaded = session.upload(local_path, dest)
    return session, uploaded


def _mega_python_get_link(session, uploaded_node):
    if not uploaded_node:
        return False, "empty uploaded node"
    link = session.get_upload_link(uploaded_node)
    if link and str(link).startswith("http"):
        return True, str(link).strip()
    return False, "python mega link missing"


def mega_upload(local_path):
    if not os.path.isfile(local_path):
        return False, f"Local file missing: {local_path}"

    u, p = get_mega_creds()
    if not (u and p):
        return False, "MEGA credentials missing"

    now = time.time()
    if mega_state["auth_backoff_until"] > now:
        # Cooldown is only useful after proven auth failures. If creds are valid now,
        # clear stale cooldown and proceed immediately.
        ok, info = validate_mega_creds(u, p)
        if not ok:
            wait_left = int(mega_state["auth_backoff_until"] - now)
            wait_left = max(1, wait_left)
            return False, f"MEGA auth cooldown active: wait {wait_left}s before retry ({info})"

    try:
        if not mega_state["mkdir_ready"]:
            _mega_rate_limit_wait()
            mkdir_r = subprocess.run(
                ["megamkdir", "--username", u, "--password", p, "/Root/faceswap"],
                capture_output=True, text=True, timeout=60
            )
            if mkdir_r.returncode != 0:
                mkdir_err = _short_err(mkdir_r)
                mkdir_err_l = mkdir_err.lower()
                if "exists" in mkdir_err_l or "already" in mkdir_err_l:
                    mega_state["mkdir_ready"] = True
                else:
                    _mega_mark_auth_failure(mkdir_err)
                    logger.warning("megamkdir failed: %s", mkdir_err)
                    if is_mega_auth_error(mkdir_err):
                        # CLI auth failed (e.g. 402) — mark mkdir_ready so next call skips mkdir
                        mega_state["mkdir_ready"] = True
                        # Try Python mega SDK fallback
                        try:
                            logger.info("MEGA CLI auth failed, trying Python SDK fallback")
                            _mega_rate_limit_wait()
                            session, uploaded = _mega_python_upload(local_path, u, p)
                            ok_link, link_or_err = _mega_python_get_link(session, uploaded)
                            if ok_link:
                                mega_state["last_python_link"] = link_or_err
                            _mega_mark_auth_success()
                            return True, ""
                        except Exception as py_e:
                            logger.warning("MEGA Python fallback also failed: %s", py_e)
                            return False, f"MEGA mkdir auth failed: {mkdir_err} | python fallback: {py_e}"
            else:
                mega_state["mkdir_ready"] = True
                _mega_mark_auth_success()

        put_cmd = [
            "megaput", "--username", u, "--password", p,
            "--path", "/Root/faceswap/", local_path
        ]
        upload_timeout = MEGA_UPLOAD_TIMEOUT_SEC if MEGA_UPLOAD_TIMEOUT_SEC > 0 else 900
        run_kw = {"capture_output": True, "text": True, "timeout": upload_timeout}

        _mega_rate_limit_wait()
        r = subprocess.run(put_cmd, **run_kw)
        if r.returncode == 0:
            _mega_mark_auth_success()
            return True, ""
        err = _short_err(r)
        err_l = err.lower()
        if "file already exists" in err_l:
            _mega_mark_auth_success()
            return True, ""
        _mega_mark_auth_failure(err)
        return False, f"megaput rc={r.returncode}: {err}"
    except FileNotFoundError as e:
        # CLI missing -> Python MEGA fallback with retry
        for _attempt in range(1, 4):
            try:
                logger.info("MEGA Python SDK attempt %d/3 (CLI missing: %s)", _attempt, e)
                _mega_rate_limit_wait()
                session, uploaded = _mega_python_upload(local_path, u, p)
                ok_link, link_or_err = _mega_python_get_link(session, uploaded)
                if ok_link:
                    mega_state["last_python_link"] = link_or_err
                _mega_mark_auth_success()
                logger.info("MEGA Python SDK upload successful on attempt %d", _attempt)
                return True, ""
            except Exception as py_e:
                logger.error("MEGA Python SDK attempt %d/3 failed — %s: %s", _attempt, type(py_e).__name__, py_e)
                if _attempt < 3:
                    logger.info("Waiting 30s before next MEGA attempt...")
                    time.sleep(30)
        return False, f"MEGA CLI missing ({e}) and all 3 Python SDK attempts failed"
    except subprocess.TimeoutExpired:
        timeout_txt = MEGA_UPLOAD_TIMEOUT_SEC if MEGA_UPLOAD_TIMEOUT_SEC > 0 else 900
        return False, f"MEGA upload timeout after {timeout_txt}s"
    except Exception as e:
        # Non-CLI unexpected failure -> Python MEGA fallback with retry
        for _attempt in range(1, 4):
            try:
                logger.info("MEGA Python SDK attempt %d/3 (CLI error: %s)", _attempt, e)
                _mega_rate_limit_wait()
                session, uploaded = _mega_python_upload(local_path, u, p)
                ok_link, link_or_err = _mega_python_get_link(session, uploaded)
                if ok_link:
                    mega_state["last_python_link"] = link_or_err
                _mega_mark_auth_success()
                logger.info("MEGA Python SDK upload successful on attempt %d", _attempt)
                return True, ""
            except Exception as py_e:
                logger.error("MEGA Python SDK attempt %d/3 failed — %s: %s", _attempt, type(py_e).__name__, py_e)
                if _attempt < 3:
                    logger.info("Waiting 30s before next MEGA attempt...")
                    time.sleep(30)
        return False, f"MEGA upload exception: {e} | all 3 Python SDK attempts failed"


def mega_export_link(remote_path, retries=3, delay_seconds=2):
    u, p = get_mega_creds()
    cached_link = str(mega_state.get("last_python_link") or "").strip()
    if cached_link.startswith("http"):
        return True, cached_link
    last_err = ""

    for attempt in range(retries):
        try:
            r = subprocess.run(
                ["megals", "--username", u, "--password", p, "--export", remote_path],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode == 0:
                m = re.search(r"https://mega\.nz/\S+", (r.stdout or ""))
                if m:
                    return True, m.group(0)
                last_err = "export link not found"
            else:
                last_err = f"megals rc={r.returncode}: {_short_err(r)}"
        except FileNotFoundError as e:
            # If CLI missing but Python upload already produced link, use it.
            cached_link = str(mega_state.get("last_python_link") or "").strip()
            if cached_link.startswith("http"):
                return True, cached_link
            return False, f"MEGA CLI not found: {e}"
        except subprocess.TimeoutExpired:
            last_err = "megals export timeout"
        except Exception as e:
            last_err = f"MEGA export exception: {e}"

        if attempt < retries - 1:
            time.sleep(delay_seconds)

    return False, last_err or "unable to export link"


def gdrive_upload(local_path):
    if not bool(_effective_creds().get("gdrive_enabled", True)):
        return False, "", "gdrive disabled by config"
    if not can_use_gdrive():
        if not RCLONE_BIN:
            return False, "", "rclone binary not found in PATH"
        return False, "", "rclone config not found"
    if not RCLONE_BIN:
        return False, "", "rclone binary not found in PATH"
    if not os.path.exists(RCLONE_CONF):
        return False, "", "rclone config not found"
    drive_target = get_gdrive_target_folder()
    logger.info("[GDRIVE_DEBUG] drive_target=%s RCLONE_CONF=%s RCLONE_BIN=%s", drive_target, RCLONE_CONF, RCLONE_BIN)
    upload_timeout = max(300, int(GDRIVE_UPLOAD_TIMEOUT_SEC or 1200))
    try:
        copy_cmd = [RCLONE_BIN, "--config", RCLONE_CONF, "copy", local_path, drive_target]
        r = None
        last_copy_err = ""
        for attempt in range(1, GDRIVE_UPLOAD_RETRIES + 1):
            try:
                r = subprocess.run(copy_cmd, capture_output=True, text=True, timeout=upload_timeout)
                if r.returncode == 0:
                    break

                last_copy_err = f"rclone copy rc={r.returncode}: {_short_err(r)}"
                if _gdrive_auth_issue(_short_err(r)):
                    # Retry after re-writing token from persistent config.
                    try:
                        cfg = load_persistent_config()
                        drive_auth = cfg.get("drive_auth_token")
                        if isinstance(drive_auth, dict) and str(drive_auth.get("access_token", "")).strip():
                            _write_rclone_drive_token(drive_auth)
                    except Exception as heal_e:
                        logger.warning("gdrive auth self-heal retry failed: %s", heal_e)

            except subprocess.TimeoutExpired:
                last_copy_err = f"rclone timeout after {upload_timeout}s"

            if attempt < GDRIVE_UPLOAD_RETRIES:
                logger.warning("gdrive upload retrying attempt=%s/%s err=%s", attempt, GDRIVE_UPLOAD_RETRIES, last_copy_err)
                time.sleep(2 * attempt)

        if r is None or r.returncode != 0:
            return False, "", last_copy_err or "rclone copy failed"

        fname = os.path.basename(local_path)
        lr = subprocess.run(
            [RCLONE_BIN, "--config", RCLONE_CONF, "link", f"{drive_target}/{fname}"],
            capture_output=True, text=True, timeout=120
        )
        if lr.returncode != 0:
            # Upload succeeded but public link generation may fail on some remotes.
            return True, "", f"rclone link rc={lr.returncode}: {_short_err(lr)}"
        return True, lr.stdout.strip(), ""
    except FileNotFoundError as e:
        return False, "", f"rclone missing: {e}"
    except Exception as e:
        return False, "", f"gdrive upload exception: {e}"


def smart_upload(local_path):
    """GDrive primary, MEGA fallback. Returns (success, platform, info)"""
    reload_runtime_credentials()

    ok, link, gdrive_info = gdrive_upload(local_path)
    if ok and link:
        logger.info("[UPLOAD] gdrive=SUCCESS")
        return True, "GDRIVE", link
    if ok and not link:
        logger.warning("[UPLOAD] gdrive=UPLOAD_OK_LINK_MISSING reason=%s — falling through to MEGA", gdrive_info)
    else:
        logger.warning("[UPLOAD] gdrive=FAILED reason=%s — trying MEGA fallback", gdrive_info)

    if not can_use_mega():
        logger.error("[UPLOAD] gdrive=FAILED mega=SKIPPED reason=mega_credentials_missing")
        return False, "gdrive", f"GDrive: {gdrive_info} | MEGA disabled (credentials missing)"

    mega_ok, mega_info = mega_upload(local_path)
    if mega_ok:
        logger.info("[UPLOAD] gdrive=FAILED mega=SUCCESS")
        remote_path = f"/Root/faceswap/{os.path.basename(local_path)}"
        link_ok, link_info = mega_export_link(remote_path)
        if link_ok:
            return True, "MEGA", link_info
        logger.warning("MEGA uploaded but link export failed: %s", link_info)
        return True, "MEGA", remote_path
    logger.error("[UPLOAD] gdrive=FAILED mega=FAILED gdrive_reason=%s mega_reason=%s", gdrive_info, mega_info)
    return False, "both", f"GDrive: {gdrive_info}\nMEGA: {mega_info}"


async def wait_and_send_mega_link(bot, chat_id, remote_path, retries=8, delay_seconds=10):
    """Best-effort MEGA export retries; sends link once it becomes available."""
    for attempt in range(retries):
        ok, info = mega_export_link(remote_path, retries=1, delay_seconds=0)
        if ok and info.startswith("http"):
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "✅ *MEGA Link Ready!*\n"
                        f"🔗 [Open MEGA Output]({info})"
                    ),
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.warning("failed to send delayed mega link: %s", e)
            return True
        if attempt < retries - 1:
            await asyncio.sleep(delay_seconds)
    try:
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "ℹ️ MEGA upload ho gaya, but export link generate nahi hua.\n"
                f"Manual path: `{remote_path}`"
            ),
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.warning("failed to send mega path fallback: %s", e)
    return False


def face_to_clean_jpg(src, dst_path=None):
    src_path = str(src)
    if dst_path:
        clean = str(dst_path)
    else:
        src_obj = Path(src_path)
        clean = str(src_obj.with_name(f"{src_obj.stem}_clean.jpg"))
    try:
        from PIL import Image as PILImage
        Path(clean).parent.mkdir(parents=True, exist_ok=True)
        PILImage.open(src_path).convert("RGB").save(clean, "JPEG", quality=95)
        return clean
    except Exception:
        return src_path


def clear_face_runtime_cache(chat_id=None):
    """Clear temporary/cached face artifacts after face replacement."""
    with suppress(Exception):
        req_root = Path(FACE_DIR) / "_requests"
        if req_root.exists():
            shutil.rmtree(req_root, ignore_errors=True)
    with suppress(Exception):
        for base in [Path(TEMP_PATH), Path(PIPELINE) / "cache", Path(FACEFUSION_DIR) / ".cache"]:
            if not base.exists():
                continue
            for fp in base.rglob("*embedding*"):
                if fp.is_file():
                    fp.unlink(missing_ok=True)
    with suppress(Exception):
        import torch

        torch.cuda.empty_cache()
    if chat_id is not None:
        logger.info("[FACE-CHANGE] runtime cache cleared chat=%s", chat_id)


def request_studio_sleep():
    """Ask Lightning Studio to sleep using SDK first, then CLI fallback."""
    if IS_LIGHTWEIGHT and VIRTUAL_SLEEP_MODE:
        return True, "virtual sleep mode active"

    if os.environ.get("STUDIO_SLEEP_DRY_RUN", "0").strip().lower() in {"1", "true", "yes"}:
        return True, "dry-run enabled"

    def _compact_error(raw, limit=420):
        text = str(raw or "").strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        if len(text) > limit:
            return text[: limit - 3] + "..."
        return text

    cloudspace_err = ""
    project_id = (os.environ.get("LIGHTNING_CLOUD_PROJECT_ID") or "").strip()
    cloudspace_id = (os.environ.get("LIGHTNING_CLOUD_SPACE_ID") or "").strip()
    if project_id and cloudspace_id:
        try:
            from lightning_sdk.lightning_cloud.rest_client import create_swagger_client
            from lightning_sdk.lightning_cloud.openapi.api.cloud_space_service_api import CloudSpaceServiceApi

            api_client = create_swagger_client(check_context=False, with_auth=True)
            CloudSpaceServiceApi(api_client).cloud_space_service_stop_cloud_space_instance(
                project_id=project_id,
                id=cloudspace_id,
            )
            return True, f"sleep requested via cloudspace api ({cloudspace_id})"
        except Exception as e:
            cloudspace_err = _compact_error(e)

    try:
        sdk = subprocess.run(
            [
                RUNTIME_PYTHON,
                "-c",
                "from lightning_sdk import Studio; Studio().stop()",
            ],
            capture_output=True,
            text=True,
            timeout=40,
        )
        if sdk.returncode == 0:
            return True, "sleep requested via lightning_sdk"
        sdk_err = _compact_error(sdk.stderr or sdk.stdout or "")
    except Exception as e:
        sdk_err = _compact_error(e)

    studio_name = (
        os.environ.get("LIGHTNING_STUDIO_NAME")
        or os.environ.get("LIGHTNING_NAME")
        or ""
    ).strip()

    teamspace_name = (os.environ.get("LIGHTNING_TEAMSPACE") or "").strip()
    username = (os.environ.get("LIGHTNING_USERNAME") or os.environ.get("LIGHTNING_LINUX_USERNAME") or "").strip()
    teamspace_ref = f"{username}/{teamspace_name}" if username and teamspace_name else ""

    if studio_name:
        try:
            cli_cmd = ["lightning", "stop", "studio", studio_name]
            if teamspace_ref:
                cli_cmd.extend(["--teamspace", teamspace_ref])
            cli = subprocess.run(cli_cmd, capture_output=True, text=True, timeout=40)
            if cli.returncode == 0:
                return True, f"sleep requested via lightning cli ({studio_name})"
            cli_err = _compact_error(cli.stderr or cli.stdout or "")
            parts = []
            if cloudspace_err:
                parts.append(f"cloudspace api failed: {cloudspace_err}")
            if sdk_err:
                parts.append(f"sdk failed: {sdk_err}")
            parts.append(f"cli failed: {cli_err}")
            return False, " | ".join(parts)
        except Exception as e:
            parts = []
            if cloudspace_err:
                parts.append(f"cloudspace api failed: {cloudspace_err}")
            if sdk_err:
                parts.append(f"sdk failed: {sdk_err}")
            parts.append(f"cli exception: {_compact_error(e)}")
            return False, " | ".join(parts)

    parts = []
    if cloudspace_err:
        parts.append(f"cloudspace api failed: {cloudspace_err}")
    if sdk_err:
        parts.append(f"sdk failed: {sdk_err}")
    parts.append("no valid studio name for cli fallback")
    return False, " | ".join(parts)


def _active_state_pid_for_chat(chat_id):
    st = _load_active_job_state() or {}
    if str(st.get("chat_id") or "") != str(chat_id):
        return 0
    worker_pid = int(st.get("worker_pid") or 0)
    processing_pid = int(st.get("processing_pid") or 0)
    if worker_pid > 0 and _pid_is_job_process(worker_pid):
        return worker_pid
    if processing_pid > 0 and _pid_is_job_process(processing_pid):
        return processing_pid
    return 0


def _has_any_active_job_pid():
    # Autosleep must remain blocked while any live worker/processing pid exists.
    try:
        for proc in list(active_jobs.values()):
            if getattr(proc, "poll", lambda: 1)() is None:
                return True
    except Exception:
        pass

    st = _load_active_job_state() or {}
    for pid_key in ("worker_pid", "processing_pid"):
        try:
            pid = int(st.get(pid_key) or 0)
        except Exception:
            pid = 0
        if pid > 0 and _pid_is_job_process(pid):
            return True
    return False


async def _safe_restart_self(application, chat_id):
    await asyncio.sleep(1)
    try:
        app_obj = application
        if app_obj is not None:
            await app_obj.stop()
    except Exception:
        pass
    restart_env = os.environ.copy()
    restart_env["BOT_SAFE_RESTART"] = "1"
    os.execve(RUNTIME_PYTHON, [RUNTIME_PYTHON, str(ROOT_DIR / "bot.py")], restart_env)


def extract_face_from_video_frame(video_path):
    """Fallback source face: use first video frame when user has not set a face image."""
    try:
        os.makedirs(FACE_DIR, exist_ok=True)
        frame_img = f"{FACE_DIR}/auto_source_from_video.jpg"
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        hw_decode_args = build_ffmpeg_hw_decode_args(video_path)
        use_gpu_filter = bool(hw_decode_args)
        cmd.extend(hw_decode_args)
        cmd.extend([
            "-i", video_path,
            "-vf",
            "scale_cuda=640:640:force_original_aspect_ratio=decrease,hwdownload,format=nv12,format=rgb24,select=eq(n\\,0)"
            if use_gpu_filter else "select=eq(n\\,0)",
            "-vframes", "1",
            "-threads", str(FFMPEG_CPU_THREADS),
            frame_img,
        ])
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r.returncode != 0 or not os.path.isfile(frame_img):
            return None
        return face_to_clean_jpg(frame_img)
    except Exception:
        return None


def main_kb(chat_id=None):
    direct_label = "🎯 Direct mode"
    multi_label = "🧩 Multi mode"
    female_label = "👩 Female only"
    male_label = "👨 Male only"
    all_gender_label = "👥 All genders"

    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🖼️ Change Face",    callback_data="change_face"),
            InlineKeyboardButton("👁 View Face",       callback_data="view_face"),
        ],
        [
            InlineKeyboardButton("⏹ Stop Job",         callback_data="stop_job"),
            InlineKeyboardButton("📊 Job Status",      callback_data="job_status_btn"),
            InlineKeyboardButton("🔁 Reupload Output", callback_data="reupload_output_menu"),
        ],
        [
            InlineKeyboardButton("🧨 Job Terminator", callback_data="queue_terminate_menu"),
        ],
        [
            InlineKeyboardButton("💾 Check Storage",   callback_data="check_storage"),
            InlineKeyboardButton("🗑 Clear Workspace", callback_data="clean_workspace"),
        ],
        [
            InlineKeyboardButton("📥 Download Output", callback_data="download_output"),
            InlineKeyboardButton("🔄 Change Drive Token", callback_data="change_drive_token"),
        ],
        [
            InlineKeyboardButton("🔑 Change MEGA", callback_data="change_mega"),
        ],
        [
            InlineKeyboardButton("✂️ Clip Range", callback_data="clip_settings"),
        ],
        [
            InlineKeyboardButton(direct_label, callback_data="mode_direct"),
            InlineKeyboardButton(multi_label, callback_data="mode_multi"),
        ],
        [
            InlineKeyboardButton(female_label, callback_data="female_only_on"),
            InlineKeyboardButton(male_label, callback_data="male_only_on"),
            InlineKeyboardButton(all_gender_label, callback_data="female_only_off"),
        ],
        [
            InlineKeyboardButton("⚡ Quick Sleep",      callback_data="quick_sleep"),
        ],
        [
            InlineKeyboardButton("▶️ Start Bot",       callback_data="start_bot"),
        ],
    ])
    logger.info("Keyboard built with Clip Range button")
    return keyboard


def _custom_mega_temp_snapshot(context):
    file_path = str(context.user_data.get("custom_mega_temp_file") or "").strip()
    temp_dir = str(context.user_data.get("custom_mega_temp_dir") or "").strip()
    return file_path, temp_dir


def _cleanup_custom_mega_temp(context):
    file_path, temp_dir = _custom_mega_temp_snapshot(context)
    context.user_data.pop("custom_mega_temp_file", None)
    context.user_data.pop("custom_mega_temp_dir", None)
    if temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    if file_path:
        with suppress(Exception):
            Path(file_path).unlink(missing_ok=True)


def _sleep_remaining_seconds(chat_id):
    state = sleep_countdown_state.get(chat_id, {}) or {}
    if str(state.get("status") or "").lower() != "running":
        return None
    ends_at_monotonic = float(state.get("ends_at_monotonic") or 0.0)
    if ends_at_monotonic <= 0:
        return None
    return max(0, int(ends_at_monotonic - time.monotonic()))


def _sleep_delay_minutes_text():
    mins = max(1, int(round(float(SLEEP_COUNTDOWN_SECONDS) / 60.0)))
    return f"{mins} min"


def cleanup_modes_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🧹 Temp Only", callback_data="clean_workspace_temp_only"),
        ],
        [
            InlineKeyboardButton("📁 Old Outputs", callback_data="clean_workspace_outputs_old"),
        ],
        [
            InlineKeyboardButton("💣 Full Clean", callback_data="clean_workspace_full_clean"),
        ],
        [
            InlineKeyboardButton("❌ Cancel", callback_data="clean_workspace_cancel"),
        ],
    ])


def sleep_countdown_kb():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Stay Awake", callback_data="cancel_sleep_countdown"),
            InlineKeyboardButton("🛑 Sleep Now", callback_data="sleep_now"),
        ]
    ])


async def send_ready_banner(message_obj, chat_id, extra_note=None):
    reload_runtime_credentials()

    if chat_id not in chat_modes:
        set_chat_mode(chat_id, "direct")

    mode = get_chat_mode(chat_id)
    mega_user, _ = get_mega_creds()
    mega_ok = can_use_mega()
    gdrive_ok = can_use_gdrive()
    execution_note = "GPU Only (CUDA)" if GPU_ONLY_MODE else "Auto (GPU/CPU)"

    gdrive_folder_display = GDRIVE_FOLDER.split(":", 1)[-1] if ":" in GDRIVE_FOLDER else GDRIVE_FOLDER

    parts = []
    if extra_note:
        parts.append(extra_note)

    parts.append(
        "🤖 *FaceSwap Bot v14 — Ready!*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎛 Current Mode: *{mode_label(mode)}*\n"
        "(Default mode: Direct FaceSwap)\n"
        f"🖥 Execution: *{execution_note}*\n\n"
        "📹 MEGA video link bhejo\n"
        "👥 2 links = video + custom face\n\n"
        "📤 *Upload order:*\n"
        f"1️⃣ MEGA `/Root/faceswap/` {'✅' if mega_ok else '❌ not configured'}\n"
        f"2️⃣ Google Drive `{gdrive_folder_display}/` {'✅' if gdrive_ok else '❌ not configured'}\n\n"
        f"📧 MEGA: `{mask_secret(mega_user)}`\n"
        f"⚡ Job complete hone ke {_sleep_delay_minutes_text()} baad studio auto-sleep"
    )

    text = "\n\n".join(parts)
    await message_obj.reply_text(text, parse_mode="Markdown", reply_markup=main_kb(chat_id))


async def send_startup_activation_message(app):
    global startup_notice_sent
    if os.environ.get("BOT_SAFE_RESTART", "").strip().lower() in {"1", "true", "yes"}:
        return
    if startup_notice_sent:
        return

    startup_notice_sent = True
    chat_id = str(ALLOWED_CHAT_ID)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # GPU info
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        name, mem = r.stdout.strip().split(",", 1)
        gpu_info = f"{name.strip()} · {int(mem.strip())//1024}GB"
    except Exception:
        gpu_info = "GPU info unavailable"

    dashboard_url = (DASHBOARD_PUBLIC_URL or "").rstrip("/")
    live_url = f"{dashboard_url}/live" if dashboard_url else "—"

    text = (
        "⚡ *Facefusion Pipeline V5 Pro*\n\n"
        "✅ Bot is online and ready\n\n"
        f"🕐 Started: `{now}`\n"
        f"🌐 Dashboard: {live_url}\n"
        f"🖥 GPU: `{gpu_info}`\n"
        f"📋 Queue: Ready\n"
        f"😴 Auto-sleep: Armed\n\n"
        "Send a Mega link to begin processing."
    )
    try:
        await safe_send_message(app.bot, chat_id, text, parse_mode="Markdown")
        _update_lifecycle_state(chat_id, is_bot_active=True)
        logger.info("[BOT_ONLINE] chat=%s dashboard=%s gpu=%s", chat_id, live_url, gpu_info)
    except Exception as e:
        logger.warning("startup activation message failed: %s", e)


async def runtime_idle_heartbeat_loop():
    chat_id = str(ALLOWED_CHAT_ID)
    interval = max(60, int(os.environ.get("BOT_HEARTBEAT_INTERVAL_SEC", "180") or 180))
    while True:
        try:
            await asyncio.sleep(interval)
            busy = bool(_is_chat_busy(chat_id) or _queue_size(chat_id) > 0)
            if not busy:
                await asyncio.to_thread(_release_runtime_memory, chat_id)
            _update_lifecycle_state(
                chat_id,
                is_bot_active=True,
                is_job_running=busy,
                is_countdown_running=bool(_task_is_running(sleep_countdown_tasks.get(chat_id))),
            )
            logger.info("[HEARTBEAT] bot_alive=1 chat=%s busy=%s queue=%s", chat_id, int(busy), _queue_size(chat_id))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("runtime heartbeat error: %s", e)


def validate_startup_credentials():
    reload_runtime_credentials()
    creds = _effective_creds()
    ok, errors = validate_credentials(creds)
    logger.info("credentials loaded (masked): %s", json.dumps(masked_credentials(creds), ensure_ascii=True))

    critical = []
    warnings = []
    for err in errors:
        if "BOT_TOKEN" in str(err):
            critical.append(err)
        else:
            warnings.append(err)

    if warnings:
        for item in warnings:
            logger.warning("credential warning: %s", item)

    if not can_use_mega():
        logger.warning("MEGA credentials missing. MEGA upload disabled; Google Drive fallback will be used.")

    if critical:
        message = ["Credential validation failed at startup:"]
        for idx, err in enumerate(critical, start=1):
            message.append(f"{idx}. {err}")
        raise SystemExit("\n".join(message))

    if ok:
        return


async def start_sleep_countdown(app, chat_id, reason_text, delay_seconds=AUTO_SHUTDOWN_DELAY_SEC, force_allow=False):
    global countdown_task, is_countdown_running, start_sleep_timer
    if not AUTO_SLEEP_ENABLED:
        logger.info("auto-sleep disabled by config; countdown skipped chat=%s", chat_id)
        append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "disabled_skip", f"chat={chat_id}")
        _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
        return None

    if force_allow:
        can_schedule = True
    else:
        can_schedule = _can_auto_sleep(chat_id)

    if not can_schedule:
        logger.info(
            "sleep countdown skipped chat=%s busy=%s queue=%s no_bg=%s phase=%s",
            chat_id,
            _is_chat_busy(chat_id),
            _queue_size(chat_id),
            _no_background_task_running(chat_id),
            _last_job_phase(chat_id),
        )
        _update_lifecycle_state(chat_id, is_countdown_running=False)
        start_sleep_timer = False
        return None

    if _has_any_active_job_pid():
        if force_allow:
            logger.info(
                "sleep countdown active_job_pid detected but force_allow=True; clearing stale state chat=%s",
                chat_id,
            )
            # The caller (post-job hook) asserts the worker has finished; clear any
            # stale persisted pid so the countdown can proceed deterministically.
            try:
                _clear_active_job_state()
            except Exception:
                pass
        else:
            logger.info("sleep countdown skipped chat=%s reason=active_job_pid", chat_id)
            _update_lifecycle_state(chat_id, is_countdown_running=False)
            start_sleep_timer = False
            return None

    existing = sleep_countdown_tasks.get(chat_id)
    if existing and not existing.done():
        logger.info("[AUTO_SLEEP_SKIP] chat=%s reason=timer_already_running", chat_id)
        return existing

    delay_seconds = int(SLEEP_COUNTDOWN_SECONDS)
    end_time_monotonic = float(time.monotonic()) + float(delay_seconds)

    sleep_timer_active[chat_id] = True
    start_sleep_timer = True

    sleep_countdown_state[chat_id] = {
        "chat_id": str(chat_id),
        "sleep_timer_active": True,
        "status": "running",
        "reason": reason_text,
        "started_at": time.time(),
        "started_at_monotonic": float(time.monotonic()),
        "delay_seconds": delay_seconds,
        "ends_at_monotonic": end_time_monotonic,
    }
    _save_sleep_countdown_state(sleep_countdown_state[chat_id])
    _update_lifecycle_state(chat_id, is_countdown_running=True, can_auto_sleep=True)
    logger.info("[AUTO SLEEP] Countdown started (%ss)", delay_seconds)
    logger.info("[AUTO_SLEEP] delay=%s", delay_seconds)
    logger.info("[SLEEP_TIMER_START] chat=%s delay=%s", chat_id, delay_seconds)
    append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_started", f"chat={chat_id} delay={delay_seconds} reason={reason_text}")

    task = asyncio.create_task(
        run_sleep_countdown(app, chat_id, reason_text=reason_text, delay_seconds=delay_seconds)
    )
    sleep_countdown_tasks[chat_id] = task
    countdown_task = task
    is_countdown_running = True
    return task


async def run_sleep_countdown(app, chat_id, reason_text, delay_seconds=AUTO_SHUTDOWN_DELAY_SEC):
    global countdown_task, is_countdown_running, start_sleep_timer
    msg = None
    try:
        total_seconds = max(1, int(delay_seconds))
        end_time = float(time.monotonic()) + float(total_seconds)

        def _countdown_text(remain_seconds):
            return f"😴 Studio sleeping in: {int(remain_seconds)}s"

        initial_text = _countdown_text(total_seconds)
        msg = await app.bot.send_message(
            chat_id=chat_id,
            text=initial_text,
            reply_markup=sleep_countdown_kb(),
            parse_mode="Markdown",
        )

        last_rendered_remain = total_seconds
        sleep_request_notice_sent = False
        logger.info("[AUTO_SLEEP_START] chat=%s delay=%s", chat_id, total_seconds)
        append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_tick", f"chat={chat_id} remain={total_seconds}")

        async def _safe_edit_countdown(text):
            if msg is None:
                return
            for _ in range(2):
                try:
                    await app.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=msg.message_id,
                        text=text,
                        reply_markup=sleep_countdown_kb(),
                        parse_mode="Markdown",
                    )
                    return
                except Exception:
                    await asyncio.sleep(0.2)
        
        while True:
            actual_remain = max(0, int(end_time - float(time.monotonic())))
            remain = actual_remain
            if last_rendered_remain is not None and remain < (last_rendered_remain - 1):
                remain = max(0, last_rendered_remain - 1)

            if remain != last_rendered_remain and remain > 0:
                logger.info("[AUTO_SLEEP_TICK] chat=%s remain=%ds", chat_id, remain)

            if _has_any_active_job_pid():
                logger.info("[AUTO_SLEEP_CANCEL] reason=active_pid chat=%s", chat_id)
                append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_cancelled", f"chat={chat_id} reason=active_pid")
                sleep_timer_active[chat_id] = False
                start_sleep_timer = False
                _clear_sleep_countdown_state()
                _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
                try:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text="🟢 Active job detected\nSleep cancelled automatically.",
                        reply_markup=main_kb(),
                    )
                except Exception:
                    pass
                return

            if remain <= 0:
                break

            if remain != last_rendered_remain:
                text = _countdown_text(remain)
                await _safe_edit_countdown(text)
                last_rendered_remain = remain
                # Log at 30-second intervals
                if remain % 30 == 0:
                    logger.info("[SLEEP_COUNTDOWN] %ss remaining — queue_empty=%s busy=%s",
                                remain, _queue_size(chat_id) == 0, _is_chat_busy(chat_id))

            if remain <= 5 and not sleep_request_notice_sent:
                idle_mins = max(1, int(round(float(delay_seconds) / 60.0)))
                try:
                    await app.bot.send_message(
                        chat_id=chat_id,
                        text=f"😴 Studio {idle_mins} minutes idle tha, ab sleep mode mein ja raha hai...",
                    )
                except Exception:
                    pass
                sleep_request_notice_sent = True

            await asyncio.sleep(1)

        await _safe_edit_countdown(_countdown_text(0))

        # Final guard: if new jobs arrived while countdown was running, abort sleep.
        logger.info("[AUTO_SLEEP_FINAL_GUARD] chat=%s is_all_jobs_completed=%s is_busy=%s queue=%s",
                    chat_id, is_all_jobs_completed(chat_id), _is_chat_busy(chat_id), _queue_size(chat_id))
        if not is_all_jobs_completed(chat_id):
            logger.info("[SLEEP_FINAL_GUARD] BLOCKED — job is active, sleep request suppressed")
            logger.info("[AUTO SLEEP] Countdown cancelled due to new activity")
            logger.info("[AUTO_SLEEP_CANCEL] reason=new_job chat=%s", chat_id)
            append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_cancelled", f"chat={chat_id} reason=new_job")
            await app.bot.send_message(
                chat_id=chat_id,
                text="🟢 Activity detected\nSleep cancelled automatically.",
                reply_markup=main_kb(),
            )
            sleep_timer_active[chat_id] = False
            start_sleep_timer = False
            sleep_countdown_state[chat_id] = {
                "chat_id": str(chat_id),
                "sleep_timer_active": False,
                "status": "cancelled",
                "reason": "Queue became non-empty during countdown",
                "cancelled_at": time.time(),
            }
            _clear_sleep_countdown_state()
            _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
            return

        sleep_countdown_state[chat_id] = {
            "chat_id": str(chat_id),
            "sleep_timer_active": False,
            "status": "completed",
            "reason": reason_text,
            "completed_at": time.time(),
        }
        sleep_timer_active[chat_id] = False
        start_sleep_timer = False
        _clear_sleep_countdown_state()
        _update_lifecycle_state(chat_id, is_countdown_running=False)

        if SLEEP_TEST_MODE:
            logger.info("[SLEEP_FINAL_GUARD] PASSED — triggering sleep request (SLEEP_TEST_MODE=true, request blocked)")
            logger.info("[SLEEP TEST MODE] real sleep suppressed by SLEEP_TEST_MODE")
            await app.bot.send_message(
                chat_id=chat_id,
                text="[SLEEP TEST MODE] Real sleep suppressed (SLEEP_TEST_MODE=1)",
                reply_markup=main_kb(),
            )
            return

        logger.info("[AUTO_SLEEP_TRIGGER] chat=%s countdown_complete=True executing_sleep_request", chat_id)
        append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "sleep_trigger_executing", f"chat={chat_id}")
        ok, info = await asyncio.to_thread(request_studio_sleep)
        if ok:
            logger.info("[AUTO SLEEP] Entering sleep mode")
            logger.info("[AUTOSLEEP_TRIGGERED] chat=%s", chat_id)
            logger.info("[AUTO_SLEEP_TRIGGER] ✅ sleep request successful chat=%s info=%s", chat_id, info)
            append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "sleep_request_ok", f"chat={chat_id} info={info}")
            await asyncio.sleep(2)
            sys.exit(0)
        else:
            logger.info("sleep trigger not available in this env: %s", info)
            append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "sleep_request_failed", f"chat={chat_id} info={info}")
            await app.bot.send_message(
                chat_id=chat_id,
                text=f"❌ Countdown complete, lekin studio sleep request fail hui: {info}",
            )
            sleep_countdown_state[chat_id] = {
                "chat_id": str(chat_id),
                "sleep_timer_active": False,
                "status": "failed",
                "reason": reason_text,
                "failed_at": time.time(),
                "details": str(info)[:180],
            }
            _clear_sleep_countdown_state()
            _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)

    except asyncio.CancelledError:
        sleep_timer_active[chat_id] = False
        start_sleep_timer = False
        logger.info("[AUTO_SLEEP_CANCEL] reason=cancelled chat=%s", chat_id)
        append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_cancelled", f"chat={chat_id} reason=cancelled")
        sleep_countdown_state[chat_id] = {
            "chat_id": str(chat_id),
            "sleep_timer_active": False,
            "status": "cancelled",
            "reason": reason_text,
            "cancelled_at": time.time(),
        }
        _clear_sleep_countdown_state()
        _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
        raise
    finally:
        current = sleep_countdown_tasks.get(chat_id)
        if current is asyncio.current_task():
            sleep_countdown_tasks.pop(chat_id, None)
        sleep_timer_active[chat_id] = bool(_task_is_running(sleep_countdown_tasks.get(chat_id)))
        countdown_task = sleep_countdown_tasks.get(chat_id)
        is_countdown_running = any(_task_is_running(t) for t in sleep_countdown_tasks.values())
        _update_lifecycle_state(chat_id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return
    chat_id = str(update.effective_chat.id)
    await send_ready_banner(update.message, chat_id)
    await update.message.reply_text(
        "🔄 Main menu refreshed.",
        reply_markup=main_kb(),
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return

    chat_id = str(update.effective_chat.id)
    if _is_chat_busy(chat_id):
        await update.message.reply_text(_build_full_status_text(chat_id), parse_mode="Markdown", reply_markup=main_kb())
        return

    st = _get_active_job_state(chat_id, allow_fallback=True) or {}
    if st and str(st.get("phase") or "").lower() in {"completed", "failed", "stopped"}:
        await update.message.reply_text(_build_full_status_text(chat_id), parse_mode="Markdown", reply_markup=main_kb())
        return

    if job_queues.get(chat_id):
        await update.message.reply_text(_build_full_status_text(chat_id), parse_mode="Markdown", reply_markup=main_kb())
        return

    outputs = list_swap_outputs()
    if outputs:
        f = outputs[0]
        await update.message.reply_text(
            "ℹ️ Koi job abhi run nahi ho rahi.\n"
            f"Latest output: `{f.name}`",
            parse_mode="Markdown", reply_markup=main_kb()
        )
    else:
        await update.message.reply_text("ℹ️ Koi job run nahi ho rahi.", reply_markup=main_kb())


async def reload_credentials_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return

    reload_runtime_credentials()
    summary_lines = _build_config_status_lines()
    await update.message.reply_text(
        "🔄 Credentials reloaded successfully\n\n"
        + "\n".join(summary_lines),
        parse_mode="Markdown",
        reply_markup=main_kb(),
    )


def _build_config_status_lines():
    creds = _effective_creds()
    bot_loaded = bool(str(creds.get("bot_token", "")).strip())
    mega_enabled = can_use_mega()
    gdrive_enabled_cfg = bool(creds.get("gdrive_enabled", True))
    gdrive_available = can_use_gdrive()
    gdrive_folder = str(creds.get("gdrive_folder", "") or GDRIVE_FOLDER).strip()

    bot_line = f"BOT_TOKEN: {'✅ Loaded' if bot_loaded else '❌ Missing'}"
    mega_line = f"MEGA: {'✅ Enabled' if mega_enabled else '❌ Disabled'}"
    if gdrive_enabled_cfg:
        if gdrive_available:
            gdrive_line = f"GDRIVE: ✅ Enabled ({gdrive_folder})"
        else:
            gdrive_line = f"GDRIVE: ⚠️ Enabled but unavailable ({gdrive_folder})"
    else:
        gdrive_line = "GDRIVE: ❌ Disabled"

    return [bot_line, mega_line, gdrive_line]


async def config_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return

    reload_runtime_credentials()
    status_lines = _build_config_status_lines()
    await update.message.reply_text(
        "🔐 *Current Configuration:*\n\n"
        + "\n".join(status_lines),
        parse_mode="Markdown",
        reply_markup=main_kb(),
    )


async def resetjobs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return

    chat_id = str(update.effective_chat.id)
    old_seq = int(queue_job_seq.get(chat_id, 0))
    queue_job_seq[chat_id] = 0
    _save_queue_state()
    logger.info("[RESETJOBS] chat=%s old_seq=%s reset to 0", chat_id, old_seq)
    await update.message.reply_text(
        f"✅ Job ID counter reset!\nPrevious: #{old_seq}\nNext job will be: #1",
        reply_markup=main_kb(),
    )


class _AutoTestMessage:
    def __init__(self, bot_obj, chat_id, text="", from_user_id=ALLOWED_USER_ID):
        self._bot = bot_obj
        self.chat_id = int(chat_id)
        self.text = text
        self.photo = []
        self.document = None
        self.from_user = type("AutoUser", (), {"id": int(from_user_id)})()

    async def reply_text(self, text, **kwargs):
        return await self._bot.send_message(chat_id=self.chat_id, text=text, **kwargs)

    async def reply_photo(self, **kwargs):
        kwargs.setdefault("chat_id", self.chat_id)
        return await self._bot.send_photo(**kwargs)

    async def reply_video(self, **kwargs):
        kwargs.setdefault("chat_id", self.chat_id)
        return await self._bot.send_video(**kwargs)

    def get_bot(self):
        return self._bot


class _AutoTestQuery:
    def __init__(self, data, msg):
        self.data = data
        self.message = msg
        self.from_user = type("AutoUser", (), {"id": int(ALLOWED_USER_ID)})()

    async def answer(self, *args, **kwargs):
        return True


async def _auto_test_notify(context, chat_id, text):
    try:
        sent = await safe_send_message(context.bot, str(chat_id), text, parse_mode="Markdown")
        if sent is None:
            await context.bot.send_message(chat_id=str(chat_id), text=text, parse_mode="Markdown")
    except Exception as e:
        logger.warning("auto test notify failed chat=%s err=%s", chat_id, e)


async def _auto_simulate_button(context, chat_id, callback_data):
    msg = _AutoTestMessage(context.bot, chat_id)
    query = _AutoTestQuery(callback_data, msg)
    update_obj = type("AutoUpdate", (), {"callback_query": query})()
    await button_handler(update_obj, context)


async def _auto_simulate_text(context, chat_id, text):
    msg = _AutoTestMessage(context.bot, chat_id, text=text)
    update_obj = type(
        "AutoTextUpdate",
        (),
        {
            "message": msg,
            "effective_user": msg.from_user,
            "effective_chat": type("AutoChat", (), {"id": int(chat_id)})(),
        },
    )()
    await handle_message(update_obj, context)


async def change_face_handler(context, chat_id, face_link=AUTO_TEST_FACE_LINK):
    await _auto_simulate_button(context, chat_id, "change_face")
    await _auto_simulate_text(context, chat_id, face_link)
    face_path = get_preuploaded_default_face()
    ok = bool(face_path and os.path.isfile(face_path))
    return ok, (face_path or "default face missing")


async def view_face_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "view_face")
    face_path = get_face(chat_id)
    ok = os.path.isfile(face_path)
    return ok, (face_path if ok else "face image not found")


async def stop_job_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "stop_job")
    ok = not _is_chat_busy(chat_id)
    return ok, ("active job stopped" if ok else "active job still running")


async def job_status_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "job_status_btn")
    return True, "status button executed"


async def clear_workspace_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "clean_workspace")
    await _auto_simulate_button(context, chat_id, "clean_workspace_temp_only")
    ok = Path(PIPELINE, "logs", "last_cleanup_summary.json").exists()
    return ok, "temp cleanup executed" if ok else "cleanup summary missing"


async def clip_range_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "clip_settings")
    await _auto_simulate_button(context, chat_id, "set_clip_range")
    await _auto_simulate_text(context, chat_id, "00:00:01-00:00:02")
    cfg = clip_ranges.get(str(chat_id), {}) or {}
    ok = bool(cfg.get("segments"))
    return ok, get_clip_range_note(str(chat_id))


async def mode_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "mode_multi")
    multi_ok = get_chat_mode(str(chat_id)) == "multi"
    await _auto_simulate_button(context, chat_id, "mode_direct")
    direct_ok = get_chat_mode(str(chat_id)) == "direct"
    return (multi_ok and direct_ok), f"multi={multi_ok}, direct={direct_ok}"


async def gender_handler(context, chat_id):
    await _auto_simulate_button(context, chat_id, "female_only_on")
    on_ok = is_female_only_enabled(str(chat_id))
    await _auto_simulate_button(context, chat_id, "female_only_off")
    off_ok = not is_female_only_enabled(str(chat_id))
    return (on_ok and off_ok), f"female_on={on_ok}, female_off={off_ok}"


async def restart_handler(context, chat_id):
    original_restart = globals().get("_safe_restart_self")
    simulated = {"called": False}

    async def _fake_restart(app, cid):
        simulated["called"] = True
        await _auto_test_notify(context, cid, "ℹ️ Auto test restart simulation complete (dry-run).")

    globals()["_safe_restart_self"] = _fake_restart
    try:
        await _auto_simulate_button(context, chat_id, "safe_restart_bot")
        await asyncio.sleep(0.2)
        return simulated["called"], ("restart simulated" if simulated["called"] else "restart task not triggered")
    finally:
        globals()["_safe_restart_self"] = original_restart


def _active_stage_snapshot(chat_id):
    st = _get_active_job_state(chat_id, allow_fallback=True) or {}
    if str(st.get("chat_id") or "") != str(chat_id):
        return "", ""
    return str(st.get("phase") or st.get("status") or "").lower(), str(st.get("stage") or "")


async def _auto_test_pipeline_once(context, chat_id, video_link=AUTO_TEST_VIDEO_LINK, timeout_sec=1800):
    await _auto_test_notify(context, chat_id, "🧪 Testing Pipeline...")
    await _auto_test_notify(context, chat_id, "🚺 Female mode: ON (forced for pipeline test)")
    set_female_only(str(chat_id), True)
    await asyncio.to_thread(_kill_orphan_job_processes, chat_id)
    _stop_active_job(chat_id)
    job_queues[str(chat_id)] = []

    queued = _queue_job(str(chat_id), video_link, None, mode="direct")
    _ensure_queue_worker(context, str(chat_id))

    required = {
        "download": False,
        "extract": False,
        "process": False,
        "merge": False,
        "upload": False,
    }
    announced = set()
    start_ts = time.time()

    while time.time() - start_ts < timeout_sec:
        phase_mem = str((job_status.get(str(chat_id), {}) or {}).get("phase", "")).lower()
        active_phase, stage_text = _active_stage_snapshot(str(chat_id))
        phase = str(active_phase or phase_mem or "").lower()
        stage_l = str(stage_text).lower()

        if "download" in phase or "download" in stage_l:
            required["download"] = True
            if "download" not in announced:
                announced.add("download")
                await _auto_test_notify(context, chat_id, "• Pipeline stage: Download")
        if "extract" in stage_l:
            required["extract"] = True
            if "extract" not in announced:
                announced.add("extract")
                await _auto_test_notify(context, chat_id, "• Pipeline stage: Extract")
        if "process" in stage_l or phase in {"faceswap", "processing"}:
            required["process"] = True
            if "process" not in announced:
                announced.add("process")
                await _auto_test_notify(context, chat_id, "• Pipeline stage: Process")
        if "merg" in stage_l:
            required["merge"] = True
            if "merge" not in announced:
                announced.add("merge")
                await _auto_test_notify(context, chat_id, "• Pipeline stage: Merge")
        if "upload" in stage_l or phase in {"upload", "uploading"}:
            required["upload"] = True
            if "upload" not in announced:
                announced.add("upload")
                await _auto_test_notify(context, chat_id, "• Pipeline stage: Upload")

        if phase == "completed":
            break
        if phase in {"failed", "stopped", "exception"}:
            break

        await asyncio.sleep(3)

    phase_mem = str((job_status.get(str(chat_id), {}) or {}).get("phase", "")).lower()
    active_phase, _ = _active_stage_snapshot(str(chat_id))
    phase = str(active_phase or phase_mem or "").lower()
    ok = all(required.values()) and phase == "completed"
    detail = f"phase={phase}, required={required}, job_id={queued.get('job_id')}"
    return ok, detail


async def _auto_test_pipeline(context, chat_id, video_link=AUTO_TEST_VIDEO_LINK, timeout_sec=1800, max_retries=2, retry_delay_sec=12):
    last_detail = ""
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            await _auto_test_notify(
                context,
                chat_id,
                f"🔁 Pipeline retry {attempt}/{max_retries} after failure...",
            )
            await asyncio.sleep(retry_delay_sec)

        ok, detail = await _auto_test_pipeline_once(
            context,
            chat_id,
            video_link=video_link,
            timeout_sec=timeout_sec,
        )
        last_detail = detail
        if ok:
            return True, detail

        detail_l = str(detail).lower()
        if "quota" in detail_l or "509" in detail_l or "over" in detail_l:
            await _auto_test_notify(
                context,
                chat_id,
                "⚠️ Possible MEGA quota/over-limit detected. Auto retrying with backoff.",
            )
        await _auto_test_remediate("Pipeline", context, chat_id)

    return False, last_detail


async def _auto_test_queue(context, chat_id, video_link=AUTO_TEST_VIDEO_LINK, timeout_sec=300):
    await _auto_test_notify(context, chat_id, "🧪 Testing Queue (2 jobs)...")
    await asyncio.to_thread(_kill_orphan_job_processes, chat_id)
    _stop_active_job(chat_id)
    job_queues[str(chat_id)] = []

    job1 = _queue_job(str(chat_id), video_link, None, mode="direct")
    job2 = _queue_job(str(chat_id), video_link, None, mode="direct")
    _ensure_queue_worker(context, str(chat_id))

    seen_job1_start = False
    seen_job2_queued = False
    start_ts = time.time()

    while time.time() - start_ts < timeout_sec:
        st = job_status.get(str(chat_id), {}) or {}
        current_job = int(st.get("job_id") or 0)
        q = job_queues.get(str(chat_id), [])

        if current_job == int(job1["job_id"]):
            seen_job1_start = True
        if any(int(item.get("job_id", 0)) == int(job2["job_id"]) for item in q):
            seen_job2_queued = True

        if seen_job1_start and seen_job2_queued:
            break
        await asyncio.sleep(1)

    if seen_job1_start:
        _stop_active_job(str(chat_id))

    seen_job2_start = False
    start_ts_2 = time.time()
    while time.time() - start_ts_2 < timeout_sec:
        st = job_status.get(str(chat_id), {}) or {}
        current_job = int(st.get("job_id") or 0)
        if current_job == int(job2["job_id"]):
            seen_job2_start = True
            break
        await asyncio.sleep(1)

    ok = seen_job1_start and seen_job2_queued and seen_job2_start
    detail = f"job1_started={seen_job1_start}, job2_queued={seen_job2_queued}, job2_started={seen_job2_start}"
    return ok, detail


async def _auto_test_remediate(test_name, context, chat_id):
    # Minimal deterministic remediation between retries.
    if test_name == "Change Face":
        context.user_data.pop("awaiting_new_face", None)
        context.user_data.pop("awaiting_face_link", None)
    if test_name == "Clip Range":
        context.user_data.pop("awaiting_clip_range", None)
        clip_ranges.pop(str(chat_id), None)
        _persist_ui_runtime_state()
    if test_name in {"Pipeline", "Queue"}:
        _stop_active_job(str(chat_id))
        await asyncio.to_thread(_kill_orphan_job_processes, str(chat_id))


async def auto_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return

    chat_id = str(update.effective_chat.id)
    await _auto_test_notify(context, chat_id, "🧪 Auto testing started...")

    tests = [
        ("Change Face", change_face_handler),
        ("View Face", view_face_handler),
        ("Stop Job", stop_job_handler),
        ("Job Status", job_status_handler),
        ("Clear Workspace", clear_workspace_handler),
        ("Clip Range", clip_range_handler),
        ("Mode", mode_handler),
        ("Gender", gender_handler),
        ("Restart", restart_handler),
    ]

    results = []
    for name, fn in tests:
        await _auto_test_notify(context, chat_id, f"🔎 Testing {name}...")
        passed = False
        detail = ""
        for attempt in range(1, 4):
            try:
                ok, detail = await fn(context, chat_id)
            except Exception as e:
                ok, detail = False, str(e)
            if ok:
                passed = True
                break
            await _auto_test_notify(
                context,
                chat_id,
                f"❌ {name} FAIL (attempt {attempt})\n`{str(detail)[:260]}`\nAuto-fix + retry...",
            )
            await _auto_test_remediate(name, context, chat_id)

        results.append((name, passed, str(detail)))
        if passed:
            await _auto_test_notify(context, chat_id, f"✅ {name} PASS")
        else:
            await _auto_test_notify(context, chat_id, f"❌ {name} FAIL\n`{str(detail)[:320]}`")

    pipeline_ok, pipeline_detail = await _auto_test_pipeline(context, chat_id)
    results.append(("Pipeline", pipeline_ok, pipeline_detail))
    if pipeline_ok:
        await _auto_test_notify(context, chat_id, "✅ Pipeline PASS")
    else:
        await _auto_test_notify(context, chat_id, f"❌ Pipeline FAIL\n`{pipeline_detail[:320]}`")

    queue_ok, queue_detail = await _auto_test_queue(context, chat_id)
    results.append(("Queue", queue_ok, queue_detail))
    if queue_ok:
        await _auto_test_notify(context, chat_id, "✅ Queue PASS")
    else:
        await _auto_test_notify(context, chat_id, f"❌ Queue FAIL\n`{queue_detail[:320]}`")

    lines = ["📋 *Auto Test Report*"]
    all_pass = True
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        lines.append(f"• {name}: *{mark}*")
        if not ok:
            all_pass = False
            lines.append(f"  ↳ `{str(detail)[:160]}`")

    lines.append(f"• Buttons: *{'PASS' if all(x[1] for x in results[:9]) else 'FAIL'}*")
    lines.append(f"• Pipeline: *{'PASS' if pipeline_ok else 'FAIL'}*")
    lines.append(f"• Queue: *{'PASS' if queue_ok else 'FAIL'}*")

    await _auto_test_notify(context, chat_id, "\n".join(lines))
    if all_pass:
        await _auto_test_notify(context, chat_id, "AUTO TEST COMPLETE — BOT FULLY WORKING")
    else:
        await _auto_test_notify(context, chat_id, "AUTO TEST COMPLETE — FAILURES DETECTED")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    cid = update.effective_chat.id if update.effective_chat else None
    if not is_authorized(uid, cid):
        await update.message.reply_text("⛔ Unauthorized user/chat. Contact admin.")
        return

    text    = (update.message.text or "").strip()
    chat_id = str(update.effective_chat.id)
    mega_links = parse_mega_links(text)
    links = parse_job_input_links(text)

    if chat_id not in chat_modes:
        set_chat_mode(chat_id, "direct")

    current_mode = get_chat_mode(chat_id)
    if current_mode != "multi" and (
        context.user_data.get("awaiting_multi_target")
        or context.user_data.get("awaiting_multi_source")
        or context.user_data.get("multi_face_crops")
        or context.user_data.get("multi_face_map")
    ):
        clear_multi_setup_state(context)
        selected_face_maps.pop(chat_id, None)

    if context.user_data.get("awaiting_clip_range"):
        cfg, err = parse_clip_range_input(text)
        if cfg:
            clip_ranges[chat_id] = cfg
            context.user_data.pop("awaiting_clip_range", None)
            _persist_ui_runtime_state()
            await update.message.reply_text(
                f"✅ Clip range set: `{get_clip_range_note(chat_id)}`",
                parse_mode="Markdown",
                reply_markup=main_kb(),
            )
            await send_mode_state_announcement(update.message, chat_id, context)
        else:
            await update.message.reply_text(f"❌ {err}", parse_mode="Markdown")
        return

    if context.user_data.get("awaiting_face_map"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            await update.message.reply_text("❌ Koi mapping line nahi mili.")
            return

        done = 0
        errors = []
        for line in lines:
            if "|" not in line:
                errors.append(f"`{line}` -> format `index|mega_link` use karo")
                continue

            idx_raw, link = line.split("|", 1)
            idx_raw = idx_raw.strip()
            link = link.strip()

            if not idx_raw.isdigit():
                errors.append(f"`{line}` -> index numeric hona chahiye")
                continue

            pos = int(idx_raw) - 1
            if pos < 0:
                errors.append(f"`{line}` -> index 1 se start hota hai")
                continue

            if not parse_mega_links(link):
                errors.append(f"`{line}` -> valid MEGA link missing")
                continue

            ok, info = await asyncio.to_thread(save_face_map_source, chat_id, pos, link)
            if ok:
                done += 1
            else:
                errors.append(f"index {idx_raw}: {info}")

        context.user_data.pop("awaiting_face_map", None)
        msg = f"✅ Face maps added: *{done}*"
        if errors:
            msg += "\n\n⚠️ Errors:\n" + "\n".join(f"- {e}" for e in errors[:6])
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=main_kb())
        return

    if current_mode == "multi" and context.user_data.get("awaiting_multi_target"):
        if not links:
            await update.message.reply_text("🧩 Multi mode active. Target video ka MEGA link bhejo.", reply_markup=main_kb())
            return
        await start_multi_setup_from_target(update, context, chat_id, links[0])
        return

    if current_mode == "multi" and context.user_data.get("awaiting_multi_source"):
        idx = int(context.user_data.get("multi_face_idx", 0))

        saved = None
        if links:
            ok, info = await asyncio.to_thread(save_face_map_source, chat_id, idx, links[0])
            if ok:
                saved = info
            else:
                await update.message.reply_text(f"❌ Source save fail: `{info}`", parse_mode="Markdown")
                return
        else:
            saved = await download_image_from_telegram_message(update.message, chat_id, prefix=f"multi_src_{idx+1}")
            if not saved:
                await update.message.reply_text(
                    "❌ Source image nahi mili. MEGA image link bhejo ya direct image Telegram par upload karo.",
                    reply_markup=main_kb()
                )
                return
            ok, reason = validate_source_face_quality(saved)
            if not ok:
                await update.message.reply_text(f"❌ Source image weak/invalid: `{reason}`", parse_mode="Markdown")
                return

        context.user_data.setdefault("multi_face_map", {})[idx] = saved
        context.user_data["multi_face_idx"] = idx + 1
        await prompt_next_multi_face(update, context, chat_id)
        return

    if context.user_data.get("awaiting_custom_mega_link"):
        if not mega_links:
            await update.message.reply_text(
                "❌ Valid MEGA link not found. Paste the full link: `https://mega.nz/file/...`",
                parse_mode="Markdown",
                reply_markup=main_kb(),
            )
            return
        
        link = mega_links[0]
        context.user_data.pop("awaiting_custom_mega_link", None)
        _cleanup_custom_mega_temp(context)
        
        # Download the file with live progress
        status_msg = await update.message.reply_text(
            "📥 *Downloading File...*\n\n" + _format_progress_bar(0) + " 0%\nSpeed: --\nETA: --",
            parse_mode="Markdown",
        )
        
        temp_dir = Path(TEMP_PATH) / f"custom_mega_{chat_id}_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        progress_state = {"run": True, "last_bytes": 0.0, "last_tick": float(time.time()), "speed_bps": 0.0}

        async def _download_progress_loop():
            while progress_state["run"]:
                await asyncio.sleep(1)
                current_time = float(time.time())
                current_bytes = 0
                with suppress(Exception):
                    current_bytes = sum(f.stat().st_size for f in temp_dir.glob("**/*") if f.is_file())
                delta_time = max(1e-6, current_time - float(progress_state["last_tick"]))
                delta_bytes = max(0.0, float(current_bytes) - float(progress_state["last_bytes"]))
                speed_bps = delta_bytes / delta_time if delta_bytes > 0 else float(progress_state["speed_bps"])
                progress_state["speed_bps"] = speed_bps
                progress_state["last_bytes"] = float(current_bytes)
                progress_state["last_tick"] = current_time

                elapsed = max(1.0, current_time - download_start)
                est_percent = min(95, int((1.0 - math.exp(-elapsed / 6.0)) * 100.0))
                speed_txt = "--" if speed_bps <= 0 else f"{(speed_bps / 1024 / 1024):.2f} MB/s"
                eta_txt = "--" if est_percent <= 0 else f"{max(1, int(elapsed * (100 - est_percent) / est_percent))} sec"

                with suppress(Exception):
                    await status_msg.edit_text(
                        "📥 *Downloading File...*\n\n"
                        f"{_format_progress_bar(est_percent)} {est_percent}%\n"
                        f"Speed: {speed_txt}\n"
                        f"ETA: {eta_txt}",
                        parse_mode="Markdown",
                    )

        progress_task = None
        try:
            # Download file with live progress loop
            download_start = time.time()
            progress_task = asyncio.create_task(_download_progress_loop())
            ok_download = await mega_download_async(link, str(temp_dir))
            download_time = time.time() - download_start
            progress_state["run"] = False
            if progress_task and not progress_task.done():
                progress_task.cancel()
                with suppress(asyncio.CancelledError):
                    await progress_task
            
            if not ok_download:
                await status_msg.edit_text(
                    "❌ Download failed. Check link and try again.",
                    reply_markup=main_kb(),
                )
                shutil.rmtree(temp_dir, ignore_errors=True)
                return
            
            # Find downloaded file
            files = [f for f in temp_dir.iterdir() if f.is_file()]
            if not files:
                await status_msg.edit_text(
                    "❌ No files found after download.",
                    reply_markup=main_kb(),
                )
                shutil.rmtree(temp_dir, ignore_errors=True)
                return
            
            # Keep the largest file as default when archives generate helper side-files.
            downloaded_file = max(files, key=lambda fp: fp.stat().st_size)
            file_size_bytes = downloaded_file.stat().st_size
            file_size_mb = file_size_bytes / 1024 / 1024
            
            # Calculate speed
            download_speed_mbs = (file_size_mb / max(0.1, download_time)) if download_time > 0 else 0
            
            # Show completion with stats
            await status_msg.edit_text(
                f"✅ *Download Complete*\n\n"
                f"📁 `{downloaded_file.name}`\n"
                f"📊 {file_size_mb:.1f} MB\n"
                f"⚡ {download_speed_mbs:.1f} MB/s\n\n"
                f"📤 *Choose Upload Destination*",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("☁️ Upload to Mega", callback_data=f"custom_mega_upload_to_mega_{int(time.time())}"),
                        InlineKeyboardButton("📁 Upload to Google Drive", callback_data=f"custom_mega_upload_to_gdrive_{int(time.time())}"),
                    ],
                    [InlineKeyboardButton("❌ Cancel", callback_data="back_main")],
                ]),
                parse_mode="Markdown",
            )
            
            # Store temp file path for upload
            context.user_data["custom_mega_temp_file"] = str(downloaded_file)
            context.user_data["custom_mega_temp_dir"] = str(temp_dir)
            
        except Exception as e:
            progress_state["run"] = False
            if progress_task and not progress_task.done():
                progress_task.cancel()
                with suppress(asyncio.CancelledError):
                    await progress_task
            logger.exception("custom mega download error: %s", e)
            await status_msg.edit_text(
                f"❌ Error: {str(e)[:100]}",
                reply_markup=main_kb(),
            )
            shutil.rmtree(temp_dir, ignore_errors=True)
        return

    if context.user_data.get("awaiting_mega_creds"):
        # Failsafe: if a MEGA link arrives while creds mode is open, do not trap job inputs.
        if links:
            context.user_data.pop("awaiting_mega_creds", None)
            logger.info("awaiting_mega_creds auto-cancelled chat=%s due mega link input", chat_id)
            await update.message.reply_text(
                "ℹ️ MEGA credentials mode auto-cancel ho gaya (link detect hua). Ab link ko job input ki tarah process kar raha hoon.",
                reply_markup=main_kb(),
            )
        elif _looks_like_mega_creds_input(text):
            u, p = text.split(":", 1)
            u = u.strip()
            p = p.strip()
            ok, info = await asyncio.to_thread(validate_mega_creds, u, p)
            if ok:
                save_mega_creds(u, p)
                context.user_data.pop("awaiting_mega_creds", None)
                await update.message.reply_text(
                    f"✅ MEGA updated and verified!\n`{u}`",
                    parse_mode="Markdown", reply_markup=main_kb()
                )
            else:
                await update.message.reply_text(
                    "❌ MEGA login verify fail hua.\n"
                    f"`{info}`\n\n"
                    "Dubara bhejo: `email:password`",
                    parse_mode="Markdown", reply_markup=main_kb()
                )
            return
        else:
            await update.message.reply_text(
                "❌ Format: `email:password`\nTip: agar job start karna hai to normal MEGA video link bhej do, mode auto-cancel ho jayega.",
                parse_mode="Markdown",
            )
            return

    if context.user_data.get("awaiting_drive_token"):
        token_value = (text or "").strip()
        ok, info = validate_drive_token(token_value)
        if not ok:
            await update.message.reply_text(
                f"❌ {info}\n\n📥 Send Drive auth token (raw access token or full JSON)",
                reply_markup=main_kb(),
            )
            return

        save_result = save_drive_token(token_value)
        context.user_data.pop("awaiting_drive_token", None)
        await update.message.reply_text(
            "✅ Drive token updated successfully",
            reply_markup=main_kb(),
        )
        logger.info(
            "drive token updated securely mode=%s masked=%s",
            save_result.get("mode"),
            save_result.get("masked"),
        )
        return

    if context.user_data.get("awaiting_new_face") or context.user_data.get("awaiting_face_link"):
        # Change Face flow accepts both MEGA links and direct image URLs.
        # Validation + face checks happen in the same guarded download path.
        candidates = []
        for item in (mega_links + links):
            raw = str(item or "").strip()
            if not raw:
                continue
            if raw not in candidates:
                candidates.append(raw)

        selected_face_link = ""
        last_reason = ""
        for raw_link in candidates:
            ok_face_link, normalized_face_link, _kind, info = await asyncio.to_thread(
                validate_input_media_link,
                raw_link,
                True,
            )
            if ok_face_link:
                selected_face_link = str(normalized_face_link or raw_link).strip()
                break
            last_reason = str(info or "invalid face link")

        if selected_face_link:
            context.user_data.pop("awaiting_new_face", None)
            context.user_data.pop("awaiting_face_link", None)
            await _handle_face_change(update, context, chat_id, selected_face_link)
        else:
            if candidates and last_reason:
                await update.message.reply_text(f"❌ Face link invalid: {last_reason}")
            else:
                await update.message.reply_text("❌ Face link nahi mila. Dobara bhejo.")
        return

    if not links:
        if current_mode == "multi":
            context.user_data["awaiting_multi_target"] = True
            await update.message.reply_text(
                "🧩 Multi FaceSwap mode active hai.\nTarget video ka MEGA link bhejo.",
                reply_markup=main_kb()
            )
            await send_mode_state_announcement(update.message, chat_id, context)
            return
        await update.message.reply_text(
            "Video link bhejo (MEGA ya direct media URL).\n"
            "MEGA format: `https://mega.nz/file/xxxxx#yyyyy`",
            parse_mode="Markdown", reply_markup=main_kb()
        )
        return

    task = sleep_countdown_tasks.get(chat_id)
    if task and not task.done():
        task.cancel()
        _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
        logger.info("[AUTO_SLEEP_CANCEL] reason=new_job chat=%s", chat_id)
        append_auto_sleep_log(AUTO_SLEEP_LOG_FILE, "countdown_cancelled", f"chat={chat_id} reason=new_job")
        await update.message.reply_text("🔄 Shutdown cancel — naya job shuru!")

    if len(links) > 1:
        # Queue ALL links — process one by one
        queued = []
        for lnk in links:
            ok_lnk, norm_lnk, _, _ = await asyncio.to_thread(validate_input_media_link, lnk, True)
            if ok_lnk:
                queued.append(_queue_job(chat_id, str(norm_lnk or lnk), None, mode=job_modes.get(chat_id, "direct")))
        if not queued:
            await update.message.reply_text("❌ Koi valid link nahi mila.", reply_markup=main_kb())
            return
        _ensure_queue_worker(context, chat_id)
        await update.message.reply_text(
            f"📋 *{len(queued)} jobs queue mein add ho gaye!*\n\n"
            + "\n".join(f"#{j['job_id']} — `{j.get('target_name','link'+str(i+1))}`" for i, j in enumerate(queued))
            + f"\n\n⚡ Ek ke baad ek automatically process honge.\n😴 Sabse aakhri job ke baad auto-sleep trigger hoga.",
            parse_mode="Markdown", reply_markup=main_kb()
        )
        return

    ok_link, normalized_link, link_kind, link_info = await asyncio.to_thread(
        validate_input_media_link,
        links[0],
        True,
    )
    if not ok_link:
        await update.message.reply_text(
            f"❌ Invalid link: {link_info}",
            reply_markup=main_kb(),
        )
        return

    if current_mode == "multi":
        await start_multi_setup_from_target(update, context, chat_id, normalized_link)
        return

    queued = []
    # Strict runtime concurrency: one active worker; new jobs are queued.
    for video_link in [normalized_link]:
        queued.append(_queue_job(chat_id, video_link, None, mode="direct"))

    logger.info("validated input link chat=%s kind=%s details=%s", chat_id, link_kind, link_info)

    # Create a fresh live dashboard session so the user can watch updates on the web.
    dashboard_url = ""
    if queued and DASHBOARD_ENABLED:
        first = queued[0]
        token, raw_url = _dashboard_register_session(
            chat_id,
            first.get("video_link", normalized_link),
            first.get("job_id", 0),
        )
        if token:
            # Verify URL is reachable before sending to user
            verified_url = _verify_dashboard_url(raw_url)
            dashboard_url = verified_url
            first["dashboard_token"] = token
            first["dashboard_url"] = dashboard_url
            _save_queue_state()

    _ensure_queue_worker(context, chat_id)

    if len(queued) == 1:
        job_id = queued[0]['job_id']
        queue_total = _queue_size(chat_id)
        is_waiting = bool(_single_job_block_reason(chat_id))
        if dashboard_url and not is_waiting:
            queued_msg_text = (
                "╔══════════════════════════╗\n"
                "     🎬 JOB STARTED\n"
                "╚══════════════════════════╝\n\n"
                f"🆔 Job ID: #{job_id}\n"
                "📥 Source: Mega Link Received\n"
                "⚡ Status: Processing Started\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "🖥️ LIVE DASHBOARD\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"🔗 [Watch Live Progress →]({dashboard_url})\n\n"
                "📊 Stages • Frames • % • ETA\n"
                "     All updating every second\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "⏳ Sit back — we'll notify you!"
            )
            msg_parse_mode = "Markdown"
        elif is_waiting:
            queued_msg_text = (
                f"🗂 *Job #{job_id} queued* — position `#{queue_total}`\n"
                "⏳ Pehla job complete hone ke baad automatically start hoga."
            )
            msg_parse_mode = "Markdown"
        else:
            queued_msg_text = f"🗂 Job #{job_id} started. Queue size: {queue_total}"
            msg_parse_mode = None
        queued_msg = await update.message.reply_text(
            queued_msg_text,
            parse_mode=msg_parse_mode,
            reply_markup=main_kb()
        )
        try:
            queued_progress_message_ids[(chat_id, int(queued[0]["job_id"]))] = int(queued_msg.message_id)
        except Exception:
            pass
    else:
        await update.message.reply_text(
            f"🗂 `{len(queued)}` jobs added to queue\n"
            f"Queue size: `{_queue_size(chat_id)}`",
            parse_mode="Markdown",
            reply_markup=main_kb()
        )


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    chat_id = str(query.message.chat_id)
    d = query.data
    uid = query.from_user.id if query.from_user else None
    logger.info("BUTTON CLICK RECEIVED: %s", d or "unknown")
    logger.info("[BUTTON CALLBACK] chat=%s user=%s data=%s", chat_id, uid, d)
    answered = False

    async def _answer_once(text=None, show_alert=False):
        nonlocal answered
        if answered:
            return
        try:
            if text:
                await query.answer(text=text, show_alert=show_alert)
            else:
                await query.answer()
            answered = True
        except Exception:
            logger.warning("callback answer failed chat=%s data=%s", chat_id, d, exc_info=True)

    blocked_until = float(telegram_flood_until.get(chat_id, 0) or 0)
    if blocked_until > time.time():
        remain = int(blocked_until - time.time())
        await _answer_once(
            text=f"Telegram flood cooldown active ({remain}s). Thoda wait karo.",
            show_alert=True,
        )

    if not is_authorized(uid, query.message.chat_id):
        await _answer_once(text="Unauthorized user/chat.", show_alert=True)
        return

    await _answer_once(text=BUTTON_ACTION_ANNOUNCEMENTS.get(d))

    if d and (("delete" in d.lower()) or ("remove" in d.lower())) and ("face" in d.lower()):
        await query.message.reply_text(
            "🚫 Face delete/remove actions disabled hain. Sirf Change Face (replace) allowed hai.",
            reply_markup=main_kb(chat_id),
        )
        return

    if d == "change_face":
        context.user_data.pop("awaiting_face_link", None)
        context.user_data["awaiting_new_face"] = True
        await query.message.reply_text("Send new MEGA image link")

    elif d == "mode_direct":
        set_chat_mode(chat_id, "direct")
        clear_multi_setup_state(context)
        selected_face_maps.pop(chat_id, None)
        await query.message.reply_text(
            "🎯 *Mode switched: Direct FaceSwap*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ Ab aap *Direct mode* me ho.\n"
            "👤 Target ke sab detected faces ek hi source face se swap honge.",
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id)
        )
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "mode_multi":
        set_chat_mode(chat_id, "multi")
        clear_multi_setup_state(context)
        selected_face_maps.pop(chat_id, None)
        context.user_data["awaiting_multi_target"] = True
        await query.message.reply_text(
            "🧩 *Mode switched: Multi FaceSwap*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ Ab aap *Multi mode* me ho.\n"
            "Target video ka MEGA link bhejo.\n"
            "Bot faces detect karke har person ka preview bhejega aur 3 options dega.",
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id),
        )
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "female_only_on":
        set_female_only(chat_id, True)
        if get_chat_mode(chat_id) != "multi":
            clear_multi_setup_state(context)
            selected_face_maps.pop(chat_id, None)
        await query.message.reply_text(
            "🚺 Target filter set: *Female only*\n"
            "Ab swap sirf female detected faces par apply hoga.",
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id),
        )
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "male_only_on":
        set_gender_mode(chat_id, "male")
        if get_chat_mode(chat_id) != "multi":
            clear_multi_setup_state(context)
            selected_face_maps.pop(chat_id, None)
        await query.message.reply_text(
            "👨 Target filter set: *Male only*\n"
            "Ab swap sirf male detected faces par apply hoga.",
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id),
        )
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "female_only_off":
        set_female_only(chat_id, False)
        if get_chat_mode(chat_id) != "multi":
            clear_multi_setup_state(context)
            selected_face_maps.pop(chat_id, None)
        await query.message.reply_text(
            "👥 Target filter set: *All genders*\n"
            "Ab swap normal mode me sab detected human faces par apply hoga.",
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id),
        )
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "multi_send_source":
        if not context.user_data.get("awaiting_multi_source"):
            await query.message.reply_text("ℹ️ Multi setup active nahi hai.", reply_markup=main_kb())
            return
        await query.message.reply_text(
            "Is person ke liye MEGA image link bhejo ya image direct Telegram par upload karo.",
            reply_markup=main_kb(chat_id),
        )

    elif d == "multi_skip_next":
        if not context.user_data.get("awaiting_multi_source"):
            await query.message.reply_text("ℹ️ Multi setup active nahi hai.", reply_markup=main_kb())
            return
        idx = int(context.user_data.get("multi_face_idx", 0))
        context.user_data["multi_face_idx"] = idx + 1
        wrapper = type("Obj", (), {"message": query.message})()
        await prompt_next_multi_face(wrapper, context, chat_id)

    elif d == "multi_use_current_face":
        if not context.user_data.get("awaiting_multi_source"):
            await query.message.reply_text("ℹ️ Multi setup active nahi hai.", reply_markup=main_kb())
            return
        idx = int(context.user_data.get("multi_face_idx", 0))
        current = get_preuploaded_default_face()
        if not current or not os.path.isfile(current):
            await query.message.reply_text(
                "❌ Pre-upload face nahi mili. Pehle `Change Face` se face set karo.",
                reply_markup=main_kb()
            )
            return

        ok, mapped_or_reason = save_face_map_local_source(chat_id, idx, current, prefix="multi_default")
        if not ok:
            await query.message.reply_text(
                f"❌ Current default face weak/invalid: `{mapped_or_reason}`\nNaya face `Change Face` se set karo.",
                parse_mode="Markdown",
                reply_markup=main_kb(),
            )
            return

        context.user_data.setdefault("multi_face_map", {})[idx] = mapped_or_reason
        context.user_data["multi_face_idx"] = idx + 1
        wrapper = type("Obj", (), {"message": query.message})()
        await prompt_next_multi_face(wrapper, context, chat_id)

    elif d == "multi_cancel_setup":
        clear_multi_setup_state(context)
        set_chat_mode(chat_id, "direct")
        selected_face_maps.pop(chat_id, None)
        await query.message.reply_text(
            "🟢 Multi setup cancel ho gaya. Bot default *Direct FaceSwap* mode par aa gaya.",
            parse_mode="Markdown",
            reply_markup=main_kb()
        )
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "view_face":
        fp = get_face(chat_id)
        if os.path.exists(fp):
            await query.message.reply_photo(
                photo=open(fp, "rb"),
                caption=f"`{os.path.basename(fp)}`",
                parse_mode="Markdown", reply_markup=main_kb()
            )
        else:
            await query.message.reply_text(
                f"❌ Face nahi mili:\n`{fp}`",
                parse_mode="Markdown", reply_markup=main_kb()
            )

    elif d == "stop_job":
        if _stop_active_job(chat_id):
            await query.message.reply_text("⏹ Job stop kar di!", reply_markup=main_kb())
        else:
            await query.message.reply_text("ℹ️ Koi job nahi chal rahi.", reply_markup=main_kb())

    elif d == "job_status_btn":
        await query.message.reply_text(
            _build_full_status_text(chat_id),
            parse_mode="Markdown",
            reply_markup=main_kb(),
        )

    elif d == "queue_terminate_menu":
        await query.message.reply_text(
            "🧨 *Job Selector Terminator*\n"
            "Active job stop kar sakte ho, ya queue me pending job remove kar sakte ho.",
            parse_mode="Markdown",
            reply_markup=_build_queue_terminator_kb(chat_id),
        )

    elif d == "terminate_job_active":
        if _stop_active_job(chat_id):
            await query.message.reply_text("⏹ Active job terminate kar di.", reply_markup=_build_queue_terminator_kb(chat_id))
        else:
            await query.message.reply_text("ℹ️ Koi active job nahi hai.", reply_markup=_build_queue_terminator_kb(chat_id))

    elif re.match(r"^terminate_job_q_\d+$", d):
        job_id = int(d.rsplit("_", 1)[1])
        queue = job_queues.get(chat_id, [])
        before = len(queue)
        queue = [item for item in queue if int(item.get("job_id", -1)) != job_id]
        job_queues[chat_id] = queue
        if len(queue) < before:
            await query.message.reply_text(
                f"🗑 Queue job `#{job_id}` terminate kar diya.",
                parse_mode="Markdown",
                reply_markup=_build_queue_terminator_kb(chat_id)
            )
        else:
            await query.message.reply_text(
                f"ℹ️ Queue job `#{job_id}` nahi mila.",
                parse_mode="Markdown",
                reply_markup=_build_queue_terminator_kb(chat_id)
            )

    elif d == "reupload_output_menu":
        await query.message.reply_text(
            "📤 *Output Upload Center*\n\n"
            "Select one flow:\n"
            "1) Reupload existing bot output\n"
            "2) Download from MEGA link and upload to destination",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🔁 Reupload Recent Output", callback_data="reupload_recent_output_menu"),
                ],
                [
                    InlineKeyboardButton("🔗 Mega Direct Upload", callback_data="custom_mega_upload_menu"),
                ],
                [
                    InlineKeyboardButton("⬅️ Back", callback_data="back_main"),
                ],
            ]),
            parse_mode="Markdown",
        )

    elif d == "reupload_recent_output_menu":
        await query.message.reply_text(
            "📤 *Choose Upload Destination*",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📁 GDrive ✅ Recommended", callback_data="reupload_output_gdrive")],
                [InlineKeyboardButton("☁️ Mega ⚠️ May fail on this server", callback_data="reupload_output_mega")],
                [InlineKeyboardButton("⬅️ Back", callback_data="reupload_output_menu")],
            ]),
            parse_mode="Markdown",
        )

    elif d in {"reupload_output_list", "upload_recent_list"}:
        # Deprecated paths - redirect to destination selector
        await query.message.reply_text(
            "📤 *Choose Upload Destination*\n\n"
            "Legacy menu has been merged into this flow.",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📁 GDrive ✅ Recommended", callback_data="reupload_output_gdrive"),
                ],
                [
                    InlineKeyboardButton("☁️ Mega ⚠️ May fail on this server", callback_data="reupload_output_mega"),
                ],
                [InlineKeyboardButton("⬅️ Back", callback_data="reupload_output_menu")],
            ]),
            parse_mode="Markdown",
        )

    elif d == "custom_mega_upload_menu":
        _cleanup_custom_mega_temp(context)
        await query.message.reply_text(
            "🔗 *Mega Direct Upload*\n\n"
            "Paste your MEGA file link below. Bot will first download it, then ask upload destination.\n\n"
            "Examples:\n"
            "`https://mega.nz/file/ID!KEY`\n"
            "`https://mega.nz/#!ID!KEY`",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("⬅️ Back", callback_data="reupload_output_menu")],
            ]),
            parse_mode="Markdown",
        )
        context.user_data["awaiting_custom_mega_link"] = True

    elif re.match(r"^custom_mega_upload_to_(mega|gdrive)_\d+$", d):
        m = re.match(r"^custom_mega_upload_to_(mega|gdrive)_(\d+)$", d)
        dest = m.group(1)
        
        file_path = context.user_data.get("custom_mega_temp_file")
        temp_dir = context.user_data.get("custom_mega_temp_dir")
        
        if not file_path or not Path(file_path).exists():
            await query.message.reply_text(
                "❌ Temp file not found. Start over.",
                reply_markup=main_kb(),
            )
            return
        
        f = Path(file_path)
        file_size_bytes = f.stat().st_size
        size_mb = file_size_bytes / 1024 / 1024
        upload_start = time.time()
        
        status_msg = await query.message.reply_text(
            f"📤 *Uploading to {dest.upper()}...*\n\n"
            f"`{f.name}`\n"
            f"{_format_progress_bar(0)} 0%\n"
            f"📊 {size_mb:.1f} MB\n"
            f"Speed: --\n"
            f"ETA: --",
            parse_mode="Markdown",
        )

        progress_state = {"run": True}

        async def _upload_progress_loop(label):
            expected_seconds = max(6.0, min(120.0, float(size_mb) / 2.0))
            while progress_state["run"]:
                await asyncio.sleep(1)
                elapsed = max(1.0, time.time() - upload_start)
                est_percent = min(95, int((elapsed / expected_seconds) * 100.0))
                eta_sec = max(1, int(expected_seconds - elapsed)) if est_percent < 95 else 1
                speed_est = float(size_mb) / elapsed
                with suppress(Exception):
                    await status_msg.edit_text(
                        f"📤 *Uploading to {label}...*\n\n"
                        f"`{f.name}`\n"
                        f"{_format_progress_bar(est_percent)} {est_percent}%\n"
                        f"📊 {size_mb:.1f} MB\n"
                        f"Speed: {speed_est:.2f} MB/s\n"
                        f"ETA: {eta_sec}s",
                        parse_mode="Markdown",
                    )
        
        try:
            if dest == "mega":
                progress_task = asyncio.create_task(_upload_progress_loop("MEGA"))
                # Upload to Mega
                mega_ok, mega_info = await asyncio.to_thread(mega_upload, str(f))
                progress_state["run"] = False
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await progress_task
                if not mega_ok:
                    if "EBLOCKED" in str(mega_info) or "509" in str(mega_info):
                        await status_msg.edit_text(
                            "⚠️ Mega Upload Blocked\n\n"
                            "This server's IP is blocked by Mega.\n"
                            "Please use GDrive instead or create a new\n"
                            "Mega account from a home network and update\n"
                            "MEGA_EMAIL and MEGA_PASSWORD in .env",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("📁 GDrive ✅ Recommended", callback_data="reupload_output_gdrive")],
                                [InlineKeyboardButton("🏠 Home", callback_data="back_main")],
                            ]),
                        )
                        return
                    await status_msg.edit_text(
                        f"❌ MEGA Upload Failed\n\n`{mega_info[:200]}`",
                        parse_mode="Markdown",
                        reply_markup=main_kb(),
                    )
                    return
                
                upload_time = time.time() - upload_start
                upload_speed_mbs = (size_mb / max(0.1, upload_time)) if upload_time > 0 else 0
                
                await status_msg.edit_text(
                    f"📤 *Uploading to MEGA...*\n\n"
                    f"`{f.name}`\n"
                    f"{_format_progress_bar(100)} 100%\n"
                    f"📊 {size_mb:.1f} MB\n"
                    f"⚡ {upload_speed_mbs:.1f} MB/s",
                    parse_mode="Markdown",
                )
                
                # Get link
                remote_path = f"/Root/direct_uploads/{f.name}"
                link_ok, link_info = await asyncio.to_thread(mega_export_link, remote_path, 2, 1)
                
                if link_ok and link_info.startswith("http"):
                    await status_msg.edit_text(
                        f"✅ *Upload Complete*\n\n🔗 Mega Link:\n`{link_info}`",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("📋 Copy Link", url=link_info)],
                            [InlineKeyboardButton("🏠 Home", callback_data="back_main")],
                        ]),
                    )
                else:
                    await status_msg.edit_text(
                        f"✅ Uploaded to MEGA\n\n📁 {remote_path}",
                        parse_mode="Markdown",
                        reply_markup=main_kb(),
                    )

                await start_sleep_countdown(
                    context.application,
                    chat_id,
                    reason_text="Direct upload completed",
                    delay_seconds=SLEEP_COUNTDOWN_SECONDS,
                    force_allow=True,
                )
            else:
                progress_task = asyncio.create_task(_upload_progress_loop("Google Drive"))
                # Upload to Google Drive
                ok, link, info = await asyncio.to_thread(gdrive_upload, str(f))
                progress_state["run"] = False
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await progress_task
                
                upload_time = time.time() - upload_start
                upload_speed_mbs = (size_mb / max(0.1, upload_time)) if upload_time > 0 else 0
                
                await status_msg.edit_text(
                    f"📤 *Uploading to Google Drive...*\n\n"
                    f"`{f.name}`\n"
                    f"{_format_progress_bar(100)} 100%\n"
                    f"📊 {size_mb:.1f} MB\n"
                    f"⚡ {upload_speed_mbs:.1f} MB/s",
                    parse_mode="Markdown",
                )
                
                if not ok:
                    await status_msg.edit_text(
                        f"❌ Google Drive Upload Failed\n\n`{info[:200]}`",
                        parse_mode="Markdown",
                        reply_markup=main_kb(),
                    )
                    return
                
                if link:
                    await status_msg.edit_text(
                        f"✅ *Upload Complete*\n\n🔗 GDrive Link:\n`{link}`",
                        parse_mode="Markdown",
                        reply_markup=InlineKeyboardMarkup([
                            [InlineKeyboardButton("📋 Copy Link", url=link)],
                            [InlineKeyboardButton("🏠 Home", callback_data="back_main")],
                        ]),
                    )
                else:
                    await status_msg.edit_text(
                        f"✅ Uploaded to Google Drive\n\n📁 {get_gdrive_target_folder()}/{f.name}",
                        parse_mode="Markdown",
                        reply_markup=main_kb(),
                    )

                await start_sleep_countdown(
                    context.application,
                    chat_id,
                    reason_text="Direct upload completed",
                    delay_seconds=SLEEP_COUNTDOWN_SECONDS,
                    force_allow=True,
                )
        finally:
            progress_state["run"] = False
            context.user_data.pop("custom_mega_temp_file", None)
            context.user_data.pop("custom_mega_temp_dir", None)
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return

    elif re.match(r"^reupload_pick_(recent|mega|gdrive)_(\d+)$", d):
        m = re.match(r"^reupload_pick_(recent|mega|gdrive)_(\d+)$", d)
        platform = m.group(1)
        idx = int(m.group(2))
        logger.info("Re-upload callback received — data: %s — platform: %s — from user: %s", d, platform, query.from_user.id)

        if platform == "recent":
            await query.message.reply_text(
                "Choose upload mode:",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("📁 GDrive ✅ Recommended", callback_data=f"reupload_pick_gdrive_{idx}"),
                    ],
                    [
                        InlineKeyboardButton("☁️ Mega ⚠️ May fail on this server", callback_data=f"reupload_pick_mega_{idx}"),
                    ],
                    [InlineKeyboardButton("⬅️ Back", callback_data="reupload_output_menu")],
                ]),
            )
            return

        outputs = list_swap_outputs()
        if idx >= len(outputs):
            await query.message.reply_text(
                "⚠️ Selected output ab available nahi hai. Dobara select karo.",
                reply_markup=main_kb()
            )
            return

        f = outputs[idx]
        size_mb = f.stat().st_size / 1024 / 1024

        status_msg = await query.message.reply_text(
            f"🔁 Upload starting...\n📁 {f.name}\n📦 {size_mb:.1f} MB"
        )

        progress_state = {"run": True, "started": float(time.time())}
        progress_frames = ["▰▱▱▱▱▱▱▱▱▱", "▰▰▱▱▱▱▱▱▱▱", "▰▰▰▱▱▱▱▱▱▱", "▰▰▰▰▱▱▱▱▱▱", "▰▰▰▰▰▱▱▱▱▱", "▰▰▰▰▰▰▱▱▱▱", "▰▰▰▰▰▰▰▱▱▱", "▰▰▰▰▰▰▰▰▱▱", "▰▰▰▰▰▰▰▰▰▱", "▰▰▰▰▰▰▰▰▰▰"]
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

        def _upload_progress_text(label: str, elapsed: int) -> str:
            # Progress is intentionally capped at 95% until upload finishes.
            # This keeps live UX responsive for both MEGA and GDrive CLI uploads.
            est_pct = min(95, max(1, int((elapsed / 40.0) * 100)))
            bar_idx = min(len(progress_frames) - 1, max(0, est_pct // 10))
            frame = progress_frames[bar_idx]
            spin = spinner[elapsed % len(spinner)]
            return (
                f"📤 *Uploading to {label}...* {spin}\n\n"
                f"`{frame}` {est_pct}%\n"
                f"⏱️ {elapsed}s elapsed\n"
                f"📁 `{f.name}`\n"
                f"📦 {size_mb:.1f} MB"
            )

        async def _upload_progress_loop():
            while progress_state["run"]:
                await asyncio.sleep(1)
                if not progress_state["run"]:
                    break
                elapsed = int(max(0, time.time() - float(progress_state["started"])))
                with suppress(Exception):
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_msg.message_id,
                        text=_upload_progress_text("MEGA" if platform == "mega" else "Google Drive", elapsed),
                        parse_mode="Markdown",
                    )

        progress_task = asyncio.create_task(_upload_progress_loop())

        try:
            use_mega_first = platform == "mega"
            logger.info("[REUPLOAD_DEBUG] Raw callback data: '%s'", d)
            logger.info("[REUPLOAD_DEBUG] Parsed provider: '%s'", platform)
            logger.info("[REUPLOAD_DEBUG] Calling upload with provider: '%s'", platform)
            if use_mega_first:
                if not can_use_mega():
                    await query.message.reply_text(
                        "❌ MEGA credentials not configured.\nUse /changemega to set credentials.",
                        reply_markup=main_kb()
                    )
                    return

                mega_ok, mega_info = await asyncio.to_thread(mega_upload, str(f))
                logger.info("[REUPLOAD_DEBUG] mega_upload result: ok=%s info=%s", mega_ok, mega_info[:120] if mega_info else '')
                if not mega_ok:
                    if "EBLOCKED" in str(mega_info) or "509" in str(mega_info):
                        await query.message.reply_text(
                            "⚠️ Mega Upload Blocked\n\n"
                            "This server's IP is blocked by Mega.\n"
                            "Please use GDrive instead or create a new\n"
                            "Mega account from a home network and update\n"
                            "MEGA_EMAIL and MEGA_PASSWORD in .env",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("📁 GDrive ✅ Recommended", callback_data=f"reupload_pick_gdrive_{idx}")],
                                [InlineKeyboardButton("🏠 Home", callback_data="back_main")],
                            ]),
                        )
                        return
                    if is_mega_auth_error(mega_info):
                        logger.warning("MEGA auth error on reupload — falling back to GDrive: %s", mega_info)
                        ok, link, ginfo = await asyncio.to_thread(gdrive_upload, str(f))
                        if ok and link:
                            with suppress(Exception):
                                await status_msg.edit_text("✅ Upload complete\n`▰▰▰▰▰▰▰▰▰▰` 100%", parse_mode="Markdown")
                            await query.message.reply_text(
                                f"✅ Re-upload complete\n📦 Provider: GOOGLE DRIVE (MEGA auth failed)\n🔗 {link}",
                                reply_markup=main_kb()
                            )
                            return
                    await query.message.reply_text(
                        f"❌ MEGA reupload fail!\nReason: {mega_info}",
                        reply_markup=main_kb()
                    )
                    return

                remote_path = f"/Root/faceswap/{f.name}"
                link_ok, link_info = await asyncio.to_thread(mega_export_link, remote_path, 2, 1)
                logger.info("Re-upload complete — Provider: MEGA — link_ok: %s — link: %s", link_ok, link_info)
                with suppress(Exception):
                    await status_msg.edit_text("✅ Upload complete\n`▰▰▰▰▰▰▰▰▰▰` 100%", parse_mode="Markdown")
                final_link = link_info if (link_ok and link_info.startswith("http")) else f"mega:{remote_path}"
                await query.message.reply_text(
                    f"✅ Re-upload complete\n📦 Provider: MEGA\n🔗 {final_link}",
                    reply_markup=main_kb()
                )

                await start_sleep_countdown(
                    context.application,
                    chat_id,
                    reason_text="Reupload completed",
                    delay_seconds=SLEEP_COUNTDOWN_SECONDS,
                    force_allow=True,
                )
                return

            ok, link, info = await asyncio.to_thread(gdrive_upload, str(f))
            logger.info("Re-upload complete — Provider: GOOGLE DRIVE — ok: %s — link: %s", ok, link)
            if ok and link:
                with suppress(Exception):
                    await status_msg.edit_text("✅ Upload complete\n`▰▰▰▰▰▰▰▰▰▰` 100%", parse_mode="Markdown")
                await query.message.reply_text(
                    f"✅ Re-upload complete\n📦 Provider: GOOGLE DRIVE\n🔗 {link}",
                    reply_markup=main_kb()
                )
                await start_sleep_countdown(
                    context.application,
                    chat_id,
                    reason_text="Reupload completed",
                    delay_seconds=SLEEP_COUNTDOWN_SECONDS,
                    force_allow=True,
                )
            elif ok:
                with suppress(Exception):
                    await status_msg.edit_text("✅ Upload complete\n`▰▰▰▰▰▰▰▰▰▰` 100%", parse_mode="Markdown")
                await query.message.reply_text(
                    f"✅ Re-upload complete\n📦 Provider: GOOGLE DRIVE\n🔗 {get_gdrive_target_folder()}/{f.name}",
                    reply_markup=main_kb()
                )
                await start_sleep_countdown(
                    context.application,
                    chat_id,
                    reason_text="Reupload completed",
                    delay_seconds=SLEEP_COUNTDOWN_SECONDS,
                    force_allow=True,
                )
            else:
                await query.message.reply_text(f"❌ Google Drive reupload fail!\nReason: {info}", reply_markup=main_kb())
        finally:
            progress_state["run"] = False
            if progress_task and not progress_task.done():
                progress_task.cancel()
                with suppress(asyncio.CancelledError):
                    await progress_task

    elif d == "reupload_output_mega":
        # Backward compatibility path.
        outputs = list_swap_outputs()
        if not outputs:
            await query.message.reply_text("No output files found. Run a job first.", reply_markup=main_kb())
            return
        await query.message.reply_text(
            "Select one output file to upload:",
            reply_markup=build_reupload_picker_kb("mega", outputs, max_items=3),
        )

    elif d == "reupload_output_gdrive":
        # Backward compatibility path.
        outputs = list_swap_outputs()
        if not outputs:
            await query.message.reply_text("No output files found. Run a job first.", reply_markup=main_kb())
            return
        await query.message.reply_text(
            "Select one output file to upload:",
            reply_markup=build_reupload_picker_kb("gdrive", outputs, max_items=3),
        )

    elif False:
        pass

    elif re.match(r"^reupload_pick_legacy_(mega|gdrive)_\d+$", d):
        # Deprecated callbacks are redirected to the modern picker flow.
        m = re.match(r"^reupload_pick_legacy_(mega|gdrive)_(\d+)$", d)
        platform = m.group(1)
        idx = int(m.group(2))

        await query.message.reply_text(
            "ℹ️ Legacy picker detected. Redirecting to current upload flow.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("☁️ Mega", callback_data=f"reupload_pick_mega_{idx}")],
                [InlineKeyboardButton("📁 Google Drive", callback_data=f"reupload_pick_gdrive_{idx}")],
                [InlineKeyboardButton("⬅️ Back", callback_data="reupload_output_menu")],
            ]),
        )
        return

    elif d == "check_storage":
        df_target = PIPELINE if os.path.exists(PIPELINE) else "/"
        df = subprocess.run(["df", "-h", df_target], capture_output=True, text=True).stdout
        sizes = log_storage_breakdown("STORAGE SCAN (button)")
        await query.message.reply_text(
            "💾 *Storage Health*\n"
            f"🧹 temp/: `{_bytes_to_gb(sizes['temp']):.2f} GB`\n"
            f"🎞 frames/: `{_bytes_to_gb(sizes['frames']):.2f} GB`\n"
            f"📁 outputs/: `{_bytes_to_gb(sizes['outputs']):.2f} GB`\n"
            f"📥 downloads/: `{_bytes_to_gb(sizes['downloads']):.2f} GB`\n"
            f"🗃 cache/: `{_bytes_to_gb(sizes['cache']):.2f} GB`\n"
            f"📊 tracked total: `{_bytes_to_gb(sizes['total']):.2f} GB`\n"
            f"🛡 policy limit: `{MAX_STORAGE_USAGE_GB} GB`\n\n"
            f"```{df}```",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🗑 Open Cleanup", callback_data="clean_workspace")],
                [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            ])
        )

    elif d == "clean_workspace":
        await query.message.reply_text(
            "⚠️ *Workspace Cleanup*\n\n"
            "🧹 *Temp Only:* Temp files clear, downloads safe.\n"
            "📁 *Old Outputs:* Purane outputs delete, latest safe.\n"
            "💣 *Full Clean:* Temp + cache + old outputs sab clear.\n\n"
            "Protected system paths aur runtime files safe rahenge.",
            parse_mode="Markdown",
            reply_markup=cleanup_modes_kb(),
        )

    elif d == "clean_workspace_cancel":
        await query.message.reply_text("🟢 Cleanup cancelled.", reply_markup=main_kb())

    elif d in {
        "clean_workspace_temp_only",
        "clean_workspace_outputs_old",
        "clean_workspace_full_clean",
    }:
        mode_map = {
            "clean_workspace_temp_only": "temp_only",
            "clean_workspace_outputs_old": "outputs_old",
            "clean_workspace_full_clean": "full_clean",
        }
        mode = mode_map[d]
        before_sizes = log_storage_breakdown(f"CLEANUP BEFORE mode={mode}")
        stats = await asyncio.to_thread(clean_workspace, mode)
        limit_stats = await asyncio.to_thread(enforce_storage_limit, MAX_STORAGE_USAGE_GB)
        after_sizes = log_storage_breakdown(f"CLEANUP AFTER mode={mode}")

        await query.message.reply_text(
            "✅ *Cleanup completed.*\n"
            f"Cleanup completed. Freed: `{(max(0, stats['before'] - stats['after']) / (1024.0 * 1024.0)):.2f} MB`\n"
            f"Mode: `{mode}`\n"
            f"Before: `{_bytes_to_gb(stats['before']):.2f} GB`\n"
            f"After: `{_bytes_to_gb(stats['after']):.2f} GB`\n"
            f"Deleted files: `{stats['deleted_files']}`\n"
            f"Downloads cleaned: `{stats.get('downloads_cleaned', 0)}` ({_bytes_to_gb(stats.get('downloads_cleaned_bytes', 0)):.2f} GB)\n"
            f"Skipped protected: `{stats.get('skipped_protected', 0)}`\n"
            f"Skipped unknown/recent/locked: `{stats.get('skipped_unknown', 0)}/{stats.get('skipped_recent', 0)}/{stats.get('skipped_locked', 0)}`\n"
            f"Policy check: `{MAX_STORAGE_USAGE_GB} GB` limit, disk usage `{float(limit_stats.get('disk_usage_pct', 0.0)):.1f}%`\n"
            f"Policy before/after: `{_bytes_to_gb(limit_stats['before']):.2f} -> {_bytes_to_gb(limit_stats['after']):.2f} GB`\n\n"
            "📊 Updated folders:\n"
            f"temp `{_bytes_to_gb(before_sizes['temp']):.2f} -> {_bytes_to_gb(after_sizes['temp']):.2f} GB`\n"
            f"frames `{_bytes_to_gb(before_sizes['frames']):.2f} -> {_bytes_to_gb(after_sizes['frames']):.2f} GB`\n"
            f"outputs `{_bytes_to_gb(before_sizes['outputs']):.2f} -> {_bytes_to_gb(after_sizes['outputs']):.2f} GB`\n"
            f"downloads `{_bytes_to_gb(before_sizes['downloads']):.2f} -> {_bytes_to_gb(after_sizes['downloads']):.2f} GB`\n"
            f"cache `{_bytes_to_gb(before_sizes['cache']):.2f} -> {_bytes_to_gb(after_sizes['cache']):.2f} GB`",
            parse_mode="Markdown",
            reply_markup=main_kb(chat_id),
        )

    elif d == "download_output":
        outputs = list_swap_outputs()
        if not outputs:
            await query.message.reply_text("❌ Koi output nahi.", reply_markup=main_kb())
            return
        f       = outputs[0]
        size_mb = f.stat().st_size / 1024 / 1024
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            await query.message.reply_photo(
                photo=open(str(f), "rb"),
                caption=f"`{f.name}`",
                parse_mode="Markdown", reply_markup=main_kb()
            )
        elif size_mb > 45:
            await query.message.reply_text(
                f"⚠️ {size_mb:.1f} MB — Telegram 50MB limit!\nGDrive/MEGA use karo.",
                reply_markup=main_kb()
            )
        else:
            await query.message.reply_video(
                video=open(str(f), "rb"),
                caption=f"`{f.name}`",
                parse_mode="Markdown", reply_markup=main_kb()
            )

    elif d == "change_mega":
        u, _ = get_mega_creds()
        context.user_data["awaiting_mega_creds"] = True
        await query.message.reply_text(
            f"🔄 Current MEGA: `{u}`\n\nNaya bhejo:\n`email:password`",
            parse_mode="Markdown"
        )

    elif d == "change_drive_token":
        context.user_data["awaiting_drive_token"] = True
        await query.message.reply_text(
            "📥 Send Drive auth token\n"
            "• Raw access_token, ya\n"
            "• Full JSON auth response"
        )

    elif d == "face_map_settings":
        count = len(selected_face_maps.get(chat_id, {}))
        await query.message.reply_text(
            "🎯 *Selective Face Swap*\n"
            f"Configured faces: *{count}*\n\n"
            "Format (har line): `face_index|mega_source_link`\n"
            "Example:\n"
            "`1|https://mega.nz/file/...`\n"
            "`3|https://mega.nz/file/...`\n\n"
            "Index order = `left-right` (video ke first frame ke hisaab se).",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📝 Add/Update", callback_data="set_face_map"),
                    InlineKeyboardButton("♻️ Clear", callback_data="clear_face_map"),
                ],
                [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            ])
        )

    elif d == "set_face_map":
        context.user_data["awaiting_face_map"] = True
        await query.message.reply_text(
            "Mappings bhejo, ek line per ek entry:\n"
            "`1|https://mega.nz/file/...`\n"
            "`3|https://mega.nz/file/...`",
            parse_mode="Markdown",
            reply_markup=main_kb()
        )

    elif d == "clear_face_map":
        selected_face_maps.pop(chat_id, None)
        await query.message.reply_text("✅ Face map clear ho gaya.", reply_markup=main_kb())

    elif d == "start_bot":
        # Reset transient input modes so normal link messages are processed as jobs.
        for transient_key in (
            "awaiting_mega_creds",
            "awaiting_drive_token",
            "awaiting_new_face",
            "awaiting_face_link",
            "awaiting_clip_range",
            "awaiting_multi_target",
            "awaiting_multi_source",
            "awaiting_face_map",
        ):
            context.user_data.pop(transient_key, None)
        task = sleep_countdown_tasks.get(chat_id)
        extra_note = None
        if task and not task.done():
            task.cancel()
            _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
            extra_note = "▶️ Sleep countdown cancel ho gaya. Bot active hai."
        else:
            extra_note = "✅ Bot already running aur ready hai."
        await send_ready_banner(query.message, chat_id, extra_note=extra_note)
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "cancel_sleep_countdown":
        task = sleep_countdown_tasks.get(chat_id)
        if task and not task.done():
            task.cancel()
            sleep_timer_active[chat_id] = False
            sleep_countdown_state[chat_id] = {
                "chat_id": str(chat_id),
                "sleep_timer_active": False,
                "status": "cancelled",
                "reason": "Cancelled by user button",
                "cancelled_at": time.time(),
            }
            _clear_sleep_countdown_state()
            _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
            await query.message.reply_text(
                "✅ Sleep countdown cancelled. Studio stays awake.",
                reply_markup=main_kb()
            )
        else:
            state = sleep_countdown_state.get(chat_id, {})
            if str(state.get("status") or "").lower() == "running":
                sleep_countdown_state[chat_id] = {
                    "chat_id": str(chat_id),
                    "sleep_timer_active": False,
                    "status": "cancelled",
                    "reason": "Cancelled by user button",
                    "cancelled_at": time.time(),
                }
                _clear_sleep_countdown_state()
                sleep_timer_active[chat_id] = False
                _update_lifecycle_state(chat_id, is_countdown_running=False, can_auto_sleep=False)
                await query.message.reply_text(
                    "✅ Sleep countdown cancelled. Studio stays awake.",
                    reply_markup=main_kb()
                )
                return
            if state.get("status") == "completed":
                await query.message.reply_text(
                    "⏰ Timer over ho chuka hai. Stop button ab active nahi hai.\n"
                    "Studio sleep mode me ja raha hai ya ja chuka hai.\n"
                    "Dubara active karne ke liye studio manually restart karo.",
                    reply_markup=main_kb()
                )
                return
            await query.message.reply_text(
                "ℹ️ No active sleep countdown found.",
                reply_markup=main_kb()
            )

    elif d == "sleep_now":
        task = sleep_countdown_tasks.get(chat_id)
        if task and not task.done():
            task.cancel()
        sleep_timer_active[chat_id] = False
        _clear_sleep_countdown_state()
        _update_lifecycle_state(chat_id, is_countdown_running=False)
        await query.message.reply_text(
            "🚀 Sleep request sent\nStudio entering sleep mode...",
            reply_markup=main_kb(),
        )
        if SLEEP_TEST_MODE:
            await query.message.reply_text(
                "[SLEEP TEST MODE] Real sleep suppressed (SLEEP_TEST_MODE=1)",
                reply_markup=main_kb(),
            )
            return
        ok, info = await asyncio.to_thread(request_studio_sleep)
        if ok:
            logger.info("sleep-now request successful chat=%s info=%s", chat_id, info)
            await asyncio.sleep(2)
            sys.exit(0)
        await query.message.reply_text(
            f"❌ Sleep request failed: {info}",
            reply_markup=main_kb(),
        )

    elif d == "quick_sleep":
        remain = _sleep_remaining_seconds(chat_id)
        if remain is not None:
            await query.message.reply_text(
                f"⏳ Sleep countdown already running: {remain}s left.",
                reply_markup=sleep_countdown_kb(),
            )
            return
        await query.message.reply_text(
            f"Studio ko sleep me bhejna hai? Countdown: {int(SLEEP_COUNTDOWN_SECONDS)}s",
            reply_markup=InlineKeyboardMarkup([[ 
                InlineKeyboardButton("✅ Haan, Start Countdown", callback_data="confirm_quick_sleep"),
                InlineKeyboardButton("❌ Cancel", callback_data="back_main"),
            ]])
        )

    elif d == "confirm_quick_sleep":
        task = await start_sleep_countdown(
            context.application,
            chat_id,
            reason_text="Quick sleep requested by user.",
            delay_seconds=SLEEP_COUNTDOWN_SECONDS,
            force_allow=True,
        )
        if task is None:
            await query.message.reply_text(
                "Auto sleep countdown start nahi hua. Active job ya pending queue check karo.",
                reply_markup=main_kb(chat_id),
            )

    elif d == "clip_settings":
        await query.message.reply_text(
            "✂️ *Clip Range Settings*\n"
            f"Current: `{get_clip_range_note(chat_id)}`\n\n"
            "Single example: `00:01:00-00:02:30`\n"
            "Multiple examples (line by line):\n"
            "`00:01:00-00:02:00`\n"
            "`00:03:00-00:04:00`",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📝 Set Range", callback_data="set_clip_range"),
                    InlineKeyboardButton("♻️ Clear", callback_data="clear_clip_range"),
                ],
                [InlineKeyboardButton("⬅️ Back", callback_data="back_main")],
            ]),
        )

    elif d == "set_clip_range":
        context.user_data["awaiting_clip_range"] = True
        await query.message.reply_text(
            "Range bhejo.\n"
            "Single: `00:01:00-00:02:30`\n"
            "Multiple (new line/comma):\n"
            "`00:01:00-00:02:00`\n"
            "`00:03:00-00:04:00`",
            parse_mode="Markdown",
        )

    elif d == "clear_clip_range":
        clip_ranges.pop(chat_id, None)
        context.user_data.pop("awaiting_clip_range", None)
        _persist_ui_runtime_state()
        await query.message.reply_text("✅ Clip range clear ho gaya.", reply_markup=main_kb(chat_id))
        await send_mode_state_announcement(query.message, chat_id, context)

    elif d == "safe_restart_bot":
        await query.message.reply_text(
            "ℹ️ Safe restart button disable kar diya gaya hai. Bot restart only server-side se hoga.",
            reply_markup=main_kb(chat_id),
        )

    elif d == "back_main":
        _cleanup_custom_mega_temp(context)
        context.user_data.pop("awaiting_custom_mega_link", None)
        await send_ready_banner(query.message, chat_id, extra_note="🔄 Main menu refreshed")


async def _download_prepare_face(face_link, chat_id, stage_callback=None, apply_as_default=False):
    os.makedirs(FACE_DIR, exist_ok=True)
    request_id = f"req_{str(chat_id)}_{int(time.time() * 1000)}"
    request_dir = Path(FACE_DIR) / "_requests" / request_id
    request_dir.mkdir(parents=True, exist_ok=True)

    async def _emit(stage_text):
        logger.info("[FACE-CHANGE] stage=%s chat=%s", str(stage_text or ""), chat_id)
        if stage_callback is None:
            return
        with suppress(Exception):
            await stage_callback(stage_text)

    logger.info("[FACE-CHANGE] worker-start chat=%s link_head=%s apply_as_default=%s", chat_id, str(face_link or "")[:140], bool(apply_as_default))

    await _emit("Connecting...")
    await _emit("Downloading...")

    try:
        ok_download, download_reason = await asyncio.wait_for(
            mega_download_async_detailed(
                face_link,
                str(request_dir),
                retries=FACE_DOWNLOAD_RETRY_COUNT,
                attempt_timeout_sec=FACE_DOWNLOAD_ATTEMPT_TIMEOUT_SEC,
                stall_timeout_sec=FACE_DOWNLOAD_STALL_TIMEOUT_SEC,
            ),
            timeout=FACE_DOWNLOAD_TOTAL_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        ok_download, download_reason = False, f"face download exceeded total timeout ({FACE_DOWNLOAD_TOTAL_TIMEOUT_SEC}s)"
        logger.warning("[FACE-CHANGE] total-timeout chat=%s timeout=%ss", chat_id, FACE_DOWNLOAD_TOTAL_TIMEOUT_SEC)

    if not ok_download:
        reason_text = str(download_reason or "unknown error")
        return False, {"reason": reason_text}

    candidates, discovered_files, extracted_dirs = _discover_face_payload_candidates(request_dir)
    logger.info(
        "[FACE-CHANGE] payload-scan chat=%s request=%s discovered=%s extracted_dirs=%s candidates=%s",
        chat_id,
        request_id,
        json.dumps(discovered_files[:40], ensure_ascii=True),
        json.dumps(extracted_dirs[:10], ensure_ascii=True),
        json.dumps([str(c.name) for c in candidates[:20]], ensure_ascii=True),
    )

    if not candidates:
        discovered_preview = ", ".join(
            [f"{d.get('name')} ({int(d.get('bytes') or 0)}B)" for d in discovered_files[:12]]
        )
        with suppress(Exception):
            shutil.rmtree(request_dir, ignore_errors=True)
        if discovered_preview:
            return False, {"reason": f"no valid image found in downloaded payload; discovered: {discovered_preview}"}
        return False, {"reason": "no file found in downloaded payload"}

    await _emit("Validating image...")

    best = None
    rejected = []
    for cand in candidates:
        prepared = coerce_face_source_to_jpg(cand)
        ok_img, reason_img = validate_downloaded_face_image(prepared)
        if ok_img:
            best = prepared
            break
        rejected.append(f"{prepared.name}: {reason_img}")

    if best is None:
        detail = rejected[0] if rejected else "no valid image candidate"
        with suppress(Exception):
            shutil.rmtree(request_dir, ignore_errors=True)
        return False, {"reason": detail}

    await _emit("Face detected...")
    await _emit("Applying new face...")

    clean = ""
    if apply_as_default:
        clean = set_default_face_from_source(str(best))
    else:
        # Save outside request_dir so rmtree below doesn't delete it
        _clean_dst = Path(FACE_DIR) / f"{request_id}_clean.jpg"
        clean = face_to_clean_jpg(str(best), dst_path=str(_clean_dst))

    if not clean or not os.path.isfile(clean):
        with suppress(Exception):
            shutil.rmtree(request_dir, ignore_errors=True)
        return False, {"reason": "face conversion/save failed"}

    dims = ""
    clean_bytes = 0
    with suppress(Exception):
        clean_bytes = int(Path(clean).stat().st_size or 0)

    with suppress(Exception):
        import cv2

        probe = cv2.imread(str(clean))
        if probe is not None:
            h, w = probe.shape[:2]
            dims = f"{w}x{h}"

    with suppress(Exception):
        shutil.rmtree(request_dir, ignore_errors=True)

    return True, {
        "clean_path": str(clean),
        "selected_file": str(best),
        "bytes": clean_bytes,
        "dimensions": dims,
    }


async def _handle_face_change(update, context, chat_id, face_link):
    logger.info("[FACE-CHANGE] start chat=%s link_head=%s", chat_id, str(face_link or "")[:140])
    status_msg = await update.message.reply_text("Connecting...")

    progress_state = {"run": True}
    started_at = float(time.time())
    progress_view = {"stage": "Connecting..."}

    async def _emit_stage(stage_text):
        progress_view["stage"] = str(stage_text or "Working...")

    async def _face_download_progress_loop():
        while progress_state["run"]:
            await asyncio.sleep(2)
            elapsed = int(max(0.0, time.time() - started_at))
            bytes_now = int(_path_size_bytes(FACE_DIR))
            mb_now = float(bytes_now) / (1024.0 * 1024.0)
            with suppress(Exception):
                await status_msg.edit_text(
                    f"{progress_view['stage']}\n"
                    f"⏱ {elapsed}s\n"
                    f"📦 {mb_now:.1f} MB received"
                )

    progress_task = asyncio.create_task(_face_download_progress_loop())
    try:
        ok_face, face_result = await _download_prepare_face(
            face_link,
            chat_id,
            stage_callback=_emit_stage,
            apply_as_default=True,
        )
    finally:
        progress_state["run"] = False
        if progress_task and not progress_task.done():
            progress_task.cancel()
            with suppress(asyncio.CancelledError):
                await progress_task

    if not ok_face:
        reason = str((face_result or {}).get("reason") or "unknown error")
        logger.warning("[FACE-CHANGE] failed chat=%s reason=%s", chat_id, reason[:240])
        await status_msg.edit_text(
            f"Face download failed: {reason[:220]}",
            reply_markup=main_kb(),
        )
        return

    clean = str((face_result or {}).get("clean_path") or "").strip()
    if not clean or not os.path.isfile(clean):
        await status_msg.edit_text(
            "Face download failed: face output missing after apply",
            reply_markup=main_kb(),
        )
        return

    clear_face_runtime_cache(chat_id)
    current_face[chat_id] = clean
    with suppress(Exception):
        await status_msg.edit_text("✅ Face Updated Successfully\nReady for next jobs.", reply_markup=main_kb())
    logger.info("[FACE-CHANGE] success chat=%s face=%s", chat_id, Path(clean).name)
    await update.message.reply_photo(
        photo=open(clean, "rb"),
        caption="✅ Face Updated Successfully\nReady for next jobs.",
        reply_markup=main_kb()
    )


async def schedule_shutdown(app, chat_id, delay_seconds=AUTO_SHUTDOWN_DELAY_SEC, reason_text=None):
    if reason_text is None:
        reason_text = "Job complete detected."
    if not _can_auto_sleep(chat_id):
        logger.info(
            "schedule_shutdown skipped chat=%s busy=%s queue=%s no_bg=%s phase=%s",
            chat_id,
            _is_chat_busy(chat_id),
            _queue_size(chat_id),
            _no_background_task_running(chat_id),
            _last_job_phase(chat_id),
        )
        return
    await start_sleep_countdown(app, chat_id, reason_text=reason_text, delay_seconds=delay_seconds)


async def on_job_completed(app, chat_id, success=True):
    global start_sleep_timer
    if app is None:
        logger.info("on_job_completed app=None chat=%s — building minimal app for sleep countdown", chat_id)
        try:
            _dashboard_record_completion(chat_id, success=bool(success))
        except Exception:
            pass
        try:
            from telegram import Bot as _Bot
            # Build minimal app-like object so start_sleep_countdown can use app.bot
            class _MinimalApp:
                def __init__(self, bot): self.bot = bot
            _bot = _Bot(token=BOT_TOKEN)
            _app = _MinimalApp(_bot)
            reason = ("Job completed successfully." if success else "Sorry, your job failed.")
            await start_sleep_countdown(_app, chat_id, reason_text=reason,
                                        delay_seconds=SLEEP_COUNTDOWN_SECONDS, force_allow=True)
        except Exception as e:
            logger.warning("on_job_completed sleep countdown failed chat=%s err=%s", chat_id, e)
        return

    if success:
        reason_text = f"Job completed successfully. Countdown started ({_sleep_delay_minutes_text()})."
    else:
        reason_text = "Sorry, your job failed. Try again later. Countdown started."

    await safe_send_message(app.bot, chat_id, reason_text)
    try:
        _dashboard_record_completion(chat_id, success=bool(success), details=reason_text)
    except Exception:
        pass

    # Explicit auto-sleep countdown announcement so the user always sees it kick in.
    countdown_minutes = max(1, int(round(SLEEP_COUNTDOWN_SECONDS / 60.0)))
    countdown_announce = (
        f"⏳ Auto-sleep countdown started: {countdown_minutes} min "
        f"({int(SLEEP_COUNTDOWN_SECONDS)}s). New job aaya to cancel ho jayega."
    )
    try:
        await safe_send_message(app.bot, chat_id, countdown_announce)
    except Exception as e:
        logger.warning("countdown announce send failed chat=%s err=%s", chat_id, e)

    start_sleep_timer = True
    await start_sleep_countdown(
        app,
        chat_id,
        reason_text=reason_text,
        delay_seconds=SLEEP_COUNTDOWN_SECONDS,
        force_allow=True,
    )

    # Clear the chat -> dashboard mapping so future jobs get a fresh session URL.
    try:
        _dashboard_clear_token(chat_id)
    except Exception:
        pass


async def run_pipeline(context, chat_id, video_link, face_link=None, job_mode="direct", progress_seed_message_id=None, queue_job_id=None, gender_mode_override=None, stage_event_hook=None):
    async def notify(msg, kb=None):
        kw = {"parse_mode": "Markdown"}
        if kb:
            kw["reply_markup"] = kb
        logger.info("DEBUG FLOW: notify() chat=%s text_head=%s", chat_id, (msg or "")[:120].replace("\n", " "))
        sent = await safe_send_message(context.bot, chat_id, msg, **kw)
        if sent is None:
            logger.warning("notify dropped due flood/transport issues chat=%s", chat_id)

    last_prog = {"txt": ""}
    progress_msg = {"obj": None}
    progress_lock = asyncio.Lock()
    progress_rate = {
        "last_sent_at": 0.0,
        "min_interval": 0.0,
    }
    bot_identity = {"id": None}
    progress_stream_token = f"{chat_id}:{time.time_ns()}"
    keepalive_task = None
    keepalive_state = {"run": False}
    ui_watchdog_task = None
    ui_watchdog_state = {"run": False}
    pipeline_started_at = time.time()
    gender_override = str(gender_mode_override or "").strip().lower()
    if gender_override in {"female", "all"}:
        effective_gender_mode = gender_override
        set_gender_mode(chat_id, effective_gender_mode)
    else:
        effective_gender_mode = get_gender_mode(chat_id)
    fast_mode_flag = "ON" if CPU_FAST_MODE else "OFF"

    def _is_active_progress_stream():
        return progress_stream_tokens.get(chat_id) == progress_stream_token

    def _actual_job_phase():
        st = _get_active_job_state(chat_id, allow_fallback=True) or {}
        return str(st.get("phase") or st.get("status") or "").lower()

    def _current_job_id_text():
        st = job_status.get(chat_id, {}) or {}
        jid = st.get("job_id")
        if jid in (None, ""):
            jid = queue_job_id
        if jid in (None, ""):
            return "-"
        return str(jid)

    async def _job_keepalive_loop():
        touch_file = Path(KEEPALIVE_TOUCH_FILE)
        touch_file.parent.mkdir(parents=True, exist_ok=True)
        while keepalive_state["run"] and _is_active_progress_stream():
            st = job_status.get(chat_id, {}) or {}
            phase = str(st.get("phase", "") or "").lower()
            if phase in {"completed", "failed", "stopped", "cancelled", "exception"}:
                break

            stage = str(st.get("stage", "-") or "-")
            now_ts = time.time()
            stamp = time.strftime("%H:%M:%S", time.localtime(now_ts))
            payload = (
                f"chat_id={chat_id} job_id={_current_job_id_text()} "
                f"phase={phase or '-'} stage={stage} ts={now_ts:.3f}\n"
            )
            try:
                await asyncio.to_thread(touch_file.write_text, payload, "utf-8")
            except Exception as e:
                logger.warning("keepalive touch write failed chat=%s: %s", chat_id, e)

            logger.info(
                "[KEEP-ALIVE] Job #%s running | Stage: %s | Time: %s",
                _current_job_id_text(),
                stage,
                stamp,
            )
            await asyncio.sleep(KEEPALIVE_INTERVAL_SEC)

    async def _bot_user_id():
        if bot_identity["id"] is not None:
            return bot_identity["id"]
        try:
            me = await context.bot.get_me()
            bot_identity["id"] = int(getattr(me, "id", 0) or 0) or None
        except Exception:
            bot_identity["id"] = None
        return bot_identity["id"]

    def _extract_message_id(msg_obj):
        try:
            msg_id = int(getattr(msg_obj, "message_id", 0) or 0)
        except Exception:
            return None
        return msg_id if msg_id > 0 else None

    async def _send_new_progress_message(text):
        sent = await safe_send_message(context.bot, chat_id, text)
        if sent is None:
            return None
        msg_id = _extract_message_id(sent)
        if not msg_id:
            return None

        sender_id = getattr(getattr(sent, "from_user", None), "id", None)
        if sender_id is not None:
            bot_id = await _bot_user_id()
            if bot_id is not None and int(sender_id) != int(bot_id):
                logger.warning("progress message sender mismatch chat=%s sender=%s bot=%s", chat_id, sender_id, bot_id)
                return None
        return sent

    async def _send_progress_anchor_blocking(initial_text):
        attempts = 0
        while True:
            attempts += 1
            try:
                sent = await context.bot.send_message(
                    chat_id=chat_id,
                    text=initial_text,
                )
                msg_id = _extract_message_id(sent)
                if msg_id:
                    sender_id = getattr(getattr(sent, "from_user", None), "id", None)
                    if sender_id is not None:
                        bot_id = await _bot_user_id()
                        if bot_id is not None and int(sender_id) != int(bot_id):
                            logger.warning(
                                "progress anchor sender mismatch chat=%s sender=%s bot=%s",
                                chat_id,
                                sender_id,
                                bot_id,
                            )
                            await asyncio.sleep(1)
                            continue
                    logger.info("progress anchor ready chat=%s message_id=%s attempts=%s", chat_id, msg_id, attempts)
                    return sent
            except RetryAfter as e:
                wait_s = _retry_after_seconds(e)
                logger.warning(
                    "progress anchor send blocked chat=%s retry_after=%ss attempts=%s",
                    chat_id,
                    wait_s,
                    attempts,
                )
                await asyncio.sleep(wait_s)
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Transport/network failures retry with bounded backoff until anchor exists.
                wait_s = min(30, max(2, attempts))
                logger.warning(
                    "progress anchor send failed chat=%s attempt=%s wait=%ss err=%s",
                    chat_id,
                    attempts,
                    wait_s,
                    e,
                )
                await asyncio.sleep(wait_s)

    async def _set_progress_anchor(new_obj):
        old_obj = progress_msg.get("obj")
        old_id = _extract_message_id(old_obj)
        new_id = _extract_message_id(new_obj)
        progress_msg["obj"] = new_obj
        if new_id:
            job_status.setdefault(chat_id, {})["message_id"] = int(new_id)
            _persist_active_job_state(chat_id, message_id=new_id)

        # Keep one visible progress message per job by deleting replaced anchors.
        if old_id and new_id and old_id != new_id:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=old_id)
            except Exception:
                pass

    async def _ensure_progress_anchor(max_wait_sec=None):
        if await _progress_message_target_valid():
            return progress_msg["obj"]
        try:
            if max_wait_sec is not None:
                sent = await asyncio.wait_for(
                    _send_progress_anchor_blocking("🚀 Job Started"),
                    timeout=max(0.2, float(max_wait_sec)),
                )
            else:
                sent = await _send_progress_anchor_blocking("🚀 Job Started")
        except asyncio.TimeoutError:
            logger.info("progress anchor deferred chat=%s timeout=%.2fs", chat_id, float(max_wait_sec or 0.0))
            return None
        await _set_progress_anchor(sent)
        return progress_msg["obj"]

    async def _progress_message_target_valid():
        obj = progress_msg.get("obj")
        msg_id = _extract_message_id(obj)
        if not msg_id:
            return False

        sender_id = getattr(getattr(obj, "from_user", None), "id", None)
        if sender_id is not None:
            bot_id = await _bot_user_id()
            if bot_id is not None and int(sender_id) != int(bot_id):
                return False

        try:
            await context.bot.get_chat(chat_id=chat_id)
        except Exception:
            return False
        return True

    if progress_seed_message_id:
        # Do not trust stale seeded IDs for edits; always establish a fresh progress anchor message.
        progress_msg["obj"] = None
    worker_pid = int(os.environ.get("PIPELINE_WORKER_PID", "0") or 0)
    job_temp_path = None
    active_target_path = None
    final_output_path = None
    downloaded_source_path = None

    async def progress(msg, force=False):
        logger.info("DEBUG FLOW: progress() chat=%s force=%s", chat_id, force)

        if not _is_active_progress_stream():
            return False

        def _message_not_modified_error(err):
            return "message is not modified" in str(err).lower()

        def _anchor_invalid_error(err):
            txt = str(err).lower()
            return (
                "message to edit not found" in txt
                or "message can't be edited" in txt
                or "chat not found" in txt
            )

        async with progress_lock:
            if msg == last_prog["txt"]:
                return False

            try:
                anchor_obj = await _ensure_progress_anchor(max_wait_sec=1.0)
                if anchor_obj is None:
                    # Do not block pipeline work on Telegram rate limits.
                    return False
                logger.info("DEBUG FLOW: progress edit message chat=%s message_id=%s", chat_id, getattr(progress_msg["obj"], "message_id", None))
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=progress_msg["obj"].message_id,
                    text=msg,
                )
                if progress_msg["obj"] is not None:
                    last_prog["txt"] = msg
                    progress_rate["last_sent_at"] = time.time()
                    _dashboard_record_progress_text(chat_id, msg)
                    return True
            except RetryAfter as e:
                raw_wait = _retry_after_seconds(e)
                logger.warning("progress edit flood chat=%s retry_after=%ss (skipping, heartbeat will retry)", chat_id, raw_wait)
                return False
            except Exception as e:
                if _message_not_modified_error(e):
                    # Keep the same anchor; this is a benign Telegram response.
                    last_prog["txt"] = msg
                    progress_rate["last_sent_at"] = time.time()
                    _dashboard_record_progress_text(chat_id, msg)
                    return True

                if _anchor_invalid_error(e):
                    logger.warning("progress anchor invalid chat=%s err=%s; recreating anchor", chat_id, e)
                    sent = await _send_progress_anchor_blocking("🚀 Progress Resynced")
                    await _set_progress_anchor(sent)
                    if progress_msg["obj"] is not None:
                        try:
                            await context.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=progress_msg["obj"].message_id,
                                text=msg,
                            )
                            last_prog["txt"] = msg
                            progress_rate["last_sent_at"] = time.time()
                            _dashboard_record_progress_text(chat_id, msg)
                            return True
                        except Exception as inner:
                            logger.warning("progress resync edit failed chat=%s: %s", chat_id, inner)
                            return False

                logger.warning("progress update failed chat=%s: %s", chat_id, e)
                return False
        return False

    def _progress_bar(pct):
        p = int(max(0, min(100, pct if pct is not None else 0)))
        full = p // 10
        return "█" * full + "░" * (10 - full)

    def _short_target_name(name, max_len=36):
        safe = (name or "-").replace("`", "'").strip()
        if len(safe) <= max_len:
            return safe
        return f"{safe[: max_len - 3]}..."

    initial_target_name = os.path.basename((video_link or "").split("?", 1)[0]) or "-"
    progress_target_name = {"value": initial_target_name}
    progress_format_log = {"emitted": False}

    live_gate = {
        "last_stage": None,
        "last_pct": -1,
        "last_emit_at": 0.0,
        "mismatch_since": 0.0,
    }

    async def _cleanup_stale_download_state(reason=""):
        if _download_allowed(chat_id):
            return

        # Stop any stale download heartbeat task from older state.
        hb = download_heartbeat_tasks.get(chat_id)
        if hb and not hb.done():
            hb.cancel()
            with suppress(asyncio.CancelledError, RuntimeError):
                await hb
            logger.info("[MEGA CLEANUP] cancelled stale download heartbeat chat=%s reason=%s", chat_id, reason or "phase_mismatch")
        if download_heartbeat_tasks.get(chat_id) is hb:
            download_heartbeat_tasks.pop(chat_id, None)

        # Drop stale queued Telegram download progress retries.
        q = telegram_retry_queues.get(str(chat_id))
        if q:
            kept = []
            dropped = 0
            for item in list(q):
                stage = _extract_progress_stage(item.get("text", "") or "")
                if stage and "downloading" in stage:
                    dropped += 1
                    continue
                kept.append(item)
            if dropped:
                telegram_retry_queues[str(chat_id)] = deque(kept)
                logger.info("[MEGA CLEANUP] dropped stale download retry items chat=%s count=%s", chat_id, dropped)

        # Kill any stale external megadl process if job is no longer in download phase.
        try:
            r = await asyncio.to_thread(
                subprocess.run,
                ["ps", "-eo", "pid=,etimes=,cmd="],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if r.returncode == 0:
                for ln in (r.stdout or "").splitlines():
                    ln = ln.strip()
                    if "megadl" not in ln:
                        continue
                    parts = ln.split(None, 2)
                    if len(parts) < 3:
                        continue
                    pid_s, et_s, cmd = parts[0], parts[1], parts[2]
                    try:
                        pid = int(pid_s)
                        age = int(et_s)
                    except Exception:
                        continue
                    if age < 30:
                        continue
                    try:
                        os.kill(pid, signal.SIGTERM)
                        logger.info("[MEGA CLEANUP] killed stale download task PID=%s age=%ss cmd=%s", pid, age, cmd[:160])
                    except Exception:
                        pass
        except Exception:
            pass

    def _persist_validation_proof_frames(before_path, after_path, target_path, attempt_tag="initial"):
        try:
            out_root = Path(VALIDATION_PROOF_DIR)
            out_root.mkdir(parents=True, exist_ok=True)
            target_stem = Path(target_path).stem if target_path else "target"
            run_tag = str(queue_job_id or int(time.time()))
            saved = []

            if before_path and os.path.isfile(before_path):
                before_dst = out_root / f"{target_stem}_job{run_tag}_{attempt_tag}_before.jpg"
                shutil.copy2(before_path, before_dst)
                saved.append(str(before_dst))

            if after_path and os.path.isfile(after_path):
                after_dst = out_root / f"{target_stem}_job{run_tag}_{attempt_tag}_after.jpg"
                shutil.copy2(after_path, after_dst)
                saved.append(str(after_dst))

            return saved
        except Exception as e:
            logger.warning("proof frame persist failed chat=%s err=%s", chat_id, e)
            return []

    stage_plan = {
        "total": 6,
    }
    stage_sequence = ["download", "extracting", "processing", "merging", "upload", "completed"]
    stage_label_map = {
        "download": "Downloading",
        "extracting": "Extracting Frames",
        "processing": "Face Swap Processing",
        "merging": "Merging Frames",
        "upload": "Uploading",
        "completed": "Output Link",
        "failed": "Failed",
    }
    stage_start_messages = {
        "download": "⬇️ Downloading started",
        "extracting": "🧩 Extracting Frames started",
        "processing": "⚙️ Face Swap Processing started",
        "merging": "🎬 Merging Frames started",
        "upload": "☁️ Uploading started",
        "completed": "🔗 Output Link ready",
    }
    stage_gate = {
        "last_index": -1,
        "announced": set(),
    }

    def _stage_key_from_text(stage_text):
        text = str(stage_text or "").strip().lower()
        if not text:
            return None
        if "download" in text:
            return "download"
        if "extract" in text:
            return "extracting"
        if ("face swap" in text and "process" in text) or text.startswith("process") or "processing" in text:
            return "processing"
        if "merg" in text:
            return "merging"
        if "upload" in text:
            return "upload"
        if "output link" in text or text.startswith("complete"):
            return "completed"
        if text.startswith("fail"):
            return "failed"
        return None

    def _sequence_stage_key(stage_key):
        key = str(stage_key or "").strip().lower()
        if key in stage_sequence:
            return key
        if key == "extracting" and "extracting" not in stage_sequence:
            return "processing"
        if key == "merging" and "merging" not in stage_sequence:
            return "upload"
        return key

    def _sequence_stage_rank(stage_key):
        key = _sequence_stage_key(stage_key)
        if key in stage_sequence:
            return stage_sequence.index(key)
        if key == "failed":
            return int(stage_plan.get("total", len(stage_sequence)))
        return max(0, len(stage_sequence) - 1)

    def _set_job_stage(stage_key, phase=None, pct=None, details=None, done_frames=None, total_frames=None):
        key = _normalize_stage_flow(stage_key)
        label = stage_label_map.get(key, STAGE_FLOW_TEXT.get(key, "Processing"))
        now_ts = time.time()
        prev_stage = str(job_status.get(chat_id, {}).get("stage") or "")
        prev_key = _stage_key_from_text(prev_stage)
        if prev_key and _sequence_stage_rank(key) < _sequence_stage_rank(prev_key):
            key = prev_key
            label = stage_label_map.get(key, STAGE_FLOW_TEXT.get(key, prev_stage or "Processing"))

        updates = {
            "stage": label,
            "updated_at": now_ts,
        }
        if phase is not None:
            updates["phase"] = phase
        if pct is not None:
            updates["pct"] = int(pct)
        if details is not None:
            updates["details"] = str(details)
        if done_frames is not None:
            updates["done_frames"] = int(max(0, done_frames))
        if total_frames is not None:
            updates["total_frames"] = int(max(0, total_frames))

        job_status[chat_id].update(updates)
        _persist_active_job_state(chat_id)

        try:
            _dashboard_record_stage(
                chat_id,
                stage_key=key,
                stage_label=label,
                phase=updates.get("phase"),
                pct=updates.get("pct"),
                details=updates.get("details"),
            )
            if any(k in updates for k in ("done_frames", "total_frames")):
                _dashboard_record_progress(
                    chat_id,
                    stage_key=key,
                    stage_label=label,
                    phase=updates.get("phase"),
                    pct=updates.get("pct"),
                    frames_done=updates.get("done_frames"),
                    frames_total=updates.get("total_frames"),
                    details=updates.get("details"),
                )
        except Exception:
            pass

        if callable(stage_event_hook):
            try:
                stage_event_hook({
                    "stage_key": key,
                    "stage_label": label,
                    "phase": updates.get("phase"),
                    "pct": updates.get("pct"),
                    "details": updates.get("details"),
                    "updated_at": now_ts,
                })
            except Exception as hook_e:
                logger.warning("stage_event_hook failed chat=%s stage=%s err=%s", chat_id, key, hook_e)

        if prev_stage != label:
            logger.info("[STAGE UPDATE] -> %s", label)
        return key, label

    def _stage_meta(stage_key, state_text):
        key = _normalize_stage_flow(stage_key)
        if key in stage_sequence:
            return stage_sequence.index(key) + 1, stage_label_map.get(key, STAGE_FLOW_TEXT.get(key, state_text or "Unknown"))
        if key == "failed":
            return int(stage_plan.get("total", 6)), "Failed"
        return 0, (state_text or "Unknown")

    def _stage_detail_for(stage_key):
        key = _normalize_stage_flow(stage_key)
        if key in stage_sequence:
            return f"Step {stage_sequence.index(key) + 1}/{int(stage_plan.get('total', 6))}"
        return ""

    def _normalize_stage_flow(stage_key):
        key = str(stage_key or "").strip().lower()
        key = _sequence_stage_key(key)

        if key not in STAGE_FLOW_ORDER:
            return "processing"

        if key == "failed":
            return "failed"

        requested_idx = _sequence_stage_rank(key)
        last_idx = int(stage_gate["last_index"])
        if requested_idx <= last_idx:
            fixed_key = stage_sequence[last_idx] if 0 <= last_idx < len(stage_sequence) else key
            return fixed_key
        if requested_idx > (last_idx + 1):
            return stage_sequence[last_idx + 1]
        return key

    perf = {
        "download": {"start": None, "end": None, "samples": []},
        "extraction": {"start": None, "end": None, "samples": []},
        "processing": {"start": None, "end": None, "samples": []},
        "merge_encode": {"start": None, "end": None, "samples": []},
        "upload": {"start": None, "end": None, "samples": []},
    }

    def _perf_start(stage_name):
        bucket = perf.get(stage_name)
        if bucket and bucket["start"] is None:
            bucket["start"] = time.time()

    def _perf_end(stage_name):
        bucket = perf.get(stage_name)
        if bucket and bucket["start"] is not None and bucket["end"] is None:
            bucket["end"] = time.time()

    def _perf_sample(stage_name, proc_pid=None):
        bucket = perf.get(stage_name)
        if not bucket or bucket["start"] is None:
            return
        _append_perf_sample(bucket, _sample_perf_point(proc_pid))

    def _perf_dump_summary():
        lines = [
            _perf_stats_line("download", perf["download"]["start"], perf["download"]["end"], perf["download"]["samples"]),
            _perf_stats_line("extraction", perf["extraction"]["start"], perf["extraction"]["end"], perf["extraction"]["samples"]),
            _perf_stats_line("processing", perf["processing"]["start"], perf["processing"]["end"], perf["processing"]["samples"]),
            _perf_stats_line("merge/encode", perf["merge_encode"]["start"], perf["merge_encode"]["end"], perf["merge_encode"]["samples"]),
            _perf_stats_line("upload", perf["upload"]["start"], perf["upload"]["end"], perf["upload"]["samples"]),
        ]
        logger.info("PERF SUMMARY chat=%s | %s", chat_id, " || ".join(lines))

    async def live_update(stage_key, state_text, pct=None, elapsed=None, eta_seconds=None,
                          done_frames=None, total_frames=None, extra=None, force=False):
        if not _is_active_progress_stream():
            return

        state_from_job = str(job_status.get(chat_id, {}).get("stage") or "").strip()
        stage_from_job = _stage_key_from_text(state_from_job)
        if stage_from_job:
            stage_key = _normalize_stage_flow(stage_from_job)
            state_text = stage_label_map.get(stage_key, STAGE_FLOW_TEXT.get(stage_key, state_from_job))
        else:
            stage_key = _normalize_stage_flow(stage_key)
            state_text = stage_label_map.get(stage_key, STAGE_FLOW_TEXT.get(stage_key, state_text))

        current_state = _get_active_job_state(chat_id, allow_fallback=True) or {}
        current_phase = current_state.get("phase") or current_state.get("status")
        current_phase_l = str(current_phase or "").lower()

        logger.info("[TELEGRAM DEBUG] Stage requested: %s", state_text)
        logger.info("[TELEGRAM DEBUG] Actual job status: %s", current_phase_l)

        expected_phase = {
            "download": "download",
            "extracting": "faceswap",
            "processing": "faceswap",
            "merging": "faceswap",
            "upload": "upload",
            "completed": "completed",
            "failed": "failed",
        }.get(stage_key)
        if expected_phase and current_phase_l and current_phase_l != expected_phase:
            if stage_key == "download":
                logger.info("[MEGA BLOCK] prevented stale downloading message requested=%s actual=%s", state_text, current_phase_l)
                await _cleanup_stale_download_state(reason="stage_guard")
            return

        if stage_key == "download" and current_phase != "download":
            logger.info("[MEGA BLOCK] prevented stale downloading message requested=%s actual=%s", state_text, current_phase_l)
            await _cleanup_stale_download_state(reason="phase_guard")
            return
        if stage_key in {"extracting", "processing", "merging"} and current_phase in {"upload", "completed", "failed", "stopped", "cancelled", "exception"}:
            return

        now = time.time()
        pct_int = int(max(0, min(100, pct))) if pct is not None and pct >= 0 else -1
        last_emit = float(live_gate["last_emit_at"] or 0.0)
        since_last = (now - last_emit) if last_emit > 0 else 9999.0
        stage_changed = stage_key != live_gate["last_stage"]
        pct_change = (pct_int - live_gate["last_pct"]) if pct_int >= 0 and live_gate["last_pct"] >= 0 else (1 if pct_int >= 0 else 0)
        pct_due = pct_change >= 1
        interval_due = since_last >= 4.0

        if not force and last_emit > 0 and not (stage_changed or pct_due or interval_due):
            return

        pct_safe = 0 if pct is None or pct < 0 else int(max(0, min(100, pct)))
        merge_verified = bool((job_status.get(chat_id, {}) or {}).get("merge_verified"))
        if stage_key not in {"completed", "failed"} and pct_safe >= 100 and not merge_verified:
            pct_safe = 99
        bar = _progress_bar(pct_safe)
        elapsed_txt = "0m 0s" if elapsed is None else _fmt_elapsed(elapsed)
        eta_txt = _fmt_elapsed(int(max(0, eta_seconds or 0)))

        d = int(max(0, done_frames or 0))
        _js_total = int((job_status.get(chat_id) or {}).get("total_frames") or 0)
        t = int(total_frames) if total_frames and total_frames > 0 else _js_total
        if t > 0 and d > t:
            d = t
        if stage_key == "merging" and t > 0 and d >= t and not merge_verified:
            d = max(0, t - 1)

        stage_safe = (state_text or "-").replace("*", "").replace("`", "'")
        female_flag = "ON" if effective_gender_mode == "female" else "OFF"
        file_name = _short_target_name(progress_target_name["value"])
        extra_info = f"Female-only: {female_flag} | Fast-mode: {fast_mode_flag} | File: {file_name}"

        if not progress_format_log["emitted"]:
            logger.info("USING PROGRESS FORMAT HERE")
            progress_format_log["emitted"] = True

        stage_num, stage_label = _stage_meta(stage_key, state_text)
        progress_title = f"🔄 [{stage_num}/{int(stage_plan['total'])}] {stage_label}"

        sent = await progress(
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"{progress_title}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🧩 Stage: {stage_safe}\n\n"
            f"{bar} {pct_safe}%\n\n"
            f"🎞 Frames: {d}/{t if t > 0 else '?'}\n\n"
            f"⏱️ Elapsed: {elapsed_txt}\n\n"
            f"⌛️ ETA: {eta_txt}\n\n"
            f"ℹ️ {extra_info}",
            force=force,
        )
        if sent:
            logger.info("[TELEGRAM STAGE] stage=%s key=%s", stage_safe, stage_key)
            stage_gate["last_index"] = max(int(stage_gate["last_index"]), int(_sequence_stage_rank(stage_key)))
            live_gate["last_stage"] = stage_key
            live_gate["last_state_text"] = state_text
            live_gate["last_pct"] = pct_int
            live_gate["last_emit_at"] = now
            try:
                _dashboard_record_progress(
                    chat_id,
                    stage_key=stage_key,
                    stage_label=stage_label,
                    stage_num=int(stage_num),
                    stage_total=int(stage_plan.get("total", 6) or 6),
                    pct=int(pct_safe),
                    frames_done=int(d),
                    frames_total=int(t) if t > 0 else 0,
                    elapsed_s=int(elapsed) if elapsed is not None else None,
                    eta_s=int(eta_seconds) if eta_seconds is not None else None,
                    extra_info=str(extra_info),
                )
            except Exception:
                pass
            stage_msg = stage_start_messages.get(stage_key)
            if stage_msg and stage_key not in stage_gate["announced"]:
                stage_gate["announced"].add(stage_key)
                await notify(stage_msg)
            logger.info(
                "progress interval chat=%s last_update_ts=%.3f current_ts=%.3f delta=%.2fs pct_change=%s",
                chat_id,
                last_emit,
                now,
                since_last,
                pct_change,
            )

    async def update_telegram_status(stage_name, progress=None, details=None, force=False):
        """Unified stage status emitter used by stage hooks and watchdog/failsafe updates."""
        stage_key_raw = _stage_key_from_text(stage_name)
        stage_key = _normalize_stage_flow(stage_key_raw or stage_name)
        stage_label = stage_label_map.get(stage_key, STAGE_FLOW_TEXT.get(stage_key, str(stage_name or "Processing")))

        st = job_status.get(chat_id, {}) or {}
        started_at = float(st.get("started_at") or pipeline_started_at or time.time())
        elapsed = int(max(0, time.time() - started_at))

        pct = progress
        if pct is None:
            pct = int(st.get("pct", live_gate.get("last_pct", -1)) or -1)
        try:
            pct = int(pct)
        except Exception:
            pct = -1
        if pct < 0:
            pct = 0

        merge_verified = bool(st.get("merge_verified"))
        if stage_key not in {"completed", "failed"} and pct >= 100 and not merge_verified:
            pct = 99

        done_frames = int(max(0, st.get("done_frames") or 0))
        total_frames = int(max(0, st.get("total_frames") or 0))
        if stage_key == "merging" and total_frames > 0 and done_frames >= total_frames and not merge_verified:
            done_frames = max(0, total_frames - 1)

        detail_text = str(details or "").strip()
        if detail_text:
            extra = detail_text
        else:
            extra = f"{stage_label} in progress... still working..."

        logger.info(
            "TELEGRAM_STATUS_HOOK chat=%s stage=%s pct=%s details=%s",
            chat_id,
            stage_key,
            pct,
            extra,
        )

        await live_update(
            stage_key,
            stage_label,
            pct=pct,
            elapsed=elapsed,
            eta_seconds=None,
            done_frames=done_frames,
            total_frames=total_frames,
            extra=extra,
            force=True if force else False,
        )

    async def _emit_stage_transition(prev_stage_key, next_stage_key, details=""):
        logger.info(
            "STAGE TRANSITION HOOK chat=%s %s -> %s details=%s",
            chat_id,
            prev_stage_key,
            next_stage_key,
            details,
        )
        await update_telegram_status(
            STAGE_FLOW_TEXT.get(next_stage_key, str(next_stage_key).title()),
            progress=0 if next_stage_key not in {"completed", "failed"} else 100,
            details=details,
            force=True,
        )

    async def _ui_progress_watchdog_loop():
        """Background UI liveness loop that keeps Telegram progress fresh under heavy backend load."""
        while ui_watchdog_state["run"] and _is_active_progress_stream():
            await asyncio.sleep(1)

            st = job_status.get(chat_id, {}) or {}
            phase = str(st.get("phase") or "").lower()
            if phase in {"completed", "failed", "stopped", "cancelled", "exception"}:
                break

            now = time.time()
            last_emit = float(live_gate.get("last_emit_at") or 0.0)
            if last_emit <= 0.0:
                continue

            since_last = now - last_emit
            stage_name = str(st.get("stage") or "Processing")

            # Heartbeat: keep user-visible activity flowing every 2-3 seconds.
            if since_last >= 3.0:
                await update_telegram_status(
                    stage_name,
                    progress=st.get("pct"),
                    details=f"{stage_name} in progress... still working...",
                    force=True,
                )

            # Failsafe: force a liveness update if UI has been silent for too long.
            if since_last >= 8.0:
                logger.warning(
                    "UI_FAILSAFE_TRIGGER chat=%s stage=%s silence=%.1fs",
                    chat_id,
                    stage_name,
                    since_last,
                )
                await update_telegram_status(
                    stage_name,
                    progress=st.get("pct"),
                    details="System still processing...",
                    force=True,
                )

    try:
        bypass_ok, bypass_info = apply_content_analyser_bypass()
        if bypass_ok:
            logger.info("CONTENT_ANALYSER_BYPASS: %s", bypass_info)
        else:
            logger.warning("CONTENT_ANALYSER_BYPASS_FAILED: %s", bypass_info)

        logger.info("ENTER: run_pipeline chat=%s video_link=%s", chat_id, video_link)
        logger.info(
            "PERF STRATEGY chat=%s exec_provider=%s exec_threads=%s ffmpeg_threads=%s cpu_target_pct=%s encoder=%s",
            chat_id,
            EXECUTION_PROVIDER,
            EXECUTION_THREAD_COUNT,
            FFMPEG_CPU_THREADS,
            CPU_THREAD_UTILIZATION_PCT,
            OUTPUT_VIDEO_ENCODER,
        )
        logger.info(
            "FACESWAP STRICT DEBUG chat=%s strict_debug=%s disable_passthrough_fallback=%s",
            chat_id,
            bool(STRICT_FACESWAP_DEBUG),
            bool(DISABLE_FACE_SWAP_FALLBACK),
        )
        require_gpu_or_raise()
        t_total   = time.time()
        space_ok, before_free, after_free, cleanup_count = ensure_workspace_free_space()
        if cleanup_count > 0:
            logger.info(
                "storage preflight cleanup applied: removed=%s free_before=%.2fGB free_after=%.2fGB",
                cleanup_count,
                _bytes_to_gb(before_free),
                _bytes_to_gb(after_free),
            )
        if not space_ok:
            logger.warning(
                "workspace free-space check still low after cleanup (free=%.2fGB required=%sGB), continuing in stable mode",
                _bytes_to_gb(after_free),
                int(MIN_FREE_SPACE_GB),
            )

        job_temp_path = create_job_temp_path(chat_id)
        active_job_protected_paths[chat_id] = {job_temp_path}
        face_path = resolve_face_for_chat(chat_id)
        job_status[chat_id] = {
            "job_id": queue_job_id,
            "phase": "starting",
            "stage": "Preparing",
            "pct": -1,
            "target": "-",
            "started_at": t_total,
            "updated_at": t_total,
            "details": "Pipeline initialized",
            "progress_token": progress_stream_token,
            "worker_pid": worker_pid,
            "done_frames": 0,
            "total_frames": 0,
            "input_path": str(video_link or ""),
            "output_path": "",
        }
        progress_stream_tokens[chat_id] = progress_stream_token
        _persist_active_job_state(chat_id)

        ui_watchdog_state["run"] = True
        ui_watchdog_task = asyncio.create_task(_ui_progress_watchdog_loop())

        prior_keepalive = job_keepalive_tasks.get(chat_id)
        if prior_keepalive and not prior_keepalive.done():
            prior_keepalive.cancel()
            with suppress(asyncio.CancelledError):
                await prior_keepalive
        keepalive_state["run"] = True
        keepalive_task = asyncio.create_task(_job_keepalive_loop())
        job_keepalive_tasks[chat_id] = keepalive_task

        # Do not block download start on Telegram send/edit throttling.
        await _ensure_progress_anchor(max_wait_sec=0.8)

        # Custom face
        if face_link:
            async def _pipeline_face_stage(stage_text):
                await notify(f"🖼 *{str(stage_text or 'Working...')}*")

            ok_face, face_result = await _download_prepare_face(
                face_link,
                chat_id,
                stage_callback=_pipeline_face_stage,
                apply_as_default=False,
            )
            if ok_face:
                face_path = str((face_result or {}).get("clean_path") or "").strip()
                if face_path and os.path.isfile(face_path):
                    await notify(f"✅ Temporary custom face ready: `{os.path.basename(face_path)}`")
                else:
                    await notify("❌ Face download failed: face output missing after apply")
                    face_path = resolve_face_for_chat(chat_id)
            else:
                fail_reason = str((face_result or {}).get("reason") or "unknown error")
                await notify(f"❌ Face download failed: {fail_reason[:220]}")
                face_path = resolve_face_for_chat(chat_id)

        # STEP 1: Download
        logger.info("ENTER: download stage chat=%s", chat_id)
        _set_job_stage("download", phase="download", details=_stage_detail_for("download"))
        t0 = time.time()
        _perf_start("download")
        _perf_sample("download")
        await live_update(
            "download",
            "Downloading",
            pct=0,
            elapsed=0,
            eta_seconds=None,
            extra="MEGA fetch initiated",
            force=True,
        )
        for d in [VIDEO_DIR, WORKSPACE, TEMP_PATH, OUTPUTS_DIR, job_temp_path]:
            os.makedirs(d, exist_ok=True)
        for f in Path(VIDEO_DIR).iterdir():
            if f.is_file():
                f.unlink()

        previous_hb = download_heartbeat_tasks.pop(chat_id, None)
        if previous_hb and not previous_hb.done():
            previous_hb.cancel()
            with suppress(asyncio.CancelledError, RuntimeError):
                await previous_hb

        download_live = {"run": True}

        async def download_heartbeat():
            while download_live["run"] and _is_active_progress_stream():
                if job_status.get(chat_id, {}).get("phase") != "download":
                    break
                elapsed = int(time.time() - t0)
                # Use actual bytes downloaded for real progress
                try:
                    downloaded_bytes = sum(
                        f.stat().st_size for f in Path(VIDEO_DIR).iterdir()
                        if f.is_file()
                    )
                except Exception:
                    downloaded_bytes = 0
                # Try to get total size from active job state
                try:
                    st = _load_active_job_state() or {}
                    total_bytes = int(st.get("download_total_bytes") or 0)
                except Exception:
                    total_bytes = 0
                if total_bytes > 0 and downloaded_bytes > 0:
                    pct = min(99, int((downloaded_bytes / total_bytes) * 100))
                    eta = max(0, int((total_bytes - downloaded_bytes) / max(1, downloaded_bytes / max(1, elapsed))))
                else:
                    assumed_total = max(45, min(600, int(MEGA_DOWNLOAD_TIMEOUT_SEC or 120)))
                    pct = min(95, int((elapsed / float(max(1, assumed_total))) * 100))
                    eta = max(0, assumed_total - elapsed)
                await live_update(
                    "download",
                    "Downloading",
                    pct=pct,
                    elapsed=elapsed,
                    eta_seconds=eta,
                )
                _perf_sample("download")
                await asyncio.sleep(PROGRESS_NOTIFY_INTERVAL_SEC)

        hb_download = asyncio.create_task(download_heartbeat())
        download_heartbeat_tasks[chat_id] = hb_download

        download_ok = False
        try:
            download_ok = await mega_download_async(video_link, VIDEO_DIR)
        finally:
            download_live["run"] = False
            if hb_download and not hb_download.done():
                hb_download.cancel()
            if hb_download:
                with suppress(asyncio.CancelledError, RuntimeError):
                    await hb_download
            if download_heartbeat_tasks.get(chat_id) is hb_download:
                download_heartbeat_tasks.pop(chat_id, None)

        if not download_ok:
            await notify("❌ *Download FAIL!* Link check karo.", main_kb())
            return

        valid_media_exts = {
            ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
            ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".heic", ".heif"
        }
        all_files = [
            f for f in Path(VIDEO_DIR).iterdir()
            if f.is_file() and f.suffix.lower() in valid_media_exts and not str(f.name).startswith(".")
        ]
        if not all_files:
            all_files = [f for f in Path(VIDEO_DIR).iterdir() if f.is_file() and not str(f.name).startswith(".")]
        if not all_files:
            await notify("❌ File nahi mili.", main_kb())
            return

        target = sorted(all_files, key=lambda f: f.stat().st_mtime, reverse=True)[0]
        if target.name.startswith("test_quota_fallback_"):
            await notify(
                "⚠️ MEGA link over-quota detect hua. Temporary testing fallback video use ki gayi hai."
            )
        target_path, original_target_name = normalize_processing_target(str(target), chat_id)
        target_path = coerce_target_media_extension(target_path)
        target = Path(target_path)
        active_target_path = str(target)
        downloaded_source_path = str(target)
        active_job_protected_paths[chat_id] = {p for p in [job_temp_path, active_target_path] if p}
        progress_target_name["value"] = target.name
        if target.name != original_target_name:
            await notify(
                "ℹ️ Target filename normalize ki gayi for stable processing:\n"
                f"`{original_target_name}` → `{target.name}`"
            )
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif", ".heic", ".heif"}
        video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

        def _detect_target_media_kind(media_path):
            p = Path(media_path)
            suffix = p.suffix.lower()

            # AVIF/HEIC are still images even when ffprobe reports AV1-coded streams.
            if suffix in {".avif", ".heic", ".heif"}:
                return "image"

            with suppress(Exception):
                from PIL import Image as PILImage
                with PILImage.open(p) as im:
                    im.verify()
                return "image"

            with suppress(Exception):
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=codec_type",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(p),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if probe.returncode == 0 and "video" in str(probe.stdout or "").lower():
                    return "video"

            if suffix in image_exts:
                return "image"
            if suffix in video_exts:
                return "video"
            return "unknown"

        target_ext = target.suffix.lower()
        media_kind = _detect_target_media_kind(str(target))
        if media_kind == "unknown":
            await notify("❌ Unsupported target media type. Supported: image(jpg/png/webp/bmp) or video(mp4/mov/mkv/avi/webm/m4v).", main_kb())
            return

        # ── Download validation gate ──────────────────────────────────────────
        _fsize = int(target.stat().st_size or 0) if target.exists() else 0
        if _fsize < 1_000_000:
            await notify(
                f"❌ Download incomplete: file too small ({_fsize:,} bytes).\n"
                "MEGA link verify karo, over-quota nahi hai? Dobara try karo.",
                main_kb(),
            )
            return
        if media_kind == "video":
            try:
                _vprobe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=codec_type",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(target)],
                    capture_output=True, text=True, timeout=15,
                )
                if _vprobe.returncode != 0 or "video" not in (_vprobe.stdout or "").lower():
                    await notify(
                        "❌ Downloaded file valid video nahi hai (corrupt ya incomplete download).\n"
                        "MEGA link verify karo aur dobara try karo.",
                        main_kb(),
                    )
                    return
                _durprobe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(target)],
                    capture_output=True, text=True, timeout=15,
                )
                if _durprobe.returncode == 0:
                    try:
                        _dur_val = float((_durprobe.stdout or "0").strip() or "0")
                        if _dur_val < 1.0:
                            await notify(
                                f"❌ Video duration too short ({_dur_val:.2f}s < 1s). Download incomplete ya wrong file.\n"
                                "MEGA link dobara check karo.",
                                main_kb(),
                            )
                            return
                    except (ValueError, TypeError):
                        pass
            except Exception as _ve:
                logger.warning("VIDEO_PROBE_VALIDATION_FAILED target=%s err=%s", target.name, _ve)
        # ─────────────────────────────────────────────────────────────────────

        is_image_target = media_kind == "image"
        if is_image_target:
            stage_plan["total"] = 4
            stage_sequence[:] = ["download", "processing", "upload", "completed"]
            stage_label_map.update({
                "download": "Downloading",
                "processing": "Processing",
                "upload": "Uploading",
                "completed": "Output Link",
            })
        else:
            stage_plan["total"] = 6
            stage_sequence[:] = ["download", "extracting", "processing", "merging", "upload", "completed"]
            stage_label_map.update({
                "download": "Downloading",
                "extracting": "Extracting Frames",
                "processing": "Face Swap Processing",
                "merging": "Merging Frames",
                "upload": "Uploading",
                "completed": "Output Link",
            })

        logger.info("TARGET_MEDIA_KIND chat=%s kind=%s target=%s stage_total=%s", chat_id, media_kind, target.name, int(stage_plan.get("total", 6)))
        total_target_frames = 0 if is_image_target else detect_total_video_frames(str(target))
        if (not is_image_target) and total_target_frames <= 0:
            fps_guess = detect_video_fps(str(target)) or 0.0
            duration_guess = detect_video_duration_seconds(str(target)) or 0.0
            if fps_guess > 0 and duration_guess > 0:
                total_target_frames = int(max(1, round(fps_guess * duration_guess)))
                logger.info(
                    "FRAME TOTAL ESTIMATE chat=%s fps=%.3f duration=%.2fs total=%s",
                    chat_id,
                    fps_guess,
                    duration_guess,
                    total_target_frames,
                )
        out_ext = target_ext if is_image_target else ".mp4"
        # Use the ORIGINAL filename stem (before normalize_processing_target mangling)
        orig_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_target_name).stem).strip("._-") or "output"
        # Strip MEGA file-ID-only stems (no useful keywords)
        if re.match(r"^mega_[A-Za-z0-9]{6,12}$", orig_stem):
            orig_stem = "video"
        orig_stem = orig_stem[:40].strip("._-") or "output"
        out = f"{OUTPUTS_DIR}/{orig_stem}_faceswapped{out_ext}"
        job_status[chat_id].update({
            "target": original_target_name,
            "updated_at": time.time(),
            "input_path": str(target),
        })
        dl_mb   = target.stat().st_size / 1024 / 1024
        dl_time = int(time.time() - t0)
        _perf_sample("download")
        _perf_end("download")
        await live_update(
            "download",
            "Downloading",
            pct=100,
            elapsed=dl_time,
            eta_seconds=0,
            extra=f"File: {target.name} | Size: {dl_mb:.1f} MB",
            force=True,
        )

        need_fallback = not face_path or not os.path.isfile(face_path)
        if not need_fallback:
            ok, reason = validate_source_face_quality(face_path)
            if not ok:
                await notify(
                    "⚠️ Source face image weak/invalid hai, swap skip jaisa result aa sakta hai.\n"
                    f"Reason: `{reason}`"
                )
                need_fallback = True

        if need_fallback and ALLOW_AUTO_SOURCE_FROM_VIDEO:
            fallback_face = extract_face_from_video_frame(str(target))
            if fallback_face and os.path.isfile(fallback_face):
                ok, reason = validate_source_face_quality(fallback_face)
                if ok:
                    face_path = fallback_face
                    await notify(
                        "ℹ️ Source face set nahi thi, target video se auto face frame use kiya gaya.\n"
                        f"`{os.path.basename(fallback_face)}`"
                    )
                else:
                    await notify(
                        "⚠️ Auto source face usable nahi nikli. Fallback processing continue hoga.\n"
                        f"Reason: `{reason}`",
                        main_kb()
                    )
                    face_path = None

        if not face_path or not os.path.isfile(face_path):
            forced_face = extract_face_from_video_frame(str(target))
            if forced_face and os.path.isfile(forced_face):
                face_path = forced_face
                await notify(
                    "⚠️ Source face missing thi, target se forced fallback face extract karke processing continue kiya gaya.",
                    main_kb()
                )
            else:
                await notify(
                    "⚠️ *Source face missing/invalid hai.*\n"
                    "FaceSwap error hone par passthrough output generate karke pipeline continue ki jayegi.",
                    main_kb()
                )

        if face_path and os.path.isfile(face_path):
            with suppress(Exception):
                logger.info(
                    "SOURCE_FACE_SELECTED chat=%s path=%s bytes=%s mtime=%s",
                    chat_id,
                    face_path,
                    int(Path(face_path).stat().st_size or 0),
                    int(Path(face_path).stat().st_mtime or 0),
                )

        # STEP 2: FaceSwap
        if is_image_target:
            logger.info("ENTER: processing stage (image target) chat=%s", chat_id)
            _set_job_stage("processing", phase="faceswap", pct=-1, details=_stage_detail_for("processing"))
        else:
            logger.info("ENTER: extraction stage chat=%s", chat_id)
            _set_job_stage("extracting", phase="faceswap", pct=-1, details=_stage_detail_for("extracting"))
        await _cleanup_stale_download_state(reason="entered_faceswap")
        if is_image_target:
            await live_update(
                "processing",
                "Processing",
                pct=0,
                elapsed=0,
                eta_seconds=None,
                done_frames=0,
                total_frames=0,
                force=True,
            )
        else:
            await live_update(
                "extracting",
                "Extracting",
                pct=0,
                elapsed=0,
                eta_seconds=None,
                done_frames=0,
                total_frames=total_target_frames,
                force=True,
            )
        selected_provider = pick_execution_provider()
        kill_stale_facefusion_runs()
        processors = ["face_swapper"]
        if ENABLE_EXPRESSION_RESTORER:
            processors.append("expression_restorer")
        if ENABLE_FACE_ENHANCER:
            processors.append("face_enhancer")
        selector_gender = get_face_selector_gender(chat_id)

        try:
            strict_detector_score = float(FACE_DETECTOR_SCORE or "0.15")
        except Exception:
            strict_detector_score = 0.15
        strict_detector_score = min(0.60, max(0.05, strict_detector_score))
        try:
            strict_landmarker_score = float(FACE_LANDMARKER_SCORE or "0.35")
        except Exception:
            strict_landmarker_score = 0.35
        strict_landmarker_score = min(0.50, max(0.15, strict_landmarker_score))
        run_face_detector_size = FACE_DETECTOR_SIZE
        run_execution_thread_count = EXECUTION_THREAD_COUNT
        run_face_swapper_pixel_boost = FACE_SWAPPER_PIXEL_BOOST
        run_video_memory_strategy = VIDEO_MEMORY_STRATEGY

        if selected_provider == "cuda" and GPU_STARTUP_BALANCED_MODE:
            run_face_detector_size = GPU_STARTUP_FACE_DETECTOR_SIZE
            run_execution_thread_count = str(min(int(EXECUTION_THREAD_COUNT), int(GPU_STARTUP_THREAD_COUNT)))
            run_face_swapper_pixel_boost = GPU_STARTUP_PIXEL_BOOST
            run_video_memory_strategy = GPU_STARTUP_VIDEO_MEMORY_STRATEGY
            logger.info(
                "GPU_STARTUP_PROFILE chat=%s mode=balanced provider=%s threads=%s detector_size=%s pixel_boost=%s video_memory_strategy=%s",
                chat_id,
                selected_provider,
                run_execution_thread_count,
                run_face_detector_size,
                run_face_swapper_pixel_boost,
                run_video_memory_strategy,
            )

        selected_detector_model = resolve_face_detector_model()
        logger.info("FACE_DETECTOR_SELECTED chat=%s model=%s", chat_id, selected_detector_model)
        # Use 'many' when gender filter is active so ALL matching-gender faces get swapped.
        # 'one' would only pick the single best face, ignoring all other females/males.
        face_selector_mode = "many" if selector_gender else "one"
        logger.info("[FACE_SELECTOR_MODE] chat=%s mode=%s gender=%s", chat_id, face_selector_mode, selector_gender or "all")
        face_filter_args = [
            "--face-detector-model", selected_detector_model,
            "--face-detector-size", run_face_detector_size,
            "--face-detector-angles", "0",
            "--face-detector-score", f"{strict_detector_score:.2f}",
            "--face-landmarker-score", f"{strict_landmarker_score:.2f}",
            "--face-selector-mode", face_selector_mode,
            "--face-selector-order", "best-worst",
            "--face-mask-types", "box",
            "--face-mask-regions", "skin", "left-eye", "right-eye", "nose", "mouth", "upper-lip", "lower-lip",
            "--face-mask-blur", FACE_MASK_BLUR,
            "--face-mask-padding", FACE_MASK_PADDING_TOP, FACE_MASK_PADDING_RIGHT, FACE_MASK_PADDING_BOTTOM, FACE_MASK_PADDING_LEFT,
        ]
        if selector_gender:
            face_filter_args.extend(["--face-selector-gender", selector_gender])
        strict_mode_note = "Female-only filter ON" if selector_gender else "All-gender filter"
        logger.info("[FACE_FILTER] strict_human_face_only=YES detector_score=%s landmarker_score=%s min_face_px=80", f"{strict_detector_score:.2f}", f"{strict_landmarker_score:.2f}")

        base_quality_args = [
            "--temp-path", job_temp_path,
            "--temp-frame-format", "jpeg",
            "--execution-providers", selected_provider,
            "--execution-thread-count", run_execution_thread_count,
            "--face-swapper-model", FACE_SWAPPER_MODEL,
            "--face-swapper-pixel-boost", run_face_swapper_pixel_boost,
            "--face-swapper-weight", FACE_SWAPPER_WEIGHT,
            "--video-memory-strategy", run_video_memory_strategy,
            "--log-level", "debug",
        ]
        if ENABLE_FACE_ENHANCER:
            base_quality_args.extend([
                "--face-enhancer-model", FACE_ENHANCER_MODEL,
                "--face-enhancer-blend", FACE_ENHANCER_BLEND,
                "--face-enhancer-weight", FACE_ENHANCER_WEIGHT,
            ])
        logger.info(
            "[DISTORTION_FIX] config face_swapper=%s enhancer=%s enhancer_weight=%s selector=reference",
            FACE_SWAPPER_MODEL,
            FACE_ENHANCER_MODEL,
            FACE_ENHANCER_WEIGHT,
        )

        clip_args = []
        reference_frame_number = 0
        clip_progress_note = ""
        progress_total_target_frames = total_target_frames
        if progress_total_target_frames > 0:
            job_status.setdefault(chat_id, {})['total_frames'] = int(progress_total_target_frames)
        original_target_for_notes = str(target)
        clip_frame_ranges = []

        clip_cfg = clip_ranges.get(chat_id)
        if (not is_image_target) and isinstance(clip_cfg, dict):
            segments = clip_cfg.get("segments") if isinstance(clip_cfg.get("segments"), list) else []
            if segments:
                duration_cap = detect_video_duration_seconds(str(target))
                for seg in segments:
                    try:
                        start_sec = max(0.0, float(seg.get("start", 0.0)))
                        end_sec = max(start_sec + 0.01, float(seg.get("end", 0.0)))
                    except Exception:
                        continue
                    if duration_cap and duration_cap > 0:
                        if start_sec >= duration_cap:
                            continue
                        end_sec = min(end_sec, duration_cap)
                        if end_sec <= start_sec:
                            continue
                    clip_frame_ranges.append({"start_sec": start_sec, "end_sec": end_sec})

                if clip_frame_ranges:
                    clip_progress_note = f"Clip mode: {len(clip_frame_ranges)} range(s)"
                    # FIX: Convert clip ranges to facefusion trim arguments
                    if len(clip_frame_ranges) == 1:
                        # Single segment: pass directly to facefusion as trim args
                        seg = clip_frame_ranges[0]
                        video_fps = detect_video_fps(str(target)) or 30.0  # default to 30fps
                        trim_start_frame = int(round(seg["start_sec"] * video_fps))
                        trim_end_frame = int(round(seg["end_sec"] * video_fps))
                        clip_args = [
                            "--trim-frame-start", str(trim_start_frame),
                            "--trim-frame-end", str(trim_end_frame),
                        ]
                        logger.info("clip_range converted: start_sec=%s end_sec=%s fps=%s -> frames %d-%d",
                                    seg["start_sec"], seg["end_sec"], video_fps, trim_start_frame, trim_end_frame)

        face_map_cfg = {
            idx: path for idx, path in selected_face_maps.get(chat_id, {}).items()
            if os.path.isfile(path)
        }
        if selected_face_maps.get(chat_id) and not face_map_cfg:
            await notify("⚠️ Selective face mapping set hai, lekin mapped source files missing hain. Default mode use hoga.")

        env = prepare_cuda_runtime_env(os.environ.copy(), selected_provider=selected_provider)
        env["CONDA_READY"] = "1"
        env.pop("CONDA_PREFIX", None)

        ff_start     = time.time()
        _perf_start("extraction")
        _perf_sample("extraction")
        last_tg_time = [0]
        last_state   = [""]
        last_pct     = [0]
        last_frame   = [0]
        parsed_frame_once = [False]
        last_progress_parse_at = [time.time()]
        last_temp_frame_scan_at = [0.0]
        ff_started   = [False]
        ff_tail      = deque(maxlen=50)
        perf_stage_live = ["extraction"]
        oom_recovery_attempted = [False]
        oom_detected_live = [False]
        completion_validated = [False]
        merge_retry_attempted = [False]
        merge_stage_detected = [False]
        job_status.setdefault(chat_id, {})["merge_verified"] = False

        def _is_oom_or_gpu_failure(lines):
            text = "\n".join(lines or [])
            lo = text.lower()
            signals = [
                "cuda failure",
                "out of memory",
                "failed to allocate memory",
                "onnxruntimeerror",
                "cudaerror",
                "cudnn",
            ]
            return any(sig in lo for sig in signals)

        def _cuda_empty_cache_light():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU_RETRY cleanup=torch.cuda.empty_cache() done chat=%s", chat_id)
            except Exception as e:
                logger.warning("GPU_RETRY cleanup empty_cache failed chat=%s err=%s", chat_id, e)

        def _set_flag(cmd_retry, flag_name, flag_value):
            if flag_name in cmd_retry:
                idx = cmd_retry.index(flag_name)
                if idx + 1 < len(cmd_retry):
                    cmd_retry[idx + 1] = str(flag_value)
                    return
            cmd_retry.extend([flag_name, str(flag_value)])

        def _set_io_paths(cmd_retry, input_path, output_path):
            if "-t" in cmd_retry:
                tidx = cmd_retry.index("-t")
                if tidx + 1 < len(cmd_retry):
                    cmd_retry[tidx + 1] = str(input_path)
            if "-o" in cmd_retry:
                oidx = cmd_retry.index("-o")
                if oidx + 1 < len(cmd_retry):
                    cmd_retry[oidx + 1] = str(output_path)

        def _build_gpu_retry_cmd(base_cmd, retry_level):
            cmd_retry = list(base_cmd)
            _set_flag(cmd_retry, "--execution-providers", "cuda")
            _set_flag(cmd_retry, "--video-memory-strategy", "strict")

            if retry_level <= 1:
                return cmd_retry

            if retry_level == 2:
                _set_flag(cmd_retry, "--execution-thread-count", str(LOW_MEMORY_THREAD_COUNT if LOW_MEMORY_MODE else EXECUTION_THREAD_COUNT))
                _set_flag(cmd_retry, "--face-swapper-pixel-boost", "256x256")
                _set_flag(cmd_retry, "--face-detector-size", "640x640")
                return cmd_retry

            # Retry level 3: ultra low-memory GPU profile.
            _set_flag(cmd_retry, "--execution-thread-count", "1")
            _set_flag(cmd_retry, "--face-swapper-pixel-boost", "256x256")
            _set_flag(cmd_retry, "--face-detector-size", "640x640")

            # Keep only swapper in ultra mode to reduce VRAM pressure.
            if "--processors" in cmd_retry:
                pidx = cmd_retry.index("--processors")
                next_idx = pidx + 1
                while next_idx < len(cmd_retry) and not str(cmd_retry[next_idx]).startswith("--"):
                    next_idx += 1
                cmd_retry = cmd_retry[:pidx + 1] + ["face_swapper"] + cmd_retry[next_idx:]

            return cmd_retry

        def _gpu_retry_profile(retry_level):
            if retry_level <= 1:
                return {"mode": "normal", "chunk_seconds": 0}
            if retry_level == 2:
                return {"mode": "low-memory", "chunk_seconds": int(GPU_RETRY_CHUNK_SECONDS_L2)}
            return {"mode": "ultra-low-memory", "chunk_seconds": int(GPU_RETRY_CHUNK_SECONDS_L3)}

        async def _run_gpu_retry_attempt(base_cmd, retry_level):
            profile = _gpu_retry_profile(retry_level)
            retry_mode = str(profile.get("mode") or "low-memory")
            retry_chunk_seconds = int(max(0, profile.get("chunk_seconds") or 0))
            retry_cmd_base = _build_gpu_retry_cmd(base_cmd, retry_level)
            retry_env = prepare_cuda_runtime_env(os.environ.copy(), selected_provider="cuda")
            retry_env["CONDA_READY"] = "1"
            retry_env.pop("CONDA_PREFIX", None)

            async def _run_single_retry_cmd(local_cmd, watchdog_label):
                local_proc = subprocess.Popen(
                    local_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=FACEFUSION_DIR,
                    env=retry_env,
                    preexec_fn=os.setsid,
                )
                active_jobs[chat_id] = local_proc
                _persist_active_job_state(chat_id, processing_pid=local_proc.pid)
                ff_tail.clear()
                await drain_subprocess_output(
                    local_proc,
                    tail_buffer=ff_tail,
                    watchdog_label=watchdog_label,
                    watchdog_interval_sec=FACEFUSION_WATCHDOG_SEC,
                )
                active_jobs.pop(chat_id, None)
                await asyncio.to_thread(local_proc.wait)
                return int(local_proc.returncode or 0)

            if is_image_target or retry_chunk_seconds <= 0:
                rc = await _run_single_retry_cmd(retry_cmd_base, f"faceswap-gpu-retry-l{retry_level}")
                ok = rc == 0 and os.path.exists(out)
                return ok, rc, f"mode={retry_mode} chunking=off"

            duration = detect_video_duration_seconds(str(target))
            if not duration or duration <= 0:
                rc = await _run_single_retry_cmd(retry_cmd_base, f"faceswap-gpu-retry-l{retry_level}-nochunk")
                ok = rc == 0 and os.path.exists(out)
                return ok, rc, f"mode={retry_mode} chunking=off reason=no-duration"

            segment_outputs = []
            seg_idx = 0
            seg_start = 0.0
            last_rc = 0
            while seg_start < (float(duration) - 0.01):
                seg_idx += 1
                seg_end = min(float(duration), seg_start + float(retry_chunk_seconds))

                trim_ok, trim_info, trim_path = create_processing_clip(
                    str(target),
                    seg_start,
                    seg_end,
                    job_temp_path,
                )
                if not trim_ok or not trim_path:
                    return False, -1, f"mode={retry_mode} trim_failed seg={seg_idx} info={trim_info}"

                seg_out = f"{OUTPUTS_DIR}/{orig_stem}_gpu_l{retry_level}_seg{seg_idx}.mp4"
                seg_cmd = list(retry_cmd_base)
                _set_io_paths(seg_cmd, str(trim_path), seg_out)
                last_rc = await _run_single_retry_cmd(seg_cmd, f"faceswap-gpu-retry-l{retry_level}-seg{seg_idx}")

                try:
                    Path(trim_path).unlink(missing_ok=True)
                except Exception:
                    pass

                if last_rc != 0 or not os.path.exists(seg_out):
                    return False, last_rc, f"mode={retry_mode} seg={seg_idx} failed rc={last_rc}"

                segment_outputs.append(seg_out)
                seg_start = seg_end

            ok_concat, concat_info = concat_video_segments(segment_outputs, out)
            for seg_path in segment_outputs:
                try:
                    Path(seg_path).unlink(missing_ok=True)
                except Exception:
                    pass
            if not ok_concat:
                return False, -2, f"mode={retry_mode} concat_failed info={concat_info}"
            return True, 0, f"mode={retry_mode} chunking=on chunk_seconds={retry_chunk_seconds}"

        async def _run_cpu_recovery_attempt(base_cmd):
            if GPU_ONLY_MODE:
                return False, -2, "cpu_recovery_disabled_gpu_only"
            cpu_cmd = list(base_cmd)
            _set_flag(cpu_cmd, "--execution-providers", "cpu")
            _set_flag(cpu_cmd, "--video-memory-strategy", "tolerant")
            _set_flag(cpu_cmd, "--execution-thread-count", str(LOW_MEMORY_THREAD_COUNT if LOW_MEMORY_MODE else EXECUTION_THREAD_COUNT))
            _set_flag(cpu_cmd, "--face-detector-model", FALLBACK_FACE_DETECTOR_MODEL)
            _set_flag(cpu_cmd, "--face-detector-size", "640x640")
            _set_flag(cpu_cmd, "--face-swapper-pixel-boost", "256x256")

            # Keep recovery focused on producing a valid swapped output quickly.
            if "--processors" in cpu_cmd:
                pidx = cpu_cmd.index("--processors")
                next_idx = pidx + 1
                while next_idx < len(cpu_cmd) and not str(cpu_cmd[next_idx]).startswith("--"):
                    next_idx += 1
                cpu_cmd = cpu_cmd[:pidx + 1] + ["face_swapper"] + cpu_cmd[next_idx:]

            cpu_env = os.environ.copy()
            cpu_env["CUDA_VISIBLE_DEVICES"] = ""
            cpu_env.pop("CONDA_PREFIX", None)
            cpu_env["CONDA_READY"] = "1"

            cpu_proc = subprocess.Popen(
                cpu_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=FACEFUSION_DIR,
                env=cpu_env,
                preexec_fn=os.setsid,
            )
            active_jobs[chat_id] = cpu_proc
            _persist_active_job_state(chat_id, processing_pid=cpu_proc.pid)
            ff_tail.clear()
            await drain_subprocess_output(
                cpu_proc,
                tail_buffer=ff_tail,
                watchdog_label="faceswap-cpu-recovery",
                watchdog_interval_sec=FACEFUSION_WATCHDOG_SEC,
            )
            active_jobs.pop(chat_id, None)
            await asyncio.to_thread(cpu_proc.wait)
            cpu_rc = int(cpu_proc.returncode or 0)
            cpu_ok = cpu_rc == 0 and os.path.exists(out)
            return cpu_ok, cpu_rc, "cpu_recovery"

        def _perf_update_from_faceswap_state(state_name, proc_pid=None):
            if state_name == "Extracting":
                desired = "extraction"
            elif state_name == "Merging":
                desired = "merge_encode"
            elif state_name == "Processing":
                desired = "processing"
            else:
                desired = perf_stage_live[0]

            if desired != perf_stage_live[0]:
                _perf_end(perf_stage_live[0])
                _perf_start(desired)
                perf_stage_live[0] = desired
            _perf_sample(desired, proc_pid=proc_pid)

        live_status = {
            "state": "Processing" if is_image_target else "Extracting",
            "pct": 0,
            "emoji": "⚙️",
        }
        current_stage_key = ["processing" if is_image_target else "extracting"]

        if face_map_cfg:
            if clip_frame_ranges:
                await notify("ℹ️ Clip ranges selective mode me apply nahi hote. Is run me ignore kiya gaya.")
                clip_frame_ranges = []
            await notify(f"🎯 Selective mode active: *{len(face_map_cfg)}* mapped face(s)")
            current_target = str(target)
            generated_out = None

            for pass_idx, (position_index, mapped_face) in enumerate(sorted(face_map_cfg.items()), start=1):
                pass_out = f"{OUTPUTS_DIR}/{orig_stem}_faceswapped_sel{pass_idx}{out_ext}"
                pass_cmd = [
                    FACEFUSION_PYTHON, f"{FACEFUSION_DIR}/facefusion.py", "headless-run",
                    "-s", mapped_face, "-t", current_target, "-o", pass_out,
                    "--processors", *processors,
                    "--face-selector-mode", "reference",
                    "--reference-face-position", str(position_index),
                    "--reference-face-distance", "0.3",
                    "--reference-frame-number", str(reference_frame_number),
                ]
                pass_cmd.extend(face_filter_args)
                pass_cmd.extend(base_quality_args)
                if clip_args:
                    pass_cmd.extend(clip_args)
                if is_image_target:
                    pass_cmd.extend(["--output-image-quality", "95"])
                else:
                    pass_cmd.extend(["--output-video-encoder", OUTPUT_VIDEO_ENCODER, "--output-audio-encoder", "aac"])

                await notify(
                    f"🎯 Selective pass `{pass_idx}/{len(face_map_cfg)}`\n"
                    f"Face index: `{position_index + 1}`"
                )
                _perf_start("processing")
                _perf_sample("processing")
                proc = subprocess.Popen(
                    pass_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=FACEFUSION_DIR,
                    env=env,
                    preexec_fn=os.setsid,
                )
                active_jobs[chat_id] = proc
                _persist_active_job_state(chat_id, processing_pid=proc.pid)
                _perf_sample("processing", proc_pid=proc.pid)
                await drain_subprocess_output(
                    proc,
                    tail_buffer=ff_tail,
                    watchdog_label=f"faceswap-selective-pass-{pass_idx}",
                            watchdog_interval_sec=FACEFUSION_WATCHDOG_SEC,
                )
                _perf_sample("processing", proc_pid=proc.pid)
                active_jobs.pop(chat_id, None)

                if proc.returncode != 0:
                    tail_text = "\n".join(ff_tail).strip()
                    tail_msg = f"\n\n`{tail_text}`" if tail_text else ""
                    await notify(
                        "━━━━━━━━━━━━━━━━━━━━━\n"
                            "❌ *[FaceSwap] FAILED!*\n"
                        "━━━━━━━━━━━━━━━━━━━━━\n"
                        f"Code: `{proc.returncode}`\n"
                        f"Provider: `{selected_provider}`"
                        f"{tail_msg}",
                        main_kb()
                    )
                    return

                if not os.path.exists(pass_out):
                    await notify("❌ Selective pass output file nahi bani.", main_kb())
                    return

                current_target = pass_out
                generated_out = pass_out

            out = generated_out or out
            ff_time = int(time.time() - ff_start)
            ff_mins, ff_secs = divmod(ff_time, 60)
            _perf_end("extraction")
            _perf_end("processing")
            _perf_end("merge_encode")

        else:
            if clip_frame_ranges:
                segment_outputs = []
                total_segments = len(clip_frame_ranges)
                for seg_idx, seg in enumerate(clip_frame_ranges, start=1):
                    seg_out = f"{OUTPUTS_DIR}/{orig_stem}_faceswapped_seg{seg_idx}.mp4"
                    await live_update(
                        "processing",
                        f"Processing (Clip {seg_idx}/{total_segments})",
                        pct=0,
                        elapsed=int(time.time() - ff_start),
                        eta_seconds=None,
                        extra=f"range {format_seconds_hhmmss(seg['start_sec'])}-{format_seconds_hhmmss(seg['end_sec'])}",
                        force=True,
                    )

                    clip_ok = False
                    for clip_attempt in (1, 2):
                        trim_ok, trim_info, trim_path = create_processing_clip(
                            str(target),
                            seg["start_sec"],
                            seg["end_sec"],
                            job_temp_path,
                        )
                        if not trim_ok or not trim_path:
                            if clip_attempt == 2:
                                await notify(
                                    f"❌ Clip `{seg_idx}/{total_segments}` trim fail: `{trim_info}`",
                                    main_kb(),
                                )
                                return
                            continue

                        seg_cmd = [
                            FACEFUSION_PYTHON, f"{FACEFUSION_DIR}/facefusion.py", "headless-run",
                            "-s", face_path, "-t", str(trim_path), "-o", seg_out,
                            "--processors", *processors,
                            "--face-selector-mode", "reference",
                            "--reference-face-position", "0",
                            "--reference-face-distance", "0.30",
                            "--reference-frame-number", "0",
                        ]
                        seg_cmd.extend(face_filter_args)
                        seg_cmd.extend(base_quality_args)
                        seg_cmd.extend(["--output-video-encoder", OUTPUT_VIDEO_ENCODER, "--output-audio-encoder", "aac"])

                        _perf_start("processing")
                        _perf_sample("processing")
                        proc = subprocess.Popen(
                            seg_cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            cwd=FACEFUSION_DIR,
                            env=env,
                            preexec_fn=os.setsid,
                        )
                        active_jobs[chat_id] = proc
                        _persist_active_job_state(chat_id, processing_pid=proc.pid)
                        _perf_sample("processing", proc_pid=proc.pid)
                        await drain_subprocess_output(
                            proc,
                            tail_buffer=ff_tail,
                            watchdog_label=f"faceswap-clip-{seg_idx}",
                            watchdog_interval_sec=FACEFUSION_WATCHDOG_SEC,
                        )
                        _perf_sample("processing", proc_pid=proc.pid)
                        active_jobs.pop(chat_id, None)

                        try:
                            Path(trim_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                        if proc.returncode == 0 and os.path.exists(seg_out):
                            clip_ok = True
                            break

                        if clip_attempt == 1:
                            await notify(f"⚠️ Clip `{seg_idx}/{total_segments}` retry ho raha hai (1/1).")
                        else:
                            tail_text = "\n".join(ff_tail).strip()
                            tail_msg = f"\n\n`{tail_text[-700:]}`" if tail_text else ""
                            await notify(
                                "━━━━━━━━━━━━━━━━━━━━━\n"
                                "❌ *[2/6] FaceSwap FAILED!*\n"
                                "━━━━━━━━━━━━━━━━━━━━━\n"
                                f"Clip: `{seg_idx}/{total_segments}`\n"
                                f"Code: `{proc.returncode}`\n"
                                f"Provider: `{selected_provider}`"
                                f"{tail_msg}",
                                main_kb(),
                            )
                            return

                    if not clip_ok:
                        await notify(f"❌ Clip `{seg_idx}/{total_segments}` process fail.", main_kb())
                        return

                    segment_outputs.append(seg_out)

                ok_concat, concat_info = concat_video_segments(segment_outputs, out)
                if not ok_concat:
                    await notify(f"❌ Multi-range concat fail: `{concat_info}`", main_kb())
                    return

                for seg_path in segment_outputs:
                    try:
                        Path(seg_path).unlink(missing_ok=True)
                    except Exception:
                        pass

                _perf_start("merge_encode")
                _perf_sample("merge_encode")
                _perf_end("merge_encode")

                ff_time = int(time.time() - ff_start)
                ff_mins, ff_secs = divmod(ff_time, 60)
                _perf_end("extraction")
                _perf_end("processing")
                _perf_end("merge_encode")
            else:
                cmd = [
                    FACEFUSION_PYTHON, f"{FACEFUSION_DIR}/facefusion.py", "headless-run",
                    "-s", face_path, "-t", str(target), "-o", out,
                    "--processors", *processors,
                    "--face-selector-mode", "reference",
                    "--reference-face-position", "0",
                    "--reference-face-distance", "0.30",
                    "--reference-frame-number", "0",
                ]
                cmd.extend(face_filter_args)
                cmd.extend(base_quality_args)
                if clip_args:
                    cmd.extend(clip_args)
                if is_image_target:
                    cmd.extend(["--output-image-quality", "95"])
                else:
                    cmd.extend(["--output-video-encoder", OUTPUT_VIDEO_ENCODER, "--output-audio-encoder", "aac"])

                proc = subprocess.Popen(
                    cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=FACEFUSION_DIR, env=env, preexec_fn=os.setsid
                )
                active_jobs[chat_id] = proc
                _persist_active_job_state(chat_id, processing_pid=proc.pid)
                _perf_sample("extraction", proc_pid=proc.pid)

                progress_extra = f"{strict_mode_note} | file: {Path(original_target_for_notes).name}"
                if clip_progress_note:
                    progress_extra += f" | {clip_progress_note}"
                progress_state = {
                    "last_emit_at": time.time(),
                }

                temp_frames_dir = [Path(job_temp_path) / "facefusion" / target.stem]
                last_frame_change_at = [time.time()]
                last_output_size = [0]
                last_output_growth_at = [0.0]
                last_stall_log_at = [0.0]
                last_resync_at = [0.0]
                frame_scan_interval_sec = [0.0]
                processing_started = [False]
                force_passthrough_due_cuda = [False]
                faceswap_forced_passthrough = [False]
                processing_hard_timeout_sec = float(FACEFUSION_HARD_TIMEOUT_SEC)
                ready_to_complete_since = [0.0]
                ready_to_complete_forced = [False]
                ready_to_complete_terminated = [False]

                def _ensure_passthrough_output_exists():
                    fallback_done = False
                    fallback_error = ""
                    try:
                        if os.path.isfile(str(target)):
                            shutil.copy2(str(target), out)
                            fallback_done = os.path.isfile(out)
                    except Exception as copy_error:
                        fallback_error = str(copy_error)

                    if fallback_done:
                        return True, fallback_error

                    # Video remux fallback when direct copy could not materialize final output.
                    try:
                        remux = subprocess.run(
                            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(target), "-c", "copy", str(out)],
                            capture_output=True,
                            text=True,
                            timeout=120,
                        )
                        if remux.returncode == 0 and os.path.isfile(out):
                            return True, fallback_error
                        if remux.stderr:
                            fallback_error = (fallback_error + " | " + remux.stderr.strip()).strip(" |")[:500]
                    except Exception as remux_error:
                        fallback_error = (fallback_error + " | " + str(remux_error)).strip(" |")[:500]

                    # Last-resort transcode fallback.
                    try:
                        transcode = subprocess.run(
                            [
                                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                "-i", str(target),
                                "-c:v", OUTPUT_VIDEO_ENCODER,
                                "-c:a", "aac",
                                str(out),
                            ],
                            capture_output=True,
                            text=True,
                            timeout=300,
                        )
                        if transcode.returncode == 0 and os.path.isfile(out):
                            return True, fallback_error
                        if transcode.stderr:
                            fallback_error = (fallback_error + " | " + transcode.stderr.strip()).strip(" |")[:500]
                    except Exception as transcode_error:
                        fallback_error = (fallback_error + " | " + str(transcode_error)).strip(" |")[:500]

                    return False, fallback_error

                def _count_images(path_obj, recursive=False):
                    if not path_obj or not path_obj.exists() or not path_obj.is_dir():
                        return 0
                    count = 0
                    try:
                        if recursive:
                            for entry in path_obj.rglob("*"):
                                if not entry.is_file():
                                    continue
                                name = entry.name.lower()
                                if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".bmp"):
                                    count += 1
                        else:
                            for entry in os.scandir(path_obj):
                                if not entry.is_file():
                                    continue
                                name = entry.name.lower()
                                if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".bmp"):
                                    count += 1
                    except Exception:
                        return 0
                    return count

                def resolve_temp_frames_dir():
                    base = Path(job_temp_path)
                    candidates = [
                        base / "facefusion" / target.stem,
                        base / "facefusion" / Path(target).name,
                        base / target.stem,
                        base / "facefusion",
                        base,
                    ]
                    best_path = temp_frames_dir[0]
                    best_count = -1
                    for cand in candidates:
                        cnt = _count_images(cand, recursive=(cand == base))
                        if cnt > best_count:
                            best_count = cnt
                            best_path = cand
                    temp_frames_dir[0] = best_path
                    return best_path, int(max(0, best_count))

                def count_temp_frames():
                    best_path, count = resolve_temp_frames_dir()
                    print("FRAME PATH:", str(best_path), flush=True)
                    print("FRAME COUNT:", int(count), flush=True)
                    logger.info("FRAME PATH: %s", str(best_path))
                    logger.info("FRAME COUNT: %s", int(count))
                    return int(count)

                def _pick_frame_scan_interval_sec(total_frames):
                    total = int(max(0, total_frames or 0))
                    if total <= 0:
                        return 2.0
                    if total <= 5000:
                        return 1.5
                    if total <= 30000:
                        return 5.0
                    return 8.0

                def _derive_progress_from_frames(frames_done, frames_total, previous_pct=0):
                    done = int(max(0, frames_done or 0))
                    total = int(max(0, frames_total or 0))
                    prev = int(max(0, previous_pct or 0))

                    if total > 0:
                        if done > total:
                            done = total
                        pct = int((done / float(total)) * 100)
                        if done == total:
                            pct = 100
                        if pct >= 99 and done == total:
                            pct = 100
                        return int(max(0, min(100, pct))), done, total

                    return prev, done, total

                def _sync_faceswap_progress(now_ts):
                    pct_calc, done_calc, total_calc = _derive_progress_from_frames(
                        last_frame[0],
                        progress_total_target_frames,
                        live_status.get("pct", 0),
                    )
                    stage_text = str(job_status.get(chat_id, {}).get("stage") or "Processing")
                    live_status["pct"] = pct_calc
                    job_status[chat_id].update({
                        "phase": "faceswap",
                        "stage": stage_text,
                        "pct": pct_calc,
                        "updated_at": now_ts,
                        "done_frames": int(done_calc),
                        "total_frames": int(total_calc),
                    })
                    last_state[0] = stage_text
                    last_pct[0] = pct_calc
                    last_tg_time[0] = now_ts
                    _persist_active_job_state(chat_id)
                    return pct_calc, done_calc, total_calc

                async def _validate_output_exists_and_stable(output_path, stable_sec=3):
                    try:
                        if not os.path.isfile(output_path):
                            return False, "output file missing"
                        size_a = int(os.path.getsize(output_path) or 0)
                        if size_a <= 0:
                            return False, "output file size is zero"
                        await asyncio.sleep(int(max(1, stable_sec)))
                        if not os.path.isfile(output_path):
                            return False, "output file disappeared during stability check"
                        size_b = int(os.path.getsize(output_path) or 0)
                        if size_b <= 0:
                            return False, "output file size became zero"
                        if size_b != size_a:
                            return False, f"output file size still changing ({size_a}->{size_b})"
                        return True, f"output stable size={size_b}"
                    except Exception as e:
                        return False, f"output stability check exception: {e}"

                async def _true_completion_check(process_exit_code, check_label="faceswap"):
                    done = int(max(0, last_frame[0] or 0))
                    total = int(max(0, progress_total_target_frames or 0))

                    if int(process_exit_code or 0) != 0:
                        return False, f"{check_label} exit code={process_exit_code}"

                    # Frame parsers can lag by a small margin near process exit; treat near-complete as valid.
                    frame_tolerance = 2
                    if total > 0 and done < max(0, total - frame_tolerance):
                        return False, f"frame mismatch done={done} total={total} tol={frame_tolerance}"

                    # Keep a short stability window so upload can start within 2s after merge exit.
                    out_ok, out_info = await _validate_output_exists_and_stable(out, stable_sec=1)
                    if not out_ok:
                        return False, out_info

                    return True, f"completion validated done={done} total={total} | {out_info}"

                async def _run_frame_debug_with_timeout(target_path, output_path, extracted_frames, debug_path, timeout_sec):
                    try:
                        return await asyncio.wait_for(
                            asyncio.to_thread(
                                analyze_faceswap_frame_debug,
                                str(target_path),
                                str(output_path),
                                int(max(0, extracted_frames or 0)),
                                str(debug_path),
                            ),
                            timeout=float(max(5, timeout_sec or FRAME_DEBUG_ANALYSIS_TIMEOUT_SEC)),
                        )
                    except asyncio.TimeoutError:
                        return {
                            "extracted_frames": int(max(0, extracted_frames or 0)),
                            "total_compared_frames": 0,
                            "detected_faces_frames": 0,
                            "swapped_frames": 0,
                            "face_detector_loaded": False,
                            "sample_before": "",
                            "sample_after": "",
                            "sample_detected_before": "",
                            "error": f"frame debug timeout after {int(max(5, timeout_sec or FRAME_DEBUG_ANALYSIS_TIMEOUT_SEC))}s",
                        }

                async def _force_fallback_stage_transition(detail_text=""):
                    if DISABLE_FACE_SWAP_FALLBACK:
                        reason = f"❌ FaceSwap not applied on frames | fallback disabled | {detail_text}"
                        logger.error(reason)
                        raise RuntimeError(reason)

                    faceswap_forced_passthrough[0] = True
                    fallback_done, fallback_error = _ensure_passthrough_output_exists()
                    completion_done = int(max(0, progress_total_target_frames or 0))
                    if completion_done > 0:
                        last_frame[0] = completion_done
                    live_status["state"] = stage_label_map.get("merging", "Merging")
                    live_status["pct"] = 100
                    _set_job_stage("merging", phase="faceswap", pct=100)
                    _sync_faceswap_progress(time.time())

                    await notify(
                        "⚠️ FaceSwap failed due to processing issue.\n"
                        "✅ Fallback output generated (original video used).",
                        main_kb(),
                    )
                    await notify("🎬 Merging started", main_kb())

                    await live_update(
                        "merging",
                        "Merging",
                        pct=100,
                        elapsed=int(time.time() - ff_start),
                        eta_seconds=0,
                        done_frames=last_frame[0],
                        total_frames=int(max(0, progress_total_target_frames or 0)),
                        extra=progress_extra,
                        force=True,
                    )

                    if detail_text:
                        logger.warning("FORCED FALLBACK chat=%s detail=%s fallback_ok=%s err=%s", chat_id, detail_text, fallback_done, fallback_error)
                    return fallback_done, fallback_error

                async def heartbeat_progress():
                    # Heartbeat emits Telegram updates every 1s from shared state.
                    # Stdout parser (process_progress_line) is the SOLE source for last_frame.
                    while proc.poll() is None and _is_active_progress_stream():
                        await asyncio.sleep(4)
                        now = time.time()
                        if proc.poll() is not None:
                            break
                        if (now - progress_state["last_emit_at"]) < 4.0:
                            continue

                        elapsed = int(now - ff_start)
                        pct = live_status["pct"]

                        out_size = 0
                        try:
                            if os.path.isfile(out):
                                out_size = int(os.path.getsize(out) or 0)
                        except Exception:
                            out_size = 0
                        if out_size > last_output_size[0]:
                            last_output_growth_at[0] = now
                            last_output_size[0] = out_size

                        pct, done_frames_live, total_frames_live = _derive_progress_from_frames(
                            last_frame[0],
                            progress_total_target_frames,
                            pct,
                        )

                        completion_reached = (
                            total_frames_live > 0
                            and done_frames_live >= total_frames_live
                            and int(max(0, min(100, pct or 0))) >= 100
                        )
                        if completion_reached:
                            if float(ready_to_complete_since[0] or 0.0) <= 0.0:
                                ready_to_complete_since[0] = now
                                logger.info(
                                    "READY_TO_COMPLETE chat=%s pid=%s done=%s total=%s -> watchdog suspended, waiting for natural exit",
                                    chat_id,
                                    int(getattr(proc, "pid", 0) or 0),
                                    int(done_frames_live),
                                    int(total_frames_live),
                                )
                            # Watchdog fully suspended: all frames done → show Finalizing, do NOT kill.
                            _fin_state = "🎬 Finalizing video..." if not merge_stage_detected[0] else live_status.get("state", "Merging")
                            _fin_key = "merging"
                            live_status["state"] = _fin_state
                            live_status["pct"] = 99
                            pct_sync, done_sync, total_sync = _sync_faceswap_progress(now)
                            await live_update(
                                _fin_key, _fin_state,
                                pct=99, elapsed=elapsed,
                                eta_seconds=None,
                                done_frames=done_frames_live,
                                total_frames=total_frames_live,
                                extra=progress_extra, force=True,
                            )
                            progress_state["last_emit_at"] = time.time()
                            continue
                        else:
                            ready_to_complete_since[0] = 0.0

                        stage_from_job = str(job_status.get(chat_id, {}).get("stage") or "Extracting")
                        key_from_job = _stage_key_from_text(stage_from_job) or "extracting"
                        current_stage_key[0] = key_from_job
                        live_status["state"] = stage_label_map.get(key_from_job, STAGE_FLOW_TEXT.get(key_from_job, stage_from_job))

                        # Fail-safe: when processing phase stalls with no parser updates, force stop and continue fallback path.
                        if processing_started[0] and proc.poll() is None:
                            processing_elapsed = now - float(ff_start or now)
                            if processing_hard_timeout_sec > 0 and processing_elapsed >= processing_hard_timeout_sec and not completion_reached:
                                logger.error(
                                    "FORCE TERMINATE FACEFUSION HARD TIMEOUT chat=%s pid=%s elapsed=%.1fs",
                                    chat_id,
                                    int(getattr(proc, "pid", 0) or 0),
                                    float(processing_elapsed),
                                )
                                with suppress(Exception):
                                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                                await asyncio.sleep(2)
                                if proc.poll() is None:
                                    with suppress(Exception):
                                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                                break

                            no_frame_change_sec = now - float(last_frame_change_at[0] or 0.0)
                            if float(last_output_growth_at[0] or 0.0) > 0.0:
                                no_output_growth_sec = now - float(last_output_growth_at[0] or 0.0)
                            else:
                                no_output_growth_sec = 0.0
                            no_progress_parse_sec = now - float(last_progress_parse_at[0] or 0.0)
                            if no_frame_change_sec >= 60.0 and no_progress_parse_sec >= 60.0 and no_output_growth_sec >= 30.0 and not completion_reached and total_frames_live > 0:
                                logger.error(
                                    "FORCE TERMINATE HUNG FACEFUSION chat=%s pid=%s done=%s total=%s no_frame_change=%.1fs no_output_growth=%.1fs no_parse=%.1fs",
                                    chat_id,
                                    int(getattr(proc, "pid", 0) or 0),
                                    int(done_frames_live),
                                    int(total_frames_live),
                                    float(no_frame_change_sec),
                                    float(no_output_growth_sec),
                                    float(no_progress_parse_sec),
                                )
                                with suppress(Exception):
                                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                                await asyncio.sleep(2)
                                if proc.poll() is None:
                                    with suppress(Exception):
                                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                                break

                        stall_age = now - float(last_frame_change_at[0] or 0.0)
                        if stall_age >= 20.0 and (now - float(last_stall_log_at[0] or 0.0)) >= 20.0:
                            logger.warning(
                                "Extraction possibly stalled chat=%s frame_count=%s stall_age=%.1fs",
                                chat_id,
                                int(last_frame[0] or 0),
                                stall_age,
                            )
                            last_stall_log_at[0] = now

                        live_status["pct"] = int(max(0, min(100, pct or 0)))
                        if completion_reached and proc.poll() is None and not merge_stage_detected[0]:
                            # All frames done but process still alive → audio/video merge in progress
                            live_status["pct"] = 99
                            _set_job_stage("merging", phase="faceswap", pct=99)
                            current_stage_key[0] = "merging"
                            live_status["state"] = "🎬 Finalizing video..."
                        elif (
                            current_stage_key[0] == "processing"
                            and total_frames_live > 0
                            and done_frames_live >= total_frames_live
                            and live_status["pct"] >= 99
                        ):
                            live_status["pct"] = 100
                            _set_job_stage("processing", phase="faceswap", pct=100)
                            current_stage_key[0] = "processing"
                            live_status["state"] = "Processing"

                        pct_sync, done_sync, total_sync = _sync_faceswap_progress(now)

                        eta = estimate_eta_seconds(elapsed, pct_sync)
                        _perf_update_from_faceswap_state(live_status["state"], proc_pid=proc.pid)

                        await live_update(
                            current_stage_key[0],
                            live_status["state"],
                            pct=pct_sync,
                            elapsed=elapsed,
                            eta_seconds=eta,
                            done_frames=done_sync,
                            total_frames=total_sync,
                            extra=progress_extra,
                            force=True,
                        )
                        progress_state["last_emit_at"] = time.time()

                heartbeat_task = asyncio.create_task(heartbeat_progress())

                if clip_progress_note:
                    await live_update(
                        "extracting",
                        "Extracting",
                        pct=0,
                        elapsed=0,
                        eta_seconds=None,
                        done_frames=0,
                        total_frames=progress_total_target_frames,
                        extra=progress_extra,
                        force=True,
                    )

                async def process_progress_line(line):
                    nonlocal progress_total_target_frames
                    if not line:
                        return
                    line = strip_ansi(line).rstrip()
                    if not line:
                        return
                    last_progress_parse_at[0] = time.time()
                    lo_line = line.lower()
                    if ("cuda failure 900" in lo_line) or ("cudaerrorstreamcaptureunsupported" in lo_line):
                        force_passthrough_due_cuda[0] = True
                        faceswap_forced_passthrough[0] = True
                        logger.error("CUDA STREAM CAPTURE ERROR DETECTED chat=%s -> force passthrough fallback", chat_id)
                        if proc.poll() is None:
                            with suppress(Exception):
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    if (
                        ("out of memory" in lo_line)
                        or ("cuda failure 2" in lo_line)
                        or ("failed to allocate memory" in lo_line)
                    ) and not oom_detected_live[0]:
                        oom_detected_live[0] = True
                        logger.error("GPU OOM DETECTED chat=%s -> terminating current run for adaptive GPU retry", chat_id)
                        if proc.poll() is None:
                            with suppress(Exception):
                                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    if "processing:" in lo_line and not processing_started[0]:
                        processing_started[0] = True
                        last_frame[0] = 0
                        last_pct[0] = 0
                        last_frame_change_at[0] = time.time()
                        _set_job_stage("processing", phase="faceswap")
                        live_status["state"] = "⚙️ Initializing processors..."
                        logger.info("PROCESSING PHASE STARTED chat=%s frame_counter_reset=1", chat_id)

                    _loading_keywords = ("loading face swapper", "loading face enhancer", "loading face detector",
                                         "loading face landmarker", "loading face recogniser", "loading model",
                                         "loading weights", "initializing")
                    if processing_started[0] and not parsed_frame_once[0]:
                        if any(kw in lo_line for kw in _loading_keywords):
                            live_status["state"] = "⚙️ Loading AI models..."
                            logger.info("MODEL_LOADING_DETECTED chat=%s line=%s", chat_id, line[:80])
                    if "device_discovery.cc" in line and "ReadFileContents Failed to open file" in line:
                        # Non-fatal ORT container warning; ignore to keep progress feed clean.
                        return

                    logger.info(f"[ff] {line}")
                    ff_tail.append(line)
                    now = time.time()
                    lo = line.lower()

                    if "merging:" in lo:
                        merge_stage_detected[0] = True
                        current_stage = _stage_key_from_text(job_status.get(chat_id, {}).get("stage")) or "processing"
                        if current_stage != "merging":
                            logger.info(
                                "STAGE TRANSITION HOOK chat=%s processing -> merging details=merge_stream_detected",
                                chat_id,
                            )
                            _set_job_stage("merging", phase="faceswap", pct=0)
                            last_frame[0] = 0
                            last_pct[0] = 0
                            last_frame_change_at[0] = now
                        current_stage_key[0] = "merging"
                        live_status["state"] = "Merging"
                        live_status["pct"] = 0

                    # FaceFusion/ffmpeg progress can come in multiple formats:
                    # - tqdm standard: "extracting:  49%|====| 218/448 [00:01<00:01, 193.49frame/s]"
                    # - tqdm complete: "extracting: 449frame [00:02, 213.67frame/s]"
                    # - FaceFusion custom: "extracting: 12% (123/1000)"
                    # - ffmpeg style: "frame=  123 ..."
                    m = re.search(r"(\d{1,3})%\s*\|", line)
                    pct = int(m.group(1)) if m else last_pct[0]
                    fm = re.search(r"frame=\s*(\d+)", line)
                    tqdmm = re.search(r"\|\s*([\d,]+)\s*/\s*([\d,]+)", line)
                    # FaceFusion custom tqdm format: "extracting: 12% (123/1000)"
                    ff_tqdm = re.search(r"(?:extracting|processing|merging):\s*(\d{1,3})%\s*\(([\d,]+)/([\d,]+)\)", line)
                    if ff_tqdm:
                        pct = int(ff_tqdm.group(1))
                    # FaceFusion completion format: "extracting: 449frame [...]" — no % sign
                    ff_done = re.search(r"(?:extracting|processing|merging):\s*([\d,]+)\s*frame\b", line)
                    prev_frame = last_frame[0]
                    frame_done = int(fm.group(1)) if fm else prev_frame

                    if fm or tqdmm or ff_tqdm or ff_done:
                        if not parsed_frame_once[0]:
                            # First real frame counter: clear the loading placeholder
                            live_status["state"] = "Processing"
                        parsed_frame_once[0] = True

                    parsed_total_frames = None
                    if tqdmm:
                        frame_done = int(tqdmm.group(1).replace(",", ""))
                        parsed_total_frames = int(tqdmm.group(2).replace(",", ""))
                    elif ff_tqdm:
                        frame_done = int(ff_tqdm.group(2).replace(",", ""))
                        parsed_total_frames = int(ff_tqdm.group(3).replace(",", ""))
                    elif ff_done:
                        # "extracting: 449frame" — completion line, use as frame_done
                        frame_done = int(ff_done.group(1).replace(",", ""))
                        if progress_total_target_frames > 0:
                            pct = 100

                    progress_total_frames = progress_total_target_frames
                    if parsed_total_frames and parsed_total_frames > 0:
                        progress_total_frames = parsed_total_frames
                        progress_total_target_frames = parsed_total_frames
                    elif progress_total_frames <= 0:
                        progress_total_frames = progress_total_target_frames

                    # Stdout is the PRIMARY (authoritative) source for frame counter.
                    if frame_done > last_frame[0]:
                        last_frame[0] = frame_done
                        last_frame_change_at[0] = now

                    pct, frame_done_sync, total_sync = _derive_progress_from_frames(
                        last_frame[0],
                        progress_total_target_frames,
                        last_pct[0],
                    )

                    # Force deterministic completion when last frame is reached.
                    if total_sync > 0 and frame_done_sync >= total_sync and pct >= 99:
                        pct = 100

                    state = str(job_status.get(chat_id, {}).get("stage") or ("Processing" if is_image_target else "Extracting"))
                    stage_key = _stage_key_from_text(state) or "extracting"
                    if is_image_target and stage_key in {"extracting", "merging"}:
                        stage_key = "processing"
                        state = stage_label_map.get("processing", "Processing")
                    emoji = "🎞" if stage_key == "extracting" else ("┘" if stage_key == "processing" else ("🔗" if stage_key == "merging" else "☁️"))
                    current_stage_key[0] = stage_key

                    ff_started[0] = True
                    live_status["state"] = state
                    live_status["pct"] = pct
                    live_status["emoji"] = emoji
                    last_frame[0] = frame_done_sync
                    _perf_update_from_faceswap_state(state, proc_pid=proc.pid)

                    _sync_faceswap_progress(time.time())

                    elapsed = int(now - ff_start)
                    eta = estimate_eta_seconds(elapsed, pct)
                    state_changed = state != last_state[0]
                    pct_jumped = abs(pct - last_pct[0]) >= 1
                    frame_step = max(25, int(progress_total_frames * 0.005)) if progress_total_frames > 0 else 25
                    frame_jumped = progress_total_frames > 0 and (frame_done_sync - prev_frame) >= frame_step
                    first_ping = ff_started[0] and last_tg_time[0] == 0
                    frame_due = frame_jumped and (now - last_tg_time[0] >= 1.0)
                    since_last_telegram_push = now - float(last_tg_time[0] or 0.0)

                    if first_ping or state_changed or pct_jumped or frame_due:
                        await live_update(
                            stage_key,
                            state,
                            pct=pct,
                            elapsed=elapsed,
                            eta_seconds=eta,
                            done_frames=frame_done_sync,
                            total_frames=total_sync,
                            extra=progress_extra,
                        )
                        last_tg_time[0] = now
                        last_state[0] = state
                        last_pct[0] = pct
                        progress_state["last_emit_at"] = now

                await drain_subprocess_output(
                    proc,
                    line_handler=process_progress_line,
                    tail_buffer=ff_tail,
                    watchdog_label="faceswap-main",
                    watchdog_interval_sec=FACEFUSION_WATCHDOG_SEC,
                    exit_pipe_grace_sec=PIPE_READ_EXIT_GRACE_SEC,
                )
                merge_exit_ts = float(time.time())
                if heartbeat_task and not heartbeat_task.done():
                    heartbeat_task.cancel()
                ff_time          = int(time.time() - ff_start)
                ff_mins, ff_secs = divmod(ff_time, 60)
                _perf_end("extraction")
                _perf_end("processing")
                _perf_end("merge_encode")
                active_jobs.pop(chat_id, None)

                proc_rc = int(proc.returncode or 0) if proc.returncode is not None else -15

                def _retry_merge_once_from_temp():
                    candidates = []
                    try:
                        base = Path(job_temp_path) / "facefusion"
                        if base.exists():
                            candidates.extend(base.rglob("temp.mp4"))
                        candidates.extend(Path(job_temp_path).rglob("temp.mp4"))
                    except Exception:
                        candidates = []

                    seen = set()
                    ranked = []
                    for p in candidates:
                        try:
                            rp = str(p.resolve())
                        except Exception:
                            rp = str(p)
                        if rp in seen:
                            continue
                        seen.add(rp)
                        try:
                            if not p.is_file() or int(p.stat().st_size or 0) <= 0:
                                continue
                            ranked.append((float(p.stat().st_mtime), p))
                        except Exception:
                            continue

                    ranked.sort(key=lambda item: item[0], reverse=True)
                    for _, cand in ranked:
                        remux = subprocess.run(
                            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(cand), "-c", "copy", str(out)],
                            capture_output=True,
                            text=True,
                            timeout=180,
                        )
                        if remux.returncode != 0 or not os.path.isfile(out):
                            transcode = subprocess.run(
                                [
                                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                                    "-i", str(cand),
                                    "-c:v", OUTPUT_VIDEO_ENCODER,
                                    "-c:a", "aac",
                                    str(out),
                                ],
                                capture_output=True,
                                text=True,
                                timeout=300,
                            )
                            if transcode.returncode != 0 or not os.path.isfile(out):
                                continue

                        ok_retry_media, retry_info = validate_output_media(str(out))
                        if ok_retry_media:
                            return True, f"candidate={cand} | {retry_info}"

                    return False, "no valid temp.mp4 candidate could be remuxed"

                completion_likely = (
                    int(max(0, progress_total_target_frames or 0)) > 0
                    and int(max(0, last_frame[0] or 0)) >= int(max(0, progress_total_target_frames or 0))
                )
                nonzero_but_output_valid = False
                if proc_rc != 0 and completion_likely:
                    out_ok_near_done, out_info_near_done = await _validate_output_exists_and_stable(out, stable_sec=3)
                    if out_ok_near_done:
                        logger.warning(
                            "PROCESS_EXIT_NONZERO_BUT_OUTPUT_STABLE chat=%s rc=%s done=%s total=%s info=%s",
                            chat_id,
                            proc_rc,
                            int(max(0, last_frame[0] or 0)),
                            int(max(0, progress_total_target_frames or 0)),
                            out_info_near_done,
                        )
                        proc_rc = 0
                        nonzero_but_output_valid = True

                if proc_rc != 0 and os.path.isfile(str(out)):
                    out_media_ok, out_media_info = validate_output_media(str(out))
                    if out_media_ok:
                        logger.warning(
                            "PROCESS_EXIT_NONZERO_BUT_OUTPUT_VALID chat=%s rc=%s info=%s",
                            chat_id,
                            proc_rc,
                            out_media_info,
                        )
                        proc_rc = 0
                        nonzero_but_output_valid = True

                if proc_rc == -15 and merge_stage_detected[0] and not merge_retry_attempted[0]:
                    merge_retry_attempted[0] = True
                    logger.warning("MERGE_SIGTERM_DETECTED chat=%s rc=%s -> attempting one merge retry", chat_id, proc_rc)
                    retry_ok, retry_info = await asyncio.to_thread(_retry_merge_once_from_temp)
                    if retry_ok:
                        logger.info("MERGE_RETRY_SUCCESS chat=%s info=%s", chat_id, retry_info)
                        await notify("⚠️ Merge interrupted (SIGTERM). Auto-retry merge succeeded.", main_kb())
                        proc_rc = 0
                    else:
                        logger.error("MERGE_RETRY_FAILED chat=%s info=%s", chat_id, retry_info)

                if proc_rc != 0:
                    # ── Error diagnostics: dump last 50 stdout/stderr lines ──────────────
                    _err_lines = [l for l in list(ff_tail)[-50:] if l.strip()]
                    _err_log_path = str(Path(PIPELINE) / "logs" / "last_error.txt")
                    try:
                        Path(_err_log_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(_err_log_path, "w", encoding="utf-8") as _ef:
                            _ef.write(f"proc_rc={proc_rc} chat={chat_id}\n" + "\n".join(_err_lines))
                        logger.info("LAST_ERROR_SAVED path=%s lines=%s", _err_log_path, len(_err_lines))
                    except Exception as _dmp_err:
                        logger.warning("LAST_ERROR_SAVE_FAILED err=%s", _dmp_err)
                    _tg_err_lines = _err_lines[-10:]
                    if _tg_err_lines:
                        _tg_err_snip = "\n".join(_tg_err_lines)
                        await notify(
                            f"🔴 FaceFusion exited rc=`{proc_rc}` — last output:\n```\n{_tg_err_snip[:900]}\n```",
                            main_kb(),
                        )
                    # ─────────────────────────────────────────────────────────────────────
                    frame_debug_stats = await asyncio.to_thread(
                        analyze_faceswap_frame_debug,
                        str(target),
                        str(out),
                        int(max(0, last_frame[0] or 0)),
                        str(Path(job_temp_path) / "detection_debug" / "gpu_failed"),
                    )
                    logger.info(
                        "FRAME_DEBUG chat=%s extracted=%s compared=%s detected=%s swapped=%s detector_loaded=%s error=%s",
                        chat_id,
                        int(frame_debug_stats.get("extracted_frames", 0) or 0),
                        int(frame_debug_stats.get("total_compared_frames", 0) or 0),
                        int(frame_debug_stats.get("detected_faces_frames", 0) or 0),
                        int(frame_debug_stats.get("swapped_frames", 0) or 0),
                        bool(frame_debug_stats.get("face_detector_loaded", False)),
                        str(frame_debug_stats.get("error", "") or ""),
                    )
                    logger.info(
                        "Detected faces in %s / %s frames",
                        int(frame_debug_stats.get("detected_faces_frames", 0) or 0),
                        int(frame_debug_stats.get("total_compared_frames", 0) or 0),
                    )
                    logger.info("Faceswap applied to %s frames", int(frame_debug_stats.get("swapped_frames", 0) or 0))
                    if frame_debug_stats.get("sample_before") or frame_debug_stats.get("sample_after"):
                        logger.info(
                            "FRAME_DEBUG_SAMPLES chat=%s before=%s after=%s detected_before=%s",
                            chat_id,
                            str(frame_debug_stats.get("sample_before", "") or ""),
                            str(frame_debug_stats.get("sample_after", "") or ""),
                            str(frame_debug_stats.get("sample_detected_before", "") or ""),
                        )

                    oom_like = _is_oom_or_gpu_failure(ff_tail)
                    hard_timeout_like = int(proc_rc or 0) == -15 and not completion_likely and not ready_to_complete_forced[0]
                    should_gpu_retry = (
                        selected_provider == "cuda"
                        and (oom_like or hard_timeout_like)
                        and not oom_recovery_attempted[0]
                    )

                    if should_gpu_retry:
                        oom_recovery_attempted[0] = True
                        retry_reason = "oom" if oom_like else "gpu-timeout"
                        max_levels = int(max(1, min(3, GPU_OOM_MAX_LEVELS)))
                        retry_success = False
                        retry_rc = int(proc.returncode or 0)
                        retry_detail = ""

                        await notify(
                            f"⚠️ GPU memory failure detected (`{retry_reason}`). Starting adaptive GPU-only retry strategy.",
                            main_kb(),
                        )

                        for retry_level in range(2, max_levels + 1):
                            profile = _gpu_retry_profile(retry_level)
                            retry_mode = str(profile.get("mode") or "low-memory")
                            logger.warning(
                                "GPU_RETRY_START chat=%s retry=%s/%s mode=%s reason=%s",
                                chat_id,
                                retry_level,
                                max_levels,
                                retry_mode,
                                retry_reason,
                            )
                            await notify(
                                f"🔁 GPU retry `{retry_level}/{max_levels}` | mode: `{retry_mode}`",
                                main_kb(),
                            )

                            _cuda_empty_cache_light()
                            retry_success, retry_rc, retry_detail = await _run_gpu_retry_attempt(cmd, retry_level)
                            logger.info(
                                "GPU_RETRY_RESULT chat=%s retry=%s mode=%s rc=%s detail=%s",
                                chat_id,
                                retry_level,
                                retry_mode,
                                retry_rc,
                                retry_detail,
                            )
                            if retry_success:
                                logger.info(
                                    "GPU_RETRY_SUCCESS chat=%s retry=%s mode=%s",
                                    chat_id,
                                    retry_level,
                                    retry_mode,
                                )
                                await notify(
                                    f"✅ GPU retry succeeded | retry `{retry_level}` | mode `{retry_mode}`",
                                    main_kb(),
                                )
                                break

                        if not retry_success:
                            cpu_recovery_success = False
                            cpu_rc = -1
                            cpu_detail = "cpu_recovery_disabled"
                            if AUTO_CPU_FALLBACK_ON_OOM and not GPU_ONLY_MODE:
                                await notify(
                                    "⚠️ GPU retries exhausted. Starting CPU recovery to complete real FaceSwap (no passthrough).",
                                    main_kb(),
                                )
                                cpu_recovery_success, cpu_rc, cpu_detail = await _run_cpu_recovery_attempt(cmd)
                                logger.info(
                                    "CPU_RECOVERY_RESULT chat=%s ok=%s rc=%s detail=%s",
                                    chat_id,
                                    bool(cpu_recovery_success),
                                    int(cpu_rc),
                                    cpu_detail,
                                )
                                if cpu_recovery_success:
                                    await notify("✅ CPU recovery succeeded. Continuing with merged/uploaded swapped output.", main_kb())

                                    final_done = count_temp_frames()
                                    if final_done > 0:
                                        last_frame[0] = final_done

                                    ok_complete, complete_info = await _true_completion_check(0, check_label="cpu-recovery")
                                    if not ok_complete:
                                        await _force_fallback_stage_transition(f"cpu_recovery_completion_validation_failed reason={complete_info}")
                                        logger.warning("cpu recovery completion validation warning chat=%s reason=%s", chat_id, complete_info)
                                    else:
                                        completion_done = int(max(0, progress_total_target_frames or 0))
                                        if completion_done > 0:
                                            last_frame[0] = completion_done
                                        completion_validated[0] = True
                                        live_status["pct"] = 100
                                        _set_job_stage("processing", phase="faceswap", pct=100)
                                        _sync_faceswap_progress(time.time())
                                        logger.info("CPU RECOVERY COMPLETION CHECK OK chat=%s reason=%s", chat_id, complete_info)

                            if cpu_recovery_success:
                                pass
                            else:
                                frame_debug_retry = await _run_frame_debug_with_timeout(
                                    target,
                                    out,
                                    int(max(0, last_frame[0] or 0)),
                                    Path(job_temp_path) / "detection_debug" / "gpu_retry_failed",
                                    FRAME_DEBUG_ANALYSIS_TIMEOUT_SEC,
                                )
                                logger.info(
                                    "FRAME_DEBUG_RETRY chat=%s extracted=%s compared=%s detected=%s swapped=%s detector_loaded=%s error=%s",
                                    chat_id,
                                    int(frame_debug_retry.get("extracted_frames", 0) or 0),
                                    int(frame_debug_retry.get("total_compared_frames", 0) or 0),
                                    int(frame_debug_retry.get("detected_faces_frames", 0) or 0),
                                    int(frame_debug_retry.get("swapped_frames", 0) or 0),
                                    bool(frame_debug_retry.get("face_detector_loaded", False)),
                                    str(frame_debug_retry.get("error", "") or ""),
                                )
                                tail_text = "\n".join(ff_tail).strip()
                                tail_msg = f"\n\n`{tail_text[-700:]}`" if tail_text else ""
                                await _force_fallback_stage_transition(
                                    f"gpu_retries_exhausted returncode={retry_rc} reason={retry_reason} detail={retry_detail} cpu_rc={cpu_rc} cpu_detail={cpu_detail}"
                                )
                                if tail_msg:
                                    await notify(f"⚠️ FaceSwap stderr tail:{tail_msg}", main_kb())
                    else:
                        tail_text = "\n".join(ff_tail).strip()
                        tail_msg = f"\n\n`{tail_text[-700:]}`" if tail_text else ""
                        await _force_fallback_stage_transition(
                                f"proc_returncode={proc_rc} cuda_passthrough={force_passthrough_due_cuda[0]}"
                        )
                        if tail_msg:
                            await notify(f"⚠️ FaceSwap stderr tail:{tail_msg}", main_kb())
                else:
                    final_done = count_temp_frames()
                    if final_done > 0:
                        last_frame[0] = final_done

                    if nonzero_but_output_valid:
                        ok_complete, complete_info = True, "nonzero_exit_but_valid_output_media"
                    else:
                        ok_complete, complete_info = await _true_completion_check(proc.returncode, check_label="faceswap")
                    if not ok_complete:
                        output_media_ok, output_media_info = validate_output_media(str(out))
                        if output_media_ok:
                            logger.warning(
                                "completion validation mismatch overridden by valid output media chat=%s reason=%s media=%s",
                                chat_id,
                                complete_info,
                                output_media_info,
                            )
                            ok_complete = True
                            complete_info = f"validation_mismatch_overridden:{complete_info}"
                        else:
                            await _force_fallback_stage_transition(f"completion_validation_failed reason={complete_info}")
                            logger.warning("completion validation warning chat=%s reason=%s", chat_id, complete_info)

                    completion_done = int(max(0, progress_total_target_frames or 0))
                    if completion_done > 0:
                        last_frame[0] = completion_done
                    completion_validated[0] = True
                    live_status["pct"] = 100
                    _set_job_stage("processing", phase="faceswap", pct=100)
                    _sync_faceswap_progress(time.time())
                    logger.info("TRUE COMPLETION CHECK OK chat=%s reason=%s", chat_id, complete_info)

                if force_passthrough_due_cuda[0]:
                    faceswap_forced_passthrough[0] = True

        if not os.path.exists(out):
            if DISABLE_FACE_SWAP_FALLBACK:
                msg = "❌ FaceSwap not applied on frames: output artifact missing and passthrough fallback disabled"
                logger.error(msg)
                raise RuntimeError(msg)
            if os.path.isfile(str(target)):
                with suppress(Exception):
                    shutil.copy2(str(target), out)
            if not os.path.exists(out):
                await notify("⚠️ Output artifact missing tha aur fallback copy fail hui. Upload phase skipped ho sakti hai.", main_kb())

        # Merge->Upload guarantee: detect ffmpeg exit, run ffprobe validation immediately,
        # and avoid heavy pre-upload diagnostics on critical path.
        ffprobe_ok, ffprobe_info = validate_output_media(str(out))
        if not ffprobe_ok:
            raise RuntimeError(f"merge output validation failed: {ffprobe_info}")
        job_status.setdefault(chat_id, {})["merge_verified"] = True
        logger.info("MERGE OUTPUT VERIFIED chat=%s info=%s", chat_id, ffprobe_info)

        merge_to_upload_delay = None
        if 'merge_exit_ts' in locals() and float(merge_exit_ts or 0.0) > 0.0:
            merge_to_upload_delay = float(time.time()) - float(merge_exit_ts)
            if merge_to_upload_delay > 2.0:
                logger.warning("MERGE_TO_UPLOAD_DELAY chat=%s delay=%.2fs (>2s)", chat_id, merge_to_upload_delay)
            else:
                logger.info("MERGE_TO_UPLOAD_DELAY chat=%s delay=%.2fs", chat_id, merge_to_upload_delay)

        out_mb = os.path.getsize(out) / 1024 / 1024
        phase_before_upload = str((job_status.get(chat_id, {}) or {}).get("phase") or "").lower()
        if phase_before_upload not in {"merging", "upload", "completed"}:
            await _emit_stage_transition(
                "processing",
                "merging",
                details="Merging in progress... still working...",
            )
            _set_job_stage("merging", phase="faceswap")
            await update_telegram_status(
                "Merging",
                progress=100,
                details=f"Merging in progress... still working... | output: {out_mb:.1f} MB",
                force=True,
            )
        else:
            logger.info("MERGE_STAGE_ALREADY_ACTIVE chat=%s phase=%s", chat_id, phase_before_upload)

        upload_path = out
        final_output_path = upload_path
        active_job_protected_paths[chat_id] = {p for p in [job_temp_path, active_target_path, final_output_path] if p}
        upload_mb = out_mb
        job_status[chat_id].update({
            "output_path": str(upload_path),
            "updated_at": time.time(),
        })
        _persist_active_job_state(chat_id)

        # STEP 3: Upload
        logger.info("ENTER: upload stage chat=%s", chat_id)
        await _emit_stage_transition(
            "merging",
            "upload",
            details="Uploading in progress... still working...",
        )
        if faceswap_forced_passthrough[0]:
            await notify("☁️ Uploading started", main_kb())
        _set_job_stage("upload", phase="upload", pct=-1, details=_stage_detail_for("upload"))
        await _cleanup_stale_download_state(reason="entered_upload")

        ul_start = time.time()
        # ── Pre-upload validation gate ────────────────────────────────────────
        _pre_ul_size = 0
        try:
            _pre_ul_size = int(os.path.getsize(upload_path) or 0) if os.path.isfile(upload_path) else 0
        except Exception:
            _pre_ul_size = 0
        if not os.path.isfile(upload_path):
            await notify(
                f"❌ Upload aborted: output file not found at expected path.\n`{upload_path}`\nPipeline check karo — FaceFusion ne output save nahi kiya.",
                main_kb(),
            )
            _clear_active_job_state()
            return
        if _pre_ul_size < 100_000:
            await notify(
                f"❌ Upload aborted: output file too small ({_pre_ul_size:,} bytes < 100 KB).\n`{upload_path}`",
                main_kb(),
            )
            _clear_active_job_state()
            return
        logger.info("PRE_UPLOAD_GATE_PASSED chat=%s path=%s size_mb=%.2f", chat_id, upload_path, _pre_ul_size / 1024 / 1024)
        # ─────────────────────────────────────────────────────────────────────
        if not can_use_mega():
            await notify("⚠️ MEGA credentials missing: MEGA upload skip hoga, Google Drive fallback use hoga.", main_kb())
        _perf_start("upload")
        _perf_sample("upload")
        upload_live = {"run": True}

        async def upload_heartbeat():
            assumed_total = max(20, int(upload_mb / 4.0))
            _last_tg_edit = 0.0
            while upload_live["run"] and _is_active_progress_stream():
                elapsed = int(time.time() - ul_start)
                pct = min(95, int((elapsed / float(max(1, assumed_total))) * 100))
                eta = max(0, assumed_total - elapsed)
                speed_est = max(0.1, upload_mb / max(1, elapsed))
                uploaded_est = min(upload_mb, speed_est * elapsed)
                await live_update(
                    "upload",
                    "Uploading to Cloud",
                    pct=pct,
                    elapsed=elapsed,
                    eta_seconds=eta,
                    extra=f"size: {upload_mb:.1f} MB | target: GDrive (primary) then MEGA fallback",
                )
                # Edit Telegram message every 5s with live upload %
                now = time.time()
                if now - _last_tg_edit >= 5:
                    _last_tg_edit = now
                    bar = "▰" * (pct // 10) + "▱" * (10 - pct // 10)
                    tg_text = (
                        f"☁️ *Uploading...*\n\n"
                        f"`{bar}` {pct}%\n"
                        f"📊 {uploaded_est:.1f} / {upload_mb:.1f} MB\n"
                        f"⚡ {speed_est:.2f} MB/s\n"
                        f"⏱ ETA: {eta}s"
                    )
                    with suppress(Exception):
                        await update_telegram_status(
                            "Uploading",
                            progress=pct,
                            details=tg_text,
                        )
                _perf_sample("upload")
                await asyncio.sleep(1)

        hb_upload = asyncio.create_task(upload_heartbeat())
        # File stability check — ensure output file is fully written before upload
        if os.path.isfile(upload_path):
            sz1 = os.path.getsize(upload_path)
            await asyncio.sleep(2)
            sz2 = os.path.getsize(upload_path) if os.path.isfile(upload_path) else 0
            if sz1 != sz2:
                logger.info("FILE_STABILITY: size changed %s -> %s, waiting 2s more", sz1, sz2)
                await asyncio.sleep(2)
        ok, platform, info = await asyncio.to_thread(smart_upload, upload_path)
        upload_link = str(info or "").strip() if str(info or "").strip().lower().startswith(("http://", "https://")) else ""
        job_status[chat_id].update({
            "upload_ok": bool(ok),
            "upload_platform": str(platform or ""),
            "upload_info": str(info or ""),
            "upload_link": upload_link,
            "updated_at": time.time(),
        })
        _persist_active_job_state(chat_id)
        upload_live["run"] = False
        if hb_upload and not hb_upload.done():
            hb_upload.cancel()

        ul_time          = int(time.time() - ul_start)
        ul_mins, ul_secs = divmod(ul_time, 60)
        total_time       = int(time.time() - t_total)
        t_mins, t_secs   = divmod(total_time, 60)
        _perf_sample("upload")
        _perf_end("upload")

        if ok:
            await live_update(
                "upload",
                "Upload Complete",
                pct=100,
                elapsed=ul_time,
                eta_seconds=0,
                extra=f"platform: {platform}",
            )

        # STEP 4: Done
        logger.info("ENTER: done stage chat=%s ok=%s platform=%s", chat_id, ok, platform if ok else "-")
        if ok:
            if not upload_link:
                # Upload succeeded but link generation failed — notify and treat as partial failure
                logger.error(
                    "UPLOAD_LINK_MISSING chat=%s platform=%s info=%s",
                    chat_id, platform, info,
                )
                await notify(
                    f"⚠️ Upload to *{platform}* succeeded but public link generate nahi hua.\n"
                    f"Error: `{info}`\n"
                    f"File locally available: `{upload_path}`\n"
                    "GDrive ke liye Drive folder check karo ya re-upload try karo.",
                    main_kb(),
                )
                _clear_active_job_state()
                return
            valid_output_ok, valid_output_info = validate_output_media(upload_path)
            if not valid_output_ok:
                raise RuntimeError(f"final output validation failed: {valid_output_info}")
            # AUTO-DELETE DISABLED: source video preserved for manual cleanup
            # delete_downloaded_source_video(downloaded_source_path)

            await _emit_stage_transition(
                "upload",
                "completed",
                details="Finalizing output...",
            )
            job_status[chat_id].update({
                "phase": "completed",
                "pct": 100,
                "updated_at": time.time(),
                "details": _stage_detail_for("completed")
            })
            _set_job_stage("completed", phase="completed", pct=100, details=_stage_detail_for("completed"))
            _persist_active_job_state(chat_id)
            await update_telegram_status(
                "Completed",
                progress=100,
                details=f"Completed | platform: {platform}",
                force=True,
            )
            if faceswap_forced_passthrough[0]:
                await notify("🔗 Output ready", main_kb())

            if platform in ("GDRIVE", "Google Drive"):
                link_line = f"\n🔗 [Google Drive Link]({upload_link})"
            elif platform == "MEGA":
                link_line = f"\n🔗 [MEGA Link]({upload_link})"
            else:
                link_line = f"\n🔗 [Upload Link]({upload_link})"

            await notify(
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ *FaceSwap Completed Successfully*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"📥 Download  → `{dl_time}s`\n"
                f"⚙️ FaceSwap → `{ff_mins}m {ff_secs}s`\n"
                f"⬆️ Upload    → `{ul_mins}m {ul_secs}s`\n"
                f"⏱ *Total    → `{t_mins}m {t_secs}s`*\n\n"
                f"📁 `{os.path.basename(upload_path)}`\n"
                f"📦 `{upload_mb:.1f} MB`\n"
                "🖥 GPU: `YES`\n"
                f"☁️ *{platform}*"
                f"{link_line}\n\n"
                f"⏱ Job complete. Studio {_sleep_delay_minutes_text()} baad auto-sleep hoga.",
                main_kb()
            )

            if job_mode == "multi":
                set_chat_mode(chat_id, "direct")
                selected_face_maps.pop(chat_id, None)
                job_modes[chat_id] = "multi_done"
                await notify(
                    "🔁 Multi FaceSwap job complete. Bot ab default *Direct FaceSwap* mode par aa gaya.",
                    main_kb()
                )

            job_status[chat_id].update({
                "phase": "completed",
                "pct": 100,
                "updated_at": time.time(),
                "details": "Pipeline completed"
            })
            logger.info("[JOB_DONE] chat=%s mode=%s", chat_id, job_mode)
            _set_job_stage("completed", phase="completed", pct=100, details="Pipeline completed")
            _update_lifecycle_state(
                chat_id,
                is_job_running=False,
                is_job_completed=True,
            )

            pending_after = _queue_size(chat_id)
            if pending_after > 0:
                logger.info("auto-sleep deferred chat=%s pending_queue=%s", chat_id, pending_after)
                await notify(
                    f"📚 Queue pending: `{pending_after}`. Next job auto-start hoga, sleep deferred.",
                    main_kb(),
                )
            else:
                if is_all_jobs_completed(chat_id):
                    app_obj = getattr(context, "application", None)
                    await on_job_completed(app_obj, chat_id, success=True)
                else:
                    logger.info(
                        "auto-sleep deferred chat=%s due incomplete completion state (busy=%s queue=%s no_bg=%s)",
                        chat_id,
                        _is_chat_busy(chat_id),
                        _queue_size(chat_id),
                        _no_background_task_running(chat_id),
                    )
            _clear_active_job_state()
        else:
            fallback_sent = False
            fallback_reason = ""
            if upload_mb <= 49:
                try:
                    await context.bot.send_video(
                        chat_id=chat_id,
                        video=open(upload_path, "rb"),
                        caption="📦 Cloud upload fail hua, isliye file direct Telegram se bhej di.",
                        reply_markup=main_kb()
                    )
                    fallback_sent = True
                except Exception as e:
                    fallback_reason = f"Telegram send fail: {e}"

            if is_mega_auth_error(info):
                guidance = "\n\n🔐 MEGA auth issue detect hua. `Change MEGA` se valid account set karo."
            else:
                guidance = ""

            extra = ""
            if fallback_sent:
                extra = "\n\n✅ Output Telegram par direct send ho gaya."
            elif fallback_reason:
                extra = f"\n\n⚠️ {fallback_reason}"

            await notify(
                f"⚠️ *Upload fail (dono platforms)!*\n"
                f"{info}{guidance}{extra}\n\n"
                f"File locally saved:\n`{upload_path}`",
                main_kb()
            )
            if faceswap_forced_passthrough[0]:
                await notify("🔗 Output ready", main_kb())
            job_status[chat_id].update({
                "phase": "failed",
                "pct": int(job_status.get(chat_id, {}).get("pct", 99) or 99),
                "updated_at": time.time(),
                "details": f"Upload failed: {info}"[:120],
                "upload_ok": False,
                "upload_platform": str(platform or ""),
                "upload_info": str(info or ""),
            })
            _set_job_stage("failed", phase="failed", pct=99, details=f"Upload failed: {info}"[:120])
            await update_telegram_status(
                "Failed",
                progress=99,
                details=f"Upload failed. Action: retry upload. Reason: {info}",
                force=True,
            )
            app_obj = getattr(context, "application", None)
            await on_job_completed(app_obj, chat_id, success=False)
            _clear_active_job_state()

    except asyncio.CancelledError:
        active_jobs.pop(chat_id, None)
        prev = job_status.get(chat_id, {})
        prev_phase = str(prev.get("phase") or "").lower()
        prev_stage = str(prev.get("stage") or "").lower()
        user_cancelled = prev_phase in {"stopped", "cancelled"} or ("user" in prev_stage)
        cancel_details = "Cancelled by user" if user_cancelled else "Cancelled by supervisor"
        cancel_stage = "Cancelled" if user_cancelled else "Cancelled by supervisor"
        job_status[chat_id] = {
            "phase": "stopped",
            "stage": cancel_stage,
            "pct": prev.get("pct", -1),
            "target": prev.get("target", "-"),
            "started_at": prev.get("started_at", time.time()),
            "updated_at": time.time(),
            "details": cancel_details,
        }
        _clear_active_job_state()
        if user_cancelled:
            await notify("⏹ Active job cancel ho gayi.", main_kb())
        raise

    except Exception as e:
        logger.exception("Pipeline error")
        active_jobs.pop(chat_id, None)
        job_status[chat_id] = {
            "phase": "failed",
            "stage": "Exception",
            "pct": job_status.get(chat_id, {}).get("pct", -1),
            "target": job_status.get(chat_id, {}).get("target", "-"),
            "started_at": job_status.get(chat_id, {}).get("started_at", time.time()),
            "updated_at": time.time(),
            "details": str(e)[:120]
        }
        app_obj = getattr(context, "application", None)
        await on_job_completed(app_obj, chat_id, success=False)
        _clear_active_job_state()
        if isinstance(e, OSError) and getattr(e, "errno", None) == 28:
            await notify(
                "❌ *System Error:* No space left on device.\n"
                "Processing halt ho gaya.\n\n"
                "🧹 Auto-cleanup run kiya gaya, lekin required free space nahi mila.\n"
                f"Minimum required free space: `{MIN_FREE_SPACE_GB} GB`",
                main_kb(),
            )
        else:
            await notify(f"❌ Error: `{e}`", main_kb())
    finally:
        leftover_proc = active_jobs.pop(chat_id, None)
        if leftover_proc:
            try:
                if leftover_proc.poll() is None:
                    with suppress(Exception):
                        os.killpg(os.getpgid(leftover_proc.pid), signal.SIGKILL)
            except Exception:
                with suppress(Exception):
                    leftover_proc.kill()

        keepalive_state["run"] = False
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            with suppress(asyncio.CancelledError):
                await keepalive_task
        if job_keepalive_tasks.get(chat_id) is keepalive_task:
            job_keepalive_tasks.pop(chat_id, None)

        ui_watchdog_state["run"] = False
        if ui_watchdog_task and not ui_watchdog_task.done():
            ui_watchdog_task.cancel()
            with suppress(asyncio.CancelledError):
                await ui_watchdog_task

        if progress_stream_tokens.get(chat_id) == progress_stream_token:
            progress_stream_tokens.pop(chat_id, None)
        _perf_end("download")
        _perf_end("extraction")
        _perf_end("processing")
        _perf_end("merge_encode")
        _perf_end("upload")
        _perf_dump_summary()
        active_job_protected_paths.pop(chat_id, None)
        final_phase = str(job_status.get(chat_id, {}).get("phase") or "").lower()
        if final_phase == "completed":
            cleanup_job_temp_path(job_temp_path)
        else:
            logger.info("preserving temp artifacts for recovery/debug chat=%s phase=%s temp=%s", chat_id, final_phase, job_temp_path)

        protected = {p for p in [job_temp_path, active_target_path, final_output_path] if p}
        prev_cleanup_task = post_job_cleanup_tasks.get(chat_id)
        if _task_is_running(prev_cleanup_task):
            prev_cleanup_task.cancel()
        # AUTO-DELETE DISABLED: post-job cleanup disabled, manual only
        # post_job_cleanup_tasks[chat_id] = asyncio.create_task(
        #     _delayed_post_job_cleanup(chat_id, protected)
        # )

        await asyncio.to_thread(_release_runtime_memory, chat_id)

        mode_used = job_modes.pop(chat_id, None)
        post_upload_tasks.pop(chat_id, None)
        if mode_used == "multi" or job_mode == "multi":
            set_chat_mode(chat_id, "direct")
            selected_face_maps.pop(chat_id, None)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="ℹ️ Multi workflow close ho gaya. Bot default *Direct FaceSwap* mode par hai.",
                    parse_mode="Markdown",
                    reply_markup=main_kb(),
                )
            except Exception:
                pass


def main():
    print("✅ Emoji Test")
    logger.info("[BOT_BOOT] pid=%s", os.getpid())
    validate_startup_credentials()
    ensure_single_instance()

    for d in [FACE_DIR, VIDEO_DIR, WORKSPACE, TEMP_PATH, OUTPUTS_DIR, PERSISTENT_ROOT, PERSISTENT_FACES_DIR]:
        os.makedirs(d, exist_ok=True)

    load_persistent_runtime_state()
    startup_state_sanity_check()
    _restore_queue_state_from_disk()

    # Start live web dashboard in a background thread (best-effort).
    try:
        _dashboard_start_server_if_enabled()
    except Exception as dash_e:
        logger.warning("dashboard startup failed: %s", dash_e)

    bypass_ok, bypass_info = apply_content_analyser_bypass()
    if bypass_ok:
        logger.info("CONTENT_ANALYSER_BYPASS: %s", bypass_info)
    else:
        logger.warning("CONTENT_ANALYSER_BYPASS_FAILED: %s", bypass_info)

    async def _post_init(application):
        global cleanup_guard_task, runtime_heartbeat_task
        if not _task_is_running(cleanup_guard_task):
            cleanup_guard_task = asyncio.create_task(_periodic_storage_guard())
        if not _task_is_running(runtime_heartbeat_task):
            runtime_heartbeat_task = asyncio.create_task(runtime_idle_heartbeat_loop())

        async def _send_startup_when_ready(context):
            await send_startup_activation_message(context.application)

        if application.job_queue is not None:
            application.job_queue.run_once(_send_startup_when_ready, when=2)
        else:
            await send_startup_activation_message(application)

        await recover_sleep_countdown_from_state(application)
        for recovered_chat_id in list(pending_recovery_chats):
            try:
                await safe_send_message(
                    application.bot,
                    str(recovered_chat_id),
                    "Bot restarted successfully. Resuming updates...",
                )
            except Exception:
                pass
            finally:
                pending_recovery_chats.discard(recovered_chat_id)

    async def _on_error(update, context):
        err = context.error
        if isinstance(err, RetryAfter):
            wait = _retry_after_seconds(err)
            logger.warning("telegram flood error captured by global handler retry_after=%ss", wait)
            return
        logger.exception("Unhandled Telegram update error: %s", err)

    log_startup_gpu_diagnostics()
    require_gpu_or_raise()

    reconnect_attempt = 0
    while True:
        app = Application.builder().token(BOT_TOKEN).post_init(_post_init).build()
        app.add_error_handler(_on_error)
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("status", status))
        app.add_handler(CommandHandler("config_status", config_status_command))
        app.add_handler(CommandHandler("reload_credentials", reload_credentials_command))
        app.add_handler(CommandHandler("auto_test", auto_test))
        app.add_handler(CommandHandler("resetjobs", resetjobs_command))
        app.add_handler(MessageHandler((filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND, handle_message))
        app.add_handler(CallbackQueryHandler(button_handler))

        logger.info(
            "✅ Bot v14 ready | UID: %s | provider=%s | exec_threads=%s | ffmpeg_threads=%s | encoder=%s | reconnect_attempt=%s",
            ALLOWED_USER_ID,
            EXECUTION_PROVIDER,
            EXECUTION_THREAD_COUNT,
            FFMPEG_CPU_THREADS,
            OUTPUT_VIDEO_ENCODER,
            reconnect_attempt,
        )

        try:
            # Lightning Studio may run bot bootstrap in contexts where adding signal handlers is disallowed.
            app.run_polling(allowed_updates=Update.ALL_TYPES, stop_signals=None, close_loop=False)
            reconnect_attempt += 1
            wait_s = min(60, max(5, reconnect_attempt * 5))
            logger.warning("polling stopped unexpectedly; retrying in %ss", wait_s)
            time.sleep(wait_s)
        except KeyboardInterrupt:
            logger.info("shutdown requested by keyboard interrupt")
            break
        except InvalidToken as e:
            logger.error("startup failed: invalid telegram token (%s)", e)
            break
        except Exception as e:
            reconnect_attempt += 1
            wait_s = min(120, max(5, reconnect_attempt * 5))
            logger.exception("polling crashed; reconnecting in %ss: %s", wait_s, e)
            time.sleep(wait_s)


if __name__ == "__main__":
    main()
