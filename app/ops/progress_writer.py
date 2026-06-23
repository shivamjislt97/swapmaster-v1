"""
Atomic progress file writer.
Writes pipeline/logs/current_job.json using field names that match
dashboard_v2.html expectations (frames_done, stage_key, elapsed_s, etc.)
"""

import json
import os
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_FILE = os.path.join(PROJECT_ROOT, "pipeline", "logs", "current_job.json")


def write_progress(data: dict) -> None:
    """Atomically write current job state to disk."""
    payload = {
        # Identity
        "job_id":        data.get("job_id", ""),
        "chat_id":       data.get("chat_id", ""),
        "queue_job_id":  data.get("queue_job_id", data.get("job_id", "")),
        "token":         data.get("token", ""),
        # Stage — dashboard uses stage_key, stage_num, stage_label
        "stage_key":     data.get("stage_key", _to_stage_key(data.get("stage", ""))),
        "stage_num":     data.get("stage_num", _stage_num(data.get("stage", ""))),
        "stage_label":   data.get("stage_label", data.get("stage_name", data.get("stage", ""))),
        "phase":         data.get("phase", data.get("status", "idle")),
        # Progress — dashboard uses pct, frames_done, frames_total
        "pct":           int(data.get("pct", data.get("percent", 0))),
        "frames_done":   int(data.get("frames_done", data.get("current_frame", 0))),
        "frames_total":  int(data.get("frames_total", data.get("total_frames", 0))),
        # Timing — dashboard uses elapsed_s, eta_s
        "elapsed_s":     int(data.get("elapsed_s", data.get("elapsed", 0))),
        "eta_s":         int(data.get("eta_s", data.get("eta_seconds", 0))),
        # GPU
        "gpu_util":      data.get("gpu_util", 0),
        "gpu_percent":   data.get("gpu_util", 0),
        "vram_gb":       data.get("vram_gb", 0),
        "gpu_name":      data.get("gpu_name", ""),
        # Speed
        "speed_fps":     data.get("speed_fps", 0),
        # State
        "completed":     data.get("completed", False),
        "success":       data.get("success", None),
        "details":       data.get("details", ""),
        # Source metadata
        "source":        data.get("source", "facefusion_direct"),
        "bot_alive":     data.get("bot_alive", True),
        "updated_at":    time.time(),
        "timestamp":     time.time(),
    }
    tmp = PROGRESS_FILE + ".tmp"
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, PROGRESS_FILE)


def read_progress() -> dict:
    """Read current progress. Returns empty dict if file missing or corrupt."""
    try:
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def clear_progress() -> None:
    write_progress({"phase": "idle", "stage_key": "queued", "stage_num": 0})


# ── helpers ──────────────────────────────────────────────────────────────────

_STAGE_KEY_MAP = {
    "download": "download", "downloading": "download",
    "validation": "validation", "validate": "validation",
    "extract": "extracting", "extracting": "extracting",
    "analysis": "analysis", "analysing": "analysis",
    "tracking": "tracking",
    "faceswap": "processing", "face swap processing": "processing",
    "processing": "processing", "swapping": "processing",
    "enhancement": "enhancement", "enhancing": "enhancement",
    "merging": "merging", "merge": "merging",
    "upload": "upload", "uploading": "upload",
    "completed": "completed", "complete": "completed",
    "failed": "failed",
}

_STAGE_NUM_MAP = {
    "download": 1, "validation": 2, "extracting": 3,
    "analysis": 4, "tracking": 5, "processing": 6,
    "enhancement": 7, "merging": 9, "upload": 10, "completed": 11,
}


def _to_stage_key(stage: str) -> str:
    return _STAGE_KEY_MAP.get(stage.lower().strip(), "processing") if stage else "queued"


def _stage_num(stage: str) -> int:
    key = _to_stage_key(stage)
    return _STAGE_NUM_MAP.get(key, 6)
