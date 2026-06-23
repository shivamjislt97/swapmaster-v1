"""
Standalone progress poller — runs independently of bot.py.

Every 2 seconds:
  1. Reads pipeline/logs/active_job_state.json
  2. Counts frames directly from FaceFusion temp dir
  3. Gets GPU stats from nvidia-smi
  4. Writes pipeline/logs/current_job.json  → feeds /ws/live
  5. Updates active session snapshot         → feeds /api/current + /api/job/{token}

Start: python ops/progress_poller.py
"""

import asyncio
import glob
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ops.progress_writer import write_progress, read_progress, _to_stage_key, _stage_num
from ops.frame_counter import get_frame_count_from_active_state
from ops.gpu_monitor import get_gpu_stats
from ops.ff_log_parser import parse_worker_log

ACTIVE_STATE_FILE = os.path.join(PROJECT_ROOT, "pipeline", "logs", "active_job_state.json")
SESSIONS_DIR      = os.path.join(PROJECT_ROOT, "pipeline", "dashboard_sessions")
POLL_INTERVAL     = 2


def _read_active_state() -> dict:
    try:
        with open(ACTIVE_STATE_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _is_bot_alive() -> bool:
    try:
        return bool(subprocess.check_output(
            ["pgrep", "-f", "python.*bot.py"], stderr=subprocess.DEVNULL
        ).strip())
    except Exception:
        return False


def _find_active_session_path(chat_id: str) -> str | None:
    """Find the most recent non-completed session snapshot for this chat."""
    best_path, best_ts = None, 0.0
    for path in glob.glob(os.path.join(SESSIONS_DIR, "*.snapshot.json")):
        try:
            with open(path) as f:
                snap = json.load(f)
            if str(snap.get("chat_id", "")) == chat_id and not snap.get("completed"):
                mtime = os.path.getmtime(path)
                if mtime > best_ts:
                    best_ts, best_path = mtime, path
        except Exception:
            continue
    return best_path


def _update_session_snapshot(snap_path: str, updates: dict) -> None:
    """Merge updates into an existing session snapshot atomically."""
    try:
        with open(snap_path) as f:
            snap = json.load(f)
        snap.update(updates)
        snap["updated_at"] = time.time()
        tmp = snap_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, ensure_ascii=False, indent=2)
        os.replace(tmp, snap_path)
    except Exception:
        pass


async def poll_once() -> None:
    state = _read_active_state()
    status = str(state.get("status", "")).lower()

    if not state or status in ("", "idle", "completed", "failed"):
        existing = read_progress()
        if existing.get("phase") not in ("idle", ""):
            write_progress({"phase": "idle", "stage_key": "queued", "source": "facefusion_direct"})
        return

    chat_id = str(state.get("chat_id", ""))
    job_id  = str(state.get("job_id", ""))

    # Direct frame count from disk
    frames = get_frame_count_from_active_state(state)

    # GPU
    gpu = get_gpu_stats()

    # Log-derived stage/FPS
    log_data = parse_worker_log(chat_id, job_id) if chat_id and job_id else {}

    # Timing — guard against start_time=0 (missing/unset) which would produce ~55-year elapsed
    start_time = float(state.get("start_time", 0))
    now = time.time()
    if start_time <= 0 or start_time > now:
        start_time = now  # treat as just-started; elapsed=0, ETA unknown
    elapsed = int(now - start_time)
    pct = frames["percent"]
    eta = int(elapsed / (pct / 100) - elapsed) if pct > 0 and elapsed > 0 else 0

    stage_raw = log_data.get("stage_from_log") or state.get("stage", "processing")
    stage_key = _to_stage_key(stage_raw)
    stage_num = _stage_num(stage_raw)

    payload = {
        "job_id":       job_id,
        "chat_id":      chat_id,
        "queue_job_id": job_id,
        "stage_key":    stage_key,
        "stage_num":    stage_num,
        "stage_label":  stage_raw,
        "phase":        status,
        "pct":          int(pct),
        "frames_done":  frames["processed_frames"],
        "frames_total": frames["total_frames"],
        "elapsed_s":    elapsed,
        "eta_s":        eta,
        "speed_fps":    log_data.get("fps_from_log", 0),
        "gpu_util":     gpu.get("gpu_util", 0),
        "vram_gb":      gpu.get("vram_used", 0),
        "gpu_name":     gpu.get("gpu_name", ""),
        "source":       "facefusion_direct",
        "bot_alive":    _is_bot_alive(),
    }

    # 1. Write current_job.json → feeds /ws/live
    write_progress(payload)

    # 2. Update active session snapshot → feeds /api/current + /api/job/{token}
    snap_path = _find_active_session_path(chat_id)
    if snap_path:
        pct_int = int(payload["pct"])
        fd = payload["frames_done"]
        ft = payload["frames_total"]
        elapsed = payload["elapsed_s"]
        eta = payload["eta_s"]

        def _fmt(sec):
            sec = max(0, int(sec or 0))
            h, r = divmod(sec, 3600)
            m, s = divmod(r, 60)
            return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"

        bar = "█" * (pct_int // 10) + "░" * (10 - pct_int // 10)
        live_progress_text = (
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔄 Face Swap Processing\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{bar} {pct_int}%\n\n"
            f"🎞 Frames: {fd:,}/{ft:,}\n\n"
            f"⏱️ Elapsed: {_fmt(elapsed)}\n\n"
            f"⌛️ ETA: {_fmt(eta)}\n\n"
            f"📡 Source: FaceFusion Direct"
        )

        _update_session_snapshot(snap_path, {
            "pct":           pct_int,
            "frames_done":   fd,
            "frames_total":  ft,
            "stage_key":     payload["stage_key"],
            "stage_num":     payload["stage_num"],
            "stage_label":   payload["stage_label"],
            "phase":         payload["phase"],
            "elapsed_s":     elapsed,
            "eta_s":         eta,
            "details":       f"Direct sync · {fd:,}/{ft:,} frames",
            "progress_text": live_progress_text,
            # extra fields dashboard reads
            "gpu_util":      payload["gpu_util"],
            "vram_gb":       payload["vram_gb"],
            "speed_fps":     payload["speed_fps"],
            "source":        "facefusion_direct",
            "bot_alive":     payload["bot_alive"],
        })


async def poll_loop() -> None:
    print(f"[POLLER] Starting — poll every {POLL_INTERVAL}s", flush=True)
    while True:
        try:
            await poll_once()
        except Exception as e:
            print(f"[POLLER] Error: {e}", flush=True)
        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(poll_loop())
