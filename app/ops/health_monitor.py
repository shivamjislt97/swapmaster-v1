#!/usr/bin/env python3
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "pipeline"
LOG_DIR = PIPELINE_DIR / "logs"
BOT_PID_FILE = LOG_DIR / "bot.pid"
GUARD_PID_FILE = LOG_DIR / "process_guard.pid"
MONITOR_PID_FILE = LOG_DIR / "health_monitor.pid"
HEALTH_LOG_FILE = LOG_DIR / "health_monitor.log"
PROCESS_TREE_LOG_FILE = LOG_DIR / "process_tree.log"
STATE_FILE = LOG_DIR / "active_job_state.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0


def _log(text: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    with HEALTH_LOG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(f"{stamp} [health_monitor] {text}\n")


def _snapshot_process_tree() -> None:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,stat=,etime=,args="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return
        lines = []
        for line in (proc.stdout or "").splitlines():
            lo = line.lower()
            if "bot.py" in lo or "process_guard.py" in lo or "job_worker.py" in lo or "facefusion.py" in lo:
                lines.append(line.strip())
        with PROCESS_TREE_LOG_FILE.open("a", encoding="utf-8") as fp:
            fp.write(f"=== {int(time.time())} ===\n")
            for item in lines:
                fp.write(item + "\n")
    except Exception:
        return


def _cleanup_stale_state() -> None:
    if not STATE_FILE.exists():
        return
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    worker_pid = int(payload.get("worker_pid") or 0)
    processing_pid = int(payload.get("processing_pid") or 0)
    phase = str(payload.get("phase") or payload.get("status") or "").lower()
    if phase in {"download", "faceswap", "processing", "upload", "starting"}:
        worker_live = worker_pid > 0 and _pid_alive(worker_pid)
        processing_live = processing_pid > 0 and _pid_alive(processing_pid)
        if not worker_live and not processing_live:
            try:
                STATE_FILE.unlink(missing_ok=True)
                _log("stale active_job_state removed (no live worker or processing pid)")
            except Exception:
                pass


def _ensure_singleton() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if MONITOR_PID_FILE.exists():
        prev = _read_pid(MONITOR_PID_FILE)
        if prev > 0 and _pid_alive(prev):
            raise SystemExit(f"health_monitor already running with pid={prev}")
    MONITOR_PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _cleanup_pidfile() -> None:
    try:
        if MONITOR_PID_FILE.exists() and _read_pid(MONITOR_PID_FILE) == os.getpid():
            MONITOR_PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def run(interval_sec: int) -> int:
    active = True

    def _handle_stop(signum, _frame):
        nonlocal active
        _log(f"signal={signum} received")
        active = False

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    _ensure_singleton()
    try:
        while active:
            bot_pid = _read_pid(BOT_PID_FILE)
            guard_pid = _read_pid(GUARD_PID_FILE)
            bot_alive = bot_pid > 0 and _pid_alive(bot_pid)
            guard_alive = guard_pid > 0 and _pid_alive(guard_pid)
            _log(f"heartbeat bot_pid={bot_pid} bot_alive={int(bot_alive)} guard_pid={guard_pid} guard_alive={int(guard_alive)}")
            _cleanup_stale_state()
            _snapshot_process_tree()
            time.sleep(max(10, int(interval_sec)))
        return 0
    finally:
        _cleanup_pidfile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime health monitor")
    parser.add_argument("--interval-sec", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(run(interval_sec=max(10, int(args.interval_sec))))


if __name__ == "__main__":
    raise SystemExit(main())
