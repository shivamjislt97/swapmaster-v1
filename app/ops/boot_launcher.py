#!/usr/bin/env python3
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_DIR = ROOT_DIR / "pipeline"
LOG_DIR = PIPELINE_DIR / "logs"
BOOT_LOG_FILE = LOG_DIR / "boot_launcher.log"
GUARD_PID_FILE = LOG_DIR / "process_guard.pid"
MONITOR_PID_FILE = LOG_DIR / "health_monitor.pid"
POLLER_PID_FILE = LOG_DIR / "progress_poller.pid"


def _log(text: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"{stamp} [boot_launcher] {text}"
    with BOOT_LOG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")
    print(line, flush=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


def _pid_matches(pid: int, expected_substr: str) -> bool:
    if pid <= 0 or not _pid_alive(pid):
        return False
    return expected_substr in _read_cmdline(pid)


def _read_pid(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0


def _spawn_if_missing(name: str, cmd: list[str], pid_file: Path, expected_substr: str) -> int:
    pid = _read_pid(pid_file)
    if _pid_matches(pid, expected_substr):
        _log(f"{name} already running pid={pid}")
        return pid
    if pid > 0:
        _log(f"stale pid cleanup name={name} pid={pid} cmd='{_read_cmdline(pid)[:120]}'")
        pid_file.unlink(missing_ok=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        env=os.environ.copy(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _log(f"started {name} pid={proc.pid}")
    if proc.poll() is not None:
        _log(f"[BOOT_FAIL] reason={name}_spawn_exit rc={proc.returncode}")
    else:
        _log(f"[BOOT] supervisor started service={name} pid={proc.pid}")
    return int(proc.pid)


def run_boot(interval_sec: int, max_backoff: int) -> int:
    guard_cmd = [sys.executable, str(ROOT_DIR / "ops" / "process_guard.py"), "--max-backoff", str(max_backoff)]
    monitor_cmd = [sys.executable, str(ROOT_DIR / "ops" / "health_monitor.py"), "--interval-sec", str(interval_sec)]
    poller_cmd = [sys.executable, str(ROOT_DIR / "ops" / "progress_poller.py")]

    _spawn_if_missing("process_guard", guard_cmd, GUARD_PID_FILE, "process_guard.py")
    _spawn_if_missing("health_monitor", monitor_cmd, MONITOR_PID_FILE, "health_monitor.py")
    _spawn_if_missing("progress_poller", poller_cmd, POLLER_PID_FILE, "progress_poller.py")
    _log("[BOOT] supervisor started")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boot launcher for bot lifecycle services")
    parser.add_argument("--interval-sec", type=int, default=120)
    parser.add_argument("--max-backoff", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(run_boot(interval_sec=max(10, int(args.interval_sec)), max_backoff=max(5, int(args.max_backoff))))


if __name__ == "__main__":
    raise SystemExit(main())
