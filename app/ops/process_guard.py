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
BOT_FILE = ROOT_DIR / "bot.py"
STATE_FILE = LOG_DIR / "process_guard_state.json"
PID_FILE = LOG_DIR / "process_guard.pid"
LOG_FILE = LOG_DIR / "process_guard.log"
BOT_STDOUT_LOG = LOG_DIR / "bot_runtime.log"


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


def _write_state(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def _log(text: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"{stamp} [process_guard] {text}"
    with LOG_FILE.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")
    print(line, flush=True)


def _ensure_singleton() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        prev = _read_pid(PID_FILE)
        if _pid_matches(prev, "process_guard.py"):
            raise SystemExit(f"process_guard already running with pid={prev}")
        if prev > 0:
            _log(f"stale guard pid cleanup pid={prev} cmd='{_read_cmdline(prev)[:120]}'")
            PID_FILE.unlink(missing_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")


def _cleanup_pidfile() -> None:
    try:
        if PID_FILE.exists() and _read_pid(PID_FILE) == os.getpid():
            PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _build_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Ensure local bin is in PATH
    local_bin = str(Path.home() / ".local" / "bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = local_bin + ":" + env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    # Inject nvidia CUDA lib paths for onnxruntime-gpu
    import sys as _sys
    from pathlib import Path as _Path
    nvidia_base = _Path(_sys.executable).parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia"
    if nvidia_base.is_dir():
        nvidia_libs = [
            str(nvidia_base / sub / "lib")
            for sub in ["cublas", "cudnn", "cuda_runtime", "cufft", "curand", "cusolver", "cusparse", "nccl", "nvjitlink"]
            if (nvidia_base / sub / "lib").is_dir()
        ]
        if nvidia_libs:
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + (f":{current}" if current else "")
    return env


def _ensure_service(name: str, script: str) -> None:
    """Start a companion service if not already running."""
    try:
        out = subprocess.check_output(["pgrep", "-f", script], stderr=subprocess.DEVNULL)
        if out.strip():
            return  # already running
    except Exception:
        pass
    log_file = LOG_DIR / f"{name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT_DIR / "ops" / script)],
            cwd=str(ROOT_DIR),
            env=_build_env(),
            stdout=lf,
            stderr=lf,
            start_new_session=True,
        )
    _log(f"[AUTO_START] {name} pid={proc.pid}")


def run_guard(max_backoff: int = 120) -> int:
    _ensure_singleton()
    restart_count = 0
    backoff_steps = [2, 5, 10]
    backoff_idx = 0
    active = True
    proc: subprocess.Popen | None = None

    def _handle_stop(signum, _frame):
        nonlocal active
        _log(f"signal={signum} received, shutting down")
        active = False
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    try:
        while active:
            restart_count += 1
            _ensure_service("progress_poller", "progress_poller.py")
            _ensure_service("health_monitor", "health_monitor.py")
            with BOT_STDOUT_LOG.open("a", encoding="utf-8") as out_fp:
                _log(f"starting bot attempt={restart_count}")
                _log(f"[BOT_RESTART] attempt={restart_count}")
                _log(f"[BOOT] bot spawn started attempt={restart_count}")
                proc = subprocess.Popen(
                    [sys.executable, str(BOT_FILE)],
                    cwd=str(ROOT_DIR),
                    env=_build_env(),
                    stdout=out_fp,
                    stderr=out_fp,
                    start_new_session=True,
                )

            _write_state(
                {
                    "guard_pid": os.getpid(),
                    "bot_pid": int(proc.pid),
                    "restart_count": int(restart_count),
                    "started_at": int(time.time()),
                    "status": "running",
                }
            )

            rc = proc.wait()
            _log(f"bot exited rc={rc}")
            _write_state(
                {
                    "guard_pid": os.getpid(),
                    "bot_pid": int(proc.pid),
                    "restart_count": int(restart_count),
                    "exited_at": int(time.time()),
                    "last_rc": int(rc),
                    "status": "exited",
                }
            )

            if not active:
                break

            if rc == 0:
                backoff_idx = 0
                _log("bot exited cleanly; restarting to keep service alive")
                continue

            _log(f"[BOT_CRASH] rc={rc}")
            _log(f"[BOOT_FAIL] reason=bot_exit_rc_{rc}")
            backoff = backoff_steps[min(backoff_idx, len(backoff_steps) - 1)]
            _log(f"[BACKOFF_WAIT] seconds={backoff}")
            _log(f"crash restart in {backoff}s")
            time.sleep(backoff)
            if backoff_idx < len(backoff_steps) - 1:
                backoff_idx += 1
            if max_backoff < backoff:
                _log(f"max_backoff override active max_backoff={max_backoff}")

        return 0
    finally:
        _cleanup_pidfile()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bot process guard")
    parser.add_argument("--max-backoff", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(run_guard(max_backoff=max(5, int(args.max_backoff))))


if __name__ == "__main__":
    raise SystemExit(main())
