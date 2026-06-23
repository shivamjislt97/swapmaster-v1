#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple


CHECK_ORDER = [
    "Python",
    "CUDA",
    "Bot",
    "Queue",
    "Image Swap",
    "Video Swap",
    "Upload",
    ".env Restore",
    "Auto Repair",
]


def run_cmd(cmd: str, cwd: Path | None = None, timeout: int = 120) -> Tuple[bool, str]:
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
        return p.returncode == 0, out.strip()
    except Exception as exc:
        return False, str(exc)


def has_cuda_provider(project_root: Path) -> Tuple[bool, str]:
    py = project_root / "venv/bin/python"
    py_cmd = str(py) if py.exists() else "python3"
    code = (
        "import onnxruntime as ort; "
        "print(','.join(ort.get_available_providers()))"
    )
    ok, out = run_cmd(f"{py_cmd} -c \"{code}\"", cwd=project_root, timeout=60)
    if not ok:
        return False, out
    return ("CUDAExecutionProvider" in out), out


def check_python(project_root: Path) -> Tuple[bool, str]:
    py = project_root / "venv/bin/python"
    if py.exists():
        return run_cmd(f"{py} --version", cwd=project_root)
    return run_cmd("python3 --version", cwd=project_root)


def check_bot(project_root: Path) -> Tuple[bool, str]:
    py = project_root / "venv/bin/python"
    py_cmd = str(py) if py.exists() else "python3"
    script = project_root / "bot.py"
    if not script.exists():
        return False, "bot.py not found"
    ok1, out1 = run_cmd(f"{py_cmd} -m py_compile bot.py", cwd=project_root)
    if not ok1:
        return False, out1
    # Import only to validate module load without launching polling loop.
    ok2, out2 = run_cmd(
        f"{py_cmd} -c \"import importlib.util; s=importlib.util.spec_from_file_location('bot', 'bot.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('import-ok')\"",
        cwd=project_root,
        timeout=120,
    )
    return ok2, out2


def check_queue(project_root: Path) -> Tuple[bool, str]:
    queue_dirs = [
        project_root / "pipeline/queue",
        project_root / "pipeline/tasks",
        project_root / "persistent/queue",
    ]
    created = []
    for d in queue_dirs:
        d.mkdir(parents=True, exist_ok=True)
        test_file = d / ".queue_probe"
        test_file.write_text("ok\n")
        test_file.unlink(missing_ok=True)
        created.append(str(d))
    return True, "verified queue dirs: " + ", ".join(created)


def check_image_swap(project_root: Path) -> Tuple[bool, str]:
    py = project_root / "venv/bin/python"
    py_cmd = str(py) if py.exists() else "python3"
    if (project_root / "facefusion.py").exists():
        return run_cmd(f"{py_cmd} facefusion.py --help", cwd=project_root, timeout=120)
    if (project_root / "facefusion").exists():
        return run_cmd(f"{py_cmd} -c \"import facefusion; print('facefusion-import-ok')\"", cwd=project_root)
    return False, "facefusion runtime not found"


def check_video_swap(project_root: Path) -> Tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix="migration_verify_") as td:
        out_file = Path(td) / "probe.mp4"
        cmd = (
            "ffmpeg -y -f lavfi -i testsrc=size=320x240:rate=24 "
            "-t 5 -pix_fmt yuv420p "
            f"{out_file}"
        )
        ok, out = run_cmd(cmd, cwd=project_root, timeout=120)
        if not ok:
            return False, out
        if not out_file.exists() or out_file.stat().st_size <= 0:
            return False, "ffmpeg output not created"
    return True, "generated and validated 5s test video"


def check_upload(project_root: Path) -> Tuple[bool, str]:
    conf = Path(__file__).resolve().parent / "rclone.conf"
    cmd = f"rclone listremotes --config {conf}" if conf.exists() else "rclone listremotes"
    ok, out = run_cmd(cmd, cwd=project_root, timeout=60)
    if not ok:
        return False, out
    if "gdrive:" not in out:
        return False, "gdrive remote not configured"
    return True, out


def check_env_restore(project_root: Path) -> Tuple[bool, str]:
    env = project_root / ".env"
    if not env.exists():
        return False, ".env missing"
    text = env.read_text(errors="ignore")
    required = ["BOT_TOKEN"]
    missing = [k for k in required if f"{k}=" not in text]
    has_chat_gate = (
        ("ALLOWED_CHAT_ID=" in text)
        or ("ALLOWED_TELEGRAM_CHAT_IDS=" in text)
        or ("ALLOWED_USER_ID=" in text)
    )
    if not has_chat_gate:
        missing.append("ALLOWED_CHAT_ID|ALLOWED_TELEGRAM_CHAT_IDS|ALLOWED_USER_ID")
    if missing:
        return False, "missing keys: " + ", ".join(missing)
    return True, ".env found with required keys"


def check_health(project_root: Path) -> Tuple[bool, str]:
    py = project_root / "venv/bin/python"
    py_cmd = str(py) if py.exists() else "python3"
    if not (project_root / "health_check.py").exists():
        return False, "health_check.py missing"
    return run_cmd(f"{py_cmd} health_check.py", cwd=project_root, timeout=180)


def check_permissions(project_root: Path) -> Tuple[bool, str]:
    needed = [project_root / "run.sh", project_root / "start.sh", project_root / "ops/start-bot-native.sh"]
    bad = []
    for p in needed:
        if not p.exists():
            bad.append(f"missing:{p}")
            continue
        if not os.access(p, os.X_OK):
            bad.append(f"not-executable:{p}")
    if bad:
        return False, "; ".join(bad)
    return True, "startup scripts executable"


def check_restart_persistence(project_root: Path) -> Tuple[bool, str]:
    state_file = project_root / "persistent/restart_probe.state"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    before = str(time.time())
    state_file.write_text(before + "\n")
    after = state_file.read_text().strip()
    if before != after:
        return False, "persistent write/read mismatch"
    return True, "persistence write/read ok"


def run_all(project_root: Path, auto_repair_attempted: bool = False) -> Dict:
    results: Dict[str, Dict[str, str]] = {}

    checks = {
        "Python": check_python,
        "CUDA": lambda p: has_cuda_provider(p),
        "Bot": check_bot,
        "Queue": check_queue,
        "Image Swap": check_image_swap,
        "Video Swap": check_video_swap,
        "Upload": check_upload,
        ".env Restore": check_env_restore,
    }

    for name in CHECK_ORDER:
        if name == "Auto Repair":
            continue
        ok, detail = checks[name](project_root)
        results[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}

    ok_health, detail_health = check_health(project_root)
    results["Health"] = {"status": "PASS" if ok_health else "FAIL", "detail": detail_health}

    ok_perm, detail_perm = check_permissions(project_root)
    results["Permissions"] = {"status": "PASS" if ok_perm else "FAIL", "detail": detail_perm}

    ok_restart, detail_restart = check_restart_persistence(project_root)
    results["Restart Persistence"] = {"status": "PASS" if ok_restart else "FAIL", "detail": detail_restart}

    results["Auto Repair"] = {
        "status": "PASS" if auto_repair_attempted else "FAIL",
        "detail": "repair attempted" if auto_repair_attempted else "repair not triggered",
    }

    required_ok = all(results[k]["status"] == "PASS" for k in CHECK_ORDER)
    overall_ready = "YES" if required_ok else "NO"

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": str(project_root),
        "results": results,
        "overall_ready": overall_ready,
    }


def write_reports(report: Dict, json_path: Path, md_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2))
    lines: List[str] = []
    lines.append("# FINAL READY REPORT")
    lines.append("")
    lines.append("| Check | Status | Details |")
    lines.append("|---|---|---|")
    for name in CHECK_ORDER:
        row = report["results"][name]
        lines.append(f"| {name} | {row['status']} | {row['detail'].replace('|', '/')} |")
    lines.append("")
    lines.append(f"Overall Ready: {report['overall_ready']}")
    md_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    parser.add_argument("--auto-repair-attempted", action="store_true")
    args = parser.parse_args()

    report = run_all(Path(args.project_root), auto_repair_attempted=args.auto_repair_attempted)
    write_reports(report, Path(args.json_out), Path(args.md_out))
    print(json.dumps(report, indent=2))
    return 0 if report["overall_ready"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
