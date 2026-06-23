#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def _load_dotenv(path: Path) -> dict:
    data = {}
    if not path.exists():
        return data
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _check_import(project_root: Path) -> tuple[bool, str]:
    sys.path.insert(0, str(project_root))
    spec = importlib.util.spec_from_file_location("bot", project_root / "bot.py")
    if spec is None or spec.loader is None:
        return False, "unable to build import spec for bot.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return True, "bot.py import ok"


def _check_dirs(project_root: Path) -> tuple[bool, str]:
    dirs = [
        project_root / "pipeline/logs",
        project_root / "pipeline/queue",
        project_root / "pipeline/workspace",
        project_root / "pipeline/workspace/output",
        project_root / "persistent",
    ]
    verified = []
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".health_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink(missing_ok=True)
        verified.append(str(directory.relative_to(project_root)))
    return True, "writable dirs: " + ", ".join(verified)


def _check_env(project_root: Path) -> tuple[bool, str]:
    dotenv = _load_dotenv(project_root / ".env")
    env = {**dotenv, **{k: v for k, v in os.environ.items() if v}}
    missing = []
    if not (env.get("BOT_TOKEN") or env.get("TELEGRAM_BOT_TOKEN")):
        missing.append("BOT_TOKEN|TELEGRAM_BOT_TOKEN")
    if not (env.get("ALLOWED_CHAT_ID") or env.get("ALLOWED_USER_ID") or env.get("ALLOWED_TELEGRAM_CHAT_IDS")):
        missing.append("ALLOWED_CHAT_ID|ALLOWED_USER_ID|ALLOWED_TELEGRAM_CHAT_IDS")
    if missing:
        return False, "missing env keys: " + ", ".join(missing)
    return True, "required env keys present"


def _check_scripts(project_root: Path) -> tuple[bool, str]:
    scripts = [project_root / "run.sh", project_root / "start.sh"]
    missing = [str(path.relative_to(project_root)) for path in scripts if not path.exists()]
    not_executable = [str(path.relative_to(project_root)) for path in scripts if path.exists() and not os.access(path, os.X_OK)]
    if missing or not_executable:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if not_executable:
            parts.append("not executable: " + ", ".join(not_executable))
        return False, "; ".join(parts)
    return True, "startup scripts executable"


def _check_facefusion(project_root: Path) -> tuple[bool, str]:
    facefusion_dir = project_root / "facefusion"
    if not facefusion_dir.is_dir():
        return False, "facefusion directory missing"
    if not (facefusion_dir / "facefusion.py").exists() and not (facefusion_dir / "facefusion/__init__.py").exists():
        return False, "facefusion entry point missing"
    return True, "facefusion runtime present"


def run(project_root: Path) -> dict:
    checks = {
        "bot_import": _check_import,
        "runtime_dirs": _check_dirs,
        "env": _check_env,
        "scripts": _check_scripts,
        "facefusion": _check_facefusion,
    }
    results = {}
    for name, fn in checks.items():
        try:
            ok, detail = fn(project_root)
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results[name] = {"status": "PASS" if ok else "FAIL", "detail": detail}
    overall = "PASS" if all(row["status"] == "PASS" for row in results.values()) else "FAIL"
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "project_root": str(project_root),
        "overall": overall,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Runtime project health check")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    args = parser.parse_args(argv)

    report = run(Path(args.project_root).resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
