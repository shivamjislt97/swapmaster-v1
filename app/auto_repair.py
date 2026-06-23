#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: str, cwd: Path | None = None) -> tuple[bool, str]:
    p = subprocess.run(cmd, shell=True, text=True, capture_output=True, cwd=str(cwd) if cwd else None)
    out = (p.stdout or "") + ("\n" + p.stderr if p.stderr else "")
    return p.returncode == 0, out.strip()


def restore_env(project_root: Path, env_snapshot: Path) -> tuple[bool, str]:
    if not env_snapshot.exists():
        return False, f"env snapshot missing: {env_snapshot}"
    marker = "# active process env snapshot"
    text = env_snapshot.read_text(errors="ignore")
    env_block = text.split(marker)[0].strip()
    if not env_block:
        return False, "env snapshot did not contain project env block"
    target = project_root / ".env"
    target.write_text(env_block + "\n")
    return True, f"restored {target}"


def fix_permissions(project_root: Path) -> tuple[bool, str]:
    paths = [project_root / "run.sh", project_root / "start.sh", project_root / "ops/start-bot-native.sh"]
    changed = []
    for p in paths:
        if p.exists():
            mode = p.stat().st_mode
            p.chmod(mode | 0o111)
            changed.append(str(p))
    return True, "chmod +x applied to: " + ", ".join(changed)


def reinstall_python_deps(project_root: Path, req_file: Path) -> tuple[bool, str]:
    py = project_root / "venv/bin/python"
    pip = project_root / "venv/bin/pip"
    if pip.exists() and req_file.exists():
        return run(f"{pip} install -r {req_file}", cwd=project_root)
    if req_file.exists():
        return run(f"python3 -m pip install -r {req_file}", cwd=project_root)
    return False, f"requirements file missing: {req_file}"


def repair_cuda_path(project_root: Path) -> tuple[bool, str]:
    candidates = []
    for base in [project_root / "venv/lib", Path("/usr/local/cuda/lib64"), Path("/usr/lib/x86_64-linux-gnu")]:
        if base.exists():
            candidates.append(str(base))
    if not candidates:
        return False, "no CUDA library directories discovered"
    old = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(candidates + ([old] if old else []))
    os.environ["LD_LIBRARY_PATH"] = merged
    return True, f"LD_LIBRARY_PATH set to: {merged}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--env-snapshot", required=True)
    parser.add_argument("--requirements", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    env_snapshot = Path(args.env_snapshot)
    req_file = Path(args.requirements)

    steps = [
        ("restore_env", restore_env(project_root, env_snapshot)),
        ("fix_permissions", fix_permissions(project_root)),
        ("reinstall_python_deps", reinstall_python_deps(project_root, req_file)),
        ("repair_cuda_path", repair_cuda_path(project_root)),
    ]

    failed = []
    for name, (ok, detail) in steps:
        print(f"[{name}] {'PASS' if ok else 'FAIL'}: {detail}")
        if not ok:
            failed.append(name)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
