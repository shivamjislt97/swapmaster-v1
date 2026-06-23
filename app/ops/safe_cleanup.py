from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterable


_MEDIA_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".3gp", ".ts",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
}

_TEMP_EXTS = {
    ".tmp", ".temp", ".part", ".partial", ".cache", ".log", ".txt", ".chunk",
    ".frame", ".dat", ".bin", ".pyc",
}


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_protected(path: Path, protected_roots: list[Path]) -> bool:
    for root in protected_roots:
        if path == root or _is_relative_to(path, root):
            return True
    return False


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (p for p in root.rglob("*") if p.is_file())


def _should_delete(path: Path, mode: str, output_dir: Path) -> bool:
    suffix = path.suffix.lower()
    name = path.name
    if mode == "outputs_old":
        return _is_relative_to(path, output_dir)
    if mode == "outputs_all":
        return _is_relative_to(path, output_dir)
    if mode == "temp_only":
        return suffix in _TEMP_EXTS or suffix in _MEDIA_EXTS
    if mode in {"full_clean", "deep_clean"}:
        # Also delete state/log files in workspace/temp (not in pipeline/logs root)
        return suffix in _TEMP_EXTS or suffix in _MEDIA_EXTS or suffix in {".json", ".jsonl"}
    return False


def disk_usage_percent(target_path: str | os.PathLike[str]) -> float:
    target = Path(target_path).resolve()
    usage = shutil.disk_usage(target)
    if usage.total <= 0:
        return 0.0
    return float(usage.used / usage.total * 100.0)


def run_cleanup(
    mode: str,
    root_dir,
    pipeline_dir,
    workspace_dir,
    temp_dir,
    output_dir,
    protected_paths=None,
    min_age_seconds: int = 0,
    audit_log_path: str | None = None,
    logger=None,
):
    mode_selected = mode if mode in {"temp_only", "outputs_old", "outputs_all", "full_clean", "deep_clean"} else "temp_only"
    root = Path(root_dir).resolve()
    pipeline = Path(pipeline_dir).resolve()
    workspace = Path(workspace_dir).resolve()
    temp = Path(temp_dir).resolve()
    outputs = Path(output_dir).resolve()
    now = time.time()

    protected_roots = [Path(p).resolve() for p in (protected_paths or []) if p]
    downloads = pipeline / "downloads"
    scan_roots: list[Path] = []
    if mode_selected == "outputs_old" or mode_selected == "outputs_all":
        scan_roots = [outputs]
    elif mode_selected == "temp_only":
        scan_roots = [temp, workspace]
    elif mode_selected == "full_clean":
        scan_roots = [temp, workspace, outputs, downloads]
    else:  # deep_clean
        scan_roots = [pipeline, workspace, temp, outputs, downloads, root]

    before = _size_bytes(pipeline)
    deleted_files = 0
    deleted_bytes = 0
    skipped_protected = 0
    skipped_unknown = 0
    skipped_recent = 0
    skipped_locked = 0
    plan = []
    seen = set()

    for scan_root in scan_roots:
        for fp in _iter_files(scan_root):
            resolved = fp.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)

            if _is_protected(resolved, protected_roots):
                skipped_protected += 1
                continue

            try:
                stat = resolved.stat()
            except OSError:
                skipped_unknown += 1
                continue

            age = now - float(stat.st_mtime)
            if min_age_seconds > 0 and age < min_age_seconds:
                skipped_recent += 1
                continue

            if not _should_delete(resolved, mode_selected, outputs):
                continue

            size = int(stat.st_size)
            zone = "outputs" if _is_relative_to(resolved, outputs) else "downloads" if "downloads" in resolved.parts else "workspace"

            try:
                resolved.unlink(missing_ok=True)
                deleted_files += 1
                deleted_bytes += size
                plan.append(
                    {
                        "action": "delete",
                        "zone": zone,
                        "path": str(resolved),
                        "size": size,
                    }
                )
            except PermissionError:
                skipped_locked += 1
            except OSError:
                skipped_unknown += 1

    after = _size_bytes(pipeline)

    # Remove empty directories left behind (skip protected roots)
    if mode_selected in {"temp_only", "full_clean", "deep_clean"}:
        for scan_root in scan_roots:
            if not scan_root.exists():
                continue
            # Walk bottom-up so nested empty dirs are caught
            for dirpath, dirnames, filenames in os.walk(str(scan_root), topdown=False):
                dp = Path(dirpath).resolve()
                if dp == scan_root.resolve():
                    continue
                if _is_protected(dp, protected_roots):
                    continue
                try:
                    if not any(dp.iterdir()):
                        dp.rmdir()
                except OSError:
                    pass
    result = {
        "mode": mode_selected,
        "before": before,
        "after": after,
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "skipped_protected": skipped_protected,
        "skipped_unknown": skipped_unknown,
        "skipped_recent": skipped_recent,
        "skipped_locked": skipped_locked,
        "plan": plan,
    }

    if audit_log_path:
        try:
            audit_fp = Path(audit_log_path)
            audit_fp.parent.mkdir(parents=True, exist_ok=True)
            with audit_fp.open("a", encoding="utf-8") as af:
                af.write(json.dumps({"ts": int(now), **result}, ensure_ascii=True) + "\n")
            summary_fp = audit_fp.parent / "last_cleanup_summary.json"
            summary_fp.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception as e:
            if logger:
                logger.warning("safe_cleanup audit write failed: %s", e)

    return result


def _size_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if p.is_file():
            try:
                total += int(p.stat().st_size)
            except OSError:
                continue
    return total
