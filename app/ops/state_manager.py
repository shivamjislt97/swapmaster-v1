import hashlib
import json
import os
import time
from pathlib import Path


class PipelineStateStore:
    def __init__(self, state_file: str | Path):
        # Accept either a file path or a directory (legacy: directory passed by mistake)
        p = Path(state_file)
        if p.is_dir() or (not p.suffix and not p.exists()):
            # Caller passed a directory — use a default state file inside it
            p = p / "logs" / "pipeline_state.json"
        self.state_file = p
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict:
        if not self.state_file.exists():
            return {}
        try:
            return json.loads(self.state_file.read_text(errors="ignore"))
        except Exception:
            return {}

    def save(self, data: dict) -> None:
        """Atomic write — safe against crash/corruption."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_name(
            f"{self.state_file.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        try:
            tmp.write_text(json.dumps(data or {}, indent=2), encoding="utf-8")
            os.replace(tmp, self.state_file)
        except Exception:
            with suppress_err():
                tmp.unlink(missing_ok=True)
            raise

    def mark_failed(
        self,
        job_key: str,
        stage: str,
        error_class: str,
        details: str,
        stack_trace: str = "",
        error_log_path: str = "",
    ) -> None:
        """Persist failure state for a job — used for deduplication/idempotency checks."""
        data = self.load()
        data.setdefault("failures", {})[job_key] = {
            "stage": stage,
            "error_class": error_class,
            "details": details[:2000],
            "stack_trace": stack_trace[:2000],
            "error_log_path": error_log_path,
            "ts": time.time(),
        }
        self.save(data)


class suppress_err:
    """Minimal context manager to suppress all exceptions (like contextlib.suppress)."""
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return True


def compute_idempotency_key(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()


def validate_output_media(path: str | Path) -> tuple[bool, str]:
    """Validate output media file exists, is non-empty, and is a valid media container."""
    p = Path(path)
    if not p.exists():
        return False, "output path missing"
    if not p.is_file():
        return False, "output path is not a file"
    size = int(p.stat().st_size or 0)
    if size <= 0:
        return False, "output file empty"
    # Minimal container format check: MP4/MOV have 'ftyp' or 'moov' box near start
    ext = p.suffix.lower()
    if ext in (".mp4", ".mov", ".m4v"):
        try:
            with open(p, "rb") as f:
                header = f.read(12)
            # MP4 boxes: first 4 bytes = size, next 4 = type
            if len(header) >= 8:
                box_type = header[4:8]
                if box_type not in (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"):
                    return False, f"invalid MP4 box type {box_type!r} — file may be corrupted"
        except Exception:
            pass  # If we can't read, let size check be sufficient
    return True, f"size={size}"
