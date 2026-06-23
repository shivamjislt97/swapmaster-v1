from pathlib import Path


def safe_cleanup(path: str | Path | None = None) -> bool:
    # Best-effort compatibility shim for restored runtime.
    return True
