#!/usr/bin/env python3
import os
import time
from pathlib import Path


class AutoSleepConfig:
    def __init__(self, enabled: bool, minutes: int, source: str):
        self.enabled = bool(enabled)
        self.minutes = int(max(1, minutes))
        self.source = str(source)

    @property
    def delay_seconds(self) -> int:
        return int(self.minutes * 60)


def load_auto_sleep_config() -> AutoSleepConfig:
    raw_enabled = str(os.environ.get("AUTO_SLEEP_ENABLED", "1")).strip().lower()
    enabled = raw_enabled in {"1", "true", "yes", "on"}

    raw_minutes = str(os.environ.get("AUTO_SLEEP_MINUTES", "30")).strip()
    source = "AUTO_SLEEP_MINUTES"
    try:
        minutes = int(raw_minutes)
    except Exception:
        minutes = 30

    # Backward compatibility for existing runtime variable.
    if "AUTO_SLEEP_MINUTES" not in os.environ:
        legacy = str(os.environ.get("AUTO_SHUTDOWN_DELAY_SEC", "")).strip()
        if legacy:
            try:
                sec = int(legacy)
                if sec > 0:
                    minutes = max(1, int(round(sec / 60.0)))
                    source = "AUTO_SHUTDOWN_DELAY_SEC"
            except Exception:
                pass

    return AutoSleepConfig(enabled=enabled, minutes=minutes, source=source)


def append_auto_sleep_log(log_path: str | Path, event: str, details: str = "") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    line = f"{ts} | {event}"
    if details:
        line += f" | {details}"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")
