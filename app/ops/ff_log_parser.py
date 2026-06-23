"""
Parse worker_trace log for FPS and stage info.
FaceFusion uses tqdm: "43%|====| 37808/87200 [4:04:37<5:37:12, 2.44frame/s]"
"""

import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "pipeline", "logs")

# tqdm line from FaceFusion worker
_TQDM_RE  = re.compile(r'\[ff\]\s+\w+:\s+(\d+)%.*?(\d+)/(\d+).*?([\d.]+)frame/s', re.I)
_ERROR_RE = re.compile(r'error|exception|traceback', re.I)


def parse_worker_log(chat_id: str, job_id: str, last_n: int = 100) -> dict:
    result = {"fps_from_log": 0.0, "stage_from_log": "", "errors": []}
    path = os.path.join(LOGS_DIR, f"worker_trace_{chat_id}_{job_id}.log")
    if not os.path.exists(path):
        return result
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines[-last_n:]):
            if not result["fps_from_log"]:
                m = _TQDM_RE.search(line)
                if m:
                    result["fps_from_log"] = float(m.group(4))
            if _ERROR_RE.search(line) and len(result["errors"]) < 3:
                result["errors"].append(line.strip()[:200])
            if result["fps_from_log"]:
                break
    except Exception:
        pass
    return result
