"""Live web dashboard that mirrors Telegram progress updates.

This module exposes:

* A small file-backed session store. Every update is written atomically so the
  Telegram bot (main process) and the pipeline worker (subprocess) can both
  emit events into the same session.
* A FastAPI/uvicorn server that serves a dashboard HTML page and JSON APIs.
  The dashboard page polls ``/api/job/{token}`` every second and rebuilds the
  view, mirroring exactly what Telegram shows.

The module is intentionally dependency-light at import time so that unit tests
can use the storage helpers without spinning up a server.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session storage helpers (cross-process safe via atomic file writes)
# ---------------------------------------------------------------------------

DEFAULT_MAX_MESSAGES = 200
DEFAULT_MAX_PROGRESS_HISTORY = 120
_SNAPSHOT_LOCK = threading.Lock()


def _sanitize_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        raise ValueError("dashboard token must be a non-empty string")
    safe = "".join(ch for ch in token if ch.isalnum() or ch in {"-", "_"})
    if not safe:
        raise ValueError(f"dashboard token contains no safe characters: {token!r}")
    return safe


def generate_session_token() -> str:
    """Return a URL-safe token used for the dashboard session URL."""
    return secrets.token_urlsafe(12)


def session_dir(sessions_root: str | Path) -> Path:
    path = Path(sessions_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_path(sessions_root: str | Path, token: str) -> Path:
    return session_dir(sessions_root) / f"{_sanitize_token(token)}.snapshot.json"


def events_path(sessions_root: str | Path, token: str) -> Path:
    return session_dir(sessions_root) / f"{_sanitize_token(token)}.events.jsonl"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use PID + nanosecond timestamp to avoid tmp name collision between bot process and worker subprocess
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("dashboard snapshot read failed path=%s err=%s", path, exc)
        return {}


def load_snapshot(sessions_root: str | Path, token: str) -> Dict[str, Any]:
    return _read_json(snapshot_path(sessions_root, token))


def list_sessions(sessions_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(sessions_root)
    if not root.exists():
        return []
    sessions: List[Dict[str, Any]] = []
    for snap_file in sorted(root.glob("*.snapshot.json")):
        snap = _read_json(snap_file)
        if not snap:
            continue
        sessions.append({
            "token": snap.get("token"),
            "chat_id": snap.get("chat_id"),
            "phase": snap.get("phase"),
            "stage_label": snap.get("stage_label"),
            "pct": snap.get("pct"),
            "completed": snap.get("completed"),
            "success": snap.get("success"),
            "created_at": snap.get("created_at"),
            "updated_at": snap.get("updated_at"),
        })
    sessions.sort(key=lambda s: float(s.get("updated_at") or 0), reverse=True)
    return sessions


def _empty_snapshot(token: str, **fields: Any) -> Dict[str, Any]:
    now = time.time()
    snap: Dict[str, Any] = {
        "token": token,
        "chat_id": "",
        "video_link": "",
        "queue_job_id": "",
        "created_at": now,
        "updated_at": now,
        "stage_key": "queued",
        "stage_label": "Queued",
        "stage_num": 0,
        "stage_total": 6,
        "phase": "queued",
        "pct": 0,
        "frames_done": 0,
        "frames_total": 0,
        "elapsed_s": 0,
        "eta_s": 0,
        "details": "",
        "extra_info": "",
        "progress_text": "",
        "messages": [],
        "progress_history": [],
        "completed": False,
        "success": None,
        "result": {},
    }
    snap.update({k: v for k, v in fields.items() if v is not None})
    return snap


def register_session(
    sessions_root: str | Path,
    token: str,
    chat_id: str | int = "",
    video_link: str = "",
    queue_job_id: str | int = "",
    **extra: Any,
) -> Dict[str, Any]:
    """Create or refresh a session snapshot before the job starts."""
    token = _sanitize_token(token)
    snap_path = snapshot_path(sessions_root, token)
    evt_path = events_path(sessions_root, token)
    with _SNAPSHOT_LOCK:
        if snap_path.exists():
            snap = _read_json(snap_path) or _empty_snapshot(token)
        else:
            snap = _empty_snapshot(token)
        snap["token"] = token
        if chat_id != "":
            snap["chat_id"] = str(chat_id)
        if video_link:
            snap["video_link"] = str(video_link)
        if queue_job_id != "":
            snap["queue_job_id"] = str(queue_job_id)
        for k, v in extra.items():
            if v is not None:
                snap[k] = v
        snap["updated_at"] = time.time()
        _atomic_write_json(snap_path, snap)
        if not evt_path.exists():
            evt_path.parent.mkdir(parents=True, exist_ok=True)
            evt_path.touch()
        _append_event(sessions_root, token, {"type": "session_registered"})
    return snap


def _append_event(sessions_root: str | Path, token: str, event: Dict[str, Any]) -> None:
    payload = {"ts": time.time(), **event}
    line = json.dumps(payload, ensure_ascii=False)
    path = events_path(sessions_root, token)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(line + "\n")


def _merge_snapshot(
    sessions_root: str | Path,
    token: str,
    updates: Dict[str, Any],
) -> Dict[str, Any]:
    token = _sanitize_token(token)
    path = snapshot_path(sessions_root, token)
    with _SNAPSHOT_LOCK:
        snap = _read_json(path)
        if not snap:
            snap = _empty_snapshot(token)
        for key, value in updates.items():
            if value is None:
                continue
            snap[key] = value
        snap["updated_at"] = time.time()
        _atomic_write_json(path, snap)
    return snap


def _truncate_text(text: str, limit: int = 4000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def record_telegram_text(
    sessions_root: str | Path,
    token: str,
    text: str,
    *,
    source: str = "notify",
    direction: str = "outgoing",
    max_messages: int = DEFAULT_MAX_MESSAGES,
) -> None:
    """Record a free-form Telegram-style message that the user would see."""
    if not token or not text:
        return
    token = _sanitize_token(token)
    text_clean = _truncate_text(text)
    entry = {
        "ts": time.time(),
        "text": text_clean,
        "source": source,
        "direction": direction,
    }
    with _SNAPSHOT_LOCK:
        path = snapshot_path(sessions_root, token)
        snap = _read_json(path) or _empty_snapshot(token)
        msgs = list(snap.get("messages") or [])
        msgs.append(entry)
        if len(msgs) > max_messages:
            msgs = msgs[-max_messages:]
        snap["messages"] = msgs
        snap["updated_at"] = time.time()
        _atomic_write_json(path, snap)
    _append_event(sessions_root, token, {"type": "telegram_message", **entry})


def record_progress_text(
    sessions_root: str | Path,
    token: str,
    text: str,
    *,
    max_history: int = DEFAULT_MAX_PROGRESS_HISTORY,
) -> None:
    """Record the rendered Telegram progress message text (for verbatim mirror)."""
    if not token or not text:
        return
    token = _sanitize_token(token)
    text_clean = _truncate_text(text)
    entry = {"ts": time.time(), "text": text_clean}
    with _SNAPSHOT_LOCK:
        path = snapshot_path(sessions_root, token)
        snap = _read_json(path) or _empty_snapshot(token)
        snap["progress_text"] = text_clean
        history = list(snap.get("progress_history") or [])
        # Avoid hammering history with duplicate consecutive frames.
        if not history or history[-1].get("text") != text_clean:
            history.append(entry)
            if len(history) > max_history:
                history = history[-max_history:]
            snap["progress_history"] = history
        snap["updated_at"] = time.time()
        _atomic_write_json(path, snap)
    _append_event(sessions_root, token, {"type": "progress_text", **entry})


def record_stage(
    sessions_root: str | Path,
    token: str,
    stage_key: str,
    stage_label: Optional[str] = None,
    *,
    phase: Optional[str] = None,
    pct: Optional[int] = None,
    details: Optional[str] = None,
) -> None:
    if not token or not stage_key:
        return
    token = _sanitize_token(token)
    updates: Dict[str, Any] = {"stage_key": str(stage_key)}
    if stage_label is not None:
        updates["stage_label"] = str(stage_label)
    if phase is not None:
        updates["phase"] = str(phase)
    if pct is not None:
        try:
            updates["pct"] = max(0, min(100, int(pct)))
        except Exception:
            pass
    if details is not None:
        updates["details"] = _truncate_text(details, 500)
    _merge_snapshot(sessions_root, token, updates)
    _append_event(sessions_root, token, {"type": "stage", **updates})


def record_progress(
    sessions_root: str | Path,
    token: str,
    *,
    stage_key: Optional[str] = None,
    stage_label: Optional[str] = None,
    stage_num: Optional[int] = None,
    stage_total: Optional[int] = None,
    pct: Optional[int] = None,
    frames_done: Optional[int] = None,
    frames_total: Optional[int] = None,
    elapsed_s: Optional[int] = None,
    eta_s: Optional[int] = None,
    extra_info: Optional[str] = None,
    phase: Optional[str] = None,
) -> None:
    if not token:
        return
    token = _sanitize_token(token)
    updates: Dict[str, Any] = {}
    if stage_key is not None:
        updates["stage_key"] = str(stage_key)
    if stage_label is not None:
        updates["stage_label"] = str(stage_label)
    if stage_num is not None:
        try:
            updates["stage_num"] = int(stage_num)
        except Exception:
            pass
    if stage_total is not None:
        try:
            updates["stage_total"] = int(stage_total)
        except Exception:
            pass
    if pct is not None:
        try:
            updates["pct"] = max(0, min(100, int(pct)))
        except Exception:
            pass
    if frames_done is not None:
        try:
            updates["frames_done"] = max(0, int(frames_done))
        except Exception:
            pass
    if frames_total is not None:
        try:
            updates["frames_total"] = max(0, int(frames_total))
        except Exception:
            pass
    if elapsed_s is not None:
        try:
            updates["elapsed_s"] = max(0, int(elapsed_s))
        except Exception:
            pass
    if eta_s is not None:
        try:
            updates["eta_s"] = max(0, int(eta_s))
        except Exception:
            pass
    if extra_info is not None:
        updates["extra_info"] = _truncate_text(extra_info, 400)
    if phase is not None:
        updates["phase"] = str(phase)
    if not updates:
        return
    _merge_snapshot(sessions_root, token, updates)
    _append_event(sessions_root, token, {"type": "progress", **updates})


def record_completion(
    sessions_root: str | Path,
    token: str,
    *,
    success: bool,
    result: Optional[Dict[str, Any]] = None,
    details: Optional[str] = None,
) -> None:
    if not token:
        return
    token = _sanitize_token(token)
    updates: Dict[str, Any] = {
        "completed": True,
        "success": bool(success),
    }
    if result is not None:
        updates["result"] = dict(result)
    if details is not None:
        updates["details"] = _truncate_text(details, 500)
    if success:
        updates["pct"] = 100
        updates["stage_key"] = "completed"
        updates["stage_label"] = "Completed"
        updates["phase"] = "completed"
    else:
        updates.setdefault("stage_label", "Failed")
        updates["phase"] = updates.get("phase") or "failed"
    _merge_snapshot(sessions_root, token, updates)
    _append_event(sessions_root, token, {"type": "completion", **updates})


def read_events_after(
    sessions_root: str | Path,
    token: str,
    offset: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Read events from the JSONL log starting at byte ``offset``.

    Returns ``(new_offset, events)``.
    """
    token = _sanitize_token(token)
    path = events_path(sessions_root, token)
    if not path.exists():
        return offset, []
    events: List[Dict[str, Any]] = []
    try:
        size = path.stat().st_size
        if offset > size:
            offset = 0
        with path.open("rb") as fp:
            fp.seek(offset)
            data = fp.read()
            new_offset = offset + len(data)
        for line in data.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
        return new_offset, events
    except Exception as exc:
        logger.warning("dashboard events read failed token=%s err=%s", token, exc)
        return offset, []


# ---------------------------------------------------------------------------
# FastAPI server (lazy imports so unit tests don't pay the cost)
# ---------------------------------------------------------------------------


_DASHBOARD_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FaceSwap AI Pipeline</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Sans:wght@400;500&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#F8FAFC;--card:#FFFFFF;--border:#E2E8F0;
  --navy:#0F172A;--green:#059669;--green-light:#22C55E;
  --warn:#EAB308;--error:#EF4444;--info:#0EA5E9;
  --slate:#64748B;--slate-light:#94A3B8;--slate-bg:#F1F5F9;
  --shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
  --shadow-md:0 4px 8px rgba(0,0,0,.06),0 2px 4px rgba(0,0,0,.04);
}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--navy);min-height:100vh;display:flex;flex-direction:column}
h1,h2,h3{font-family:'Plus Jakarta Sans',sans-serif}
.mono{font-family:'Fira Code',monospace}

/* HEADER */
header{
  background:var(--card);border-bottom:1px solid var(--border);
  padding:0 24px;height:64px;display:flex;align-items:center;gap:16px;
  position:sticky;top:0;z-index:10;box-shadow:var(--shadow);
}
.logo{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.25rem;font-weight:800;color:var(--navy);letter-spacing:-.02em}
.logo span{color:var(--green)}
.spacer{flex:1}
.live-badge{
  display:inline-flex;align-items:center;gap:6px;
  background:#EFF6FF;border:1px solid #BFDBFE;color:var(--info);
  padding:5px 12px;border-radius:20px;font-size:.75rem;font-weight:600;letter-spacing:.04em;
  font-family:'Plus Jakarta Sans',sans-serif;
}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--info);animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.8)}}
.live-badge.success{background:#F0FDF4;border-color:#BBF7D0;color:var(--green-light)}
.live-badge.success .live-dot{background:var(--green-light);animation:none}
.live-badge.error{background:#FEF2F2;border-color:#FECACA;color:var(--error)}
.live-badge.error .live-dot{background:var(--error);animation:none}
.live-badge.idle{background:var(--slate-bg);border-color:var(--border);color:var(--slate)}
.live-badge.idle .live-dot{background:var(--slate-light);animation:none}
.elapsed-chip{
  font-family:'Fira Code',monospace;font-size:.8rem;color:var(--slate);
  background:var(--slate-bg);border:1px solid var(--border);
  padding:4px 10px;border-radius:8px;
}

/* LAYOUT */
.wrap{max-width:960px;margin:0 auto;padding:24px 16px;flex:1;display:flex;flex-direction:column;gap:20px}

/* HERO CARD */
.hero-card{
  background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:32px;box-shadow:var(--shadow-md);
  display:flex;align-items:center;gap:40px;
}
@media(max-width:640px){.hero-card{flex-direction:column;gap:24px;padding:24px}}
.ring-wrap{position:relative;width:140px;height:140px;flex-shrink:0}
.ring-wrap svg{transform:rotate(-90deg)}
.ring-bg{fill:none;stroke:#E2E8F0;stroke-width:10}
.ring-fill{fill:none;stroke:var(--green);stroke-width:10;stroke-linecap:round;
  stroke-dasharray:376;stroke-dashoffset:376;transition:stroke-dashoffset .6s cubic-bezier(.4,0,.2,1)}
.ring-pct{
  position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
  font-family:'Fira Code',monospace;font-size:1.8rem;font-weight:500;color:var(--navy);line-height:1;
}
.ring-pct small{font-size:.65rem;color:var(--slate);font-family:'DM Sans',sans-serif;margin-top:2px}
.hero-info{flex:1}
.hero-stage{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.4rem;font-weight:700;color:var(--navy);margin-bottom:6px}
.hero-eta{font-size:.9rem;color:var(--slate);margin-bottom:20px}
.hero-stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
@media(max-width:400px){.hero-stats{grid-template-columns:1fr}}
.hstat{background:var(--slate-bg);border-radius:10px;padding:10px 14px}
.hstat-label{font-size:.65rem;color:var(--slate);text-transform:uppercase;letter-spacing:.08em;margin-bottom:3px}
.hstat-val{font-family:'Fira Code',monospace;font-size:.95rem;font-weight:500;color:var(--navy)}

/* PIPELINE */
.pipeline-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:24px;box-shadow:var(--shadow)}
.section-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:.7rem;font-weight:700;color:var(--slate);text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px}
.steps{display:flex;align-items:flex-start;overflow-x:auto;padding-bottom:4px;gap:0}
.step{display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;min-width:72px;position:relative}
.step:not(:last-child)::after{
  content:'';position:absolute;top:14px;left:calc(50% + 16px);
  width:calc(100% - 32px);height:2px;background:var(--border);z-index:0;
}
.step.done:not(:last-child)::after{background:var(--green-light)}
.step.active:not(:last-child)::after{background:var(--green)}
.step-dot{
  width:28px;height:28px;border-radius:50%;border:2px solid var(--border);
  background:var(--bg);display:flex;align-items:center;justify-content:center;
  font-size:.7rem;font-weight:700;color:var(--slate-light);z-index:1;position:relative;transition:all .3s;
}
.step.done .step-dot{background:var(--green-light);border-color:var(--green-light);color:#fff;font-size:.8rem}
.step.active .step-dot{background:var(--navy);border-color:var(--navy);color:#fff;box-shadow:0 0 0 3px rgba(5,150,105,.2)}
.step.failed .step-dot{background:var(--error);border-color:var(--error);color:#fff}
.step-name{font-size:.62rem;color:var(--slate-light);text-align:center;white-space:nowrap;font-family:'DM Sans',sans-serif}
.step.done .step-name{color:var(--green-light);font-weight:500}
.step.active .step-name{color:var(--navy);font-weight:600}

/* STATS GRID */
.stats-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:640px){.stats-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:400px){.stats-grid{grid-template-columns:1fr}}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;box-shadow:var(--shadow)}
.stat-label{font-size:.65rem;color:var(--slate);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;font-family:'Plus Jakarta Sans',sans-serif;font-weight:600}
.stat-val{font-family:'Fira Code',monospace;font-size:1.3rem;font-weight:500;color:var(--navy)}
.stat-sub{font-size:.72rem;color:var(--slate-light);margin-top:4px}
.vram-bar{height:6px;background:var(--slate-bg);border-radius:99px;overflow:hidden;margin-top:8px}
.vram-fill{height:100%;background:var(--green);border-radius:99px;transition:width .5s ease;width:0%}

/* LOG */
.log-card{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;box-shadow:var(--shadow)}
.log-header{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.log-header .section-title{margin:0}
.log-count{font-size:.7rem;color:var(--slate);background:var(--slate-bg);padding:2px 8px;border-radius:99px;font-family:'Fira Code',monospace}
.log-body{max-height:280px;overflow-y:auto;padding:8px}
.log-entry{display:flex;gap:10px;padding:8px 10px;border-radius:8px;font-size:.82rem;line-height:1.5;border-bottom:1px solid var(--border)}
.log-entry:last-child{border-bottom:none}
.log-entry:hover{background:var(--slate-bg)}
.log-time{font-family:'Fira Code',monospace;color:var(--slate-light);font-size:.7rem;white-space:nowrap;padding-top:2px;min-width:56px}
.log-text{color:var(--navy);word-break:break-word;white-space:pre-wrap;font-size:.82rem}
.log-empty{padding:32px;text-align:center;color:var(--slate-light);font-size:.85rem}

/* COMPLETION */
.complete-card{
  background:linear-gradient(135deg,#F0FDF4,#ECFDF5);
  border:1px solid #BBF7D0;border-radius:16px;padding:40px;
  text-align:center;box-shadow:var(--shadow-md);display:none;
}
.complete-card.show{display:block}
.complete-icon{font-size:3rem;margin-bottom:16px}
.complete-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.5rem;font-weight:800;color:var(--navy);margin-bottom:8px}
.complete-sub{color:var(--slate);margin-bottom:24px}
.btn-primary{
  display:inline-block;background:var(--navy);color:#fff;
  font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;font-size:.9rem;
  padding:12px 28px;border-radius:10px;text-decoration:none;
  box-shadow:var(--shadow-md);transition:opacity .2s;
}
.btn-primary:hover{opacity:.85}

/* IDLE */
.idle-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:60px 40px;text-align:center;box-shadow:var(--shadow-md)}
.idle-ring{width:80px;height:80px;border-radius:50%;border:3px solid #D1FAE5;border-top-color:var(--green);animation:spin 1.2s linear infinite;margin:0 auto 24px}
@keyframes spin{to{transform:rotate(360deg)}}
.idle-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.4rem;font-weight:700;color:var(--navy);margin-bottom:8px}
.idle-sub{color:var(--slate);font-size:.9rem}

footer{text-align:center;padding:16px;color:var(--slate-light);font-size:.72rem;border-top:1px solid var(--border);font-family:'Fira Code',monospace}

/* TECH STRIP — GPU + Models */
.tech-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px 24px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:12px}
.tech-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.tech-row-label{font-family:'Plus Jakarta Sans',sans-serif;font-size:.68rem;font-weight:700;color:var(--slate);text-transform:uppercase;letter-spacing:.1em;margin-right:4px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:5px 10px;border-radius:6px;font-size:.74rem;font-weight:500;font-family:'Plus Jakarta Sans',sans-serif;border:1px solid transparent;white-space:nowrap}
.chip-tech{background:var(--navy);color:#fff}
.chip-model{background:#F0FDF4;color:#15803D;border-color:#BBF7D0}
.chip-dot{width:5px;height:5px;border-radius:50%;background:currentColor;opacity:.85}

/* ERROR STATE */
.error-card{background:#FEF2F2;border:1px solid #FECACA;border-radius:16px;padding:32px;text-align:center;box-shadow:var(--shadow-md);display:none}
.error-card.show{display:block}
.error-icon{font-size:2.5rem;margin-bottom:12px}
.error-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.4rem;font-weight:700;color:var(--error);margin-bottom:8px}
.error-msg{color:#991B1B;font-size:.92rem;margin-bottom:16px;line-height:1.5;max-width:520px;margin-left:auto;margin-right:auto}
.error-meta{display:inline-flex;gap:10px;flex-wrap:wrap;justify-content:center;font-family:'Fira Code',monospace;font-size:.78rem;color:#991B1B}
.error-meta span{background:#FEE2E2;border:1px solid #FECACA;padding:4px 10px;border-radius:6px}
</style>
</head>
<body>
<header>
  <h1 class="logo">⚡ FaceSwap <span>AI Pipeline</span></h1>
  <div class="spacer"></div>
  <span class="elapsed-chip mono" id="elapsed-chip">0m 0s</span>
  <span class="live-badge idle" id="live-badge"><span class="live-dot"></span><span id="badge-text">Connecting</span></span>
</header>

<div class="wrap">
  <!-- IDLE STATE -->
  <div class="idle-card" id="idle-card" style="display:none">
    <div class="idle-ring"></div>
    <h2 class="idle-title">Waiting for next job…</h2>
    <p class="idle-sub">Send a MEGA link to the bot to start processing</p>
  </div>

  <!-- HERO -->
  <div class="hero-card" id="hero-card">
    <div class="ring-wrap">
      <svg width="140" height="140" viewBox="0 0 140 140">
        <circle class="ring-bg" cx="70" cy="70" r="60"/>
        <circle class="ring-fill" id="ring-fill" cx="70" cy="70" r="60"/>
      </svg>
      <div class="ring-pct"><span id="pct-num">0%</span><small>complete</small></div>
    </div>
    <div class="hero-info">
      <div class="hero-stage" id="stage-label">Initializing…</div>
      <div class="hero-eta" id="eta-text">Calculating ETA…</div>
      <div class="hero-stats">
        <div class="hstat"><div class="hstat-label">Frames</div><div class="hstat-val" id="frames">— / —</div></div>
        <div class="hstat"><div class="hstat-label">Phase</div><div class="hstat-val" id="phase">—</div></div>
      </div>
    </div>
  </div>

  <!-- PIPELINE -->
  <div class="pipeline-card">
    <div class="section-title">Pipeline Stages</div>
    <div class="steps" id="steps"></div>
  </div>

  <!-- STATS GRID -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Frames Done</div>
      <div class="stat-val mono" id="s-frames">0</div>
      <div class="stat-sub" id="s-frames-total">of ? total</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Elapsed Time</div>
      <div class="stat-val mono" id="s-elapsed">0m 0s</div>
      <div class="stat-sub">processing time</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">ETA Remaining</div>
      <div class="stat-val mono" id="s-eta">—</div>
      <div class="stat-sub">estimated</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Progress</div>
      <div class="stat-val mono" id="s-pct">0%</div>
      <div class="vram-bar"><div class="vram-fill" id="s-bar"></div></div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Job ID</div>
      <div class="stat-val mono" id="s-jobid">—</div>
      <div class="stat-sub">queue job</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Stage</div>
      <div class="stat-val mono" id="s-stage" style="font-size:.85rem">—</div>
      <div class="stat-sub" id="s-details">—</div>
    </div>
  </div>

  <!-- TECH STRIP: GPU + MODELS -->
  <div class="tech-card">
    <div class="tech-row">
      <span class="tech-row-label">GPU Stack</span>
      <span class="chip chip-tech"><span class="chip-dot"></span><span id="chip-gpu">Tesla T4</span></span>
      <span class="chip chip-tech"><span class="chip-dot"></span>CUDA</span>
      <span class="chip chip-tech"><span class="chip-dot"></span>ONNX Runtime</span>
      <span class="chip chip-tech"><span class="chip-dot"></span>h264_nvenc</span>
    </div>
    <div class="tech-row">
      <span class="tech-row-label">Model Stack</span>
      <span class="chip chip-model"><span class="chip-dot"></span>YOLOFace</span>
      <span class="chip chip-model"><span class="chip-dot"></span>HyperSwap</span>
      <span class="chip chip-model"><span class="chip-dot"></span>GFPGAN 1.4</span>
      <span class="chip chip-model"><span class="chip-dot"></span>FairFace</span>
    </div>
  </div>

  <!-- ERROR STATE -->
  <div class="error-card" id="error-card">
    <div class="error-icon">⚠️</div>
    <h2 class="error-title" id="error-title">Job Failed</h2>
    <p class="error-msg" id="error-msg">An error occurred during processing.</p>
    <div class="error-meta">
      <span id="error-stage-meta">Stage: —</span>
      <span id="error-phase-meta">Phase: —</span>
    </div>
  </div>

  <!-- COMPLETION -->
  <div class="complete-card" id="complete-card">
    <div class="complete-icon">✅</div>
    <h2 class="complete-title">Job Completed!</h2>
    <p class="complete-sub" id="complete-sub">Your video has been processed successfully.</p>
  </div>

  <!-- LOG -->
  <div class="log-card">
    <div class="log-header">
      <div class="section-title">Activity Log</div>
      <span class="log-count" id="log-count">0</span>
    </div>
    <div class="log-body" id="log-body">
      <div class="log-empty">Waiting for updates…</div>
    </div>
  </div>
</div>

<footer>Auto-refreshes every second &nbsp;·&nbsp; token: __TOKEN__</footer>

<script>
const TOKEN = '__TOKEN__';
const STAGES = [
  {key:'download',  label:'Download'},
  {key:'extracting',label:'Extract'},
  {key:'processing',label:'Process'},
  {key:'merging',   label:'Merge'},
  {key:'upload',    label:'Upload'},
  {key:'completed', label:'Done'},
];
const CIRC = 2 * Math.PI * 60; // 376.99

const stepsEl = document.getElementById('steps');
STAGES.forEach((s,i) => {
  const d = document.createElement('div');
  d.className = 'step'; d.id = 'step-' + s.key;
  d.innerHTML = `<div class="step-dot">${i+1}</div><div class="step-name">${s.label}</div>`;
  stepsEl.appendChild(d);
});

function fmt(sec){
  sec = Math.max(0, Math.floor(+sec||0));
  const h = Math.floor(sec/3600), m = Math.floor((sec%3600)/60), s = sec%60;
  return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
}
function esc(t){return String(t??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function timeStr(ts){if(!ts)return '';return new Date(+ts*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'})}

let lastMsgCount = -1;

function render(snap){
  const pct = Math.max(0, Math.min(100, +snap.pct||0));
  const completed = !!snap.completed;
  const failed = snap.success === false;

  // Badge
  const badge = document.getElementById('live-badge');
  const badgeText = document.getElementById('badge-text');
  if(completed){
    badge.className = failed ? 'live-badge error' : 'live-badge success';
    badgeText.textContent = failed ? 'Failed' : 'Completed';
  } else {
    badge.className = 'live-badge';
    badgeText.textContent = '● LIVE';
  }

  // Elapsed chip
  document.getElementById('elapsed-chip').textContent = fmt(snap.elapsed_s);

  // Idle vs active
  const hasData = snap.stage_key && snap.stage_key !== 'queued';
  document.getElementById('idle-card').style.display = (!hasData && !completed) ? 'block' : 'none';
  document.getElementById('hero-card').style.display = (hasData || completed) ? 'flex' : 'none';

  // Ring
  const offset = CIRC - (pct / 100) * CIRC;
  document.getElementById('ring-fill').style.strokeDashoffset = offset;
  document.getElementById('ring-fill').style.stroke = failed ? '#EF4444' : '#059669';
  document.getElementById('pct-num').textContent = pct + '%';

  // Hero
  document.getElementById('stage-label').textContent = snap.stage_label || 'Processing…';
  document.getElementById('eta-text').textContent = snap.eta_s > 0 ? `ETA: ${fmt(snap.eta_s)} remaining` : 'Calculating ETA…';
  document.getElementById('phase').textContent = snap.phase || '—';
  const fd = +snap.frames_done||0, ft = +snap.frames_total||0;
  document.getElementById('frames').textContent = ft > 0 ? `${fd} / ${ft}` : (fd > 0 ? `${fd} / ?` : '— / —');

  // Stats
  document.getElementById('s-frames').textContent = fd;
  document.getElementById('s-frames-total').textContent = ft > 0 ? `of ${ft} total` : 'of ? total';
  document.getElementById('s-elapsed').textContent = fmt(snap.elapsed_s);
  document.getElementById('s-eta').textContent = snap.eta_s > 0 ? fmt(snap.eta_s) : '—';
  document.getElementById('s-pct').textContent = pct + '%';
  document.getElementById('s-bar').style.width = pct + '%';
  document.getElementById('s-jobid').textContent = snap.queue_job_id || '—';
  document.getElementById('s-stage').textContent = snap.stage_key || '—';
  document.getElementById('s-details').textContent = snap.details || snap.extra_info || '—';

  // Pipeline
  const activeIdx = STAGES.findIndex(s => s.key === snap.stage_key);
  STAGES.forEach((s,i) => {
    const el = document.getElementById('step-' + s.key);
    el.className = 'step';
    if(completed){ el.classList.add(failed ? 'failed' : 'done'); }
    else if(activeIdx >= 0){
      if(i < activeIdx) el.classList.add('done');
      else if(i === activeIdx) el.classList.add('active');
    }
    el.querySelector('.step-dot').textContent = (completed && !failed && i < STAGES.length) ? '✓' : (i+1);
  });

  // Completion card
  const cc = document.getElementById('complete-card');
  if(completed && !failed){ cc.classList.add('show'); }
  else { cc.classList.remove('show'); }

  // Error card
  const ec = document.getElementById('error-card');
  if(failed){
    ec.classList.add('show');
    document.getElementById('error-title').textContent = (snap.stage_label && snap.stage_label !== 'Failed') ? `Failed at ${snap.stage_label}` : 'Job Failed';
    document.getElementById('error-msg').textContent = snap.details || 'An error occurred during processing. Check the activity log for details.';
    document.getElementById('error-stage-meta').textContent = `Stage: ${snap.stage_key || '—'}`;
    document.getElementById('error-phase-meta').textContent = `Phase: ${snap.phase || '—'}`;
  } else {
    ec.classList.remove('show');
  }

  // Log
  const msgs = Array.isArray(snap.messages) ? snap.messages : [];
  if(msgs.length !== lastMsgCount){
    lastMsgCount = msgs.length;
    document.getElementById('log-count').textContent = msgs.length;
    const body = document.getElementById('log-body');
    if(msgs.length === 0){
      body.innerHTML = '<div class="log-empty">Waiting for updates…</div>';
    } else {
      body.innerHTML = msgs.slice(-8).reverse().map(m =>
        `<div class="log-entry"><span class="log-time">${esc(timeStr(m.ts))}</span><span class="log-text">${esc(m.text||'')}</span></div>`
      ).join('');
    }
  }
}

async function tick(){
  try{
    const r = await fetch(`/api/job/${encodeURIComponent(TOKEN)}?_=${Date.now()}`,{cache:'no-store'});
    if(!r.ok){ document.getElementById('badge-text').textContent='Error '+r.status; return; }
    render(await r.json());
  } catch(e){
    document.getElementById('live-badge').className='live-badge idle';
    document.getElementById('badge-text').textContent='Reconnecting…';
  }
}
tick();
setInterval(tick,1000);
</script>
</body>
</html>
"""


def _render_dashboard_html(token: str) -> str:
    return _DASHBOARD_HTML_TEMPLATE.replace("__TOKEN__", _sanitize_token(token))


_INDEX_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset='utf-8'>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FaceSwap AI Pipeline · Sessions</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=DM+Sans:wght@400;500&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#F8FAFC;--card:#FFFFFF;--border:#E2E8F0;--navy:#0F172A;--green:#059669;--green-light:#22C55E;--error:#EF4444;--info:#0EA5E9;--slate:#64748B;--slate-light:#94A3B8;--slate-bg:#F1F5F9;--shadow:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--navy);min-height:100vh;display:flex;flex-direction:column}
header{background:var(--card);border-bottom:1px solid var(--border);padding:0 24px;height:64px;display:flex;align-items:center;gap:16px;box-shadow:var(--shadow)}
.logo{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.25rem;font-weight:800;color:var(--navy);letter-spacing:-.02em}
.logo span{color:var(--green)}
.spacer{flex:1}
.live-badge{display:inline-flex;align-items:center;gap:6px;background:#EFF6FF;border:1px solid #BFDBFE;color:var(--info);padding:5px 12px;border-radius:20px;font-size:.75rem;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--info);animation:blink 1.2s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}
.wrap{max-width:1100px;margin:0 auto;padding:32px 16px;width:100%;flex:1}
h1{font-family:'Plus Jakarta Sans',sans-serif;font-size:1.5rem;font-weight:700;margin-bottom:8px}
.sub{color:var(--slate);font-size:.9rem;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;box-shadow:var(--shadow);overflow:hidden}
.empty{padding:48px 24px;text-align:center;color:var(--slate)}
.empty-icon{font-size:2.5rem;margin-bottom:12px;opacity:.5}
table{width:100%;border-collapse:collapse}
th,td{padding:14px 18px;border-bottom:1px solid var(--border);text-align:left;font-size:.88rem}
th{font-family:'Plus Jakarta Sans',sans-serif;color:var(--slate);text-transform:uppercase;letter-spacing:.08em;font-size:.68rem;font-weight:700;background:var(--slate-bg)}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--slate-bg)}
a{color:var(--green);text-decoration:none;font-family:'Fira Code',monospace;font-size:.82rem;font-weight:500}
a:hover{text-decoration:underline}
.pill{display:inline-block;padding:3px 10px;border-radius:99px;font-size:.7rem;font-weight:600;font-family:'Plus Jakarta Sans',sans-serif}
.pill-active{background:#EFF6FF;color:var(--info);border:1px solid #BFDBFE}
.pill-done{background:#F0FDF4;color:var(--green-light);border:1px solid #BBF7D0}
.pill-fail{background:#FEF2F2;color:var(--error);border:1px solid #FECACA}
.pill-idle{background:var(--slate-bg);color:var(--slate);border:1px solid var(--border)}
.pct-bar{display:inline-flex;align-items:center;gap:8px}
.pct-track{width:60px;height:6px;background:var(--slate-bg);border-radius:99px;overflow:hidden}
.pct-fill{height:100%;background:var(--green);border-radius:99px}
.pct-num{font-family:'Fira Code',monospace;font-size:.78rem;color:var(--navy);min-width:34px}
footer{text-align:center;padding:16px;color:var(--slate-light);font-size:.72rem;border-top:1px solid var(--border);font-family:'Fira Code',monospace}
</style></head><body>
<header>
  <div class="logo">⚡ FaceSwap <span>AI Pipeline</span></div>
  <div class="spacer"></div>
  <span class="live-badge"><span class="live-dot"></span>Sessions</span>
</header>
<div class="wrap">
  <h1>Live Sessions</h1>
  <p class="sub">Click any session to view live progress dashboard</p>
  <div class="card">
  __BODY__
  </div>
</div>
<footer>FaceSwap AI · Verdana Health design system</footer>
</body></html>
"""


def _render_index_html(sessions: List[Dict[str, Any]]) -> str:
    if not sessions:
        body = (
            "<div class='empty'><div class='empty-icon'>\U0001F4ED</div>"
            "<div>No sessions yet</div>"
            "<div style='margin-top:6px;font-size:.82rem;color:var(--slate-light)'>"
            "Send a MEGA video link to the bot to start a session.</div></div>"
        )
    else:
        rows = []
        for s in sessions:
            tok = _sanitize_token(str(s.get("token") or ""))
            link = f"<a href='/job/{tok}'>{tok}</a>" if tok else "-"
            phase_raw = str(s.get("phase") or "-").lower()
            stage = str(s.get("stage_label") or "-")
            chat = str(s.get("chat_id") or "-")
            completed = bool(s.get("completed"))
            success = s.get("success")
            try:
                pct_int = max(0, min(100, int(s.get("pct") or 0)))
            except Exception:
                pct_int = 0
            if completed and success is False:
                pill_cls, pill_txt = "pill pill-fail", "Failed"
            elif completed:
                pill_cls, pill_txt = "pill pill-done", "Completed"
            elif phase_raw in {"", "-", "queued"}:
                pill_cls, pill_txt = "pill pill-idle", "Queued"
            else:
                pill_cls, pill_txt = "pill pill-active", "Live"
            pct_html = (
                f"<div class='pct-bar'><div class='pct-track'>"
                f"<div class='pct-fill' style='width:{pct_int}%'></div></div>"
                f"<span class='pct-num'>{pct_int}%</span></div>"
            )
            rows.append(
                f"<tr><td>{link}</td><td><span class='{pill_cls}'>{pill_txt}</span></td>"
                f"<td>{stage}</td><td>{pct_html}</td><td style='color:var(--slate);font-family:Fira Code,monospace;font-size:.78rem'>{chat}</td></tr>"
            )
        body = (
            "<table><thead><tr>"
            "<th>Session</th><th>Status</th><th>Stage</th><th>Progress</th><th>Chat</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    return _INDEX_HTML_TEMPLATE.replace("__BODY__", body)


_CURRENT_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FaceFusion Live</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#080b12;color:#e2e8f0;min-height:100vh;display:flex;align-items:center;justify-content:center}
.wrap{text-align:center;padding:40px 20px}
.logo{font-size:1.4rem;font-weight:700;margin-bottom:32px;color:#e2e8f0}
.logo span{color:#4f8ef7}
.spinner{width:48px;height:48px;border:3px solid #1e2a40;border-top-color:#4f8ef7;border-radius:50%;animation:spin 0.8s linear infinite;margin:0 auto 24px}
@keyframes spin{to{transform:rotate(360deg)}}
.status{font-size:1rem;color:#64748b;margin-bottom:8px}
.sub{font-size:.8rem;color:#334155}
.found{display:none}
.found .check{font-size:2.5rem;margin-bottom:12px}
.found .msg{font-size:1rem;color:#22d07a}
</style>
</head>
<body>
<div class="wrap">
  <div class="logo">Face<span>Fusion</span> Live</div>
  <div id="idle">
    <div class="spinner"></div>
    <div class="status" id="status">Looking for active job…</div>
    <div class="sub" id="sub">Checking every 2 seconds</div>
  </div>
  <div class="found" id="found">
    <div class="check">✅</div>
    <div class="msg">Job found! Redirecting…</div>
  </div>
</div>
<script>
let lastToken = null;
async function poll() {
  try {
    const r = await fetch('/api/current?_=' + Date.now(), {cache:'no-store'});
    if (!r.ok) return;
    const d = await r.json();
    if (d.token) {
      if (d.token !== lastToken) {
        lastToken = d.token;
        document.getElementById('idle').style.display = 'none';
        document.getElementById('found').style.display = 'block';
        setTimeout(() => { window.location.href = '/job/' + encodeURIComponent(d.token); }, 800);
        return;
      }
    } else {
      document.getElementById('status').textContent = 'Waiting for a job to start…';
      document.getElementById('sub').textContent = 'Send a MEGA link to the bot';
    }
  } catch(e) {}
}
poll();
setInterval(poll, 2000);
</script>
</body>
</html>
"""


_CURRENT_JOB_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceSwap AI — Live Pipeline</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#02040a;--surface:#0a0f1c;--surface-2:#111827;--border:rgba(0,212,255,.18);--cyan:#00d4ff;--green:#00ff9f;--yellow:#fbbf24;--red:#f87171;--purple:#a78bfa;--orange:#fb923c;--pink:#f472b6;--text:#e2e8f0;--text-dim:#94a3b8;--text-faint:#475569}
*{box-sizing:border-box;margin:0;padding:0}html{scroll-behavior:smooth}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden}
.mono{font-family:'JetBrains Mono',monospace}
@keyframes scanline{0%{transform:translateY(-100%)}100%{transform:translateY(100vh)}}
@keyframes breathe{0%,100%{opacity:.15;transform:scale(1)}50%{opacity:.28;transform:scale(1.2)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes flash{0%,100%{opacity:1}50%{opacity:.4}}
@keyframes slideIn{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse-ring{0%{box-shadow:0 0 0 0 rgba(0,255,159,.45)}70%{box-shadow:0 0 0 10px rgba(0,255,159,0)}100%{box-shadow:0 0 0 0 rgba(0,255,159,0)}}
@keyframes bar-shine{0%{background-position:-200% 0}100%{background-position:200% 0}}
.grid-overlay{position:fixed;inset:0;z-index:-3;background-image:linear-gradient(rgba(0,212,255,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.035) 1px,transparent 1px);background-size:60px 60px;pointer-events:none}
.orb{position:fixed;border-radius:50%;filter:blur(60px);pointer-events:none;z-index:-2;animation:breathe 8s ease-in-out infinite}.orb-1{width:500px;height:500px;background:var(--cyan);top:-150px;left:-100px}.orb-2{width:400px;height:400px;background:var(--purple);bottom:-120px;right:-80px;animation-delay:-3s}.scanline{position:fixed;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.3;z-index:-1;animation:scanline 10s linear infinite}
.page{max-width:1280px;margin:0 auto;padding:20px}
header{display:flex;align-items:center;gap:14px;margin-bottom:24px;flex-wrap:wrap}
.brand{font-size:22px;font-weight:900;letter-spacing:-.02em;background:linear-gradient(90deg,var(--cyan),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.status-bar{margin-left:auto;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.pill{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:6px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface);display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.pill .dot{width:7px;height:7px;border-radius:50%}
.pill.live{border-color:rgba(0,255,159,.4);color:var(--green)}.pill.live .dot{background:var(--green);animation:pulse-ring 1.5s infinite}
.pill.idle{border-color:var(--border);color:var(--text-dim)}.pill.idle .dot{background:var(--text-dim)}
.pill.err{border-color:rgba(248,113,113,.4);color:var(--red)}.pill.err .dot{background:var(--red)}
.pill.done{border-color:rgba(0,255,159,.4);color:var(--green)}.pill.done .dot{background:var(--green)}
.layout{display:grid;grid-template-columns:320px 1fr;gap:20px}
.sidebar{display:flex;flex-direction:column;gap:16px}
.main{display:flex;flex-direction:column;gap:20px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:20px;position:relative;overflow:hidden;animation:slideIn .5s ease}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.3}
.card.glow-cyan{box-shadow:0 0 30px rgba(0,212,255,.08),inset 0 1px 0 rgba(0,212,255,.1)}
.card.glow-green{box-shadow:0 0 30px rgba(0,255,159,.08),inset 0 1px 0 rgba(0,255,159,.1)}
.card.glow-red{box-shadow:0 0 30px rgba(248,113,113,.08),inset 0 1px 0 rgba(248,113,113,.1)}
.ring-wrap{position:relative;width:220px;height:220px;margin:0 auto 18px}
.ring-wrap svg{transform:rotate(-90deg);width:220px;height:220px;filter:drop-shadow(0 0 8px currentColor)}
.ring-bg{fill:none;stroke:rgba(255,255,255,.08);stroke-width:12}
.ring-fg{fill:none;stroke-width:12;stroke-linecap:round;transition:stroke-dashoffset .6s ease,stroke .3s ease}
.pct{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.pct .big{font-family:'JetBrains Mono',monospace;font-size:48px;font-weight:700;line-height:1}
.pct .small{font-size:11px;color:var(--text-dim);margin-top:4px;text-transform:uppercase;letter-spacing:.1em}
.job-title{text-align:center;font-size:20px;font-weight:800;margin-bottom:4px}
.job-meta{text-align:center;font-size:12px;color:var(--text-dim);margin-bottom:18px}
.btn{width:100%;padding:12px;border-radius:12px;border:1px solid var(--border);background:var(--surface-2);color:var(--text);font-family:inherit;font-size:13px;font-weight:600;cursor:pointer;margin-top:8px;transition:all .2s}
.btn:hover{background:var(--border)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:8px}
.stat-item{background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:14px;text-align:center}
.stat-item .label{font-size:10px;color:var(--text-dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.stat-item .val{font-family:'JetBrains Mono',monospace;font-size:20px;font-weight:700}
.stat-item .sub{font-size:11px;color:var(--text-faint);margin-top:2px}
.section-title{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.12em;color:var(--text-dim);margin-bottom:14px;display:flex;align-items:center;gap:8px}
.section-title::before{content:'';width:4px;height:16px;background:var(--cyan);border-radius:2px}
.pipeline{display:flex;align-items:center;gap:0;overflow-x:auto;padding-bottom:8px}
.p-stage{min-width:100px;text-align:center;padding:10px 6px;border-radius:12px;border:1px solid transparent;position:relative;opacity:.3;transition:all .3s}
.p-stage.done,.p-stage.active,.p-stage.error{opacity:1}
.p-stage .icon{width:44px;height:44px;border-radius:50%;border:1px solid currentColor;display:flex;align-items:center;justify-content:center;margin:0 auto 8px;font-size:18px;background:var(--bg)}
.p-stage.done .icon{background:currentColor;color:var(--bg);box-shadow:0 0 16px currentColor}
.p-stage.active .icon{box-shadow:0 0 20px currentColor,0 0 40px currentColor;animation:flash 1.2s infinite}
.p-stage.error .icon{background:var(--red);color:#fff;border-color:var(--red);box-shadow:0 0 20px var(--red)}
.p-stage .name{font-size:10px;font-weight:600;line-height:1.3}
.p-conn{min-width:24px;height:2px;background:rgba(255,255,255,.1);position:relative}
.p-conn.on{background:linear-gradient(90deg,var(--cyan),var(--green));box-shadow:0 0 8px var(--cyan)}
.terminal{background:var(--bg);border:1px solid var(--border);border-radius:16px;overflow:hidden}
.term-head{background:var(--surface-2);padding:12px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.term-dots{display:flex;gap:6px}
.term-dot{width:10px;height:10px;border-radius:50%}
.term-body{height:240px;overflow:auto;padding:12px;font-size:12px;line-height:1.7}
.log-row{display:grid;grid-template-columns:68px 72px 1fr;gap:10px;padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.log-row .t{color:var(--text-faint);font-size:11px}.log-row .ty{font-weight:700;font-size:11px}
.log-row .m{color:var(--text-dim)}.ty-info{color:var(--cyan)}.ty-warn{color:var(--yellow)}.ty-error{color:var(--red)}.ty-success{color:var(--green)}
.progress-bar{height:10px;background:var(--surface-2);border-radius:999px;overflow:hidden;margin-top:14px;position:relative}
.progress-bar i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--cyan),var(--green));box-shadow:0 0 14px var(--cyan);width:0%;transition:width .8s ease;position:relative}
.progress-bar i::after{content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);background-size:200% 100%;animation:bar-shine 2s infinite}
.tech{display:flex;gap:8px;flex-wrap:wrap}
.tech span{padding:6px 10px;border-radius:8px;font-size:11px;font-weight:700;border:1px solid var(--border);background:var(--surface-2)}
.tech span.on{background:rgba(0,255,159,.12);border-color:rgba(0,255,159,.3);color:var(--green);box-shadow:0 0 10px rgba(0,255,159,.15)}
.empty-state{text-align:center;padding:60px 20px}
.empty-state .spinner{width:64px;height:64px;border:3px solid rgba(0,212,255,.2);border-top-color:var(--cyan);border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 20px}
.empty-state h2{font-size:20px;margin-bottom:8px}
.empty-state p{color:var(--text-dim);font-size:14px;max-width:400px;margin:0 auto;line-height:1.6}
.done-card{text-align:center;padding:40px 20px}
.done-card .icon{font-size:56px;margin-bottom:12px}
.done-card h2{font-size:26px;margin-bottom:8px;color:var(--green);text-shadow:0 0 20px rgba(0,255,159,.3)}
.done-card p{color:var(--text-dim);margin-bottom:20px}
.done-card a{display:inline-block;padding:12px 24px;border-radius:12px;background:linear-gradient(90deg,var(--cyan),var(--green));color:var(--bg);font-weight:800;text-decoration:none;transition:opacity .2s}
.done-card a:hover{opacity:.85}
.error-card{text-align:center;padding:40px 20px}
.error-card .icon{font-size:48px;margin-bottom:12px}
.error-card h2{font-size:24px;margin-bottom:8px;color:var(--red)}
.error-card p{color:var(--text-dim);max-width:480px;margin:0 auto 16px;line-height:1.6}
@media(max-width:900px){.layout{grid-template-columns:1fr}.ring-wrap{width:180px;height:180px}.ring-wrap svg{width:180px;height:180px}.pct .big{font-size:38px}.stats{grid-template-columns:1fr}.pipeline{gap:8px}.p-conn{display:none}}
@media(max-width:520px){.page{padding:14px}.brand{font-size:18px}.status-bar{order:3;width:100%;margin-left:0;margin-top:8px}.job-title{font-size:17px}.term-body{height:180px}}
</style>
</head>
<body>
<div class="grid-overlay"></div><div class="orb orb-1"></div><div class="orb orb-2"></div><div class="scanline"></div>
<div class="page">
<header><div class="brand">⚡ FaceSwap AI Pipeline</div><div class="status-bar"><span class="pill idle" id="live-pill"><span class="dot"></span><span id="live-text">CONNECTING</span></span><span class="pill mono" id="clock">--:--:--</span><span class="pill mono" id="elapsed">00:00</span></div></header>

<div id="idle" class="empty-state" style="display:none"><div class="spinner"></div><h2>Waiting for current job</h2><p>Send a MEGA link to the Telegram bot. This page will automatically switch to live mode when a job starts.</p></div>

<div id="app" class="layout" style="display:none">
<aside class="sidebar">
<div class="card glow-cyan" id="ring-card">
<div class="ring-wrap"><svg viewBox="0 0 220 220"><circle class="ring-bg" cx="110" cy="110" r="94"/><circle class="ring-fg" id="ring" cx="110" cy="110" r="94" stroke="var(--cyan)" stroke-dasharray="590" stroke-dashoffset="590"/></svg><div class="pct"><div class="big" id="pct">0%</div><div class="small">complete</div></div></div>
<div class="job-title" id="stage-title">Initializing...</div>
<div class="job-meta" id="stage-sub">Fetching pipeline state</div>
<div class="stats"><div class="stat-item"><div class="label">Job ID</div><div class="val mono" id="job-id" style="font-size:16px">—</div></div><div class="stat-item"><div class="label">Phase</div><div class="val mono" id="phase" style="font-size:16px">—</div></div></div>
</div>
</aside>

<div class="main">
<div class="card">
<div class="section-title">Live Metrics</div>
<div class="stats" style="grid-template-columns:repeat(4,1fr)"><div class="stat-item"><div class="label">Frames</div><div class="val mono cyan" id="frames">0 / 0</div><div class="sub" id="frame-note">waiting</div></div><div class="stat-item"><div class="label">Speed</div><div class="val mono orange" id="speed">— fps</div><div class="sub">processing rate</div></div><div class="stat-item"><div class="label">GPU</div><div class="val mono purple" id="vram">Ready</div><div class="sub" id="vram-note">VRAM active</div></div><div class="stat-item"><div class="label">ETA</div><div class="val mono green" id="eta">—</div><div class="sub" id="timer-note">estimated remaining</div></div></div>
<div class="progress-bar"><i id="bar"></i></div>
</div>

<div class="card">
<div class="section-title">Stage Pipeline</div>
<div class="pipeline" id="pipeline"></div>
</div>

<div class="card" id="done-card" style="display:none"><div class="done-card"><div class="icon">✅</div><h2>Job Complete</h2><p id="done-msg">Output is ready.</p><a id="result-link" href="#" target="_blank" style="display:none">Open Output</a></div></div>
<div class="card glow-red" id="error-card" style="display:none"><div class="error-card"><div class="icon">⚠️</div><h2 id="error-title">Job Failed</h2><p id="error-text">Check the activity log for details.</p></div></div>

<div class="card">
<div class="section-title">GPU Stack & Models</div>
<div class="tech" id="tech"><span>CUDA</span><span>ONNX Runtime</span><span>h264_nvenc</span><span class="on">YOLOFace</span><span class="on">HyperSwap</span><span class="on">GFPGAN</span><span class="on">FairFace</span></div>
</div>

<div class="terminal">
<div class="term-head"><div class="term-dots"><div class="term-dot" style="background:var(--red)"></div><div class="term-dot" style="background:var(--yellow)"></div><div class="term-dot" style="background:var(--green)"></div></div><div style="font-size:12px;color:var(--text-dim);margin-left:8px">pipeline.log</div></div>
<div class="term-body" id="log-body"><div style="color:var(--text-faint);text-align:center;padding:40px 0">Waiting for live logs...</div></div>
</div>
</div>
</div>
</div>

<script>
const FIXED_TOKEN='__TOKEN__';
const API=FIXED_TOKEN?'/api/job/'+encodeURIComponent(FIXED_TOKEN):'/api/current';
const STAGES=[['download','Download','var(--cyan)'],['validation','Validate','var(--green)'],['extracting','Extract','var(--yellow)'],['analysis','Analyze','var(--orange)'],['tracking','Track','var(--pink)'],['processing','Swap','var(--purple)'],['enhancement','Enhance','var(--blue)'],['merging','Merge','var(--green)'],['upload','Upload','var(--red)'],['completed','Done','var(--cyan)']];
const KEYMAP={download:'download',extracting:'extracting',faceswap:'processing',processing:'processing',merging:'merging',merge:'merging',upload:'upload',completed:'completed',failed:'upload',queued:'download'};
const CIRC=2*Math.PI*94;let lastFrames=null,lastTs=null,speedHistory=[],lastLogLen=-1;
function fmt(sec){sec=Math.max(0,Math.floor(+sec||0));const h=Math.floor(sec/3600),m=Math.floor((sec%3600)/60),s=sec%60;return h?`${h}h ${m}m`:`${m}:${String(s).padStart(2,'0')}`}
function esc(v){return String(v??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function tstr(ts){return ts?new Date(ts*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}):'--:--:--'}
function renderPipeline(activeKey,completed,failed){const mapped=KEYMAP[activeKey]||activeKey||'download';const idx=STAGES.findIndex(s=>s[0]===mapped);document.getElementById('pipeline').innerHTML=STAGES.map((s,i)=>{let cls='',icon=i+1;if(completed&&!failed){cls=' done';icon='✓'}else if(failed&&i<=Math.max(0,idx)){cls=i===idx?' error':' done';icon=i<idx?'✓':'!'}else if(idx>=0){if(i<idx){cls=' done';icon='✓'}else if(i===idx){cls=' active';icon='●'}}return `<div class="p-stage${cls}" style="color:${s[2]}"><div class="icon">${icon}</div><div class="name">${s[1]}</div></div>${i<STAGES.length-1?'<div class="p-conn '+(i<idx||completed&&!failed||(failed&&i<idx)?'on':'')+'"></div>':''}`}).join('');}
function renderLog(msgs,snap){const rows=(Array.isArray(msgs)?msgs:[]).slice(-10).reverse();if(rows.length===lastLogLen)return;lastLogLen=rows.length;if(!rows.length){document.getElementById('log-body').innerHTML='<div style="color:var(--text-faint);text-align:center;padding:40px 0">Waiting for live logs...</div>';return}document.getElementById('log-body').innerHTML=rows.map(m=>{const t=String(m.text||'');let ty='info';if(/fail|error|🔴|❌/i.test(t))ty='error';else if(/warn|retry|⚠/i.test(t))ty='warn';else if(/success|complete|✅/i.test(t))ty='success';return `<div class="log-row"><span class="t">${tstr(m.ts)}</span><span class="ty ty-${ty}">[${ty.toUpperCase()}]</span><span class="m">${esc(t)}</span></div>`}).join('');document.getElementById('log-body').scrollTop=document.getElementById('log-body').scrollHeight;}
function render(snap){if(!snap||!snap.token){document.getElementById('idle').style.display='block';document.getElementById('app').style.display='none';const p=document.getElementById('live-pill');p.className='pill idle';document.getElementById('live-text').textContent='IDLE';return}document.getElementById('idle').style.display='none';document.getElementById('app').style.display='grid';const pct=Math.max(0,Math.min(100,parseInt(snap.pct||0,10))),failed=snap.success===false,completed=!!snap.completed;const ring=document.getElementById('ring');ring.style.strokeDashoffset=CIRC-(pct/100)*CIRC;ring.style.stroke=failed?'var(--red)':completed?'var(--green)':'var(--cyan)';document.getElementById('pct').textContent=`${pct}%`;document.getElementById('stage-title').textContent=snap.stage_label||(snap.active?'Processing':'Current Job');document.getElementById('stage-sub').textContent=snap.details||snap.extra_info||(completed?'Job finished.':'Live progress from pipeline.');document.getElementById('job-id').textContent=snap.queue_job_id||'—';document.getElementById('phase').textContent=snap.phase||'—';document.getElementById('frames').textContent=`${snap.frames_done||0} / ${snap.frames_total||0}`;document.getElementById('frame-note').textContent=snap.frames_total?`${pct}% of detected frames`:'waiting for frame scan';document.getElementById('eta').textContent=snap.eta_s?fmt(snap.eta_s):(completed?'Done':'—');document.getElementById('timer-note').textContent=completed?`elapsed ${fmt(snap.elapsed_s)}`:'estimated remaining';document.getElementById('elapsed').textContent=fmt(snap.elapsed_s);const now=Date.now()/1000,fd=Number(snap.frames_done||0);if(lastFrames!==null&&lastTs&&fd>=lastFrames){const fps=(fd-lastFrames)/Math.max(.1,now-lastTs);if(isFinite(fps)){speedHistory.push(fps);if(speedHistory.length>12)speedHistory.shift();document.getElementById('speed').textContent=fps>0?`${fps.toFixed(1)} fps`:'— fps';}}lastFrames=fd;lastTs=now;document.getElementById('vram').textContent=completed?'Idle':'Active';document.getElementById('vram-note').textContent=completed?'GPU idle':'GPU processing';document.getElementById('bar').style.width=`${pct}%`;document.getElementById('bar').style.background=failed?'linear-gradient(90deg,var(--red),#ff5555)':completed?'linear-gradient(90deg,var(--green),#55ffaa)':'linear-gradient(90deg,var(--cyan),var(--green))';document.getElementById('bar').style.boxShadow=failed?'0 0 14px var(--red)':completed?'0 0 14px var(--green)':'0 0 14px var(--cyan)';const pill=document.getElementById('live-pill');if(failed){pill.className='pill err';document.getElementById('live-text').textContent='FAILED'}else if(completed){pill.className='pill done';document.getElementById('live-text').textContent='DONE'}else{pill.className='pill live';document.getElementById('live-text').textContent='LIVE'}renderPipeline(snap.stage_key||snap.phase,completed,failed);document.getElementById('done-card').style.display=completed&&!failed?'block':'none';document.getElementById('error-card').style.display=failed?'block':'none';if(failed){document.getElementById('error-title').textContent=`Failed at ${snap.stage_label||snap.phase||'job'}`;document.getElementById('error-text').textContent=snap.details||'The pipeline reported a failure.'}const result=snap.result||{},url=result.url||result.link||result.mega_link||result.gdrive_link||'',lnk=document.getElementById('result-link');if(url){lnk.href=url;lnk.style.display='inline-block'}else{lnk.style.display='none'}renderLog(snap.messages,snap);}
async function tick(){try{const r=await fetch(`${API}?_=${Date.now()}`,{cache:'no-store'});if(!r.ok){if(r.status===404)render({token:null});return}render(await r.json())}catch(e){const p=document.getElementById('live-pill');if(p){p.className='pill err';document.getElementById('live-text').textContent='ERROR'}}}tick();setInterval(tick,1000);setInterval(()=>{document.getElementById('clock').textContent=new Date().toLocaleTimeString('en-GB',{hour12:false})},1000);
</script>
</body>
</html>"""


def _render_current_job_html(token: str = "") -> str:
    safe = _sanitize_token(token) if token else ""
    return _CURRENT_JOB_HTML.replace("__TOKEN__", safe)


_PIPELINE_STAGE_BY_KEY = {
    "queued": 0,
    "download": 1,
    "validation": 2,
    "extracting": 3,
    "analysis": 4,
    "tracking": 5,
    "processing": 6,
    "faceswap": 6,
    "enhancement": 7,
    "merge_validate": 8,
    "merging": 9,
    "merge": 9,
    "upload": 10,
    "completed": 11,
    "failed": 10,
}


def _format_duration(seconds: Any) -> str:
    try:
        seconds_i = max(0, int(float(seconds or 0)))
    except Exception:
        seconds_i = 0
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _load_all_snapshots(sessions_root: str | Path) -> List[Dict[str, Any]]:
    root = Path(sessions_root)
    if not root.exists():
        return []
    snaps: List[Dict[str, Any]] = []
    for snap_file in root.glob("*.snapshot.json"):
        snap = _read_json(snap_file)
        if snap:
            snaps.append(snap)
    snaps.sort(key=lambda item: float(item.get("updated_at") or 0), reverse=True)
    return snaps


def _gpu_telemetry() -> Dict[str, Any]:
    try:
        import subprocess
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        line = (proc.stdout or "").splitlines()[0]
        util, mem, name = [part.strip() for part in line.split(",", 2)]
        return {"gpu_percent": int(float(util)), "vram_gb": round(float(mem) / 1024, 1), "gpu_name": name}
    except Exception:
        return {"gpu_percent": 0, "vram_gb": 0, "gpu_name": "GPU"}


def _pipeline_job_from_snapshot(snap: Dict[str, Any], gpu: Dict[str, Any], active_token: str = "") -> Dict[str, Any]:
    stage_key = str(snap.get("stage_key") or snap.get("phase") or "queued").lower()
    try:
        stage_num = int(snap.get("stage_num") or 0)
    except Exception:
        stage_num = 0
    stage = _PIPELINE_STAGE_BY_KEY.get(stage_key, stage_num if 0 <= stage_num <= 11 else 1)
    if snap.get("completed"):
        status = "done" if snap.get("success") is not False else "error"
        stage = 11 if status == "done" else max(1, stage)
    elif stage_key == "queued":
        status = "queued"
    else:
        status = "processing"
    token = str(snap.get("token") or "")
    gpu_active = status == "processing" and (not active_token or token == active_token)
    return {
        "id": str(snap.get("queue_job_id") or token or "JOB"),
        "token": token,
        "status": status,
        "stage": stage,
        "stage_key": stage_key,
        "stage_label": snap.get("stage_label") or stage_key.title(),
        "frames_total": int(snap.get("frames_total") or 0),
        "frames_processed": int(snap.get("frames_done") or 0),
        "percent": int(snap.get("pct") or 0),
        "gpu_percent": int(gpu.get("gpu_percent") or 0) if gpu_active else 0,
        "vram_gb": float(gpu.get("vram_gb") or 0) if gpu_active else 0,
        "gpu_name": gpu.get("gpu_name") or "GPU",
        "eta": _format_duration(snap.get("eta_s") or 0) if snap.get("eta_s") else "—",
        "eta_seconds": int(snap.get("eta_s") or 0),
        "telegram_user": str(snap.get("telegram_user") or snap.get("chat_id") or "operator"),
        "mega_link": str(snap.get("video_link") or ""),
        "started_at": snap.get("created_at"),
        "updated_at": snap.get("updated_at"),
        "details": snap.get("details") or snap.get("extra_info") or "",
    }


def _pipeline_status_payload(sessions_root: str | Path, token: str = "") -> Dict[str, Any]:
    snaps = _load_all_snapshots(sessions_root)
    if token:
        safe = _sanitize_token(token)
        snaps = [snap for snap in snaps if str(snap.get("token") or "") == safe]
    active_snap = next((snap for snap in snaps if not snap.get("completed")), snaps[0] if snaps else {})
    active_token = str(active_snap.get("token") or "")
    gpu = _gpu_telemetry()
    jobs = [_pipeline_job_from_snapshot(snap, gpu, active_token) for snap in snaps[:25]]
    completed = [snap for snap in snaps if snap.get("completed") and snap.get("success") is not False]
    total_frames = sum(int(snap.get("frames_done") or 0) for snap in snaps)
    created_values = [float(snap.get("created_at") or 0) for snap in snaps if snap.get("created_at")]
    uptime = int(max(0, time.time() - min(created_values))) if created_values else 0
    active_gpu_values = [job["gpu_percent"] for job in jobs if job.get("gpu_percent")]
    return {
        "jobs": jobs,
        "active_stage": jobs[0].get("stage") if jobs else 0,
        "session": {
            "jobs_completed": len(completed),
            "total_frames": total_frames,
            "uptime_seconds": uptime,
            "avg_gpu": int(sum(active_gpu_values) / len(active_gpu_values)) if active_gpu_values else int(gpu.get("gpu_percent") or 0),
            "gpu_device": gpu.get("gpu_name") or "GPU",
        },
    }


def _pipeline_logs_payload(sessions_root: str | Path, job_id: str = "", limit: int = 120) -> Dict[str, Any]:
    snaps = _load_all_snapshots(sessions_root)
    target = None
    job_id = str(job_id or "")
    for snap in snaps:
        if job_id and job_id in {str(snap.get("queue_job_id") or ""), str(snap.get("token") or "")}:
            target = snap
            break
    if target is None:
        target = next((snap for snap in snaps if not snap.get("completed")), snaps[0] if snaps else None)
    logs: List[Dict[str, Any]] = []
    if target:
        for msg in list(target.get("messages") or [])[-limit:]:
            text = str(msg.get("text") or "")
            lowered = text.lower()
            if "error" in lowered or "fail" in lowered or "❌" in text:
                kind = "error"
            elif "warn" in lowered or "retry" in lowered or "⚠" in text:
                kind = "warn"
            elif "success" in lowered or "complete" in lowered or "✅" in text:
                kind = "success"
            else:
                kind = "info"
            ts = float(msg.get("ts") or target.get("updated_at") or time.time())
            logs.append({"time": time.strftime("%H:%M:%S", time.localtime(ts)), "type": kind, "msg": text})
    return {"logs": logs}


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


class DashboardServer:
    """Manage a uvicorn server running in a dedicated background thread."""

    def __init__(
        self,
        sessions_root: str | Path,
        host: str = "0.0.0.0",
        port: int = 8765,
        public_url: Optional[str] = None,
        root_path: str = "",
        log_level: str = "warning",
    ) -> None:
        self.sessions_root = str(sessions_root)
        self.host = str(host or "0.0.0.0")
        self.port = int(port or 8765)
        self.public_url = (public_url or f"http://localhost:{self.port}").rstrip("/")
        self.root_path = root_path.rstrip("/") if root_path else ""
        self.log_level = log_level
        self._server: Any = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Any = None
        self._started = threading.Event()

    def public_session_url(self, token: str) -> str:
        return f"{self.public_url}/job/{_sanitize_token(token)}"

    def build_app(self):
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

        sessions_root = self.sessions_root
        root_path = self.root_path
        from fastapi.middleware.cors import CORSMiddleware
        app = FastAPI(title="FaceFusion Live Dashboard", docs_url=None, redoc_url=None, root_path=root_path)
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @app.get("/healthz", response_class=PlainTextResponse)
        async def healthz() -> str:
            return "ok"

        # App-compatibility aliases
        @app.get("/health", response_class=PlainTextResponse)
        async def health() -> str:
            return "ok"

        @app.get("/ping", response_class=PlainTextResponse)
        async def ping() -> str:
            return "ok"

        @app.get("/api/health")
        async def api_health():
            return JSONResponse({"status": "ok", "server": "FaceFusion Dashboard"})

        @app.get("/api/ping")
        async def api_ping():
            return JSONResponse({"status": "ok"})

        @app.get("/api/current")
        async def api_current():
            """Return the most recent active (non-completed) session, or latest if all done."""
            sessions = list_sessions(sessions_root)
            # Prefer active job first
            active = [s for s in sessions if not s.get("completed")]
            target = active[0] if active else (sessions[0] if sessions else None)
            if not target or not target.get("token"):
                return JSONResponse({"token": None, "active": False})
            snap = load_snapshot(sessions_root, target["token"])
            snap["active"] = not snap.get("completed", False)
            return JSONResponse(snap)

        @app.get("/api/pipeline/status")
        async def api_pipeline_status(token: str = ""):
            return JSONResponse(_pipeline_status_payload(sessions_root, token))

        @app.get("/api/pipeline/logs")
        async def api_pipeline_logs(job_id: str = "", limit: int = 120):
            return JSONResponse(_pipeline_logs_payload(sessions_root, job_id, max(1, min(500, int(limit or 120)))))

        # --- Direct sync endpoints (bot-independent) ---

        @app.get("/api/status")
        async def api_status():
            """
            Reads pipeline/logs/current_job.json written by progress_poller.py.
            Bot-independent — works even if bot.py is frozen.
            """
            try:
                from ops.progress_writer import read_progress
                from ops.gpu_monitor import get_gpu_stats
            except ImportError:
                return JSONResponse({"error": "progress_writer not available"}, status_code=503)
            data = read_progress()
            gpu = get_gpu_stats()
            data["gpu_util"]  = gpu.get("gpu_util",  data.get("gpu_util", 0))
            data["vram_gb"]   = gpu.get("vram_used", data.get("vram_gb", 0))
            data["gpu_name"]  = gpu.get("gpu_name",  "Tesla T4")
            return JSONResponse(data)

        @app.websocket("/ws/live")
        async def ws_live(websocket: WebSocket):
            """
            WebSocket that streams current_job.json every second.
            Bot-independent — reads directly from disk.
            """
            await websocket.accept()
            try:
                from ops.progress_writer import read_progress
                import asyncio as _asyncio
                last_sent: dict = {}
                while True:
                    data = read_progress()
                    if data != last_sent:
                        await websocket.send_json(data)
                        last_sent = data.copy()
                    await _asyncio.sleep(1)
            except WebSocketDisconnect:
                return
            except Exception:
                return

        @app.websocket("/ws/pipeline")
        async def ws_pipeline(websocket: WebSocket):
            await websocket.accept()
            try:
                import asyncio
                while True:
                    await websocket.send_json(_pipeline_status_payload(sessions_root))
                    await asyncio.sleep(1.2)
            except WebSocketDisconnect:
                return

        @app.get("/current", response_class=HTMLResponse)
        async def current_page() -> str:
            return _render_current_job_html()

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return _render_current_job_html()

        @app.get("/live", response_class=HTMLResponse)
        async def live_monitor() -> str:
            live_html_path = Path(__file__).resolve().parent.parent / "scripts" / "live_monitor.html"
            if live_html_path.exists():
                return HTMLResponse(live_html_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>live_monitor.html not found</h1>", status_code=404)

        # Also register prefixed routes for Lightning AI proxy (forwards /PORT/path as-is)
        if root_path:
            @app.get(f"{root_path}/live", response_class=HTMLResponse)
            async def live_monitor_prefixed() -> str:
                return await live_monitor()

            @app.get(f"{root_path}/", response_class=HTMLResponse)
            async def index_prefixed() -> str:
                return await index()

            @app.get(f"{root_path}", response_class=HTMLResponse)
            async def index_prefixed_noslash() -> str:
                return await index()

            @app.get(f"{root_path}/healthz", response_class=PlainTextResponse)
            async def healthz_prefixed() -> str:
                return "ok"

            @app.get(f"{root_path}/health", response_class=PlainTextResponse)
            async def health_prefixed() -> str:
                return "ok"

            @app.get(f"{root_path}/ping", response_class=PlainTextResponse)
            async def ping_prefixed() -> str:
                return "ok"

            @app.get(f"{root_path}/api/health")
            async def api_health_prefixed():
                return JSONResponse({"status": "ok", "server": "FaceFusion Dashboard"})

            @app.get(f"{root_path}/api/status")
            async def api_status_prefixed():
                return await api_status()

            @app.websocket(f"{root_path}/ws/live")
            async def ws_live_prefixed(websocket: WebSocket):
                return await ws_live(websocket)

            @app.websocket(f"{root_path}/ws/pipeline")
            async def ws_pipeline_prefixed(websocket: WebSocket):
                return await ws_pipeline(websocket)

        @app.get("/job/{token}", response_class=HTMLResponse)
        async def job_page(token: str) -> str:
            try:
                token = _sanitize_token(token)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid token")
            return _render_current_job_html(token)

        @app.get("/api/sessions")
        async def api_sessions():
            return JSONResponse(list_sessions(sessions_root))

        @app.get("/api/job/{token}")
        async def api_job(token: str):
            try:
                token = _sanitize_token(token)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid token")
            snap = load_snapshot(sessions_root, token)
            if not snap:
                raise HTTPException(status_code=404, detail="session not found")
            return JSONResponse(snap)

        @app.get("/api/job/{token}/events")
        async def api_events(token: str, since: int = 0):
            try:
                token = _sanitize_token(token)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid token")
            new_offset, events = read_events_after(sessions_root, token, max(0, int(since or 0)))
            return JSONResponse({"offset": new_offset, "events": events})

        @app.get("/dashboard_v2.html", response_class=HTMLResponse)
        async def serve_dashboard_v2() -> str:
            from fastapi.responses import FileResponse
            p = Path(__file__).resolve().parent.parent / "dashboard_v2.html"
            if not p.exists():
                raise HTTPException(status_code=404, detail="dashboard_v2.html not found")
            return FileResponse(str(p), media_type="text/html")

        if root_path:
            @app.get(f"{root_path}/dashboard_v2.html", response_class=HTMLResponse)
            async def serve_dashboard_v2_prefixed() -> str:
                return await serve_dashboard_v2()

        return app

    def start(self) -> "DashboardServer":
        if self._thread and self._thread.is_alive():
            return self

        try:
            import uvicorn
        except Exception as exc:  # pragma: no cover - import guard
            logger.warning("dashboard server disabled: uvicorn import failed: %s", exc)
            return self

        app = self.build_app()
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            access_log=False,
            loop="asyncio",
            lifespan="off",
            proxy_headers=True,
            forwarded_allow_ips="*",
            root_path=self.root_path,
        )
        server = uvicorn.Server(config)
        self._server = server

        def _run() -> None:
            try:
                server.run()
            except SystemExit:
                pass
            except Exception as run_exc:  # pragma: no cover - defensive
                logger.warning("dashboard server crashed: %s", run_exc)

        thread = threading.Thread(target=_run, name="dashboard-server", daemon=True)
        thread.start()
        self._thread = thread

        # Wait briefly for the server to start so callers can log a usable URL.
        for _ in range(50):
            if getattr(server, "started", False):
                self._started.set()
                break
            time.sleep(0.05)

        return self

    def stop(self) -> None:
        srv = self._server
        if srv is None:
            return
        try:
            srv.should_exit = True
        except Exception:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        self._thread = None
        self._server = None


def start_dashboard_server(
    sessions_root: str | Path,
    host: str = "0.0.0.0",
    port: int = 8765,
    public_url: Optional[str] = None,
    root_path: str = "",
    log_level: str = "warning",
) -> Optional[DashboardServer]:
    """Start the dashboard server and return the server instance.

    Returns ``None`` if the server could not be started, which keeps the bot
    fully functional when the dashboard is disabled or fails to import.
    """
    try:
        server = DashboardServer(
            sessions_root=sessions_root,
            host=host,
            port=port,
            public_url=public_url,
            root_path=root_path,
            log_level=log_level,
        )
        server.start()
        return server
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("dashboard server start failed: %s", exc)
        return None


__all__ = [
    "DashboardServer",
    "DEFAULT_MAX_MESSAGES",
    "DEFAULT_MAX_PROGRESS_HISTORY",
    "events_path",
    "generate_session_token",
    "list_sessions",
    "load_snapshot",
    "read_events_after",
    "record_completion",
    "record_progress",
    "record_progress_text",
    "record_stage",
    "record_telegram_text",
    "register_session",
    "session_dir",
    "snapshot_path",
    "start_dashboard_server",
]
