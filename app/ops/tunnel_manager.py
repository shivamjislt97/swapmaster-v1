#!/usr/bin/env python3
"""
Tunnel manager — ngrok primary, Lightning SDK fallback.
Singleton. Auto-recovers. Updates .env + tunnel_url.txt.
"""
import os, re, subprocess, time, logging, json, urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
LOG_DIR  = ROOT_DIR / "pipeline" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [tunnel_manager] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "tunnel_manager.log", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)

PORT           = int(os.environ.get("DASHBOARD_PORT", "8765"))
ENV_FILE       = ROOT_DIR / ".env"
PID_FILE       = LOG_DIR / "tunnel_manager.pid"
URL_FILE       = LOG_DIR / "tunnel_url.txt"
NGROK_BIN      = os.environ.get("NGROK_BIN", "/tmp/ngrok")
NGROK_API      = "http://localhost:4040/api/tunnels"
CHECK_INTERVAL = 30


def _ensure_singleton() -> None:
    if PID_FILE.exists():
        try:
            old = int(PID_FILE.read_text().strip())
            if old != os.getpid():
                try: os.kill(old, 9)
                except ProcessLookupError: pass
                log.info("[SINGLETON] Killed old pid=%d", old)
        except Exception: pass
    try:
        r = subprocess.run(["pgrep", "-f", "tunnel_manager.py"], capture_output=True, text=True)
        for line in r.stdout.strip().splitlines():
            try:
                pid = int(line.strip())
                if pid != os.getpid():
                    os.kill(pid, 9)
                    log.info("[SINGLETON] Killed stray pid=%d", pid)
            except Exception: pass
    except Exception: pass
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    log.info("[SINGLETON] PID=%d registered", os.getpid())


def _ngrok_get_url() -> str:
    """Return current ngrok HTTPS tunnel URL if alive."""
    try:
        r = urllib.request.urlopen(NGROK_API, timeout=3)
        d = json.loads(r.read())
        for t in d.get("tunnels", []):
            u = str(t.get("public_url", ""))
            if u.startswith("https") and f":{PORT}" in str(t.get("config", {}).get("addr", "")):
                return u
        # Also accept any https tunnel if only one exists
        for t in d.get("tunnels", []):
            u = str(t.get("public_url", ""))
            if u.startswith("https"):
                return u
    except Exception:
        pass
    return ""


def _ngrok_start() -> str:
    """Start ngrok tunnel, return URL or empty."""
    if not Path(NGROK_BIN).is_file():
        log.warning("[NGROK] binary not found at %s", NGROK_BIN)
        return ""
    # Kill stale ngrok
    subprocess.run(["pkill", "-f", f"ngrok.*{PORT}"], capture_output=True)
    time.sleep(2)
    try:
        subprocess.Popen(
            [NGROK_BIN, "http", str(PORT), "--log=stdout"],
            stdout=open(LOG_DIR / "ngrok.log", "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(20):
            time.sleep(1)
            url = _ngrok_get_url()
            if url:
                log.info("[NGROK] Tunnel started: %s", url)
                return url
    except Exception as e:
        log.error("[NGROK] Start failed: %s", e)
    return ""


def _lightning_get_url() -> str:
    """Get Lightning SDK registered URL for current studio."""
    try:
        from lightning_sdk import Studio
        from lightning_sdk.lightning_cloud.rest_client import create_swagger_client
        from lightning_sdk.lightning_cloud.openapi.api.cloud_space_service_api import CloudSpaceServiceApi
        project_id = os.environ.get("LIGHTNING_CLOUD_PROJECT_ID", "")
        space_id   = os.environ.get("LIGHTNING_CLOUD_SPACE_ID", "")
        if not (project_id and space_id):
            return ""
        client = create_swagger_client(check_context=False, with_auth=True)
        cs_api = CloudSpaceServiceApi(client)
        studio_name = ""
        for cs in cs_api.cloud_space_service_list_cloud_spaces(project_id=project_id).cloudspaces:
            if cs.id == space_id:
                studio_name = getattr(cs, "name", "")
                break
        if not studio_name:
            return ""
        studio = Studio(name=studio_name)
        # Check existing
        for ep in studio.list_ports():
            ep_ports = ep.get("ports", []) if isinstance(ep, dict) else getattr(ep, "ports", [])
            urls     = ep.get("urls",  []) if isinstance(ep, dict) else getattr(ep, "urls",  [])
            if str(PORT) in [str(p) for p in ep_ports] and urls:
                return str(urls[0]).rstrip("/")
        # Register
        for ep in studio.add_ports(PORT):
            urls = ep.get("urls", []) if isinstance(ep, dict) else getattr(ep, "urls", [])
            if urls:
                return str(urls[0]).rstrip("/")
    except Exception as e:
        log.debug("[LIGHTNING] %s", e)
    return ""


def _verify(url: str) -> bool:
    try:
        r = urllib.request.urlopen(f"{url}/healthz", timeout=8)
        return r.read().decode().strip() in ("ok", "OK")
    except Exception:
        return False


def _save_url(url: str) -> None:
    URL_FILE.write_text(url, encoding="utf-8")
    if ENV_FILE.exists():
        content = ENV_FILE.read_text(encoding="utf-8")
        if "DASHBOARD_PUBLIC_URL=" in content:
            content = re.sub(r"^DASHBOARD_PUBLIC_URL=.*", f"DASHBOARD_PUBLIC_URL={url}", content, flags=re.MULTILINE)
        else:
            content = content.rstrip() + f"\nDASHBOARD_PUBLIC_URL={url}\n"
        ENV_FILE.write_text(content, encoding="utf-8")
    log.info("[URL_SAVED] %s", url)


def run() -> None:
    _ensure_singleton()
    active_url = ""

    while True:
        # 1. Check ngrok (primary)
        url = _ngrok_get_url()
        if url and _verify(url):
            if url != active_url:
                active_url = url
                _save_url(url)
                log.info("[ACTIVE] ngrok URL: %s", url)
        else:
            # ngrok dead → restart
            log.info("[NGROK] Not alive — starting...")
            url = _ngrok_start()
            if url and _verify(url):
                active_url = url
                _save_url(url)
                log.info("[ACTIVE] ngrok restarted: %s", url)
            else:
                # 2. Lightning fallback
                log.info("[FALLBACK] Trying Lightning SDK...")
                url = _lightning_get_url()
                if url and _verify(url):
                    active_url = url
                    _save_url(url)
                    log.info("[ACTIVE] Lightning URL: %s", url)
                else:
                    log.warning("[TUNNEL] All methods failed — retrying in %ds", CHECK_INTERVAL)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    run()
