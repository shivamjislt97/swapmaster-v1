import os
from pathlib import Path


def _read_dotenv(dotenv_path: Path) -> dict:
    data = {}
    if not dotenv_path.exists():
        return data
    for raw in dotenv_path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def mask_secret(value: str, keep: int = 4) -> str:
    text = str(value or "")
    if len(text) <= keep:
        return "*" * len(text)
    return ("*" * (len(text) - keep)) + text[-keep:]


def resolve_credentials() -> dict:
    # Look for .env in project root (parent of app/)
    root = Path(__file__).resolve().parents[1]  # app/
    project_root = root.parent  # project root
    env_data = _read_dotenv(project_root / ".env")
    # Also check app/.env for backward compatibility
    if not env_data:
        env_data = _read_dotenv(root / ".env")

    def pick(*keys: str, default: str = "") -> str:
        for key in keys:
            val = os.environ.get(key)
            if val:
                return str(val).strip()
            val = env_data.get(key)
            if val:
                return str(val).strip()
        return default

    gdrive_folder = pick("GDRIVE_FOLDER", default="gdrive:masterswap")
    remote_name = pick("GDRIVE_REMOTE_NAME", default="gdrive")
    if gdrive_folder and ":" not in gdrive_folder:
        gdrive_folder = f"{remote_name}:{gdrive_folder}"

    return {
        "bot_token": pick("BOT_TOKEN", "TELEGRAM_BOT_TOKEN"),
        "gdrive_remote_name": remote_name,
        "gdrive_folder": gdrive_folder,
        "default_face_link": pick("DEFAULT_FACE_MEGA_LINK", "DEFAULT_FACE_LINK"),
        "mega_email": pick("MEGA_EMAIL"),
        "mega_password": pick("MEGA_PASSWORD"),
    }


def validate_credentials(creds: dict) -> tuple[bool, list]:
    missing = []
    if not str((creds or {}).get("bot_token", "")).strip():
        missing.append("bot_token")
    return (len(missing) == 0, missing)


def reload_credentials() -> dict:
    return resolve_credentials()


def masked_credentials(creds: dict) -> dict:
    data = dict(creds or {})
    for key in ["bot_token", "mega_email", "mega_password"]:
        if key in data:
            data[key] = mask_secret(data.get(key, ""))
    return data
