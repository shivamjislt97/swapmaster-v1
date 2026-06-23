#!/usr/bin/env python3
"""Upload a file to Google Drive using rclone (already configured in the pipeline)."""
import subprocess, sys, os
from pathlib import Path

def upload(local_path: str, folder_name: str = "FacefusionBackups") -> None:
    remote = os.environ.get("GDRIVE_REMOTE_NAME", "gdrive")
    dest = f"{remote}:{folder_name}"
    rclone = os.environ.get("RCLONE_BIN", "rclone")
    conf = os.environ.get("RCLONE_CONF", str(Path.home() / ".config/rclone/rclone.conf"))

    print(f"Uploading {local_path} → {dest}")
    r = subprocess.run(
        [rclone, "--config", conf, "copy", local_path, dest, "--progress"],
        check=True
    )

    fname = Path(local_path).name
    lr = subprocess.run(
        [rclone, "--config", conf, "link", f"{dest}/{fname}"],
        capture_output=True, text=True
    )
    link = lr.stdout.strip()
    print(f"Upload complete. Link: {link or '(link generation not supported)'}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: gdrive_upload.py <file_path> [folder_name]")
        sys.exit(1)
    upload(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "FacefusionBackups")
