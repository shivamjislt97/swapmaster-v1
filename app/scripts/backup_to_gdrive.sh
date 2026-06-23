#!/bin/bash
set -euo pipefail

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="facefusion_pipeline_backup_${TIMESTAMP}.tar.gz"
RCLONE="${PROJECT}/bin/rclone"
CONF="${PROJECT}/rclone.conf"
BASE="$(basename "$PROJECT")"

echo "Creating backup: ${BACKUP_NAME}"

tar -czf "/tmp/${BACKUP_NAME}" \
    --exclude="${BASE}/.env" \
    --exclude="${BASE}/pipeline/workspace" \
    --exclude="${BASE}/pipeline/downloads" \
    --exclude="${BASE}/pipeline/logs" \
    --exclude="${BASE}/pipeline/frames" \
    --exclude="${BASE}/persistent/jobs" \
    --exclude="${BASE}/outputs" \
    --exclude="${BASE}/workspace" \
    --exclude="${BASE}/__pycache__" \
    --exclude="${BASE}/.git" \
    --exclude="${BASE}/rclone.conf" \
    --exclude="${BASE}/.mega_creds" \
    --exclude="${BASE}/bin/rclone" \
    --exclude="*.mp4" --exclude="*.avi" --exclude="*.mkv" \
    --exclude="*.jpg" --exclude="*.jpeg" --exclude="*.png" --exclude="*.gif" \
    --exclude="*.pyc" --exclude="__pycache__" \
    -C "$(dirname "$PROJECT")" "$BASE"

SIZE=$(du -sh "/tmp/${BACKUP_NAME}" | cut -f1)
echo "Archive: /tmp/${BACKUP_NAME} (${SIZE})"

echo "Uploading to GDrive..."
RCLONE_BIN="$RCLONE" RCLONE_CONF="$CONF" python3 "$PROJECT/scripts/gdrive_upload.py" "/tmp/${BACKUP_NAME}" "FacefusionBackups"

rm -f "/tmp/${BACKUP_NAME}"
echo "Done: ${BACKUP_NAME}"
