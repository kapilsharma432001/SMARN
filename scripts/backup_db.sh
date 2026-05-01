#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${SMARN_APP_DIR:-$(pwd)}"
BACKUP_DIR="${SMARN_BACKUP_DIR:-${APP_DIR}/backups}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_FILE="${BACKUP_DIR}/smarn_${TIMESTAMP}.dump"
TMP_BACKUP_FILE="${BACKUP_FILE}.tmp"

cd "$APP_DIR"
mkdir -p "$BACKUP_DIR"
trap 'rm -f "$TMP_BACKUP_FILE"' EXIT

echo "==> Creating database backup: ${BACKUP_FILE}"
docker compose exec -T db sh -c \
  'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$TMP_BACKUP_FILE"

mv "$TMP_BACKUP_FILE" "$BACKUP_FILE"

echo "==> Backup created: ${BACKUP_FILE}"
