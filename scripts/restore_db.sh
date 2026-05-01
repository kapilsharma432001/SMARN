#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/restore_db.sh <backup.dump>" >&2
  exit 1
fi

APP_DIR="${SMARN_APP_DIR:-$(pwd)}"
BACKUP_FILE="$1"

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "Backup file not found: ${BACKUP_FILE}" >&2
  exit 1
fi

cd "$APP_DIR"

echo "This will restore ${BACKUP_FILE} into the database configured for the db container."
echo "It may overwrite existing data. Type RESTORE to continue:"
read -r CONFIRMATION

if [[ "$CONFIRMATION" != "RESTORE" ]]; then
  echo "Restore cancelled."
  exit 1
fi

RESTORE_FILE="/tmp/smarn_restore_$(date -u +%Y%m%dT%H%M%SZ).dump"

docker compose cp "$BACKUP_FILE" "db:${RESTORE_FILE}"
docker compose exec -T db sh -c \
  'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists "$1"' \
  sh "$RESTORE_FILE"
docker compose exec -T db rm -f "$RESTORE_FILE"

echo "Restore completed."
