#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="${SMARN_APP_DIR:-$(pwd)}"
BRANCH="${SMARN_DEPLOY_BRANCH:-main}"

cd "$APP_DIR"

echo "==> Pulling latest code from ${BRANCH}"
git fetch origin "$BRANCH"
git pull --ff-only origin "$BRANCH"

echo "==> Starting database"
docker compose up -d db

echo "==> Building updated application image"
docker compose build api bot

echo "==> Running database migrations"
docker compose run --rm --no-deps api alembic upgrade head

echo "==> Restarting api and bot"
docker compose up -d --no-deps api bot

echo "==> Container status"
docker compose ps
