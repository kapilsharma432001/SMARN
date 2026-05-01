# Deploy SMARN on an Oracle Always Free Ubuntu VM

This guide keeps the deployment simple: one Ubuntu VM, Docker Compose, PostgreSQL
with pgvector, the FastAPI API, and the Telegram bot. No Kubernetes, Terraform,
Nginx, domain, HTTPS, or public UI is needed yet.

Warnings:

- Oracle server infrastructure may be free when you stay within Always Free
  limits, but OpenAI API usage is not free.
- Oracle says idle Always Free compute instances may be reclaimed according to
  its policy. See Oracle's Always Free resource documentation:
  https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm
- Do not commit `.env`, database backups, or secrets.

## 1. Create the Oracle VM

1. Create or sign in to an Oracle Cloud account:
   https://www.oracle.com/cloud/free/
2. Create an Ubuntu compute instance in your home region.
3. Pick an Always Free eligible shape when available.
4. Add your SSH public key during VM creation.
5. Keep SSH open for your IP. The API does not need public exposure yet.

Oracle capacity and labels change over time. Use Oracle's current console labels
and official Free Tier docs as the source of truth.

## 2. Install Docker

SSH into the server:

```bash
ssh ubuntu@YOUR_SERVER_IP
```

Install Docker and the Compose plugin:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Log out and back in so the `docker` group change applies, then verify:

```bash
docker --version
docker compose version
```

## 3. Clone SMARN

```bash
mkdir -p ~/apps
cd ~/apps
git clone git@github.com:YOUR_GITHUB_USER/YOUR_SMARN_REPO.git SMARN
cd SMARN
```

## 4. Create `.env` on the server

Create `.env` only on the server:

```bash
nano .env
```

Example:

```dotenv
APP_ENV=production
LOG_LEVEL=INFO

POSTGRES_DB=smarn
POSTGRES_USER=smarn
POSTGRES_PASSWORD=replace-with-a-long-random-password

OPENAI_API_KEY=replace-on-server-only
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe

TELEGRAM_BOT_TOKEN=replace-on-server-only

EMBEDDING_DIMENSIONS=1536
MEMORY_SEARCH_LIMIT=5
MEMORY_RELEVANCE_MAX_DISTANCE=0.75
REVIEW_TIMEZONE=Asia/Kolkata
```

Keep this file out of Git. The repository `.gitignore` excludes `.env`.
Docker Compose builds the internal `DATABASE_URL` for `api` and `bot` from the
Postgres values above so containers connect to `db:5432`.

For now, keep `POSTGRES_PASSWORD` URL-safe and alphanumeric because Docker
Compose builds `DATABASE_URL` from it.

## 5. Run the first deployment manually

From the repo directory on the server:

```bash
bash scripts/deploy.sh
```

The script:

- pulls the latest `main`
- starts `db`
- builds the app image
- runs `alembic upgrade head`
- restarts `api` and `bot`
- prints `docker compose ps`

It does not run `docker compose down -v` and does not delete database volumes.

## 6. Check containers and logs

```bash
docker compose ps
docker compose logs -f bot
docker compose logs -f api
```

The bot logs structured JSON events such as:

```text
telegram_voice_received
telegram_voice_downloaded
voice_transcription_started
voice_memory_saved
telegram_voice_reply_sent
```

The API is bound to localhost in `docker-compose.yml`:

```text
127.0.0.1:8000:8000
```

This means it is available on the VPS itself, but not publicly exposed. That is
intentional for now.

Health check from the server:

```bash
curl http://127.0.0.1:8000/health
```

## 7. Test the bot

In Telegram, send:

```text
/start
/remember Fixed TFG-231 duplicate ingestion bug using composite key upsert
/ask What did I do for TFG-231?
/daily_review
/weekly_review
```

Send a short voice note and watch logs:

```bash
docker compose logs -f bot
```

## 8. Create a database backup

Backups are manual for now:

```bash
bash scripts/backup_db.sh
```

Backup files are written under:

```text
backups/
```

The repository ignores `backups/`, `*.dump`, and `*.sql`.

Copy backups off the VM periodically:

```bash
scp ubuntu@YOUR_SERVER_IP:~/apps/SMARN/backups/smarn_YYYYMMDDTHHMMSSZ.dump .
```

## 9. Optional restore

Restore is intentionally manual and asks for confirmation:

```bash
docker compose stop api bot
bash scripts/restore_db.sh backups/smarn_YYYYMMDDTHHMMSSZ.dump
docker compose up -d api bot
```

Do not run restore unless you are intentionally replacing the current database
contents.

## 10. Configure GitHub Secrets

In GitHub, open:

```text
Repository -> Settings -> Secrets and variables -> Actions -> New repository secret
```

Add:

```text
SMARN_SERVER_HOST
SMARN_SERVER_USER
SMARN_SERVER_SSH_KEY
SMARN_SERVER_PORT
SMARN_APP_DIR
```

Example values:

```text
SMARN_SERVER_HOST=YOUR_SERVER_IP
SMARN_SERVER_USER=ubuntu
SMARN_SERVER_PORT=22
SMARN_APP_DIR=/home/ubuntu/apps/SMARN
```

`SMARN_SERVER_SSH_KEY` should be a private key that can SSH into the server.
The matching public key must be in `~/.ssh/authorized_keys` on the VM.

Do not add `OPENAI_API_KEY` or `TELEGRAM_BOT_TOKEN` to GitHub Actions. They
belong only in the server `.env`.

## 11. Automatic deploys

Two workflows are included:

- `.github/workflows/ci.yml` runs tests for pull requests.
- `.github/workflows/deploy.yml` runs tests on pushes to `main`, then SSHes to
  the server and runs `bash scripts/deploy.sh`.

Deployment is conservative:

- tests must pass first
- build and migrations run before app/bot containers are restarted
- database volumes are not deleted
- backups are not automatic yet

After pushing to `main`, check:

```bash
docker compose ps
docker compose logs -f bot
```
