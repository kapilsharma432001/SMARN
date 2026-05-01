# SMARN

SMARN: Self Memory Archive & Recall Network.

SMARN stores Telegram text and voice memories in PostgreSQL with pgvector,
enriches saved memories, answers questions through humanized recall, and can
generate daily or weekly memory reviews.

## Stack

- Python 3.12+
- FastAPI
- PostgreSQL + pgvector
- SQLAlchemy 2.x
- Alembic
- Pydantic v2 / pydantic-settings
- python-telegram-bot
- Docker Compose
- pytest

## Project layout

```text
src/smarn/
  api/          FastAPI routes and dependencies
  db/           SQLAlchemy base, session, and models
  memories/     Memory service, repository, and embedding provider
  telegram/     Telegram bot entrypoint and command handlers
alembic/        Database migrations
tests/          Basic test suite
```

## Memory schema

`memory_entries` stores:

```text
id
user_id
source
raw_text
summary
category
tags
importance_score
embedding
created_at
updated_at
deleted_at
```

Allowed categories:

- `personal`
- `work`
- `learning`
- `command`
- `idea`
- `reminder_candidate`
- `unknown`

## Local setup

Create an environment file:

```bash
cp .env.example .env
```

Install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Start PostgreSQL with pgvector:

```bash
docker compose up -d db
```

Run migrations:

```bash
alembic upgrade head
```

Start the API:

```bash
uvicorn smarn.main:create_app --factory --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Run the Telegram bot:

```bash
export TELEGRAM_BOT_TOKEN=your-token
python -m smarn.telegram.bot
```

Bot logs are written to stdout as structured JSON. Use `LOG_LEVEL=DEBUG` or
`LOG_LEVEL=INFO` to control verbosity.

## Docker Compose

Run the database, API, and Telegram bot together:

```bash
docker compose up --build
```

The API runs migrations on startup and listens on `http://127.0.0.1:8000`.
PostgreSQL is not exposed publicly.

For 24/7 VPS deployment, see:

```text
docs/deployment/oracle-free-vm.md
```

## Bot commands

- `/start` shows a short usage prompt.
- `/remember <text>` stores and enriches a memory for the Telegram user.
- `/ask <question>` retrieves relevant memories for that Telegram user and returns a natural answer.
- Voice notes are transcribed, enriched, and saved as `telegram_voice` memories.
- `/daily_review` summarizes memories from today.
- `/weekly_review` summarizes memories from the past 7 days.

## Notes

`raw_text` is kept unchanged. If LLM enrichment is unavailable, SMARN still saves
the raw memory with `unknown` category, no tags, and importance score `1`.
Voice transcription uses the configured transcription provider and does not store
audio files beyond temporary processing. 

## Tests

```bash
pytest
```
