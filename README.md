# SMARN

SMARN: Self Memory Archive & Recall Network.

MVP v0.1 stores Telegram memories in PostgreSQL with pgvector and retrieves relevant memories through `/ask`.

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

## Docker Compose

Run the API and database together:

```bash
docker compose up --build
```

The API will run migrations on startup and listen at `http://localhost:8000`.

## Bot commands

- `/start` shows a short usage prompt.
- `/remember <text>` stores a memory for the Telegram user.
- `/ask <question>` retrieves the closest stored memories for that Telegram user.

## Notes

The MVP uses a deterministic local hash embedding provider so the pgvector storage and retrieval path works without adding an external AI provider yet. Replace `HashEmbeddingProvider` when adding production embeddings.

## Tests

```bash
pytest
```
