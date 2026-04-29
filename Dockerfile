FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .

COPY alembic.ini /app/alembic.ini
COPY alembic /app/alembic

CMD ["uvicorn", "smarn.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
