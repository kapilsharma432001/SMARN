from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.categories import MemoryCategory, coerce_memory_category
from smarn.memories.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from smarn.memories.repository import MemoryRepository


@dataclass(frozen=True)
class MemorySearchResult:
    id: uuid.UUID
    raw_text: str
    summary: str | None
    category: MemoryCategory
    tags: list[str]
    created_at: datetime
    score: float


class MemoryService:
    def __init__(
        self,
        session: Session,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = MemoryRepository(session)
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            api_key = self.settings.openai_api_key
            if api_key is None:
                raise ValueError("OPENAI_API_KEY must be set in the environment.")
            self.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key.get_secret_value(),
                dimensions=self.settings.embedding_dimensions,
            )

    def remember(
        self,
        raw_text: str,
        *,
        user_id: str | None = None,
        source: str = "telegram",
        summary: str | None = None,
        category: MemoryCategory | str | None = None,
        tags: list[str] | None = None,
    ) -> uuid.UUID:
        cleaned = raw_text.strip()
        if not cleaned:
            raise ValueError("Memory content cannot be empty.")

        cleaned_summary = summary.strip() if summary else None

        entry = self.repository.create(
            raw_text=cleaned,
            embedding=self.embedding_provider.embed(cleaned),
            source=source,
            user_id=user_id,
            summary=cleaned_summary or None,
            category=coerce_memory_category(category),
            tags=tags,
        )
        return entry.id

    def ask(
        self,
        question: str,
        *,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemorySearchResult]:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty.")

        rows = self.repository.search(
            embedding=self.embedding_provider.embed(cleaned),
            user_id=user_id,
            limit=limit or self.settings.memory_search_limit,
        )

        return [
            MemorySearchResult(
                id=entry.id,
                raw_text=entry.raw_text,
                summary=entry.summary,
                category=entry.category,
                tags=entry.tags,
                created_at=entry.created_at,
                score=score,
            )
            for entry, score in rows
        ]
