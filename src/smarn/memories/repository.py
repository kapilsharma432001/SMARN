from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from smarn.db.models import MemoryEntry
from smarn.memories.categories import MemoryCategory


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self,
        *,
        raw_text: str,
        embedding: list[float],
        source: str = "telegram",
        user_id: str | None = None,
        summary: str | None = None,
        category: MemoryCategory = MemoryCategory.UNKNOWN,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            user_id=user_id,
            embedding=embedding,
            raw_text=raw_text,
            summary=summary,
            category=category,
            tags=tags or [],
            source=source,
        )
        self.session.add(entry)
        self.session.flush()
        return entry

    def search(
        self,
        *,
        embedding: list[float],
        limit: int,
        user_id: str | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        distance = MemoryEntry.embedding.cosine_distance(embedding).label("distance")
        statement = (
            select(MemoryEntry, distance)
            .where(MemoryEntry.deleted_at.is_(None))
            .order_by(distance)
            .limit(limit)
        )

        if user_id is not None:
            statement = statement.where(MemoryEntry.user_id == user_id)

        rows = self.session.execute(statement).all()
        return [(entry, float(score)) for entry, score in rows]
