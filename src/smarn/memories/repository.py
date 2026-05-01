from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from smarn.db.models import MemoryEntry, MemoryObservation
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
        importance_score: int = 1,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            user_id=user_id,
            embedding=embedding,
            raw_text=raw_text,
            summary=summary,
            category=category,
            tags=tags or [],
            importance_score=importance_score,
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
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        category: MemoryCategory | None = None,
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
        if start_at is not None:
            statement = statement.where(MemoryEntry.created_at >= start_at)
        if end_at is not None:
            statement = statement.where(MemoryEntry.created_at < end_at)
        if category is not None:
            statement = statement.where(MemoryEntry.category == category)

        rows = self.session.execute(statement).all()
        return [(entry, float(score)) for entry, score in rows]

    def list_created_between(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        user_id: str | None = None,
        category: MemoryCategory | None = None,
    ) -> list[MemoryEntry]:
        statement = (
            select(MemoryEntry)
            .where(MemoryEntry.deleted_at.is_(None))
            .where(MemoryEntry.created_at >= start_at)
            .where(MemoryEntry.created_at < end_at)
            .order_by(MemoryEntry.created_at.asc())
        )

        if user_id is not None:
            statement = statement.where(MemoryEntry.user_id == user_id)
        if category is not None:
            statement = statement.where(MemoryEntry.category == category)

        return list(self.session.scalars(statement).all())


class ObservationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(
        self,
        *,
        memory_id: object,
        user_id: str | None,
        observations: Sequence[object],
    ) -> list[MemoryObservation]:
        rows: list[MemoryObservation] = []
        for observation in observations:
            row = MemoryObservation(
                memory_id=memory_id,
                user_id=user_id,
                observation_type=getattr(observation, "observation_type"),
                label=getattr(observation, "label"),
                value_text=getattr(observation, "value_text"),
                value_number=_to_decimal(getattr(observation, "value_number")),
                unit=getattr(observation, "unit"),
                occurred_at=getattr(observation, "occurred_at"),
                confidence=getattr(observation, "confidence"),
                metadata_=getattr(observation, "metadata"),
            )
            self.session.add(row)
            rows.append(row)
        if rows:
            self.session.flush()
        return rows

    def list_for_analytics(
        self,
        *,
        user_id: str | None,
        start_at: datetime,
        end_at: datetime,
        observation_types: Sequence[str] | None = None,
    ) -> list[MemoryObservation]:
        observed_at = func.coalesce(
            MemoryObservation.occurred_at,
            MemoryObservation.created_at,
        )
        statement = (
            select(MemoryObservation)
            .where(observed_at >= start_at)
            .where(observed_at < end_at)
            .order_by(observed_at.asc())
        )
        if user_id is not None:
            statement = statement.where(MemoryObservation.user_id == user_id)
        if observation_types:
            statement = statement.where(
                MemoryObservation.observation_type.in_(list(observation_types))
            )
        return list(self.session.scalars(statement).all())


def _to_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except Exception:
        return None
