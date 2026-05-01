from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from smarn.config import get_settings
from smarn.db.base import Base
from smarn.memories.categories import MemoryCategory, memory_category_values


class MemoryEntry(Base):
    __tablename__ = "memory_entries"
    __table_args__ = (
        CheckConstraint(
            "importance_score >= 1 AND importance_score <= 5",
            name="ck_memory_entries_importance_score_range",
        ),
        Index("ix_memory_entries_created_at", "created_at"),
        Index("ix_memory_entries_deleted_at", "deleted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="telegram")
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    category: Mapped[MemoryCategory] = mapped_column(
        Enum(
            MemoryCategory,
            name="memory_category",
            values_callable=lambda _: memory_category_values(),
        ),
        nullable=False,
        default=MemoryCategory.UNKNOWN,
        server_default=MemoryCategory.UNKNOWN.value,
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String()),
        nullable=False,
        default=list,
        server_default=text("'{}'::text[]"),
    )
    importance_score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=text("1"),
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(get_settings().embedding_dimensions),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryObservation(Base):
    __tablename__ = "memory_observations"
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_observations_confidence_range",
        ),
        Index(
            "ix_memory_observations_user_type_occurred",
            "user_id",
            "observation_type",
            "occurred_at",
        ),
        Index("ix_memory_observations_user_created", "user_id", "created_at"),
        Index(
            "ix_memory_observations_metadata_gin",
            "metadata",
            postgresql_using="gin",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    memory_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("memory_entries.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str | None] = mapped_column(String(64), index=True)
    observation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str | None] = mapped_column(String(255))
    value_text: Mapped[str | None] = mapped_column(Text)
    value_number: Mapped[Decimal | None] = mapped_column(Numeric())
    unit: Mapped[str | None] = mapped_column(String(64))
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    metadata_: Mapped[dict[str, object]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
