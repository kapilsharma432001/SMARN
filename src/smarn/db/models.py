from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column

from smarn.config import get_settings
from smarn.db.base import Base
from smarn.memories.categories import MemoryCategory, memory_category_values


class MemoryEntry(Base):
    __tablename__ = "memory_entries"
    __table_args__ = (
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
