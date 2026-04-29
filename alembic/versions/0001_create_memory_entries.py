"""create memory entries

Revision ID: 0001_create_memory_entries
Revises:
Create Date: 2026-04-29 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_create_memory_entries"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    memory_category = postgresql.ENUM(
        "personal",
        "work",
        "learning",
        "command",
        "idea",
        "reminder_candidate",
        "unknown",
        name="memory_category",
        create_type=False,
    )
    postgresql.ENUM(
        "personal",
        "work",
        "learning",
        "command",
        "idea",
        "reminder_candidate",
        "unknown",
        name="memory_category",
    ).create(op.get_bind(), checkfirst=True)

    op.create_table(
        "memory_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "category",
            memory_category,
            server_default="unknown",
            nullable=False,
        ),
        sa.Column(
            "tags",
            postgresql.ARRAY(sa.String()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_memory_entries_user_id"),
        "memory_entries",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_entries_created_at",
        "memory_entries",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_memory_entries_deleted_at",
        "memory_entries",
        ["deleted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_memory_entries_deleted_at", table_name="memory_entries")
    op.drop_index("ix_memory_entries_created_at", table_name="memory_entries")
    op.drop_index(op.f("ix_memory_entries_user_id"), table_name="memory_entries")
    op.drop_table("memory_entries")
    postgresql.ENUM(name="memory_category").drop(op.get_bind(), checkfirst=True)
