"""add indexes and memory observations

Revision ID: 0003_add_indexes_and_observations
Revises: 0002_add_memory_importance_score
Create Date: 2026-05-01 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0003_add_indexes_and_observations"
down_revision: str | None = "0002_add_memory_importance_score"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("observation_type", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=True),
        sa.Column("value_text", sa.Text(), nullable=True),
        sa.Column("value_number", sa.Numeric(), nullable=True),
        sa.Column("unit", sa.String(length=64), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_memory_observations_confidence_range",
        ),
        sa.ForeignKeyConstraint(
            ["memory_id"],
            ["memory_entries.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_memory_observations_user_id"),
        "memory_observations",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        "ix_memory_observations_user_type_occurred",
        "memory_observations",
        ["user_id", "observation_type", "occurred_at"],
        unique=False,
    )
    op.create_index(
        "ix_memory_observations_user_created",
        "memory_observations",
        ["user_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_memory_observations_metadata_gin",
        "memory_observations",
        ["metadata"],
        unique=False,
        postgresql_using="gin",
    )

    with context.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            ix_memory_entries_user_created_desc
            ON memory_entries (user_id, created_at DESC)
            WHERE deleted_at IS NULL
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            ix_memory_entries_user_category_created_desc
            ON memory_entries (user_id, category, created_at DESC)
            WHERE deleted_at IS NULL
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            ix_memory_entries_tags_gin
            ON memory_entries USING gin (tags)
            WHERE deleted_at IS NULL
            """
        )
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            ix_memory_entries_embedding_hnsw
            ON memory_entries USING hnsw (embedding vector_cosine_ops)
            WHERE deleted_at IS NULL
            """
        )


def downgrade() -> None:
    with context.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_memory_entries_embedding_hnsw"
        )
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_memory_entries_tags_gin")
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "ix_memory_entries_user_category_created_desc"
        )
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_memory_entries_user_created_desc"
        )

    op.drop_index(
        "ix_memory_observations_metadata_gin",
        table_name="memory_observations",
    )
    op.drop_index(
        "ix_memory_observations_user_created",
        table_name="memory_observations",
    )
    op.drop_index(
        "ix_memory_observations_user_type_occurred",
        table_name="memory_observations",
    )
    op.drop_index(
        op.f("ix_memory_observations_user_id"),
        table_name="memory_observations",
    )
    op.drop_table("memory_observations")
