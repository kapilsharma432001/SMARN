"""add indexes and memory observations

Revision ID: 0003_observations_indexes
Revises: 0002_add_memory_importance_score
Create Date: 2026-05-01 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import context, op

revision: str = "0003_observations_indexes"
down_revision: str | None = "0002_add_memory_importance_score"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_observations (
            id UUID NOT NULL,
            memory_id UUID NOT NULL,
            user_id VARCHAR(64),
            observation_type VARCHAR(64) NOT NULL,
            label VARCHAR(255),
            value_text TEXT,
            value_number NUMERIC,
            unit VARCHAR(64),
            occurred_at TIMESTAMP WITH TIME ZONE,
            confidence FLOAT NOT NULL,
            metadata JSONB DEFAULT '{}'::jsonb NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT ck_memory_observations_confidence_range
                CHECK (confidence >= 0 AND confidence <= 1),
            FOREIGN KEY(memory_id) REFERENCES memory_entries (id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_observations_user_id
        ON memory_observations (user_id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_observations_user_type_occurred
        ON memory_observations (user_id, observation_type, occurred_at)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_observations_user_created
        ON memory_observations (user_id, created_at)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_memory_observations_metadata_gin
        ON memory_observations USING gin (metadata)
        """
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

    op.execute("DROP INDEX IF EXISTS ix_memory_observations_metadata_gin")
    op.execute("DROP INDEX IF EXISTS ix_memory_observations_user_created")
    op.execute("DROP INDEX IF EXISTS ix_memory_observations_user_type_occurred")
    op.execute("DROP INDEX IF EXISTS ix_memory_observations_user_id")
    op.execute("DROP TABLE IF EXISTS memory_observations")
