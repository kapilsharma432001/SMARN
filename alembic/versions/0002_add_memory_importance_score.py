"""add memory importance score

Revision ID: 0002_add_memory_importance_score
Revises: 0001_create_memory_entries
Create Date: 2026-04-30 00:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0002_add_memory_importance_score"
down_revision: str | None = "0001_create_memory_entries"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "memory_entries",
        sa.Column(
            "importance_score",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_memory_entries_importance_score_range",
        "memory_entries",
        "importance_score >= 1 AND importance_score <= 5",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_memory_entries_importance_score_range",
        "memory_entries",
        type_="check",
    )
    op.drop_column("memory_entries", "importance_score")
