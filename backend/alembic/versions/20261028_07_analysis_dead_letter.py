"""Add analysis dead letter table.

Revision ID: 20261028_07
Revises: 20260721_06
Create Date: 2026-10-28 16:00:00
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "20261028_07"
down_revision: Union[str, None] = "20260721_06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_dead_letters",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("task_id", sa.String(length=36), nullable=False),
        sa.Column("media_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("goal", sa.String(length=500), nullable=True),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("payload", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("task_id", name="uq_dead_letter_task"),
    )
    op.create_index("ix_analysis_dead_letters_task_id", "analysis_dead_letters", ["task_id"])
    op.create_index("ix_analysis_dead_letters_media_id", "analysis_dead_letters", ["media_id"])
    op.create_index("ix_dead_letter_created", "analysis_dead_letters", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_dead_letter_created", table_name="analysis_dead_letters")
    op.drop_index("ix_analysis_dead_letters_media_id", table_name="analysis_dead_letters")
    op.drop_index("ix_analysis_dead_letters_task_id", table_name="analysis_dead_letters")
    op.drop_table("analysis_dead_letters")
