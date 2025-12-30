"""add has_error to trajectories

Revision ID: 20251231_000003
Revises: 20251231_000002
Create Date: 2025-12-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20251231_000003"
down_revision = "20251231_000002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "trajectories",
        sa.Column("has_error", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index(
        "idx_trajectories_has_error",
        "trajectories",
        ["has_error"],
    )


def downgrade() -> None:
    op.drop_index("idx_trajectories_has_error", table_name="trajectories")
    op.drop_column("trajectories", "has_error")
