"""add agent_type to trajectories

Revision ID: 20251231_000002
Revises: 20251229_000001
Create Date: 2025-12-31 00:00:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20251231_000002"
down_revision = "20251229_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("trajectories", sa.Column("agent_type", sa.Text(), nullable=True))
    op.create_index(
        "idx_trajectories_agent_type",
        "trajectories",
        ["agent_type"],
    )


def downgrade() -> None:
    op.drop_index("idx_trajectories_agent_type", table_name="trajectories")
    op.drop_column("trajectories", "agent_type")
