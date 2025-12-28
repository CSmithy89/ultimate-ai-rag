"""create trajectory tables

Revision ID: 20251229_000001
Revises: 
Create Date: 2025-12-29 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20251229_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "trajectories",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_table(
        "trajectory_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "trajectory_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("trajectories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tenant_id", sa.Text(), nullable=False, server_default=sa.text("'unknown'")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_trajectory_events_trajectory_id",
        "trajectory_events",
        ["trajectory_id"],
    )
    op.create_index(
        "idx_trajectory_events_created_at",
        "trajectory_events",
        ["created_at"],
    )
    op.create_index(
        "idx_trajectory_events_event_type",
        "trajectory_events",
        ["event_type"],
    )
    op.create_index(
        "idx_trajectories_session_id",
        "trajectories",
        ["session_id"],
    )
    op.create_index(
        "idx_trajectories_tenant_id",
        "trajectories",
        ["tenant_id"],
    )
    op.create_index(
        "idx_trajectory_events_tenant_id",
        "trajectory_events",
        ["tenant_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_trajectory_events_tenant_id", table_name="trajectory_events")
    op.drop_index("idx_trajectories_tenant_id", table_name="trajectories")
    op.drop_index("idx_trajectories_session_id", table_name="trajectories")
    op.drop_index("idx_trajectory_events_event_type", table_name="trajectory_events")
    op.drop_index("idx_trajectory_events_created_at", table_name="trajectory_events")
    op.drop_index("idx_trajectory_events_trajectory_id", table_name="trajectory_events")
    op.drop_table("trajectory_events")
    op.drop_table("trajectories")
