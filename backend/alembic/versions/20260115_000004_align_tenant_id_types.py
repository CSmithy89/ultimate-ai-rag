"""align tenant_id types to UUID

Revision ID: 20260115_000004
Revises: 20251231_000003
Create Date: 2026-01-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260115_000004"
down_revision = "20251231_000003"
branch_labels = None
depends_on = None

UUID_REGEX = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
ZERO_UUID = "00000000-0000-0000-0000-000000000000"


def _normalize_tenant_ids(table_name: str) -> None:
    op.execute(
        f"""
        UPDATE {table_name}
        SET tenant_id = '{ZERO_UUID}'
        WHERE tenant_id IS NULL OR tenant_id !~* '{UUID_REGEX}'
        """
    )


def upgrade() -> None:
    for table in ("trajectories", "trajectory_events", "llm_usage_events", "llm_cost_alerts"):
        _normalize_tenant_ids(table)

    op.alter_column(
        "trajectories",
        "tenant_id",
        existing_type=sa.Text(),
        type_=postgresql.UUID(as_uuid=True),
        server_default=None,
        postgresql_using="tenant_id::uuid",
    )
    op.alter_column(
        "trajectory_events",
        "tenant_id",
        existing_type=sa.Text(),
        type_=postgresql.UUID(as_uuid=True),
        server_default=None,
        postgresql_using="tenant_id::uuid",
    )
    op.alter_column(
        "llm_usage_events",
        "tenant_id",
        existing_type=sa.Text(),
        type_=postgresql.UUID(as_uuid=True),
        postgresql_using="tenant_id::uuid",
    )
    op.alter_column(
        "llm_cost_alerts",
        "tenant_id",
        existing_type=sa.Text(),
        type_=postgresql.UUID(as_uuid=True),
        postgresql_using="tenant_id::uuid",
    )


def downgrade() -> None:
    op.alter_column(
        "llm_cost_alerts",
        "tenant_id",
        existing_type=postgresql.UUID(as_uuid=True),
        type_=sa.Text(),
        postgresql_using="tenant_id::text",
    )
    op.alter_column(
        "llm_usage_events",
        "tenant_id",
        existing_type=postgresql.UUID(as_uuid=True),
        type_=sa.Text(),
        postgresql_using="tenant_id::text",
    )
    op.alter_column(
        "trajectory_events",
        "tenant_id",
        existing_type=postgresql.UUID(as_uuid=True),
        type_=sa.Text(),
        server_default="unknown",
        postgresql_using="tenant_id::text",
    )
    op.alter_column(
        "trajectories",
        "tenant_id",
        existing_type=postgresql.UUID(as_uuid=True),
        type_=sa.Text(),
        server_default="unknown",
        postgresql_using="tenant_id::text",
    )
