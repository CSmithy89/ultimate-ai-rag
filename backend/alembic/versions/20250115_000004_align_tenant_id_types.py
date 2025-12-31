"""align tenant_id types to UUID

Revision ID: 20250115_000004
Revises: 20251231_000003
Create Date: 2025-01-15 00:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20250115_000004"
down_revision = "20251231_000003"
branch_labels = None
depends_on = None

UUID_REGEX = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


def _assert_valid_tenant_ids(table_name: str) -> None:
    bind = op.get_bind()
    result = bind.execute(
        sa.text(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            WHERE tenant_id IS NULL OR tenant_id !~* :uuid_regex
            """
        ),
        {"uuid_regex": UUID_REGEX},
    )
    invalid_count = result.scalar() or 0
    if invalid_count:
        raise RuntimeError(
            f"{table_name} contains {invalid_count} invalid tenant_id values; "
            "clean the data before running this migration."
        )


def upgrade() -> None:
    for table in ("trajectories", "trajectory_events", "llm_usage_events", "llm_cost_alerts"):
        _assert_valid_tenant_ids(table)

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
