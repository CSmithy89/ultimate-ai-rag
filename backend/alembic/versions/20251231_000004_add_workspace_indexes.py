"""add workspace indexes

Revision ID: 20251231_000004
Revises: 20251231_000003
Create Date: 2025-12-31 00:00:00
"""

from alembic import op


revision = "20251231_000004"
down_revision = "20251231_000003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'workspace_items'
            ) THEN
                CREATE INDEX IF NOT EXISTS idx_workspace_items_tenant_content_id
                ON workspace_items(tenant_id, content_id);
            END IF;
        END $$;
        """
    )
    # workspace_shares already has a primary key on id and an index on tenant_id.


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_workspace_items_tenant_content_id")
