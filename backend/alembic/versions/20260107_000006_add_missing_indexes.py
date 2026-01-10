"""add missing indexes for Epic 20 features

Revision ID: 20260107_000006
Revises: 20260106_000005
Create Date: 2026-01-07 00:00:00

Epic 20: Add missing database indexes for performance optimization.

PostgreSQL indexes added:
- scoped_memories(tenant_id, created_at) - for time-based queries

Neo4j indexes (run manually or via application startup):
- CREATE INDEX entity_tenant_idx IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id)
- CREATE INDEX community_tenant_idx IF NOT EXISTS FOR (c:Community) ON (c.tenant_id)
- CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.tenant_id, c.level)
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260107_000006"
down_revision = "20260106_000005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add missing time-based query index for scoped_memories
    op.create_index(
        "idx_scoped_memories_tenant_created",
        "scoped_memories",
        ["tenant_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_scoped_memories_tenant_created", table_name="scoped_memories")
