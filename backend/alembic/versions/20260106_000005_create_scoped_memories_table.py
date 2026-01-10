"""create scoped_memories table

Revision ID: 20260106_000005
Revises: 20250115_000004
Create Date: 2026-01-06 00:00:00

Epic 20 Story 20-A1: Scoped memory hierarchy for user, session, agent, and global memories.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "20260106_000005"
down_revision = "20250115_000004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Ensure pgvector extension exists (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create scope enum type
    scope_enum = postgresql.ENUM(
        "user", "session", "agent", "global",
        name="memory_scope",
        create_type=False,
    )
    scope_enum.create(op.get_bind(), checkfirst=True)

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("scoped_memories"):
        op.create_table(
            "scoped_memories",
            sa.Column(
                "id",
                postgresql.UUID(as_uuid=True),
                primary_key=True,
                nullable=False,
            ),
            sa.Column(
                "tenant_id",
                postgresql.UUID(as_uuid=True),
                nullable=False,
            ),
            sa.Column(
                "scope",
                postgresql.ENUM(
                    "user", "session", "agent", "global",
                    name="memory_scope",
                    create_type=False,
                ),
                nullable=False,
            ),
            sa.Column(
                "user_id",
                postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
            sa.Column(
                "session_id",
                postgresql.UUID(as_uuid=True),
                nullable=True,
            ),
            sa.Column(
                "agent_id",
                sa.Text(),
                nullable=True,
            ),
            sa.Column(
                "content",
                sa.Text(),
                nullable=False,
            ),
            sa.Column(
                "importance",
                sa.Float(),
                nullable=False,
                server_default="1.0",
            ),
            sa.Column(
                "metadata",
                postgresql.JSONB(),
                nullable=True,
                server_default="{}",
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "accessed_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "access_count",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )

    # Create pgvector column using raw SQL (1536-dimension for OpenAI ada-002)
    # Note: Using raw SQL because SQLAlchemy doesn't natively support pgvector type
    op.execute(
        "ALTER TABLE scoped_memories "
        "ADD COLUMN IF NOT EXISTS embedding vector(1536)"
    )

    # Multi-tenant index (required for all queries per CLAUDE.md)
    # Use raw SQL for IF NOT EXISTS support to be idempotent
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_scoped_memories_tenant_id "
        "ON scoped_memories(tenant_id)"
    )

    # Scope-based filtering indexes
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_scoped_memories_tenant_scope "
        "ON scoped_memories(tenant_id, scope)"
    )
    op.create_index(
        "idx_scoped_memories_tenant_user",
        "scoped_memories",
        ["tenant_id", "user_id"],
    )
    op.create_index(
        "idx_scoped_memories_tenant_session",
        "scoped_memories",
        ["tenant_id", "session_id"],
    )
    op.create_index(
        "idx_scoped_memories_tenant_agent",
        "scoped_memories",
        ["tenant_id", "agent_id"],
    )

    # Composite index for hierarchical scope queries
    op.create_index(
        "idx_scoped_memories_hierarchy",
        "scoped_memories",
        ["tenant_id", "scope", "user_id", "session_id"],
    )

    # Importance-based sorting (for consolidation)
    op.create_index(
        "idx_scoped_memories_importance",
        "scoped_memories",
        ["tenant_id", "importance"],
    )

    # Accessed_at for decay calculations
    op.create_index(
        "idx_scoped_memories_accessed_at",
        "scoped_memories",
        ["tenant_id", "accessed_at"],
    )

    # Vector similarity search index (HNSW for approximate nearest neighbor)
    # Ensure we replace any existing index with the desired implementation
    op.execute("DROP INDEX IF EXISTS idx_scoped_memories_embedding")
    op.execute(
        "CREATE INDEX idx_scoped_memories_embedding "
        "ON scoped_memories USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_scoped_memories_embedding")
    op.drop_index("idx_scoped_memories_accessed_at", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_importance", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_hierarchy", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_tenant_agent", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_tenant_session", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_tenant_user", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_tenant_scope", table_name="scoped_memories")
    op.drop_index("idx_scoped_memories_tenant_id", table_name="scoped_memories")

    # Drop table
    op.drop_table("scoped_memories")

    # Drop enum type
    op.execute("DROP TYPE IF EXISTS memory_scope")
