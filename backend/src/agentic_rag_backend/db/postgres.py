"""PostgreSQL async client for documents, jobs, chunks, and workspace tables."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import asyncpg
import structlog

from agentic_rag_backend.core.errors import DatabaseError
from agentic_rag_backend.models.ingest import JobProgress, JobStatus, JobStatusEnum, JobType

logger = structlog.get_logger(__name__)

WORKSPACE_MAX_CONTENT_BYTES = 100_000


def _validate_workspace_content(content: str) -> None:
    byte_size = len(content.encode("utf-8"))
    if byte_size > WORKSPACE_MAX_CONTENT_BYTES:
        raise ValueError(
            f"Content size ({byte_size} bytes) exceeds maximum of {WORKSPACE_MAX_CONTENT_BYTES} bytes"
        )


import math

def _validate_embedding(embedding: list[float], expected_dim: int = 1536) -> None:
    """Validate embedding dimension and values.
    
    Args:
        embedding: Vector to validate
        expected_dim: Expected number of dimensions (default 1536)
        
    Raises:
        ValueError: If dimension mismatch or contains non-finite values (NaN/Inf)
    """
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
        )
    
    for i, val in enumerate(embedding):
        if not math.isfinite(val):
            raise ValueError(f"Invalid embedding value at index {i}: {val}")


class PostgresClient:
    """
    Async PostgreSQL client for managing documents, ingestion jobs, and chunks.

    Implements multi-tenancy through tenant_id filtering on all queries.
    Supports pgvector for chunk embedding storage and similarity search.
    """

    def __init__(self, url: str) -> None:
        """
        Initialize PostgreSQL client.

        Args:
            url: PostgreSQL connection URL (e.g., postgresql://user:pass@localhost/db)
        """
        self.url = url
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.url,
                min_size=2,
                max_size=10,
            )
            logger.info("postgres_connected", url=self.url.split("@")[-1])

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("postgres_disconnected")

    @property
    def pool(self) -> asyncpg.Pool:
        """Get the connection pool, raising error if not connected."""
        if self._pool is None:
            raise DatabaseError("connection", "PostgreSQL pool not connected")
        return self._pool

    async def create_tables(self) -> None:
        """
        Create required tables if they don't exist.

        This creates the documents, ingestion_jobs, and chunks tables with proper
        indexes for multi-tenancy, status filtering, and vector similarity search.
        """
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

            # Create documents table with Story 4.2 columns (file_size, page_count)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    source_type VARCHAR(20) NOT NULL,
                    source_url TEXT,
                    filename TEXT,
                    content_hash VARCHAR(64) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    file_size BIGINT,
                    page_count INTEGER,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (tenant_id, content_hash)
                )
            """)

            # Create indexes for documents
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
                ON documents(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_status
                ON documents(status)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_content_hash
                ON documents(content_hash)
            """)

            # Create ingestion_jobs table with Story 4.2 column (processing_time_ms)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    document_id UUID REFERENCES documents(id),
                    job_type VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'queued',
                    progress JSONB,
                    error_message TEXT,
                    processing_time_ms INTEGER,
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes for ingestion_jobs
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_tenant_id
                ON ingestion_jobs(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_status
                ON ingestion_jobs(status)
            """)

            # Story 4.3: Create chunks table with pgvector embedding
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes for chunks table
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_tenant_id
                ON chunks(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id)
            """)

            # Create IVFFlat index for vector similarity search
            # Note: This requires at least some data to be present for optimal list sizing
            # Using a conservative list size for initial setup
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                    ON chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except asyncpg.PostgresError:
                # IVFFlat index creation may fail if no data exists yet
                # Will be created when sufficient data is available
                logger.warning("ivfflat_index_skipped", reason="may require data to exist first")

            # Epic 20: Hierarchical chunk storage for small-to-big retrieval
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS hierarchical_chunks (
                    id TEXT PRIMARY KEY,
                    tenant_id UUID NOT NULL,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    level INTEGER NOT NULL,
                    parent_id TEXT,
                    child_ids TEXT[],
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indexes for hierarchical chunks table
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hierarchical_chunks_tenant_id
                ON hierarchical_chunks(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hierarchical_chunks_document_id
                ON hierarchical_chunks(document_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hierarchical_chunks_level
                ON hierarchical_chunks(level)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hierarchical_chunks_parent_id
                ON hierarchical_chunks(parent_id)
            """)

            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_hierarchical_chunks_embedding
                    ON hierarchical_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except asyncpg.PostgresError:
                logger.warning(
                    "ivfflat_index_skipped",
                    reason="hierarchical chunks may require data to exist first",
                )

            # Epic 8: LLM cost monitoring tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_usage_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    trajectory_id UUID,
                    model_id TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    input_cost_usd NUMERIC(12, 6) NOT NULL,
                    output_cost_usd NUMERIC(12, 6) NOT NULL,
                    total_cost_usd NUMERIC(12, 6) NOT NULL,
                    complexity TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                ALTER TABLE llm_usage_events
                ADD COLUMN IF NOT EXISTS baseline_model_id TEXT
            """)
            await conn.execute("""
                ALTER TABLE llm_usage_events
                ADD COLUMN IF NOT EXISTS baseline_total_cost_usd NUMERIC(12, 6)
            """)
            await conn.execute("""
                ALTER TABLE llm_usage_events
                ADD COLUMN IF NOT EXISTS savings_usd NUMERIC(12, 6)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_usage_events_tenant_id
                ON llm_usage_events(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_usage_events_created_at
                ON llm_usage_events(created_at)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_llm_usage_events_model_id
                ON llm_usage_events(model_id)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cost_alerts (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    daily_threshold_usd NUMERIC(12, 2),
                    monthly_threshold_usd NUMERIC(12, 2),
                    enabled BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_llm_cost_alerts_tenant_id
                ON llm_cost_alerts(tenant_id)
            """)

            # Epic 11: Workspace persistence tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_items (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    content_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    title TEXT,
                    query TEXT,
                    sources JSONB,
                    session_id TEXT,
                    trajectory_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspace_items_tenant_id
                ON workspace_items(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspace_items_content_id
                ON workspace_items(content_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspace_items_tenant_content_id
                ON workspace_items(tenant_id, content_id)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_shares (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    content_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    title TEXT,
                    query TEXT,
                    sources JSONB,
                    token TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspace_shares_tenant_id
                ON workspace_shares(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspace_shares_expires_at
                ON workspace_shares(expires_at)
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_bookmarks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    content_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    title TEXT,
                    query TEXT,
                    session_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_workspace_bookmarks_tenant_id
                ON workspace_bookmarks(tenant_id)
            """)

            # Epic 20: Scoped memories table (Story 20-A1)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scoped_memories (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    scope VARCHAR(20) NOT NULL,
                    user_id UUID,
                    session_id UUID,
                    agent_id VARCHAR(100),
                    content TEXT NOT NULL,
                    importance NUMERIC(3,2) NOT NULL DEFAULT 1.0,
                    metadata JSONB DEFAULT '{}',
                    embedding vector(1536),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    accessed_at TIMESTAMPTZ DEFAULT NOW(),
                    access_count INTEGER DEFAULT 0
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoped_memories_tenant_id
                ON scoped_memories(tenant_id)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoped_memories_scope
                ON scoped_memories(scope)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoped_memories_user_id
                ON scoped_memories(user_id) WHERE user_id IS NOT NULL
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoped_memories_session_id
                ON scoped_memories(session_id) WHERE session_id IS NOT NULL
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoped_memories_agent_id
                ON scoped_memories(agent_id) WHERE agent_id IS NOT NULL
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scoped_memories_tenant_scope
                ON scoped_memories(tenant_id, scope)
            """)
            # Vector similarity index for memory search
            try:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_scoped_memories_embedding
                    ON scoped_memories USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
            except asyncpg.PostgresError:
                # IVFFlat index creation may fail if no data exists yet
                logger.warning("scoped_memories_ivfflat_index_skipped", reason="may require data to exist first")

            logger.info("tables_created")

    async def create_document(
        self,
        tenant_id: UUID,
        source_type: str,
        content_hash: str,
        source_url: Optional[str] = None,
        filename: Optional[str] = None,
        file_size: Optional[int] = None,
        page_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """
        Create a new document record.

        Args:
            tenant_id: Tenant identifier
            source_type: Type of source ('url', 'pdf', 'text')
            content_hash: SHA-256 hash of content for deduplication
            source_url: Source URL for web documents
            filename: Filename for uploaded documents
            file_size: File size in bytes (for PDF documents)
            page_count: Number of pages (for PDF documents)
            metadata: Additional document metadata

        Returns:
            UUID of the created document

        Raises:
            DatabaseError: If creation fails
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO documents (tenant_id, source_type, source_url, filename, content_hash, file_size, page_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (tenant_id, content_hash) DO UPDATE
                    SET updated_at = NOW()
                    RETURNING id
                    """,
                    tenant_id,
                    source_type,
                    source_url,
                    filename,
                    content_hash,
                    file_size,
                    page_count,
                    metadata,
                )
                doc_id = row["id"]
                logger.info("document_created", document_id=str(doc_id), tenant_id=str(tenant_id))
                return doc_id
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_document", str(e)) from e

    async def get_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get a document by ID with tenant filtering.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier for access control

        Returns:
            Document record as dictionary, or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM documents
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    document_id,
                    tenant_id,
                )
                return dict(row) if row else None
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_document", str(e)) from e

    async def update_document_status(
        self,
        document_id: UUID,
        tenant_id: UUID,
        status: str,
    ) -> bool:
        """
        Update document status.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier
            status: New status value

        Returns:
            True if updated, False if not found
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE documents
                    SET status = $3, updated_at = NOW()
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    document_id,
                    tenant_id,
                    status,
                )
                return result == "UPDATE 1"
        except asyncpg.PostgresError as e:
            raise DatabaseError("update_document_status", str(e)) from e

    async def update_document_page_count(
        self,
        document_id: UUID,
        tenant_id: UUID,
        page_count: int,
    ) -> bool:
        """
        Update document page count after parsing.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier
            page_count: Number of pages in the document

        Returns:
            True if updated, False if not found
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE documents
                    SET page_count = $3, updated_at = NOW()
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    document_id,
                    tenant_id,
                    page_count,
                )
                return result == "UPDATE 1"
        except asyncpg.PostgresError as e:
            raise DatabaseError("update_document_page_count", str(e)) from e

    async def create_job(
        self,
        tenant_id: UUID,
        job_type: JobType,
        document_id: Optional[UUID] = None,
    ) -> UUID:
        """
        Create a new ingestion job.

        Args:
            tenant_id: Tenant identifier
            job_type: Type of job (crawl, parse, index)
            document_id: Optional associated document ID

        Returns:
            UUID of the created job

        Raises:
            DatabaseError: If creation fails
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO ingestion_jobs (tenant_id, document_id, job_type, status)
                    VALUES ($1, $2, $3, 'queued')
                    RETURNING id, created_at
                    """,
                    tenant_id,
                    document_id,
                    job_type.value,
                )
                job_id = row["id"]
                logger.info(
                    "job_created",
                    job_id=str(job_id),
                    tenant_id=str(tenant_id),
                    job_type=job_type.value,
                )
                return job_id
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_job", str(e)) from e

    async def get_job(
        self,
        job_id: UUID,
        tenant_id: UUID,
    ) -> Optional[JobStatus]:
        """
        Get job status by ID with tenant filtering.

        Args:
            job_id: Job UUID
            tenant_id: Tenant identifier for access control

        Returns:
            JobStatus model, or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT * FROM ingestion_jobs
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    job_id,
                    tenant_id,
                )
                if not row:
                    return None

                progress = None
                if row["progress"]:
                    progress = JobProgress(**row["progress"])

                return JobStatus(
                    job_id=row["id"],
                    tenant_id=row["tenant_id"],
                    job_type=JobType(row["job_type"]),
                    status=JobStatusEnum(row["status"]),
                    progress=progress,
                    error_message=row["error_message"],
                    created_at=row["created_at"],
                    started_at=row["started_at"],
                    completed_at=row["completed_at"],
                )
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_job", str(e)) from e

    async def update_job_status(
        self,
        job_id: UUID,
        tenant_id: UUID,
        status: JobStatusEnum,
        progress: Optional[dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update job status and progress.

        Args:
            job_id: Job UUID
            tenant_id: Tenant identifier
            status: New status value
            progress: Optional progress metrics
            error_message: Optional error message for failed jobs

        Returns:
            True if updated, False if not found
        """
        try:
            async with self.pool.acquire() as conn:
                # Build the update query dynamically
                if status == JobStatusEnum.RUNNING:
                    result = await conn.execute(
                        """
                        UPDATE ingestion_jobs
                        SET status = $3, progress = $4, started_at = NOW()
                        WHERE id = $1 AND tenant_id = $2
                        """,
                        job_id,
                        tenant_id,
                        status.value,
                        progress,
                    )
                elif status in (JobStatusEnum.COMPLETED, JobStatusEnum.FAILED):
                    result = await conn.execute(
                        """
                        UPDATE ingestion_jobs
                        SET status = $3, progress = $4, error_message = $5, completed_at = NOW()
                        WHERE id = $1 AND tenant_id = $2
                        """,
                        job_id,
                        tenant_id,
                        status.value,
                        progress,
                        error_message,
                    )
                else:
                    result = await conn.execute(
                        """
                        UPDATE ingestion_jobs
                        SET status = $3, progress = $4
                        WHERE id = $1 AND tenant_id = $2
                        """,
                        job_id,
                        tenant_id,
                        status.value,
                        progress,
                    )

                updated = result == "UPDATE 1"
                if updated:
                    logger.info(
                        "job_status_updated",
                        job_id=str(job_id),
                        status=status.value,
                    )
                return updated
        except asyncpg.PostgresError as e:
            raise DatabaseError("update_job_status", str(e)) from e

    async def update_job_processing_time(
        self,
        job_id: UUID,
        tenant_id: UUID,
        processing_time_ms: int,
    ) -> bool:
        """
        Update job processing time for NFR2 performance tracking.

        Args:
            job_id: Job UUID
            tenant_id: Tenant identifier
            processing_time_ms: Processing time in milliseconds

        Returns:
            True if updated, False if not found
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE ingestion_jobs
                    SET processing_time_ms = $3
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    job_id,
                    tenant_id,
                    processing_time_ms,
                )
                updated = result == "UPDATE 1"
                if updated:
                    logger.info(
                        "job_processing_time_updated",
                        job_id=str(job_id),
                        processing_time_ms=processing_time_ms,
                    )
                return updated
        except asyncpg.PostgresError as e:
            raise DatabaseError("update_job_processing_time", str(e)) from e

    async def list_jobs(
        self,
        tenant_id: UUID,
        status: Optional[JobStatusEnum] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[JobStatus], int]:
        """
        List jobs for a tenant with optional status filter.

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            Tuple of (list of JobStatus models, total count)
        """
        try:
            async with self.pool.acquire() as conn:
                # Get total count with the same WHERE clause
                if status:
                    count_row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) as total FROM ingestion_jobs
                        WHERE tenant_id = $1 AND status = $2
                        """,
                        tenant_id,
                        status.value,
                    )
                    rows = await conn.fetch(
                        """
                        SELECT * FROM ingestion_jobs
                        WHERE tenant_id = $1 AND status = $2
                        ORDER BY created_at DESC
                        LIMIT $3 OFFSET $4
                        """,
                        tenant_id,
                        status.value,
                        limit,
                        offset,
                    )
                else:
                    count_row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) as total FROM ingestion_jobs
                        WHERE tenant_id = $1
                        """,
                        tenant_id,
                    )
                    rows = await conn.fetch(
                        """
                        SELECT * FROM ingestion_jobs
                        WHERE tenant_id = $1
                        ORDER BY created_at DESC
                        LIMIT $2 OFFSET $3
                        """,
                        tenant_id,
                        limit,
                        offset,
                    )

                total = count_row["total"] if count_row else 0

                jobs = []
                for row in rows:
                    progress = None
                    if row["progress"]:
                        progress = JobProgress(**row["progress"])

                    jobs.append(JobStatus(
                        job_id=row["id"],
                        tenant_id=row["tenant_id"],
                        job_type=JobType(row["job_type"]),
                        status=JobStatusEnum(row["status"]),
                        progress=progress,
                        error_message=row["error_message"],
                        created_at=row["created_at"],
                        started_at=row["started_at"],
                        completed_at=row["completed_at"],
                    ))

                return jobs, total
        except asyncpg.PostgresError as e:
            raise DatabaseError("list_jobs", str(e)) from e

    # Story 4.3: Chunk storage methods

    async def create_chunk(
        self,
        tenant_id: UUID,
        document_id: UUID,
        content: str,
        chunk_index: int,
        token_count: int,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> UUID:
        """
        Create a new chunk with embedding.

        Args:
            tenant_id: Tenant identifier
            document_id: Parent document ID
            content: Chunk text content
            chunk_index: Position in document
            token_count: Number of tokens in chunk
            embedding: Optional 1536-dim embedding vector
            metadata: Optional additional metadata

        Returns:
            UUID of the created chunk

        Raises:
            DatabaseError: If creation fails
        """
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding list to string format for pgvector
                embedding_str = None
                if embedding is not None:
                    _validate_embedding(embedding)
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                row = await conn.fetchrow(
                    """
                    INSERT INTO chunks (tenant_id, document_id, content, chunk_index, token_count, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::vector, $7)
                    RETURNING id
                    """,
                    tenant_id,
                    document_id,
                    content,
                    chunk_index,
                    token_count,
                    embedding_str,
                    metadata,
                )
                chunk_id = row["id"]
                logger.debug(
                    "chunk_created",
                    chunk_id=str(chunk_id),
                    document_id=str(document_id),
                    chunk_index=chunk_index,
                )
                return chunk_id
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_chunk", str(e)) from e

    async def get_chunk(
        self,
        chunk_id: UUID,
        tenant_id: UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get a chunk by ID with tenant filtering.

        Args:
            chunk_id: Chunk UUID
            tenant_id: Tenant identifier for access control

        Returns:
            Chunk record as dictionary, or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT id, tenant_id, document_id, content, chunk_index, token_count, metadata, created_at
                    FROM chunks
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    chunk_id,
                    tenant_id,
                )
                return dict(row) if row else None
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_chunk", str(e)) from e

    async def get_chunks_by_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
    ) -> list[dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier for access control

        Returns:
            List of chunk records ordered by chunk_index
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, tenant_id, document_id, content, chunk_index, token_count, metadata, created_at
                    FROM chunks
                    WHERE document_id = $1 AND tenant_id = $2
                    ORDER BY chunk_index
                    """,
                    document_id,
                    tenant_id,
                )
                return [dict(row) for row in rows]
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_chunks_by_document", str(e)) from e

    async def search_similar_chunks(
        self,
        tenant_id: UUID,
        embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Returns:
            List of chunk records with similarity scores
        """
        try:
            _validate_embedding(embedding)
            async with self.pool.acquire() as conn:
                # Convert embedding to string format
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                rows = await conn.fetch(
                    """
                    SELECT
                        id, tenant_id, document_id, content, chunk_index, token_count, metadata, created_at,
                        1 - (embedding <=> $2::vector) as similarity
                    FROM chunks
                    WHERE tenant_id = $1
                        AND embedding IS NOT NULL
                        AND 1 - (embedding <=> $2::vector) >= $3
                    ORDER BY embedding <=> $2::vector
                    LIMIT $4
                    """,
                    tenant_id,
                    embedding_str,
                    similarity_threshold,
                    limit,
                )
                return [dict(row) for row in rows]
        except asyncpg.PostgresError as e:
            raise DatabaseError("search_similar_chunks", str(e)) from e

    async def delete_chunks_by_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
    ) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier

        Returns:
            Number of chunks deleted
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM chunks
                    WHERE document_id = $1 AND tenant_id = $2
                    """,
                    document_id,
                    tenant_id,
                )
                # Parse "DELETE N" result
                count = int(result.split(" ")[1]) if result else 0
                logger.info(
                    "chunks_deleted",
                    document_id=str(document_id),
                    count=count,
                )
                return count
        except asyncpg.PostgresError as e:
            raise DatabaseError("delete_chunks_by_document", str(e)) from e

    # Epic 20: Hierarchical chunk storage methods

    async def create_hierarchical_chunk(
        self,
        tenant_id: UUID,
        document_id: UUID,
        chunk_id: str,
        level: int,
        content: str,
        chunk_index: int,
        token_count: int,
        start_char: int,
        end_char: int,
        parent_id: Optional[str] = None,
        child_ids: Optional[list[str]] = None,
        embedding: Optional[list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Create or update a hierarchical chunk.

        Args:
            tenant_id: Tenant identifier
            document_id: Parent document ID
            chunk_id: Deterministic chunk identifier
            level: Hierarchy level (0 = smallest)
            content: Chunk text content
            chunk_index: Position within the level
            token_count: Number of tokens in chunk
            start_char: Starting character offset
            end_char: Ending character offset
            parent_id: Optional parent chunk ID
            child_ids: Optional list of child chunk IDs
            embedding: Optional embedding vector
            metadata: Optional metadata payload

        Returns:
            Chunk ID that was stored
        """
        try:
            async with self.pool.acquire() as conn:
                embedding_str = None
                if embedding is not None:
                    _validate_embedding(embedding)
                    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                await conn.execute(
                    """
                    INSERT INTO hierarchical_chunks (
                        id, tenant_id, document_id, level, parent_id, child_ids,
                        content, chunk_index, token_count, start_char, end_char,
                        embedding, metadata
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::vector, $13)
                    ON CONFLICT (id) DO UPDATE SET
                        tenant_id = EXCLUDED.tenant_id,
                        document_id = EXCLUDED.document_id,
                        level = EXCLUDED.level,
                        parent_id = EXCLUDED.parent_id,
                        child_ids = EXCLUDED.child_ids,
                        content = EXCLUDED.content,
                        chunk_index = EXCLUDED.chunk_index,
                        token_count = EXCLUDED.token_count,
                        start_char = EXCLUDED.start_char,
                        end_char = EXCLUDED.end_char,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    chunk_id,
                    tenant_id,
                    document_id,
                    level,
                    parent_id,
                    child_ids,
                    content,
                    chunk_index,
                    token_count,
                    start_char,
                    end_char,
                    embedding_str,
                    metadata,
                )
                logger.debug(
                    "hierarchical_chunk_created",
                    chunk_id=chunk_id,
                    document_id=str(document_id),
                    level=level,
                )
                return chunk_id
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_hierarchical_chunk", str(e)) from e

    async def get_hierarchical_chunk(
        self,
        chunk_id: str,
        tenant_id: UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get a hierarchical chunk by ID.

        Args:
            chunk_id: Chunk identifier
            tenant_id: Tenant identifier for access control

        Returns:
            Chunk record as dictionary, or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        id, tenant_id, document_id, level, parent_id, child_ids,
                        content, chunk_index, token_count, start_char, end_char,
                        metadata, created_at
                    FROM hierarchical_chunks
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    chunk_id,
                    tenant_id,
                )
                return dict(row) if row else None
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_hierarchical_chunk", str(e)) from e

    async def get_hierarchical_chunks_by_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
        level: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Get hierarchical chunks for a document.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier for access control
            level: Optional hierarchy level filter

        Returns:
            List of hierarchical chunk records ordered by level, chunk_index
        """
        try:
            async with self.pool.acquire() as conn:
                if level is None:
                    rows = await conn.fetch(
                        """
                        SELECT
                            id, tenant_id, document_id, level, parent_id, child_ids,
                            content, chunk_index, token_count, start_char, end_char,
                            metadata, created_at
                        FROM hierarchical_chunks
                        WHERE document_id = $1 AND tenant_id = $2
                        ORDER BY level, chunk_index
                        """,
                        document_id,
                        tenant_id,
                    )
                else:
                    rows = await conn.fetch(
                        """
                        SELECT
                            id, tenant_id, document_id, level, parent_id, child_ids,
                            content, chunk_index, token_count, start_char, end_char,
                            metadata, created_at
                        FROM hierarchical_chunks
                        WHERE document_id = $1 AND tenant_id = $2 AND level = $3
                        ORDER BY chunk_index
                        """,
                        document_id,
                        tenant_id,
                        level,
                    )
                return [dict(row) for row in rows]
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_hierarchical_chunks_by_document", str(e)) from e

    async def search_similar_hierarchical_chunks(
        self,
        tenant_id: UUID,
        embedding: list[float],
        level: int,
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Search for similar hierarchical chunks using cosine similarity.

        Args:
            tenant_id: Tenant identifier for filtering
            embedding: Query embedding vector (1536 dimensions)
            level: Hierarchy level to search
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of chunk records with similarity scores
        """
        try:
            _validate_embedding(embedding)
            async with self.pool.acquire() as conn:
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                rows = await conn.fetch(
                    """
                    SELECT
                        id, tenant_id, document_id, level, parent_id, child_ids,
                        content, chunk_index, token_count, start_char, end_char,
                        metadata, created_at,
                        1 - (embedding <=> $2::vector) as similarity
                    FROM hierarchical_chunks
                    WHERE tenant_id = $1
                        AND level = $3
                        AND embedding IS NOT NULL
                        AND 1 - (embedding <=> $2::vector) >= $4
                    ORDER BY embedding <=> $2::vector
                    LIMIT $5
                    """,
                    tenant_id,
                    embedding_str,
                    level,
                    similarity_threshold,
                    limit,
                )
                return [dict(row) for row in rows]
        except asyncpg.PostgresError as e:
            raise DatabaseError("search_similar_hierarchical_chunks", str(e)) from e

    async def delete_hierarchical_chunks_by_document(
        self,
        document_id: UUID,
        tenant_id: UUID,
    ) -> int:
        """
        Delete all hierarchical chunks for a document.

        Args:
            document_id: Document UUID
            tenant_id: Tenant identifier

        Returns:
            Number of chunks deleted
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    DELETE FROM hierarchical_chunks
                    WHERE document_id = $1 AND tenant_id = $2
                    """,
                    document_id,
                    tenant_id,
                )
                count = int(result.split(" ")[1]) if result else 0
                logger.info(
                    "hierarchical_chunks_deleted",
                    document_id=str(document_id),
                    count=count,
                )
                return count
        except asyncpg.PostgresError as e:
            raise DatabaseError("delete_hierarchical_chunks_by_document", str(e)) from e

    async def get_chunk_count(
        self,
        tenant_id: UUID,
        document_id: Optional[UUID] = None,
    ) -> int:
        """
        Get the count of chunks for a tenant or document.

        Args:
            tenant_id: Tenant identifier
            document_id: Optional document filter

        Returns:
            Number of chunks
        """
        try:
            async with self.pool.acquire() as conn:
                if document_id:
                    row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) as count FROM chunks
                        WHERE tenant_id = $1 AND document_id = $2
                        """,
                        tenant_id,
                        document_id,
                    )
                else:
                    row = await conn.fetchrow(
                        """
                        SELECT COUNT(*) as count FROM chunks
                        WHERE tenant_id = $1
                        """,
                        tenant_id,
                    )
                return row["count"] if row else 0
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_chunk_count", str(e)) from e

    async def create_workspace_item(
        self,
        workspace_id: UUID,
        tenant_id: UUID,
        content_id: str,
        content: str,
        title: Optional[str],
        query: Optional[str],
        sources: Optional[list[dict[str, Any]]],
        session_id: Optional[str],
        trajectory_id: Optional[str],
    ) -> datetime:
        """
        Create a workspace item for saved content.

        Args:
            workspace_id: Workspace item UUID
            tenant_id: Tenant identifier
            content_id: Original content identifier
            content: Saved content
            title: Optional title
            query: Optional query
            sources: Optional sources metadata
            session_id: Optional session ID
            trajectory_id: Optional trajectory ID

        Returns:
            created_at timestamp
        """
        try:
            _validate_workspace_content(content)
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO workspace_items (
                        id,
                        tenant_id,
                        content_id,
                        content,
                        title,
                        query,
                        sources,
                        session_id,
                        trajectory_id
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING created_at
                    """,
                    workspace_id,
                    tenant_id,
                    content_id,
                    content,
                    title,
                    query,
                    sources,
                    session_id,
                    trajectory_id,
                )
                return row["created_at"]
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_workspace_item", str(e)) from e

    async def get_workspace_item(
        self,
        tenant_id: UUID,
        workspace_id: UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get a workspace item by ID with tenant filtering.

        Args:
            tenant_id: Tenant identifier
            workspace_id: Workspace item UUID

        Returns:
            Workspace item record, or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        content_id,
                        content,
                        title,
                        query,
                        sources,
                        session_id,
                        trajectory_id,
                        created_at
                    FROM workspace_items
                    WHERE id = $1 AND tenant_id = $2
                    """,
                    workspace_id,
                    tenant_id,
                )
                return dict(row) if row else None
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_workspace_item", str(e)) from e

    async def create_workspace_share(
        self,
        share_id: UUID,
        tenant_id: UUID,
        content_id: str,
        content: str,
        title: Optional[str],
        query: Optional[str],
        sources: Optional[list[dict[str, Any]]],
        token: str,
        expires_at: Optional[datetime],
    ) -> datetime:
        """
        Create a shareable workspace item.

        Args:
            share_id: Share UUID
            tenant_id: Tenant identifier
            content_id: Original content identifier
            content: Shared content
            title: Optional title
            query: Optional query
            sources: Optional sources metadata
            token: Signed token
            expires_at: Expiration timestamp

        Returns:
            created_at timestamp
        """
        try:
            _validate_workspace_content(content)
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO workspace_shares (
                        id,
                        tenant_id,
                        content_id,
                        content,
                        title,
                        query,
                        sources,
                        token,
                        expires_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    RETURNING created_at
                    """,
                    share_id,
                    tenant_id,
                    content_id,
                    content,
                    title,
                    query,
                    sources,
                    token,
                    expires_at,
                )
                return row["created_at"]
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_workspace_share", str(e)) from e

    async def get_workspace_share(
        self,
        share_id: UUID,
    ) -> Optional[dict[str, Any]]:
        """
        Get a shared workspace item by share ID.

        Args:
            share_id: Share UUID

        Returns:
            Share record, or None if not found
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT
                        id,
                        tenant_id,
                        content_id,
                        content,
                        title,
                        query,
                        sources,
                        token,
                        created_at,
                        expires_at
                    FROM workspace_shares
                    WHERE id = $1
                    """,
                    share_id,
                )
                return dict(row) if row else None
        except asyncpg.PostgresError as e:
            raise DatabaseError("get_workspace_share", str(e)) from e

    async def create_workspace_bookmark(
        self,
        bookmark_id: UUID,
        tenant_id: UUID,
        content_id: str,
        content: str,
        title: Optional[str],
        query: Optional[str],
        session_id: Optional[str],
    ) -> datetime:
        """
        Create a workspace bookmark.

        Args:
            bookmark_id: Bookmark UUID
            tenant_id: Tenant identifier
            content_id: Original content identifier
            content: Bookmarked content
            title: Optional title
            query: Optional query
            session_id: Optional session ID

        Returns:
            created_at timestamp
        """
        try:
            _validate_workspace_content(content)
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO workspace_bookmarks (
                        id,
                        tenant_id,
                        content_id,
                        content,
                        title,
                        query,
                        session_id
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING created_at
                    """,
                    bookmark_id,
                    tenant_id,
                    content_id,
                    content,
                    title,
                    query,
                    session_id,
                )
                return row["created_at"]
        except asyncpg.PostgresError as e:
            raise DatabaseError("create_workspace_bookmark", str(e)) from e

    async def list_workspace_bookmarks(
        self,
        tenant_id: UUID,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        List workspace bookmarks for a tenant.

        Args:
            tenant_id: Tenant identifier
            limit: Maximum number of bookmarks to return
            offset: Pagination offset

        Returns:
            Tuple of bookmark rows and total count
        """
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        content_id,
                        title,
                        query,
                        created_at
                    FROM workspace_bookmarks
                    WHERE tenant_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    tenant_id,
                    limit,
                    offset,
                )
                count_row = await conn.fetchrow(
                    """
                    SELECT COUNT(*) AS count
                    FROM workspace_bookmarks
                    WHERE tenant_id = $1
                    """,
                    tenant_id,
                )
                total = count_row["count"] if count_row else 0
                return [dict(row) for row in rows], total
        except asyncpg.PostgresError as e:
            raise DatabaseError("list_workspace_bookmarks", str(e)) from e


# Global PostgreSQL client instance
_postgres_client: Optional[PostgresClient] = None


async def get_postgres_client(url: Optional[str] = None) -> PostgresClient:
    """
    Get or create the global PostgreSQL client instance.

    Args:
        url: PostgreSQL connection URL. Required on first call.

    Returns:
        PostgresClient instance
    """
    global _postgres_client
    if _postgres_client is None:
        if url is None:
            raise DatabaseError("init", "Database URL required for first initialization")
        _postgres_client = PostgresClient(url)
        await _postgres_client.connect()
        await _postgres_client.create_tables()
    return _postgres_client


async def close_postgres_client() -> None:
    """Close the global PostgreSQL client connection."""
    global _postgres_client
    if _postgres_client is not None:
        await _postgres_client.disconnect()
        _postgres_client = None
