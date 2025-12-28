"""PostgreSQL async client for documents and jobs tables."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

import asyncpg
import structlog

from agentic_rag_backend.core.errors import DatabaseError
from agentic_rag_backend.models.ingest import JobProgress, JobStatus, JobStatusEnum, JobType

logger = structlog.get_logger(__name__)


class PostgresClient:
    """
    Async PostgreSQL client for managing documents and ingestion jobs.

    Implements multi-tenancy through tenant_id filtering on all queries.
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

        This creates the documents and ingestion_jobs tables with proper
        indexes for multi-tenancy and status filtering.
        """
        async with self.pool.acquire() as conn:
            # Create documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    source_type VARCHAR(20) NOT NULL,
                    source_url TEXT,
                    filename TEXT,
                    content_hash VARCHAR(64) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
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

            # Create ingestion_jobs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id UUID NOT NULL,
                    document_id UUID REFERENCES documents(id),
                    job_type VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'queued',
                    progress JSONB,
                    error_message TEXT,
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

            logger.info("tables_created")

    async def create_document(
        self,
        tenant_id: UUID,
        source_type: str,
        content_hash: str,
        source_url: Optional[str] = None,
        filename: Optional[str] = None,
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
                    INSERT INTO documents (tenant_id, source_type, source_url, filename, content_hash, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, content_hash) DO UPDATE
                    SET updated_at = NOW()
                    RETURNING id
                    """,
                    tenant_id,
                    source_type,
                    source_url,
                    filename,
                    content_hash,
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

    async def list_jobs(
        self,
        tenant_id: UUID,
        status: Optional[JobStatusEnum] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobStatus]:
        """
        List jobs for a tenant with optional status filter.

        Args:
            tenant_id: Tenant identifier
            status: Optional status filter
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip

        Returns:
            List of JobStatus models
        """
        try:
            async with self.pool.acquire() as conn:
                if status:
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

                return jobs
        except asyncpg.PostgresError as e:
            raise DatabaseError("list_jobs", str(e)) from e


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
