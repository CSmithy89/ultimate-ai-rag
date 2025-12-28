"""Async index worker for document indexing pipeline.

This worker consumes jobs from the Redis Streams 'index.jobs' queue,
processes documents through the IndexerAgent, and stores results
in pgvector (chunks) and Neo4j (entities/relationships).
"""

import asyncio
import json
import time
from typing import Optional
from uuid import UUID

import structlog

from agentic_rag_backend.agents.indexer import IndexerAgent
from agentic_rag_backend.config import load_settings
from agentic_rag_backend.core.errors import AppError, ExtractionError
from agentic_rag_backend.db.neo4j import Neo4jClient, get_neo4j_client
from agentic_rag_backend.db.postgres import PostgresClient, get_postgres_client
from agentic_rag_backend.db.redis import (
    INDEX_CONSUMER_GROUP,
    INDEX_JOBS_STREAM,
    RedisClient,
    get_redis_client,
)
from agentic_rag_backend.indexing.embeddings import EmbeddingGenerator
from agentic_rag_backend.indexing.entity_extractor import EntityExtractor
from agentic_rag_backend.models.ingest import JobStatusEnum

logger = structlog.get_logger(__name__)


async def process_index_job(
    job_data: dict,
    postgres: PostgresClient,
    neo4j: Neo4jClient,
    indexer_agent: IndexerAgent,
) -> None:
    """
    Process a single indexing job.

    This function:
    1. Updates job status to running
    2. Runs the IndexerAgent to chunk, embed, extract entities
    3. Updates job status and progress
    4. Logs trajectory for debugging

    Args:
        job_data: Job data from Redis Stream containing:
            - job_id: UUID of the job
            - tenant_id: Tenant identifier
            - document_id: Document UUID
            - content: Document text content
            - content_hash: Content hash for deduplication
            - metadata: Document metadata
            - source_type: Source type (url, pdf, text)
            - filename: Original filename (optional)
        postgres: PostgreSQL client for status updates
        neo4j: Neo4j client for graph operations
        indexer_agent: IndexerAgent instance

    Raises:
        ExtractionError: If indexing fails
        AppError: For other application errors
    """
    job_id = UUID(job_data["job_id"])
    tenant_id = UUID(job_data["tenant_id"])
    document_id = UUID(job_data["document_id"])
    content = job_data.get("content", "")
    content_hash = job_data.get("content_hash", "")
    source_type = job_data.get("source_type", "text")
    filename = job_data.get("filename")

    # Parse metadata if it's a string
    metadata = job_data.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    logger.info(
        "index_job_started",
        job_id=str(job_id),
        tenant_id=str(tenant_id),
        document_id=str(document_id),
        content_length=len(content),
    )

    start_time = time.perf_counter()

    try:
        # Update job status to running
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.RUNNING,
            progress={
                "chunks_processed": 0,
                "total_chunks": 0,
                "entities_extracted": 0,
                "relationships_extracted": 0,
            },
        )

        # Validate content
        if not content or not content.strip():
            raise ExtractionError(str(document_id), "Document content is empty")

        # Add source info to metadata
        enriched_metadata = {
            **metadata,
            "source_type": source_type,
            "content_hash": content_hash,
        }
        if filename:
            enriched_metadata["filename"] = filename

        # Run the indexer agent
        result = await indexer_agent.index_document(
            document_id=document_id,
            tenant_id=tenant_id,
            content=content,
            metadata=enriched_metadata,
        )

        # Calculate total processing time
        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Update job processing time
        await postgres.update_job_processing_time(
            job_id=job_id,
            tenant_id=tenant_id,
            processing_time_ms=processing_time_ms,
        )

        # Update document status
        await postgres.update_document_status(
            document_id=document_id,
            tenant_id=tenant_id,
            status="completed",
        )

        # Update job status to completed
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.COMPLETED,
            progress={
                "chunks_processed": result.chunks_created,
                "total_chunks": result.chunks_created,
                "entities_extracted": result.entities_extracted,
                "relationships_extracted": result.relationships_extracted,
                "entities_deduplicated": result.entities_deduplicated,
                "processing_time_ms": processing_time_ms,
            },
        )

        # Log trajectory info if available
        trajectory = indexer_agent.get_trajectory()
        if trajectory:
            logger.info(
                "index_job_trajectory",
                job_id=str(job_id),
                trajectory_id=trajectory.run_id,
                entries_count=len(trajectory.entries),
            )

        logger.info(
            "index_job_completed",
            job_id=str(job_id),
            document_id=str(document_id),
            chunks_created=result.chunks_created,
            entities_extracted=result.entities_extracted,
            relationships_extracted=result.relationships_extracted,
            processing_time_ms=processing_time_ms,
        )

    except AppError as e:
        # Update document status to failed
        await postgres.update_document_status(
            document_id=document_id,
            tenant_id=tenant_id,
            status="failed",
        )

        # Update job status to failed
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.FAILED,
            error_message=str(e),
        )
        logger.error(
            "index_job_failed",
            job_id=str(job_id),
            error=str(e),
            error_code=e.code.value if hasattr(e, "code") else None,
        )
        raise

    except Exception as e:
        # Update document status to failed
        await postgres.update_document_status(
            document_id=document_id,
            tenant_id=tenant_id,
            status="failed",
        )

        # Update job status to failed for unexpected errors
        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.FAILED,
            error_message=f"Unexpected error: {str(e)}",
        )
        logger.error(
            "index_job_failed_unexpected",
            job_id=str(job_id),
            error=str(e),
        )
        raise ExtractionError(str(document_id), str(e))


async def run_index_worker(
    consumer_name: str = "index-worker-1",
    batch_size: int = 1,
) -> None:
    """
    Run the index worker as a long-running async task.

    This function:
    1. Connects to Redis, PostgreSQL, and Neo4j
    2. Initializes the IndexerAgent with all dependencies
    3. Continuously consumes jobs from index.jobs stream
    4. Processes each job through the IndexerAgent
    5. Acknowledges jobs after processing

    Args:
        consumer_name: Unique identifier for this worker instance
        batch_size: Number of jobs to fetch at once
    """
    settings = load_settings()

    logger.info(
        "index_worker_starting",
        consumer_name=consumer_name,
        stream=INDEX_JOBS_STREAM,
        group=INDEX_CONSUMER_GROUP,
    )

    # Get database clients
    redis = await get_redis_client(settings.redis_url)
    postgres = await get_postgres_client(settings.database_url)
    neo4j = await get_neo4j_client(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    # Initialize components for IndexerAgent
    embedding_generator = EmbeddingGenerator(
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )

    entity_extractor = EntityExtractor(
        api_key=settings.openai_api_key,
        model=settings.entity_extraction_model,
    )

    # Create IndexerAgent
    indexer_agent = IndexerAgent(
        postgres=postgres,
        neo4j=neo4j,
        embedding_generator=embedding_generator,
        entity_extractor=entity_extractor,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        similarity_threshold=settings.entity_similarity_threshold,
    )

    logger.info(
        "index_worker_initialized",
        consumer_name=consumer_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        entity_model=settings.entity_extraction_model,
        embedding_model=settings.embedding_model,
    )

    # Consume jobs from stream
    async for job_data in redis.consume_jobs(
        stream=INDEX_JOBS_STREAM,
        group=INDEX_CONSUMER_GROUP,
        consumer=consumer_name,
        count=batch_size,
        block_ms=5000,  # 5 second timeout
    ):
        try:
            await process_index_job(
                job_data=job_data,
                postgres=postgres,
                neo4j=neo4j,
                indexer_agent=indexer_agent,
            )
        except Exception as e:
            # Log but don't crash the worker
            logger.error(
                "index_worker_job_error",
                job_id=job_data.get("job_id"),
                error=str(e),
            )
            # Continue processing other jobs


async def main() -> None:
    """Entry point for running the index worker as a standalone process."""
    import signal
    import sys

    # Handle graceful shutdown
    def signal_handler(sig: int, frame: Optional[object]) -> None:
        logger.info("index_worker_shutdown_requested")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await run_index_worker()
    except Exception as e:
        logger.error("index_worker_crashed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
