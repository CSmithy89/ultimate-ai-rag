"""Async index worker for document indexing pipeline.

This worker consumes jobs from the Redis Streams 'index.jobs' queue,
stores chunk embeddings in Postgres, and ingests documents into Graphiti
for knowledge graph extraction.
"""

import asyncio
import hashlib
import os
import time
from typing import Optional
from uuid import UUID

import structlog

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.core.errors import AppError, ExtractionError, IngestionError
from agentic_rag_backend.db.graphiti import (
    GRAPHITI_AVAILABLE,
    GraphitiClient,
    create_graphiti_client,
)
from agentic_rag_backend.db.postgres import PostgresClient, get_postgres_client
from agentic_rag_backend.db.redis import (
    INDEX_CONSUMER_GROUP,
    INDEX_JOBS_STREAM,
    get_redis_client,
)
from agentic_rag_backend.embeddings import EmbeddingGenerator
from agentic_rag_backend.llm.providers import get_embedding_adapter
from agentic_rag_backend.indexing.chunker import chunk_document
from agentic_rag_backend.indexing.graphiti_ingestion import ingest_document_as_episode
from agentic_rag_backend.llm import UnsupportedLLMProviderError, get_llm_adapter
from agentic_rag_backend.models.documents import (
    SourceType,
    UnifiedDocument,
    parse_document_metadata,
)
from agentic_rag_backend.models.ingest import JobStatusEnum
from agentic_rag_backend.ops import CostTracker

logger = structlog.get_logger(__name__)


async def process_index_job(
    job_data: dict,
    postgres: PostgresClient,
    embedding_generator: EmbeddingGenerator,
    graphiti_client: Optional[GraphitiClient],
    skip_graphiti_ingestion: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """
    Process a single indexing job.

    This function:
    1. Updates job status to running
    2. Chunks the document and stores embeddings in Postgres
    3. Ingests the document as a Graphiti episode
    4. Updates job status and progress

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
        embedding_generator: Embedding generator for chunk storage
        graphiti_client: Connected Graphiti client for graph ingestion
        skip_graphiti_ingestion: Skip Graphiti ingestion when disabled
        chunk_size: Target chunk size in tokens
        chunk_overlap: Token overlap between chunks

    Raises:
        ExtractionError: If indexing fails
        AppError: For other application errors
    """
    job_id = UUID(job_data["job_id"])
    tenant_id = UUID(job_data["tenant_id"])
    document_id = UUID(job_data["document_id"])
    content = job_data.get("content", "")
    content_hash = job_data.get("content_hash") or ""
    source_type = job_data.get("source_type", "text")
    filename = job_data.get("filename")

    if not content_hash:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    try:
        source_type_enum = SourceType(source_type)
    except ValueError:
        source_type_enum = SourceType.TEXT

    extra_fields = {
        "content_hash": content_hash,
        "source_type": source_type,
    }
    if filename:
        extra_fields["filename"] = filename

    metadata_model = parse_document_metadata(
        job_data.get("metadata", {}),
        extra_fields=extra_fields,
        log=logger,
        log_context={
            "job_id": str(job_id),
            "document_id": str(document_id),
        },
    )

    document = UnifiedDocument(
        id=document_id,
        tenant_id=tenant_id,
        source_type=source_type_enum,
        source_url=metadata_model.source_url,
        filename=filename,
        content=content,
        content_hash=content_hash,
        metadata=metadata_model,
    )

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

        chunks = chunk_document(
            content=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            await postgres.update_job_processing_time(
                job_id=job_id,
                tenant_id=tenant_id,
                processing_time_ms=processing_time_ms,
            )
            await postgres.update_document_status(
                document_id=document_id,
                tenant_id=tenant_id,
                status="completed",
            )
            await postgres.update_job_status(
                job_id=job_id,
                tenant_id=tenant_id,
                status=JobStatusEnum.COMPLETED,
                progress={
                    "chunks_processed": 0,
                    "total_chunks": 0,
                    "entities_extracted": 0,
                    "relationships_extracted": 0,
                    "processing_time_ms": processing_time_ms,
                },
            )
            logger.info(
                "index_job_completed_empty",
                job_id=str(job_id),
                document_id=str(document_id),
                processing_time_ms=processing_time_ms,
            )
            return

        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.RUNNING,
            progress={
                "chunks_processed": 0,
                "total_chunks": len(chunks),
                "entities_extracted": 0,
                "relationships_extracted": 0,
            },
        )

        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await embedding_generator.generate_embeddings(
            chunk_texts,
            tenant_id=str(tenant_id),
        )

        for chunk, embedding in zip(chunks, embeddings):
            await postgres.create_chunk(
                tenant_id=tenant_id,
                document_id=document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                embedding=embedding,
                metadata={
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                },
            )

        ingestion_result = None
        if graphiti_client is None or not graphiti_client.is_connected:
            if skip_graphiti_ingestion:
                logger.warning(
                    "graphiti_ingestion_skipped",
                    document_id=str(document_id),
                    tenant_id=str(tenant_id),
                )
            else:
                raise IngestionError(str(document_id), "Graphiti client not available")
        else:
            ingestion_result = await ingest_document_as_episode(
                graphiti_client=graphiti_client,
                document=document,
            )

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
        entities_extracted = 0
        relationships_extracted = 0
        if ingestion_result is not None:
            entities_extracted = ingestion_result.entities_extracted
            relationships_extracted = ingestion_result.edges_created

        await postgres.update_job_status(
            job_id=job_id,
            tenant_id=tenant_id,
            status=JobStatusEnum.COMPLETED,
            progress={
                "chunks_processed": len(chunks),
                "total_chunks": len(chunks),
                "entities_extracted": entities_extracted,
                "relationships_extracted": relationships_extracted,
                "processing_time_ms": processing_time_ms,
            },
        )

        logger.info(
            "index_job_completed",
            job_id=str(job_id),
            document_id=str(document_id),
            chunks_created=len(chunks),
            entities_extracted=entities_extracted,
            relationships_extracted=relationships_extracted,
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
        raise ExtractionError(str(document_id), str(e)) from e


async def run_index_worker(
    consumer_name: str = "index-worker-1",
    batch_size: int = 1,
) -> None:
    """
    Run the index worker as a long-running async task.

    This function:
    1. Connects to Redis and PostgreSQL
    2. Initializes embedding and Graphiti clients
    3. Continuously consumes jobs from index.jobs stream
    4. Processes each job through chunking + Graphiti ingestion
    5. Acknowledges jobs after processing

    Args:
        consumer_name: Unique identifier for this worker instance
        batch_size: Number of jobs to fetch at once
    """
    settings = get_settings()
    try:
        llm_adapter = get_llm_adapter(settings)
    except UnsupportedLLMProviderError as exc:
        logger.error(
            "llm_provider_unsupported",
            provider=settings.llm_provider,
            error=str(exc),
        )
        raise

    logger.info(
        "index_worker_starting",
        consumer_name=consumer_name,
        stream=INDEX_JOBS_STREAM,
        group=INDEX_CONSUMER_GROUP,
    )

    # Get database clients
    redis = await get_redis_client(settings.redis_url)
    postgres = await get_postgres_client(settings.database_url)
    cost_tracker = CostTracker(
        postgres.pool,
        pricing_json=settings.model_pricing_json,
    )

    # Get embedding adapter for multi-provider support
    embedding_adapter = get_embedding_adapter(settings)

    skip_graphiti_ingestion = os.getenv("SKIP_GRAPHITI") == "1"
    graphiti_client: Optional[GraphitiClient] = None
    if skip_graphiti_ingestion:
        logger.warning("graphiti_ingestion_disabled")
    elif GRAPHITI_AVAILABLE:
        try:
            graphiti_client = await create_graphiti_client(
                uri=settings.neo4j_uri,
                user=settings.neo4j_user,
                password=settings.neo4j_password,
                llm_provider=llm_adapter.provider,
                llm_api_key=llm_adapter.api_key,
                llm_base_url=llm_adapter.base_url,
                embedding_provider=settings.embedding_provider,
                embedding_api_key=settings.embedding_api_key,
                embedding_base_url=settings.embedding_base_url,
                embedding_model=settings.graphiti_embedding_model,
                llm_model=settings.graphiti_llm_model,
            )
            logger.info("graphiti_worker_connected")
        except Exception as e:
            logger.error("graphiti_worker_connect_failed", error=str(e))
            raise RuntimeError("Graphiti unavailable for index worker") from e
    else:
        raise RuntimeError("graphiti-core is not installed for index worker")

    # Create embedding generator using the multi-provider adapter
    embedding_generator = EmbeddingGenerator.from_adapter(
        adapter=embedding_adapter,
        cost_tracker=cost_tracker,
    )

    logger.info(
        "index_worker_initialized",
        consumer_name=consumer_name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
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
                embedding_generator=embedding_generator,
                graphiti_client=graphiti_client,
                skip_graphiti_ingestion=skip_graphiti_ingestion,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
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
