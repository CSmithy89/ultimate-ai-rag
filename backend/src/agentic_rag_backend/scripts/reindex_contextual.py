"""Reindex existing chunks with contextual enrichment.

This script re-embeds existing content in the database with contextual
retrieval enabled, improving retrieval accuracy by 35-67%.

Usage:
    uv run python -m agentic_rag_backend.scripts.reindex_contextual \\
        --batch-size 100 \\
        --tenant-id your-tenant
"""

import argparse
import asyncio
import sys
import time
from typing import Optional
from uuid import UUID

import structlog

from agentic_rag_backend.config import get_settings
from agentic_rag_backend.db.postgres import PostgresClient, get_postgres_client
from agentic_rag_backend.embeddings import EmbeddingGenerator
from agentic_rag_backend.indexing.chunker import ChunkData
from agentic_rag_backend.indexing.contextual import (
    ContextualChunkEnricher,
    DocumentContext,
    create_contextual_enricher,
)
from agentic_rag_backend.llm.providers import get_embedding_adapter
from agentic_rag_backend.ops import CostTracker

logger = structlog.get_logger(__name__)


async def get_documents_for_reindexing(
    postgres: PostgresClient,
    tenant_id: UUID,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Get documents that need reindexing.

    Args:
        postgres: PostgreSQL client
        tenant_id: Tenant identifier
        limit: Maximum number of documents to return
        offset: Number of documents to skip

    Returns:
        List of document records
    """
    async with postgres.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, content, metadata, filename
                FROM documents
                WHERE tenant_id = %s
                ORDER BY created_at
                LIMIT %s OFFSET %s
                """,
                (str(tenant_id), limit, offset),
            )
            rows = await cur.fetchall()
            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "metadata": row[2] or {},
                    "filename": row[3],
                }
                for row in rows
            ]


async def get_chunks_for_document(
    postgres: PostgresClient,
    document_id: UUID,
    tenant_id: UUID,
) -> list[dict]:
    """Get all chunks for a document.

    Args:
        postgres: PostgreSQL client
        document_id: Document identifier
        tenant_id: Tenant identifier

    Returns:
        List of chunk records
    """
    async with postgres.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, content, chunk_index, token_count, metadata
                FROM chunks
                WHERE document_id = %s AND tenant_id = %s
                ORDER BY chunk_index
                """,
                (str(document_id), str(tenant_id)),
            )
            rows = await cur.fetchall()
            return [
                {
                    "id": row[0],
                    "content": row[1],
                    "chunk_index": row[2],
                    "token_count": row[3],
                    "metadata": row[4] or {},
                }
                for row in rows
            ]


async def update_chunks_in_transaction(
    postgres: PostgresClient,
    updates: list[tuple[UUID, UUID, list[float], dict]],
) -> None:
    """Update multiple chunks atomically in a single transaction.

    This ensures all chunks for a document are updated together,
    preventing inconsistent state if the script crashes mid-document.

    Args:
        postgres: PostgreSQL client
        updates: List of (chunk_id, tenant_id, embedding, metadata) tuples
    """
    if not updates:
        return

    async with postgres.pool.connection() as conn:
        # Use a transaction to ensure atomicity
        async with conn.transaction():
            async with conn.cursor() as cur:
                for chunk_id, tenant_id, embedding, metadata in updates:
                    await cur.execute(
                        """
                        UPDATE chunks
                        SET embedding = %s, metadata = %s
                        WHERE id = %s AND tenant_id = %s
                        """,
                        (embedding, metadata, str(chunk_id), str(tenant_id)),
                    )


async def reindex_document(
    document: dict,
    tenant_id: UUID,
    postgres: PostgresClient,
    embedding_generator: EmbeddingGenerator,
    contextual_enricher: ContextualChunkEnricher,
) -> dict:
    """Reindex a single document with contextual enrichment.

    Args:
        document: Document record
        tenant_id: Tenant identifier
        postgres: PostgreSQL client
        embedding_generator: Embedding generator
        contextual_enricher: Contextual chunk enricher

    Returns:
        Summary of the reindexing operation
    """
    document_id = UUID(str(document["id"]))
    metadata = document.get("metadata", {})
    filename = document.get("filename")

    # Get existing chunks
    chunks_data = await get_chunks_for_document(postgres, document_id, tenant_id)
    if not chunks_data:
        return {
            "document_id": str(document_id),
            "status": "skipped",
            "reason": "no_chunks",
            "chunks_processed": 0,
        }

    # Convert to ChunkData objects
    chunks = [
        ChunkData(
            content=c["content"],
            chunk_index=c["chunk_index"],
            token_count=c["token_count"],
            start_char=c["metadata"].get("start_char", 0),
            end_char=c["metadata"].get("end_char", len(c["content"])),
        )
        for c in chunks_data
    ]

    # Create document context
    doc_context = DocumentContext(
        title=metadata.get("title") or filename,
        summary=metadata.get("description"),
        full_content=document.get("content", ""),
    )

    # Enrich chunks
    start_time = time.perf_counter()
    enriched_chunks = await contextual_enricher.enrich_chunks(chunks, doc_context)
    enrichment_time_ms = (time.perf_counter() - start_time) * 1000

    # Generate new embeddings from enriched content
    enriched_texts = [ec.enriched_content for ec in enriched_chunks]
    embeddings = await embedding_generator.generate_embeddings(
        enriched_texts,
        tenant_id=str(tenant_id),
    )

    # Validate list lengths before zip to prevent silent truncation
    if not (len(chunks_data) == len(enriched_chunks) == len(embeddings)):
        logger.error(
            "list_length_mismatch",
            document_id=str(document_id),
            chunks_data_len=len(chunks_data),
            enriched_chunks_len=len(enriched_chunks),
            embeddings_len=len(embeddings),
        )
        return {
            "document_id": str(document_id),
            "status": "error",
            "reason": "list_length_mismatch",
            "chunks_processed": 0,
        }

    # Build update batch for transactional update
    reindex_timestamp = time.time()
    updates = []

    for chunk_record, enriched, embedding in zip(chunks_data, enriched_chunks, embeddings):
        updated_metadata = chunk_record.get("metadata", {}).copy()
        updated_metadata["contextual_enrichment"] = True
        updated_metadata["context"] = enriched.context
        updated_metadata["context_generation_ms"] = enriched.context_generation_ms
        updated_metadata["reindexed_at"] = reindex_timestamp

        updates.append((
            UUID(str(chunk_record["id"])),
            tenant_id,
            embedding,
            updated_metadata,
        ))

    # Execute all DB updates atomically in a single transaction
    await update_chunks_in_transaction(postgres, updates)

    return {
        "document_id": str(document_id),
        "status": "success",
        "chunks_processed": len(chunks),
        "enrichment_time_ms": round(enrichment_time_ms, 2),
    }


async def reindex_all(
    tenant_id: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> None:
    """Reindex all documents for a tenant with contextual enrichment.

    Args:
        tenant_id: Tenant identifier
        batch_size: Number of documents to process per batch
        dry_run: If True, only report what would be done
    """
    settings = get_settings()

    if not settings.contextual_retrieval_enabled:
        logger.error(
            "contextual_retrieval_disabled",
            message="Set CONTEXTUAL_RETRIEVAL_ENABLED=true to enable",
        )
        sys.exit(1)

    tenant_uuid = UUID(tenant_id)

    logger.info(
        "reindex_starting",
        tenant_id=tenant_id,
        batch_size=batch_size,
        dry_run=dry_run,
        contextual_model=settings.contextual_model,
    )

    # Initialize clients
    postgres = await get_postgres_client(settings.database_url)
    embedding_adapter = get_embedding_adapter(settings)

    cost_tracker = CostTracker(
        postgres.pool,
        pricing_json=settings.model_pricing_json,
    )

    embedding_generator = EmbeddingGenerator.from_adapter(
        adapter=embedding_adapter,
        cost_tracker=cost_tracker,
    )

    contextual_enricher = create_contextual_enricher(settings)
    if not contextual_enricher:
        logger.error("contextual_enricher_creation_failed")
        sys.exit(1)

    # Process documents in batches
    offset = 0
    total_documents = 0
    total_chunks = 0
    start_time = time.perf_counter()

    while True:
        documents = await get_documents_for_reindexing(
            postgres, tenant_uuid, batch_size, offset
        )

        if not documents:
            break

        logger.info(
            "batch_starting",
            batch_offset=offset,
            batch_size=len(documents),
        )

        for doc in documents:
            if dry_run:
                chunks = await get_chunks_for_document(
                    postgres, UUID(str(doc["id"])), tenant_uuid
                )
                logger.info(
                    "dry_run_document",
                    document_id=str(doc["id"]),
                    chunks=len(chunks),
                )
                total_chunks += len(chunks)
            else:
                result = await reindex_document(
                    doc,
                    tenant_uuid,
                    postgres,
                    embedding_generator,
                    contextual_enricher,
                )
                total_chunks += result.get("chunks_processed", 0)
                logger.info("document_reindexed", **result)

            total_documents += 1

        offset += batch_size

    elapsed_seconds = time.perf_counter() - start_time

    logger.info(
        "reindex_complete",
        total_documents=total_documents,
        total_chunks=total_chunks,
        elapsed_seconds=round(elapsed_seconds, 2),
        dry_run=dry_run,
    )

    # Cleanup database connection
    await postgres.close()


def main() -> None:
    """Entry point for the reindexing script."""
    parser = argparse.ArgumentParser(
        description="Reindex existing chunks with contextual enrichment"
    )
    parser.add_argument(
        "--tenant-id",
        required=True,
        help="Tenant identifier",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of documents to process per batch (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done, don't make changes",
    )

    args = parser.parse_args()

    asyncio.run(reindex_all(
        tenant_id=args.tenant_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
