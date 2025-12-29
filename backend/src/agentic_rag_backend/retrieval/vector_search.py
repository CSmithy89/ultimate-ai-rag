from __future__ import annotations

from typing import Optional
from uuid import UUID

import structlog

from agentic_rag_backend.core.errors import AppError
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.indexing.embeddings import EmbeddingGenerator

from .types import VectorHit

logger = structlog.get_logger(__name__)

DEFAULT_VECTOR_LIMIT = 8  # Tuned for concise evidence payloads.
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Balanced precision/recall for MVP.


class VectorSearchService:
    """Semantic vector search over pgvector chunks."""

    def __init__(
        self,
        postgres: Optional[PostgresClient],
        embedding_generator: Optional[EmbeddingGenerator],
        limit: int = DEFAULT_VECTOR_LIMIT,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self.postgres = postgres
        self.embedding_generator = embedding_generator
        self.limit = limit
        self.similarity_threshold = similarity_threshold

    async def search(self, query: str, tenant_id: str) -> list[VectorHit]:
        """Search for similar chunks and return vector hits."""
        if not self.postgres or not self.embedding_generator:
            logger.warning("vector_search_unavailable")
            return []

        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            logger.warning("vector_search_invalid_tenant_id", tenant_id=tenant_id)
            return []

        try:
            embedding = await self.embedding_generator.generate_embedding(query)
            rows = await self.postgres.search_similar_chunks(
                tenant_id=tenant_uuid,
                embedding=embedding,
                limit=self.limit,
                similarity_threshold=self.similarity_threshold,
            )
        except AppError as exc:
            logger.error("vector_search_failed", error=str(exc))
            raise
        except Exception as exc:
            logger.error("vector_search_unexpected_error", error=str(exc))
            raise

        hits: list[VectorHit] = []
        for row in rows:
            hits.append(
                VectorHit(
                    chunk_id=str(row.get("id")),
                    document_id=str(row.get("document_id")),
                    content=row.get("content", ""),
                    similarity=float(row.get("similarity", 0.0)),
                    metadata=row.get("metadata"),
                )
            )

        logger.info(
            "vector_search_completed",
            tenant_id=tenant_id,
            hits=len(hits),
        )
        return hits
