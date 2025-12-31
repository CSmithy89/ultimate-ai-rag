from __future__ import annotations

import asyncio
from typing import Optional
from uuid import UUID

import structlog

from agentic_rag_backend.core.errors import AppError
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.embeddings import EmbeddingGenerator

from .cache import TTLCache, hash_cache_key
from .constants import (
    DEFAULT_RETRIEVAL_CACHE_SIZE,
    DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS,
    DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_LIMIT,
)
from .types import VectorHit

logger = structlog.get_logger(__name__)


class VectorSearchService:
    """Semantic vector search over pgvector chunks."""

    def __init__(
        self,
        postgres: Optional[PostgresClient],
        embedding_generator: Optional[EmbeddingGenerator],
        limit: int = DEFAULT_VECTOR_LIMIT,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        timeout_seconds: float = DEFAULT_RETRIEVAL_TIMEOUT_SECONDS,
        cache_ttl_seconds: float = DEFAULT_RETRIEVAL_CACHE_TTL_SECONDS,
        cache_size: int = DEFAULT_RETRIEVAL_CACHE_SIZE,
    ) -> None:
        self.postgres = postgres
        self.embedding_generator = embedding_generator
        self.limit = limit
        self.similarity_threshold = similarity_threshold
        self.timeout_seconds = timeout_seconds
        self._cache = (
            TTLCache[list[VectorHit]](max_size=cache_size, ttl_seconds=cache_ttl_seconds)
            if cache_ttl_seconds > 0
            else None
        )

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

        query_hash = hash_cache_key(query)
        cache_key = (tenant_id, query_hash, self.limit, self.similarity_threshold)
        if self._cache:
            cached_hits = self._cache.get(cache_key)
            if cached_hits is not None:
                logger.debug("vector_search_cache_hit", tenant_id=tenant_id)
                return cached_hits

        try:
            embedding = await self._await_with_timeout(
                self.embedding_generator.generate_embedding(query),
                "vector_search_embedding_timeout",
            )
            rows = await self._await_with_timeout(
                self.postgres.search_similar_chunks(
                    tenant_id=tenant_uuid,
                    embedding=embedding,
                    limit=self.limit,
                    similarity_threshold=self.similarity_threshold,
                ),
                "vector_search_query_timeout",
            )
        except asyncio.TimeoutError as exc:
            logger.warning("vector_search_timeout", tenant_id=tenant_id, error=str(exc))
            raise
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
        if self._cache:
            self._cache.set(cache_key, hits)
        return hits

    async def _await_with_timeout(self, awaitable, event: str):
        if self.timeout_seconds <= 0:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout=self.timeout_seconds)
        except asyncio.TimeoutError as exc:
            logger.warning(event, error=str(exc))
            raise
