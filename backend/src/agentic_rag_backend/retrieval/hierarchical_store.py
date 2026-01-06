"""Postgres-backed hierarchical chunk store for small-to-big retrieval."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

import structlog

from agentic_rag_backend.db.postgres import PostgresClient

from .constants import DEFAULT_SIMILARITY_THRESHOLD

logger = structlog.get_logger(__name__)


class PostgresHierarchicalChunkStore:
    """ChunkStore implementation using Postgres hierarchical_chunks table."""

    def __init__(
        self,
        postgres: PostgresClient,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._postgres = postgres
        self._similarity_threshold = similarity_threshold

    async def get(self, chunk_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        """Get a hierarchical chunk by ID."""
        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            logger.warning("hierarchical_chunk_invalid_tenant_id", tenant_id=tenant_id)
            return None
        return await self._postgres.get_hierarchical_chunk(chunk_id, tenant_uuid)

    async def get_parent(self, chunk_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        """Get parent chunk if present."""
        chunk = await self.get(chunk_id, tenant_id)
        if not chunk:
            return None
        parent_id = chunk.get("parent_id")
        if not parent_id:
            return None
        return await self.get(parent_id, tenant_id)

    async def search_by_embedding(
        self,
        embedding: list[float],
        tenant_id: str,
        level: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search chunks by embedding at specific level."""
        try:
            tenant_uuid = UUID(tenant_id)
        except ValueError:
            logger.warning("hierarchical_chunk_invalid_tenant_id", tenant_id=tenant_id)
            return []

        rows = await self._postgres.search_similar_hierarchical_chunks(
            tenant_id=tenant_uuid,
            embedding=embedding,
            level=level,
            limit=limit,
            similarity_threshold=self._similarity_threshold,
        )

        results: list[dict[str, Any]] = []
        for row in rows:
            row_data = dict(row)
            row_data["score"] = float(row_data.pop("similarity", 0.0))
            results.append(row_data)
        return results
