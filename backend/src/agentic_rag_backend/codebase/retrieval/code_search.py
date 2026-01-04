"""Semantic search over indexed codebase chunks."""

from __future__ import annotations

from typing import Any, Optional

import structlog

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.embeddings import EmbeddingGenerator
from agentic_rag_backend.retrieval.vector_search import VectorSearchService

logger = structlog.get_logger(__name__)


class CodeSearchService:
    """Perform semantic search over indexed code chunks."""

    def __init__(
        self,
        postgres: PostgresClient,
        embedding_generator: EmbeddingGenerator,
        neo4j: Optional[Neo4jClient] = None,
    ) -> None:
        self._vector_search = VectorSearchService(postgres, embedding_generator)
        self._neo4j = neo4j

    async def search(
        self,
        tenant_id: str,
        query: str,
        limit: int = 10,
        include_relationships: bool = True,
    ) -> list[dict[str, Any]]:
        self._vector_search.limit = max(limit * 2, self._vector_search.limit)
        hits = await self._vector_search.search(query, tenant_id)
        code_hits = [
            hit for hit in hits
            if hit.metadata and hit.metadata.get("source_type") == "codebase"
        ]

        results: list[dict[str, Any]] = []
        for hit in code_hits[:limit]:
            metadata = hit.metadata or {}
            relationships: list[dict[str, Any]] = []

            symbol_entity_id = metadata.get("symbol_entity_id")
            if include_relationships and symbol_entity_id and self._neo4j:
                try:
                    relationships = await self._neo4j.get_entity_relationships(
                        entity_id=symbol_entity_id,
                        tenant_id=tenant_id,
                        limit=25,
                    )
                except Exception as exc:
                    logger.warning("codebase_relationship_fetch_failed", error=str(exc))

            results.append({
                "symbol_name": metadata.get("symbol_name", ""),
                "symbol_type": metadata.get("symbol_type", ""),
                "file_path": metadata.get("file_path", ""),
                "line_start": metadata.get("line_start", 0),
                "line_end": metadata.get("line_end", 0),
                "content": hit.content,
                "score": hit.similarity,
                "relationships": relationships,
            })

        return results
