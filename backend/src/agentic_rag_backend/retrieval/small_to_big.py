"""Small-to-Big Retrieval (Story 20-C3).

This module implements the small-to-big retrieval pattern:
- Search on small chunks (Level 0) for precise matching
- Return parent chunks at a configurable level for complete context

The pattern solves the precision vs. context trade-off:
- Small chunks match queries precisely but lack context
- Large chunks provide context but match imprecisely
- Small-to-big gets the best of both worlds

Key Features:
- Vector search on Level 0 (embedding level) chunks
- Parent retrieval at configurable return level
- Deduplication of overlapping parents
- Tracks which small chunks matched for transparency
- Feature flag: HIERARCHICAL_CHUNKS_ENABLED

Configuration:
- SMALL_TO_BIG_RETURN_LEVEL: Level to return (default: 2 = 1024 tokens)
- HIERARCHICAL_CHUNKS_ENABLED: Enable/disable feature (default: false)

Performance target: <100ms additional latency over standard search
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import structlog

logger = structlog.get_logger(__name__)

# Default configuration
DEFAULT_RETURN_LEVEL = 2
DEFAULT_TOP_K = 10


@dataclass
class SmallToBigResult:
    """Result from small-to-big retrieval.

    Attributes:
        id: Chunk identifier
        content: Chunk text content
        level: Hierarchy level of returned chunk
        score: Combined relevance score
        matched_child_ids: IDs of Level 0 chunks that matched
        matched_scores: Scores of matched children
        document_id: Source document identifier
        token_count: Number of tokens in content
        metadata: Additional metadata
    """

    id: str
    content: str
    level: int
    score: float
    matched_child_ids: list[str] = field(default_factory=list)
    matched_scores: list[float] = field(default_factory=list)
    document_id: str = ""
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "level": self.level,
            "score": self.score,
            "matched_child_ids": self.matched_child_ids,
            "matched_scores": self.matched_scores,
            "document_id": self.document_id,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class SmallToBigRetrievalResult:
    """Full result of small-to-big retrieval operation.

    Attributes:
        query: Original query string
        results: List of SmallToBigResult objects
        matched_at_level: Level where matching occurred (always 0 for small-to-big)
        returned_at_level: Level of returned chunks
        total_matches: Total number of Level 0 matches before deduplication
        processing_time_ms: Time taken for retrieval
        tenant_id: Tenant identifier
    """

    query: str
    results: list[SmallToBigResult]
    matched_at_level: int
    returned_at_level: int
    total_matches: int
    processing_time_ms: int
    tenant_id: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "matched_at_level": self.matched_at_level,
            "returned_at_level": self.returned_at_level,
            "total_matches": self.total_matches,
            "processing_time_ms": self.processing_time_ms,
            "tenant_id": self.tenant_id,
        }


class ChunkStore(Protocol):
    """Protocol for chunk storage operations."""

    async def get(self, chunk_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        """Get chunk by ID."""
        ...

    async def get_parent(self, chunk_id: str, tenant_id: str) -> Optional[dict[str, Any]]:
        """Get parent chunk."""
        ...

    async def search_by_embedding(
        self,
        embedding: list[float],
        tenant_id: str,
        level: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search chunks by embedding at specific level."""
        ...


class VectorSearch(Protocol):
    """Protocol for vector search operations."""

    async def search(
        self,
        query: str,
        tenant_id: str,
        limit: int,
        filter_metadata: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute vector search."""
        ...


class SmallToBigRetriever:
    """Retrieves small chunks but returns parent context.

    This class implements the small-to-big retrieval pattern:
    1. Search Level 0 (smallest) chunks for precise matching
    2. Traverse up hierarchy to target return level
    3. Deduplicate overlapping parent chunks
    4. Return parent chunks with matched child info

    Attributes:
        chunk_store: Storage for hierarchical chunks
        return_level: Level at which to return chunks (default: 2)
        embedding_level: Level used for embedding search (default: 0)
    """

    def __init__(
        self,
        chunk_store: ChunkStore,
        return_level: int = DEFAULT_RETURN_LEVEL,
        embedding_level: int = 0,
    ) -> None:
        """Initialize SmallToBigRetriever.

        Args:
            chunk_store: Storage backend for hierarchical chunks
            return_level: Level at which to return chunks (default: 2)
            embedding_level: Level used for embedding search (default: 0)
        """
        self._chunk_store = chunk_store
        self.return_level = return_level
        self.embedding_level = embedding_level

        logger.debug(
            "small_to_big_retriever_initialized",
            return_level=return_level,
            embedding_level=embedding_level,
        )

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
        return_level: Optional[int] = None,
    ) -> SmallToBigRetrievalResult:
        """Execute small-to-big retrieval.

        This is the main entry point that:
        1. Searches Level 0 chunks using the embedding
        2. Groups matches by ancestor at return level
        3. Deduplicates and ranks parent chunks
        4. Returns enriched results with child match info

        Args:
            query: Natural language query
            tenant_id: Tenant identifier for multi-tenancy
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            return_level: Override the return level (default: use configured)

        Returns:
            SmallToBigRetrievalResult with parent chunks and match info
        """
        start_time = time.perf_counter()
        effective_return_level = return_level if return_level is not None else self.return_level

        # Validate tenant_id - must be non-empty for multi-tenancy
        if not tenant_id or not tenant_id.strip():
            logger.warning(
                "small_to_big_invalid_tenant_id",
                tenant_id=tenant_id,
                query=query[:50] if query else "",
            )
            return SmallToBigRetrievalResult(
                query=query,
                results=[],
                matched_at_level=self.embedding_level,
                returned_at_level=effective_return_level,
                total_matches=0,
                processing_time_ms=0,
                tenant_id=tenant_id or "",
            )

        logger.info(
            "small_to_big_retrieval_started",
            query=query[:100],
            tenant_id=tenant_id,
            return_level=effective_return_level,
            top_k=top_k,
        )

        try:
            # Step 1: Search small chunks at embedding level
            small_matches = await self._search_small_chunks(
                embedding=embedding,
                tenant_id=tenant_id,
                limit=top_k * 3,  # Get more matches to allow for deduplication
            )

            if not small_matches:
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                return SmallToBigRetrievalResult(
                    query=query,
                    results=[],
                    matched_at_level=self.embedding_level,
                    returned_at_level=effective_return_level,
                    total_matches=0,
                    processing_time_ms=processing_time_ms,
                    tenant_id=tenant_id,
                )

            # Step 2: Get ancestors at return level for each match
            parent_to_children: dict[str, list[tuple[str, float]]] = {}

            for match in small_matches:
                chunk_id = match.get("id", "")
                score = match.get("score", 0.0)
                chunk_level = match.get("level", 0)

                # Get ancestor at return level
                ancestor = await self._get_ancestor_at_level(
                    chunk_id=chunk_id,
                    current_level=chunk_level,
                    target_level=effective_return_level,
                    tenant_id=tenant_id,
                )

                if ancestor:
                    parent_id = ancestor.get("id", "")
                    if parent_id not in parent_to_children:
                        parent_to_children[parent_id] = []
                    parent_to_children[parent_id].append((chunk_id, score))

            # Step 3: Deduplicate and rank by combined score
            results = await self._create_results(
                parent_to_children=parent_to_children,
                tenant_id=tenant_id,
                top_k=top_k,
            )

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)

            logger.info(
                "small_to_big_retrieval_completed",
                query=query[:100],
                tenant_id=tenant_id,
                small_matches=len(small_matches),
                parent_results=len(results),
                processing_time_ms=processing_time_ms,
            )

            return SmallToBigRetrievalResult(
                query=query,
                results=results,
                matched_at_level=self.embedding_level,
                returned_at_level=effective_return_level,
                total_matches=len(small_matches),
                processing_time_ms=processing_time_ms,
                tenant_id=tenant_id,
            )

        except Exception as e:
            logger.error(
                "small_to_big_retrieval_failed",
                query=query[:100],
                tenant_id=tenant_id,
                error=str(e),
            )
            raise

    async def _search_small_chunks(
        self,
        embedding: list[float],
        tenant_id: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Search for small chunks at embedding level.

        Args:
            embedding: Query embedding vector
            tenant_id: Tenant identifier
            limit: Maximum results

        Returns:
            List of matching chunk dictionaries
        """
        return await self._chunk_store.search_by_embedding(
            embedding=embedding,
            tenant_id=tenant_id,
            level=self.embedding_level,
            limit=limit,
        )

    async def _get_ancestor_at_level(
        self,
        chunk_id: str,
        current_level: int,
        target_level: int,
        tenant_id: str,
    ) -> Optional[dict[str, Any]]:
        """Traverse up hierarchy to find ancestor at target level.

        Args:
            chunk_id: Starting chunk ID
            current_level: Current chunk's level
            target_level: Target ancestor level
            tenant_id: Tenant identifier

        Returns:
            Ancestor chunk dictionary, or None if not found
        """
        if current_level >= target_level:
            # Already at or above target level, return the chunk itself
            return await self._chunk_store.get(chunk_id, tenant_id)

        # Traverse up the hierarchy
        current_id = chunk_id
        current = await self._chunk_store.get(current_id, tenant_id)

        while current and current.get("level", 0) < target_level:
            parent_id = current.get("parent_id")
            if not parent_id:
                # No more parents, return current
                break
            current = await self._chunk_store.get(parent_id, tenant_id)
            if current:
                current_id = parent_id

        return current

    async def _create_results(
        self,
        parent_to_children: dict[str, list[tuple[str, float]]],
        tenant_id: str,
        top_k: int,
    ) -> list[SmallToBigResult]:
        """Create SmallToBigResult objects from parent-children mapping.

        Args:
            parent_to_children: Mapping of parent IDs to list of (child_id, score)
            tenant_id: Tenant identifier
            top_k: Maximum results to return

        Returns:
            Sorted list of SmallToBigResult objects
        """
        results = []

        # Fetch all parent chunks in parallel
        parent_ids = list(parent_to_children.keys())
        parent_tasks = [
            self._chunk_store.get(parent_id, tenant_id)
            for parent_id in parent_ids
        ]
        parent_chunks = await asyncio.gather(*parent_tasks, return_exceptions=True)

        for parent_id, parent_chunk in zip(parent_ids, parent_chunks):
            if isinstance(parent_chunk, Exception) or not parent_chunk:
                continue

            children = parent_to_children[parent_id]
            child_ids = [c[0] for c in children]
            child_scores = [c[1] for c in children]

            # Combined score: max of children + bonus for multiple matches
            max_score = max(child_scores) if child_scores else 0.0
            match_bonus = min(0.1 * (len(children) - 1), 0.2)  # Up to 0.2 bonus
            combined_score = min(1.0, max_score + match_bonus)

            result = SmallToBigResult(
                id=parent_chunk.get("id", ""),
                content=parent_chunk.get("content", ""),
                level=parent_chunk.get("level", 0),
                score=round(combined_score, 3),
                matched_child_ids=child_ids,
                matched_scores=[round(s, 3) for s in child_scores],
                document_id=parent_chunk.get("document_id", ""),
                token_count=parent_chunk.get("token_count", 0),
                metadata=parent_chunk.get("metadata", {}),
            )
            results.append(result)

        # Sort by score descending and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


class SmallToBigAdapter:
    """Adapter for SmallToBigRetriever with configuration and feature flag.

    This adapter provides:
    - Feature flag checking (HIERARCHICAL_CHUNKS_ENABLED)
    - Configuration from Settings
    - Graceful fallback when disabled
    """

    def __init__(
        self,
        retriever: Optional[SmallToBigRetriever],
        enabled: bool = False,
        return_level: int = DEFAULT_RETURN_LEVEL,
    ) -> None:
        """Initialize SmallToBigAdapter.

        Args:
            retriever: SmallToBigRetriever instance (or None if disabled)
            enabled: Whether small-to-big retrieval is enabled
            return_level: Configured return level
        """
        self._retriever = retriever
        self.enabled = enabled
        self.return_level = return_level

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        embedding: list[float],
        top_k: int = DEFAULT_TOP_K,
        return_level: Optional[int] = None,
    ) -> Optional[SmallToBigRetrievalResult]:
        """Execute small-to-big retrieval if enabled.

        Args:
            query: Natural language query
            tenant_id: Tenant identifier
            embedding: Query embedding vector
            top_k: Maximum results
            return_level: Override return level

        Returns:
            SmallToBigRetrievalResult if enabled and successful, None otherwise
        """
        if not self.enabled or not self._retriever:
            logger.debug(
                "small_to_big_retrieval_disabled",
                enabled=self.enabled,
                has_retriever=self._retriever is not None,
            )
            return None

        return await self._retriever.retrieve(
            query=query,
            tenant_id=tenant_id,
            embedding=embedding,
            top_k=top_k,
            return_level=return_level or self.return_level,
        )


def get_small_to_big_adapter(
    chunk_store: Optional[ChunkStore],
    settings: Any,
) -> SmallToBigAdapter:
    """Factory function to create SmallToBigAdapter from settings.

    Args:
        chunk_store: Chunk storage backend
        settings: Application settings

    Returns:
        Configured SmallToBigAdapter instance
    """
    enabled = getattr(settings, "hierarchical_chunks_enabled", False)
    return_level = getattr(settings, "small_to_big_return_level", DEFAULT_RETURN_LEVEL)
    embedding_level = getattr(settings, "hierarchical_embedding_level", 0)

    retriever = None
    if enabled and chunk_store:
        retriever = SmallToBigRetriever(
            chunk_store=chunk_store,
            return_level=return_level,
            embedding_level=embedding_level,
        )

    return SmallToBigAdapter(
        retriever=retriever,
        enabled=enabled,
        return_level=return_level,
    )
