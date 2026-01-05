"""Memory Consolidation for Epic 20 Memory Platform (Story 20-A2).

This module implements memory consolidation that:
- Deduplicates similar memories using embedding similarity
- Applies importance decay based on time and access frequency
- Removes memories below the minimum importance threshold
- Supports batch processing for large datasets

The consolidation process is designed to be tenant-isolated and
can run on a schedule or be triggered manually via API.
"""

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

from .models import ConsolidationResult, MemoryScope, ScopedMemory

if TYPE_CHECKING:
    from .store import ScopedMemoryStore

logger = structlog.get_logger(__name__)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector

    Returns:
        Similarity score 0.0-1.0 (higher is more similar)
    """
    if not vec_a or not vec_b:
        return 0.0

    a = np.array(vec_a)
    b = np.array(vec_b)

    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def calculate_importance(
    current_importance: float,
    access_count: int,
    days_since_access: float,
    decay_half_life_days: int = 30,
    access_boost_factor: float = 0.1,
) -> float:
    """Calculate new importance after applying decay and access boost.

    Formula:
    - decay_factor = 2 ** (-days_since_access / half_life_days)
    - access_boost = min(1.0, 0.5 + (access_count * access_boost_factor))
    - new_importance = importance * decay_factor * access_boost

    Args:
        current_importance: Current importance score (0.0-1.0)
        access_count: Number of times the memory was accessed
        days_since_access: Days since last access
        decay_half_life_days: Days for importance to halve (default: 30)
        access_boost_factor: Boost per access (default: 0.1)

    Returns:
        New importance value clamped to 0.0-1.0
    """
    # Exponential decay: importance halves every half_life_days
    decay_factor = 2 ** (-days_since_access / decay_half_life_days)

    # Access boost: more accesses = slower effective decay
    # Base boost is 0.5, can reach 1.0 with enough accesses
    access_boost = min(1.0, 0.5 + (access_count * access_boost_factor))

    new_importance = current_importance * decay_factor * access_boost

    return max(0.0, min(1.0, new_importance))


class MemoryConsolidator:
    """Consolidate and manage memory lifecycle.

    The consolidator performs:
    1. Importance decay - Reduce importance based on time since last access
    2. Duplicate merging - Merge similar memories (by embedding similarity)
    3. Cleanup - Remove memories below minimum importance threshold

    All operations are tenant-isolated via tenant_id filtering.

    Attributes:
        store: ScopedMemoryStore instance for memory operations
        similarity_threshold: Minimum similarity to consider memories as duplicates
        decay_half_life_days: Days for importance to decay by half
        min_importance: Threshold below which memories are removed
        batch_size: Number of memories to process per batch
    """

    def __init__(
        self,
        store: "ScopedMemoryStore",
        similarity_threshold: float = 0.9,
        decay_half_life_days: int = 30,
        min_importance: float = 0.1,
        consolidation_batch_size: int = 100,
    ) -> None:
        """Initialize the memory consolidator.

        Args:
            store: ScopedMemoryStore instance
            similarity_threshold: Similarity threshold for duplicate detection (0.0-1.0)
            decay_half_life_days: Days for importance to halve
            min_importance: Minimum importance threshold (memories below are removed)
            consolidation_batch_size: Number of memories to process per batch
        """
        self.store = store
        self.similarity_threshold = similarity_threshold
        self.decay_half_life_days = decay_half_life_days
        self.min_importance = min_importance
        self.batch_size = consolidation_batch_size

        # Track last consolidation status for API queries
        self._last_run_at: Optional[datetime] = None
        self._last_result: Optional[ConsolidationResult] = None

    @property
    def last_run_at(self) -> Optional[datetime]:
        """Get timestamp of last consolidation run."""
        return self._last_run_at

    @property
    def last_result(self) -> Optional[ConsolidationResult]:
        """Get result of last consolidation run."""
        return self._last_result

    async def consolidate(
        self,
        tenant_id: str,
        scope: Optional[MemoryScope] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> ConsolidationResult:
        """Run consolidation for a tenant with optional scope filtering.

        Consolidation steps:
        1. Apply importance decay based on time and access frequency
        2. Find and merge similar memories (>threshold similarity)
        3. Remove memories below minimum importance threshold

        Args:
            tenant_id: Tenant identifier (required)
            scope: Optional scope to consolidate (None for all scopes)
            user_id: User ID for USER/SESSION scope
            session_id: Session ID for SESSION scope
            agent_id: Agent ID for AGENT scope

        Returns:
            ConsolidationResult with counts of processed, merged, decayed, removed
        """
        start_time = time.perf_counter()

        logger.info(
            "memory_consolidation_started",
            tenant_id=tenant_id,
            scope=scope.value if scope else "all",
        )

        # Get all memories in scope
        memories = await self._get_scope_memories(
            tenant_id=tenant_id,
            scope=scope,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )

        if not memories:
            result = ConsolidationResult(
                memories_processed=0,
                duplicates_merged=0,
                memories_decayed=0,
                memories_removed=0,
                processing_time_ms=0,
                scope=scope,
                tenant_id=tenant_id,
            )
            self._last_run_at = datetime.now(timezone.utc)
            self._last_result = result
            return result

        # Step 1: Apply importance decay
        decayed_count = await self._apply_importance_decay(memories, tenant_id)

        # Step 2: Find and merge similar memories
        merged_count = await self._merge_similar_memories(memories, tenant_id)

        # Step 3: Remove low-importance memories
        # Re-fetch memories after decay to get updated importance values
        memories = await self._get_scope_memories(
            tenant_id=tenant_id,
            scope=scope,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )
        removed_count = await self._remove_low_importance(memories, tenant_id)

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        result = ConsolidationResult(
            memories_processed=len(memories),
            duplicates_merged=merged_count,
            memories_decayed=decayed_count,
            memories_removed=removed_count,
            processing_time_ms=processing_time_ms,
            scope=scope,
            tenant_id=tenant_id,
        )

        self._last_run_at = datetime.now(timezone.utc)
        self._last_result = result

        logger.info(
            "memory_consolidation_complete",
            tenant_id=tenant_id,
            scope=scope.value if scope else "all",
            memories_processed=len(memories),
            duplicates_merged=merged_count,
            memories_decayed=decayed_count,
            memories_removed=removed_count,
            processing_time_ms=f"{processing_time_ms:.2f}",
        )

        return result

    async def consolidate_all_tenants(self) -> list[ConsolidationResult]:
        """Consolidate memories for all tenants.

        This is called by the scheduler to process all tenants.
        Each tenant is processed independently with full tenant isolation.

        Returns:
            List of ConsolidationResult for each tenant processed
        """
        logger.info("memory_consolidation_all_tenants_started")
        start_time = time.perf_counter()

        # Get unique tenant IDs from the store
        tenant_ids = await self._get_all_tenant_ids()

        results = []
        for tenant_id in tenant_ids:
            try:
                result = await self.consolidate(tenant_id=tenant_id)
                results.append(result)
            except Exception as e:
                logger.error(
                    "memory_consolidation_tenant_failed",
                    tenant_id=tenant_id,
                    error=str(e),
                )

        total_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "memory_consolidation_all_tenants_complete",
            tenants_processed=len(tenant_ids),
            total_processing_time_ms=f"{total_time_ms:.2f}",
        )

        return results

    async def _get_scope_memories(
        self,
        tenant_id: str,
        scope: Optional[MemoryScope],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[ScopedMemory]:
        """Retrieve all memories for a given scope context.

        Uses batch processing to handle large datasets efficiently.

        Args:
            tenant_id: Tenant identifier
            scope: Memory scope (None for all scopes)
            user_id: Optional user filter
            session_id: Optional session filter
            agent_id: Optional agent filter

        Returns:
            List of ScopedMemory objects
        """
        all_memories: list[ScopedMemory] = []
        offset = 0

        while True:
            memories, total = await self.store.list_memories(
                tenant_id=tenant_id,
                scope=scope,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                limit=self.batch_size,
                offset=offset,
            )

            if not memories:
                break

            # Fetch embeddings for each memory (needed for similarity comparison)
            # The list_memories doesn't include embeddings by default
            for memory in memories:
                if memory.embedding is None:
                    # Fetch full memory to get embedding
                    full_memory = await self.store.get_memory(
                        memory_id=str(memory.id),
                        tenant_id=tenant_id,
                    )
                    if full_memory and full_memory.embedding:
                        memory.embedding = full_memory.embedding

            all_memories.extend(memories)
            offset += len(memories)

            if len(memories) < self.batch_size:
                break

        return all_memories

    async def _get_all_tenant_ids(self) -> list[str]:
        """Get all unique tenant IDs with memories.

        Returns:
            List of tenant ID strings
        """
        # Query PostgreSQL for distinct tenant_ids
        async with self.store._postgres.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT tenant_id FROM scoped_memories"
            )
            return [str(row["tenant_id"]) for row in rows]

    async def _apply_importance_decay(
        self,
        memories: list[ScopedMemory],
        tenant_id: str,
    ) -> int:
        """Apply time-based importance decay to memories.

        Uses exponential decay with access frequency boost.

        Args:
            memories: List of memories to process
            tenant_id: Tenant identifier

        Returns:
            Number of memories with updated importance
        """
        decayed_count = 0
        now = datetime.now(timezone.utc)

        for memory in memories:
            # Calculate days since last access
            days_since_access = (now - memory.accessed_at).total_seconds() / 86400

            # Calculate new importance
            new_importance = calculate_importance(
                current_importance=memory.importance,
                access_count=memory.access_count,
                days_since_access=days_since_access,
                decay_half_life_days=self.decay_half_life_days,
            )

            # Only update if importance actually changed (with small epsilon for float comparison)
            if abs(new_importance - memory.importance) > 0.001:
                await self.store.update_memory(
                    memory_id=str(memory.id),
                    tenant_id=tenant_id,
                    importance=new_importance,
                )
                memory.importance = new_importance
                decayed_count += 1

        return decayed_count

    async def _merge_similar_memories(
        self,
        memories: list[ScopedMemory],
        tenant_id: str,
    ) -> int:
        """Find and merge similar memories based on embedding similarity.

        Memories with similarity >= threshold are merged:
        - Primary memory is kept (highest importance)
        - Secondary memories are deleted
        - Metadata is combined
        - Access counts are summed

        Args:
            memories: List of memories to process
            tenant_id: Tenant identifier

        Returns:
            Number of memories merged (deleted after merge)
        """
        merged_count = 0
        processed_ids: set[str] = set()

        # Filter to memories with embeddings only
        memories_with_embeddings = [m for m in memories if m.embedding]

        for i, primary in enumerate(memories_with_embeddings):
            primary_id = str(primary.id)
            if primary_id in processed_ids:
                continue

            # Find similar memories
            similar_memories: list[ScopedMemory] = []

            for secondary in memories_with_embeddings[i + 1:]:
                secondary_id = str(secondary.id)
                if secondary_id in processed_ids:
                    continue

                # Must be same scope for merging
                if secondary.scope != primary.scope:
                    continue

                similarity = cosine_similarity(
                    primary.embedding or [],
                    secondary.embedding or [],
                )

                if similarity >= self.similarity_threshold:
                    similar_memories.append(secondary)

            if similar_memories:
                # Merge all similar memories into primary
                merged_metadata = await self._merge_memories(primary, similar_memories)

                # Update primary with merged data
                await self.store.update_memory(
                    memory_id=primary_id,
                    tenant_id=tenant_id,
                    importance=merged_metadata["importance"],
                    metadata=merged_metadata["metadata"],
                )

                # Delete secondary memories
                for secondary in similar_memories:
                    secondary_id = str(secondary.id)
                    await self.store.delete_memory(
                        memory_id=secondary_id,
                        tenant_id=tenant_id,
                    )
                    processed_ids.add(secondary_id)
                    merged_count += 1

                logger.debug(
                    "memories_merged",
                    primary_id=primary_id,
                    merged_count=len(similar_memories),
                    tenant_id=tenant_id,
                )

        return merged_count

    async def _merge_memories(
        self,
        primary: ScopedMemory,
        similar: list[ScopedMemory],
    ) -> dict:
        """Create merged data from primary and similar memories.

        Merge strategy:
        - Keep primary content (most important/recent)
        - Use max importance across all memories
        - Sum access counts
        - Merge metadata (primary takes precedence)

        Args:
            primary: Primary memory to keep
            similar: List of similar memories to merge

        Returns:
            Dictionary with merged importance, access_count, and metadata
        """
        # Combine importance (max of all)
        all_importance = [primary.importance] + [s.importance for s in similar]
        combined_importance = max(all_importance)

        # Combine access counts
        combined_access = primary.access_count + sum(s.access_count for s in similar)

        # Merge metadata (primary takes precedence)
        merged_metadata = {}
        for s in reversed(similar):
            merged_metadata.update(s.metadata)
        merged_metadata.update(primary.metadata)

        # Add merge tracking
        merged_metadata["merged_count"] = merged_metadata.get("merged_count", 0) + len(similar)
        merged_metadata["merged_at"] = datetime.now(timezone.utc).isoformat()
        merged_metadata["merged_from_ids"] = [str(s.id) for s in similar]

        return {
            "importance": combined_importance,
            "access_count": combined_access,
            "metadata": merged_metadata,
        }

    async def _remove_low_importance(
        self,
        memories: list[ScopedMemory],
        tenant_id: str,
    ) -> int:
        """Remove memories with importance below threshold.

        Args:
            memories: List of memories to check
            tenant_id: Tenant identifier

        Returns:
            Number of memories removed
        """
        removed_count = 0

        for memory in memories:
            if memory.importance < self.min_importance:
                deleted = await self.store.delete_memory(
                    memory_id=str(memory.id),
                    tenant_id=tenant_id,
                )
                if deleted:
                    removed_count += 1
                    logger.debug(
                        "low_importance_memory_removed",
                        memory_id=str(memory.id),
                        importance=memory.importance,
                        threshold=self.min_importance,
                        tenant_id=tenant_id,
                    )

        return removed_count

    def find_similar_memories(
        self,
        target_embedding: list[float],
        memories: list[ScopedMemory],
        threshold: Optional[float] = None,
    ) -> list[tuple[ScopedMemory, float]]:
        """Find memories similar to a target embedding.

        Utility method for testing and inspection.

        Args:
            target_embedding: Target embedding vector
            memories: List of memories to compare against
            threshold: Similarity threshold (default: self.similarity_threshold)

        Returns:
            List of (memory, similarity_score) tuples above threshold
        """
        if threshold is None:
            threshold = self.similarity_threshold

        similar: list[tuple[ScopedMemory, float]] = []

        for memory in memories:
            if not memory.embedding:
                continue

            similarity = cosine_similarity(target_embedding, memory.embedding)
            if similarity >= threshold:
                similar.append((memory, similarity))

        # Sort by similarity descending
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    def cleanup_low_importance(
        self,
        memories: list[ScopedMemory],
    ) -> list[ScopedMemory]:
        """Identify memories below importance threshold (sync utility method).

        This is a utility method for testing; actual cleanup uses _remove_low_importance.

        Args:
            memories: List of memories to filter

        Returns:
            List of memories below min_importance threshold
        """
        return [m for m in memories if m.importance < self.min_importance]
