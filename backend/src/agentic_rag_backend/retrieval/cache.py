"""TTL cache implementations for retrieval operations.

This module provides caching utilities for various retrieval operations including:
- General TTL cache for any value type
- Specialized reranking result cache (Story 19-G1)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from time import monotonic
from typing import Generic, Hashable, Optional, TypeVar, TYPE_CHECKING

import structlog

from agentic_rag_backend.observability.metrics import (
    record_reranker_cache_hit,
    record_reranker_cache_miss,
    set_reranker_cache_size,
)

if TYPE_CHECKING:
    from .reranking import RerankedHit

T = TypeVar("T")
logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry(Generic[T]):
    value: T
    expires_at: float


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0

    @property
    def total_requests(self) -> int:
        """Total cache requests (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0.0 to 1.0)."""
        total = self.total_requests
        if total == 0:
            return 0.0
        return self.hits / total

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0


class TTLCache(Generic[T]):
    """Simple size-bounded TTL cache."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._store: OrderedDict[Hashable, CacheEntry[T]] = OrderedDict()
        self._stats = CacheStats()

    def get(self, key: Hashable) -> T | None:
        now = monotonic()
        entry = self._store.get(key)
        if not entry:
            self._stats.misses += 1
            return None
        if entry.expires_at <= now:
            self._store.pop(key, None)
            self._stats.misses += 1
            return None
        self._store.move_to_end(key)
        self._stats.hits += 1
        return entry.value

    def set(self, key: Hashable, value: T) -> None:
        now = monotonic()
        self._store[key] = CacheEntry(value=value, expires_at=now + self.ttl_seconds)
        self._store.move_to_end(key)
        self._prune(now)

    def _prune(self, now: float) -> None:
        expired = [key for key, entry in self._store.items() if entry.expires_at <= now]
        for key in expired:
            self._store.pop(key, None)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._store)

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._store.clear()


def hash_cache_key(value: str) -> str:
    """Hash cache keys to avoid storing raw query strings."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# =============================================================================
# Story 19-G1: Reranking Result Cache
# =============================================================================


def generate_reranker_cache_key(
    query_text: str,
    document_ids: list[str],
    reranker_model: str,
    tenant_id: str,
) -> str:
    """Generate a cache key for reranking results.

    The cache key is a SHA-256 hash of the query, sorted document IDs,
    reranker model, and tenant ID.

    Args:
        query_text: The search query
        document_ids: List of document IDs being reranked
        reranker_model: The reranker model name
        tenant_id: Tenant identifier for multi-tenancy isolation

    Returns:
        SHA-256 hash string for use as cache key
    """
    # Sort document IDs to ensure consistent ordering
    sorted_doc_ids = sorted(document_ids)
    key_components = [
        query_text,
        "|".join(sorted_doc_ids),
        reranker_model,
        tenant_id,
    ]
    combined = "\n".join(key_components)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class RerankerCache:
    """Cache for reranking results with tenant isolation.

    This cache stores reranked results keyed by query + document set + model + tenant.
    It helps avoid redundant reranking calls for identical query-document combinations.

    Implements Story 19-G1 acceptance criteria:
    - Reranked results cached by query hash + document set
    - Cache TTL is configurable (default 5 minutes)
    - Cache hit rate is logged and exposed as metric
    - Repeated identical queries return faster
    - Cache respects tenant isolation

    Usage:
        cache = RerankerCache(
            enabled=True,
            ttl_seconds=300,
            max_size=1000,
        )

        # Check cache
        cached = cache.get(query, doc_ids, model, tenant_id)
        if cached:
            return cached

        # Perform reranking
        reranked = await reranker.rerank(...)

        # Store in cache
        cache.set(query, doc_ids, model, tenant_id, reranked)
    """

    def __init__(
        self,
        enabled: bool = False,
        ttl_seconds: int = 300,
        max_size: int = 1000,
    ) -> None:
        """Initialize the reranker cache.

        Args:
            enabled: Whether caching is enabled
            ttl_seconds: Time-to-live for cache entries in seconds (default: 300)
            max_size: Maximum number of cached entries (default: 1000)
        """
        self._enabled = enabled
        self._cache: TTLCache[list] = TTLCache(max_size=max_size, ttl_seconds=ttl_seconds)

        logger.info(
            "reranker_cache_initialized",
            enabled=enabled,
            ttl_seconds=ttl_seconds,
            max_size=max_size,
        )

    @property
    def enabled(self) -> bool:
        """Whether caching is enabled."""
        return self._enabled

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._cache.stats

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return self._cache.size

    def get(
        self,
        query_text: str,
        document_ids: list[str],
        reranker_model: str,
        tenant_id: str,
    ) -> Optional[list]:
        """Get cached reranking results if available.

        Args:
            query_text: The search query
            document_ids: List of document IDs being reranked
            reranker_model: The reranker model name
            tenant_id: Tenant identifier

        Returns:
            Cached list of RerankedHit objects, or None if not cached
        """
        if not self._enabled:
            return None

        cache_key = generate_reranker_cache_key(
            query_text, document_ids, reranker_model, tenant_id
        )
        result = self._cache.get(cache_key)

        if result is not None:
            # Record cache hit metric
            record_reranker_cache_hit(tenant_id)
            logger.debug(
                "reranker_cache_hit",
                query_preview=query_text[:50],
                doc_count=len(document_ids),
                tenant_id=tenant_id,
            )
        else:
            # Record cache miss metric
            record_reranker_cache_miss(tenant_id)
            logger.debug(
                "reranker_cache_miss",
                query_preview=query_text[:50],
                doc_count=len(document_ids),
                tenant_id=tenant_id,
            )

        return result

    def set(
        self,
        query_text: str,
        document_ids: list[str],
        reranker_model: str,
        tenant_id: str,
        reranked_results: list,
    ) -> None:
        """Store reranking results in cache.

        Args:
            query_text: The search query
            document_ids: List of document IDs being reranked
            reranker_model: The reranker model name
            tenant_id: Tenant identifier
            reranked_results: List of RerankedHit objects to cache
        """
        if not self._enabled:
            return

        cache_key = generate_reranker_cache_key(
            query_text, document_ids, reranker_model, tenant_id
        )
        self._cache.set(cache_key, reranked_results)

        # Update cache size metric
        set_reranker_cache_size(self._cache.size)

        logger.debug(
            "reranker_cache_set",
            query_preview=query_text[:50],
            doc_count=len(document_ids),
            result_count=len(reranked_results),
            tenant_id=tenant_id,
        )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("reranker_cache_cleared")

    def log_stats(self) -> None:
        """Log current cache statistics."""
        stats = self._cache.stats
        logger.info(
            "reranker_cache_stats",
            hits=stats.hits,
            misses=stats.misses,
            hit_rate=round(stats.hit_rate, 4),
            cache_size=self.size,
        )
