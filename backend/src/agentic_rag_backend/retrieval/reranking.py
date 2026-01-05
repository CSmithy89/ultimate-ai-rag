"""Cross-encoder reranking for improved retrieval precision.

Supports:
- Cohere Rerank API (rerank-v3.5, 100+ languages, 32K context)
- FlashRank (local CPU-optimized, no API cost)

Story 19-G1: Adds caching support for reranking results
Story 19-G3: Adds model preloading support for FlashRank

Usage:
    from agentic_rag_backend.retrieval.reranking import (
        create_reranker_client,
        get_reranker_adapter,
        RerankerProviderType,
    )

    adapter = get_reranker_adapter(settings)
    client = create_reranker_client(adapter)
    reranked = await client.rerank(query, hits, top_k=10)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, TYPE_CHECKING

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .types import VectorHit
from .cache import RerankerCache
from ..observability.metrics import (
    record_retrieval_latency,
    record_reranking_improvement,
)

logger = structlog.get_logger(__name__)

# Global reranker cache instance (Story 19-G1)
_reranker_cache: Optional[RerankerCache] = None

if TYPE_CHECKING:
    from ..config import Settings


class RerankerProviderType(str, Enum):
    """Reranker provider types."""
    COHERE = "cohere"
    FLASHRANK = "flashrank"


@dataclass(frozen=True)
class RerankerProviderAdapter:
    """Adapter for reranker providers."""

    provider: RerankerProviderType
    api_key: Optional[str]
    model: str
    top_k: int = 10
    # Story 19-G3: Model preloading
    preload: bool = False


@dataclass(frozen=True)
class RerankedHit:
    """A vector hit with reranking score."""

    hit: VectorHit
    rerank_score: float
    original_rank: int


class RerankerClient(ABC):
    """Abstract base class for reranker clients."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> list[RerankedHit]:
        """Rerank vector hits using cross-encoder scoring.

        Args:
            query: The search query
            hits: Vector hits to rerank
            top_k: Number of top results to return
            tenant_id: Tenant identifier for metrics
            strategy: Retrieval strategy for metrics labeling

        Returns:
            Reranked hits with scores, sorted by relevance
        """
        pass

    @abstractmethod
    def get_model(self) -> str:
        """Get the model name used for reranking."""
        pass


class CachedRerankerClient(RerankerClient):
    """Caching wrapper for reranker clients."""

    def __init__(self, inner: RerankerClient, cache: RerankerCache) -> None:
        self._inner = inner
        self._cache = cache

    async def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> list[RerankedHit]:
        if not hits:
            return []
        if tenant_id is None:
            return await self._inner.rerank(
                query=query,
                hits=hits,
                top_k=top_k,
                tenant_id=tenant_id,
                strategy=strategy,
            )

        document_ids = [hit.document_id for hit in hits]
        chunk_ids = [hit.chunk_id for hit in hits]
        cached = self._cache.get(
            query_text=query,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
            reranker_model=self._inner.get_model(),
            tenant_id=tenant_id,
            top_k=top_k,
        )
        if cached is not None:
            return cached

        reranked = await self._inner.rerank(
            query=query,
            hits=hits,
            top_k=top_k,
            tenant_id=tenant_id,
            strategy=strategy,
        )
        self._cache.set(
            query_text=query,
            document_ids=document_ids,
            chunk_ids=chunk_ids,
            reranker_model=self._inner.get_model(),
            tenant_id=tenant_id,
            top_k=top_k,
            reranked_results=reranked,
        )
        return reranked

    def get_model(self) -> str:
        return self._inner.get_model()


class CohereRerankerClient(RerankerClient):
    """Cohere Rerank API client.

    Uses Cohere's cross-encoder models for high-accuracy reranking.
    Supports 100+ languages and 32K context window.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-v3.5",
    ) -> None:
        try:
            import cohere
            self._client = cohere.AsyncClient(api_key=api_key)
        except ImportError:
            raise RuntimeError(
                "cohere package required for Cohere reranking. "
                "Install with: uv add cohere"
            )
        self._model = model
        logger.info("cohere_reranker_initialized", model=model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            "cohere_rerank_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        ),
    )
    async def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> list[RerankedHit]:
        """Rerank using Cohere API."""
        if not hits:
            return []

        start_time = time.perf_counter()

        # Prepare documents for reranking
        documents = [hit.content for hit in hits]

        try:
            response = await self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(hits)),
                return_documents=False,
            )
        except Exception as e:
            logger.error("cohere_rerank_failed", error=str(e))
            raise

        elapsed_seconds = time.perf_counter() - start_time
        elapsed_ms = elapsed_seconds * 1000

        # Map results back to VectorHit objects
        reranked = []
        for result in response.results:
            original_idx = result.index
            reranked.append(RerankedHit(
                hit=hits[original_idx],
                rerank_score=result.relevance_score,
                original_rank=original_idx,
            ))

        # Record metrics
        if tenant_id is not None:
            record_retrieval_latency(
                strategy=strategy,
                phase="rerank",
                tenant_id=tenant_id,
                duration_seconds=elapsed_seconds,
            )
            # Record improvement ratio if we have results
            if reranked and hits:
                pre_score = hits[0].similarity  # Best original score
                post_score = reranked[0].rerank_score  # Best reranked score
                record_reranking_improvement(
                    tenant_id=tenant_id,
                    pre_score=pre_score,
                    post_score=post_score,
                )

        logger.info(
            "cohere_rerank_complete",
            input_count=len(hits),
            output_count=len(reranked),
            latency_ms=round(elapsed_ms, 2),
            model=self._model,
        )

        return reranked

    def get_model(self) -> str:
        return self._model


class FlashRankRerankerClient(RerankerClient):
    """FlashRank local reranker client.

    Uses CPU-optimized models for cost-effective local reranking.
    No API costs, good for cost-sensitive deployments.

    Story 19-G3: Supports lazy loading or eager preloading of the model.
    """

    def __init__(
        self,
        model: str = "ms-marco-MiniLM-L-12-v2",
        preload: bool = False,
    ) -> None:
        """Initialize the FlashRank reranker client.

        Args:
            model: Model name to use for reranking
            preload: If True, load the model immediately (Story 19-G3)
        """
        self._model = model
        self._ranker = None
        self._preload = preload
        self._model_loaded = False

        if preload:
            self._ensure_model_loaded()
        else:
            logger.info("flashrank_reranker_created", model=model, preload=False)

    def _ensure_model_loaded(self) -> None:
        """Lazily load the FlashRank model.

        Story 19-G3: Model loading happens on first use unless preload=True.
        """
        if self._model_loaded:
            return

        start_time = time.perf_counter()
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=self._model)
            self._model_loaded = True
            load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "flashrank_model_loaded",
                model=self._model,
                load_time_ms=round(load_time_ms, 2),
                preloaded=self._preload,
            )
        except ImportError:
            raise RuntimeError(
                "flashrank package required for FlashRank reranking. "
                "Install with: uv add flashrank"
            )

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded (for health checks).

        Story 19-G3: Health checks can wait for model load completion.
        """
        return self._model_loaded

    async def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int = 10,
        tenant_id: Optional[str] = None,
        strategy: str = "hybrid",
    ) -> list[RerankedHit]:
        """Rerank using local FlashRank model."""
        if not hits:
            return []

        # Story 19-G3: Ensure model is loaded (lazy loading on first use)
        self._ensure_model_loaded()

        start_time = time.perf_counter()

        # Prepare passages for reranking
        passages = [{"id": i, "text": hit.content} for i, hit in enumerate(hits)]

        try:
            # FlashRank is synchronous, run in thread
            import asyncio
            results = await asyncio.to_thread(
                self._ranker.rerank,
                query,
                passages,
            )
        except Exception as e:
            logger.error("flashrank_rerank_failed", error=str(e))
            raise

        elapsed_seconds = time.perf_counter() - start_time
        elapsed_ms = elapsed_seconds * 1000

        # Map results back to VectorHit objects
        reranked = []
        for result in results[:top_k]:
            original_idx = result["id"]
            reranked.append(RerankedHit(
                hit=hits[original_idx],
                rerank_score=result["score"],
                original_rank=original_idx,
            ))

        # Record metrics
        if tenant_id is not None:
            record_retrieval_latency(
                strategy=strategy,
                phase="rerank",
                tenant_id=tenant_id,
                duration_seconds=elapsed_seconds,
            )
            # Record improvement ratio if we have results
            if reranked and hits:
                pre_score = hits[0].similarity  # Best original score
                post_score = reranked[0].rerank_score  # Best reranked score
                record_reranking_improvement(
                    tenant_id=tenant_id,
                    pre_score=pre_score,
                    post_score=post_score,
                )

        logger.info(
            "flashrank_rerank_complete",
            input_count=len(hits),
            output_count=len(reranked),
            latency_ms=round(elapsed_ms, 2),
            model=self._model,
        )

        return reranked

    def get_model(self) -> str:
        return self._model


def create_reranker_client(adapter: RerankerProviderAdapter) -> RerankerClient:
    """Factory function to create the appropriate reranker client.

    Args:
        adapter: Reranker provider adapter with config

    Returns:
        Configured RerankerClient instance

    Raises:
        ValueError: If provider is not supported
    """
    if adapter.provider == RerankerProviderType.COHERE:
        if not adapter.api_key:
            raise ValueError("COHERE_API_KEY is required for Cohere reranking.")
        client: RerankerClient = CohereRerankerClient(
            api_key=adapter.api_key,
            model=adapter.model,
        )
    elif adapter.provider == RerankerProviderType.FLASHRANK:
        # Story 19-G3: Pass preload flag to FlashRank client
        client = FlashRankRerankerClient(
            model=adapter.model,
            preload=adapter.preload,
        )
    else:
        raise ValueError(f"Unsupported reranker provider: {adapter.provider}")

    cache = get_reranker_cache()
    if cache and cache.enabled:
        return CachedRerankerClient(client, cache)
    return client


def get_reranker_adapter(settings: Settings) -> RerankerProviderAdapter:
    """Create reranker adapter from settings.

    Args:
        settings: Application settings

    Returns:
        RerankerProviderAdapter configured from settings
    """
    try:
        provider = RerankerProviderType(settings.reranker_provider)
    except ValueError:
        raise ValueError(
            f"RERANKER_PROVIDER must be cohere or flashrank. "
            f"Got {settings.reranker_provider!r}."
        )

    return RerankerProviderAdapter(
        provider=provider,
        api_key=settings.cohere_api_key if provider == RerankerProviderType.COHERE else None,
        model=settings.reranker_model,
        top_k=settings.reranker_top_k,
        # Story 19-G3: Model preloading from settings
        preload=settings.reranker_preload_model,
    )


# =============================================================================
# Story 19-G1: Reranking Cache Functions
# =============================================================================


def init_reranker_cache(settings: Settings) -> RerankerCache:
    """Initialize the global reranker cache from settings.

    Args:
        settings: Application settings

    Returns:
        Configured RerankerCache instance
    """
    global _reranker_cache
    _reranker_cache = RerankerCache(
        enabled=settings.reranker_cache_enabled,
        ttl_seconds=settings.reranker_cache_ttl_seconds,
        max_size=settings.reranker_cache_max_size,
    )
    return _reranker_cache


def get_reranker_cache() -> Optional[RerankerCache]:
    """Get the global reranker cache instance.

    Returns:
        The global RerankerCache instance, or None if not initialized
    """
    return _reranker_cache
