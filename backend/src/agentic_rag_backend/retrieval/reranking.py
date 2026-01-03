"""Cross-encoder reranking for improved retrieval precision.

Supports:
- Cohere Rerank API (rerank-v3.5, 100+ languages, 32K context)
- FlashRank (local CPU-optimized, no API cost)

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

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .types import VectorHit

logger = structlog.get_logger(__name__)


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
    ) -> list[RerankedHit]:
        """Rerank vector hits using cross-encoder scoring.

        Args:
            query: The search query
            hits: Vector hits to rerank
            top_k: Number of top results to return

        Returns:
            Reranked hits with scores, sorted by relevance
        """
        pass

    @abstractmethod
    def get_model(self) -> str:
        """Get the model name used for reranking."""
        pass


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

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Map results back to VectorHit objects
        reranked = []
        for result in response.results:
            original_idx = result.index
            reranked.append(RerankedHit(
                hit=hits[original_idx],
                rerank_score=result.relevance_score,
                original_rank=original_idx,
            ))

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
    """

    def __init__(
        self,
        model: str = "ms-marco-MiniLM-L-12-v2",
    ) -> None:
        try:
            from flashrank import Ranker
            self._ranker = Ranker(model_name=model)
        except ImportError:
            raise RuntimeError(
                "flashrank package required for FlashRank reranking. "
                "Install with: uv add flashrank"
            )
        self._model = model
        logger.info("flashrank_reranker_initialized", model=model)

    async def rerank(
        self,
        query: str,
        hits: list[VectorHit],
        top_k: int = 10,
    ) -> list[RerankedHit]:
        """Rerank using local FlashRank model."""
        if not hits:
            return []

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

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Map results back to VectorHit objects
        reranked = []
        for result in results[:top_k]:
            original_idx = result["id"]
            reranked.append(RerankedHit(
                hit=hits[original_idx],
                rerank_score=result["score"],
                original_rank=original_idx,
            ))

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
        return CohereRerankerClient(
            api_key=adapter.api_key,
            model=adapter.model,
        )
    elif adapter.provider == RerankerProviderType.FLASHRANK:
        return FlashRankRerankerClient(
            model=adapter.model,
        )
    else:
        raise ValueError(f"Unsupported reranker provider: {adapter.provider}")


def get_reranker_adapter(settings: "Settings") -> RerankerProviderAdapter:
    """Create reranker adapter from settings.

    Args:
        settings: Application settings

    Returns:
        RerankerProviderAdapter configured from settings
    """
    from ..config import Settings  # Avoid circular import at runtime

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
    )
