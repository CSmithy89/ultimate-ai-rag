"""Unified retrieval pipeline for advanced RAG workflows.

This module centralizes retrieval operations so multiple entrypoints
(Orchestrator, MCP tools, APIs) share the same logic and feature flags.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Any

import structlog

from .types import VectorHit, GraphTraversalResult
from .graph_traversal import GraphTraversalService
from .graphiti_retrieval import GraphitiSearchResult, graphiti_search
from .graph_rerankers import GraphReranker
from .reranking import RerankerClient, RerankedHit
from .small_to_big import SmallToBigAdapter
from .vector_search import VectorSearchService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..db.graphiti import GraphitiClient
else:  # pragma: no cover - optional dependency
    GraphitiClient = Any  # type: ignore

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class VectorSearchResult:
    """Vector search result with optional reranking details."""

    hits: list[VectorHit]
    original_hits: list[VectorHit]
    reranked: list[RerankedHit] | None
    reranking_applied: bool
    reranker_model: str | None


@dataclass(frozen=True)
class HybridRetrievalResult:
    """Hybrid retrieval result with vector hits and Graphiti graph context."""

    vector: VectorSearchResult
    graphiti: GraphitiSearchResult


class RetrievalPipeline:
    """Unified retrieval operations shared across entrypoints."""

    def __init__(
        self,
        vector_search: Optional[VectorSearchService],
        graph_traversal: Optional[GraphTraversalService],
        graphiti_client: Optional["GraphitiClient"] = None,
        reranker: Optional[RerankerClient] = None,
        reranker_top_k: int = 10,
        small_to_big: Optional[SmallToBigAdapter] = None,
        graph_reranker: Optional[GraphReranker] = None,
    ) -> None:
        self._vector_search = vector_search
        self._graph_traversal = graph_traversal
        self._graphiti = graphiti_client
        self._reranker = reranker
        self._reranker_top_k = reranker_top_k
        self._small_to_big = small_to_big
        self._graph_reranker = graph_reranker

    async def vector_search(
        self,
        query: str,
        tenant_id: str,
        use_reranking: bool = True,
        top_k: Optional[int] = None,
        strategy: str = "vector",
    ) -> VectorSearchResult:
        """Run vector search with optional reranking."""
        if not self._vector_search:
            logger.warning("pipeline_vector_search_unavailable")
            return VectorSearchResult(
                hits=[],
                original_hits=[],
                reranked=None,
                reranking_applied=False,
                reranker_model=None,
            )

        if self._small_to_big and self._small_to_big.enabled:
            hits = await self._run_small_to_big(query, tenant_id, top_k)
            if hits:
                result = await self._apply_reranking(
                    query=query,
                    tenant_id=tenant_id,
                    hits=hits,
                    use_reranking=use_reranking,
                    top_k=top_k,
                    strategy=strategy,
                )
                return await self._apply_graph_reranking(query, tenant_id, result)

        hits = await self._vector_search.search(query, tenant_id)
        result = await self._apply_reranking(
            query=query,
            tenant_id=tenant_id,
            hits=hits,
            use_reranking=use_reranking,
            top_k=top_k,
            strategy=strategy,
        )
        return await self._apply_graph_reranking(query, tenant_id, result)

    async def _apply_reranking(
        self,
        query: str,
        tenant_id: str,
        hits: list[VectorHit],
        use_reranking: bool,
        top_k: Optional[int],
        strategy: str,
    ) -> VectorSearchResult:
        original_hits = list(hits)

        if self._reranker and use_reranking and hits:
            try:
                reranked = await self._reranker.rerank(
                    query=query,
                    hits=hits,
                    top_k=top_k if top_k is not None else self._reranker_top_k,
                    tenant_id=tenant_id,
                    strategy=strategy,
                )
                hits = [r.hit for r in reranked]
                return VectorSearchResult(
                    hits=hits,
                    original_hits=original_hits,
                    reranked=reranked,
                    reranking_applied=True,
                    reranker_model=self._reranker.get_model(),
                )
            except Exception as exc:  # pragma: no cover - best-effort fallback
                logger.warning("pipeline_reranking_failed", error=str(exc))

        return VectorSearchResult(
            hits=hits,
            original_hits=original_hits,
            reranked=None,
            reranking_applied=False,
            reranker_model=self._reranker.get_model() if self._reranker else None,
        )

    async def _run_small_to_big(
        self,
        query: str,
        tenant_id: str,
        top_k: Optional[int],
    ) -> list[VectorHit]:
        if not self._vector_search:
            return []
        try:
            embedding = await self._vector_search.generate_embedding(query, tenant_id)
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.warning("pipeline_small_to_big_embedding_failed", error=str(exc))
            return []

        effective_limit = top_k
        if effective_limit is None:
            effective_limit = getattr(self._vector_search, "limit", self._reranker_top_k)

        try:
            result = await self._small_to_big.retrieve(
                query=query,
                tenant_id=tenant_id,
                embedding=embedding,
                top_k=effective_limit,
            ) if self._small_to_big else None
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.warning("pipeline_small_to_big_failed", error=str(exc))
            return []

        if not result or not result.results:
            return []

        hits: list[VectorHit] = []
        for item in result.results:
            metadata = dict(item.metadata or {})
            metadata.update({
                "hierarchy_level": item.level,
                "matched_child_ids": item.matched_child_ids,
                "matched_scores": item.matched_scores,
                "retrieval": "small_to_big",
            })
            hits.append(
                VectorHit(
                    chunk_id=item.id,
                    document_id=item.document_id,
                    content=item.content,
                    similarity=item.score,
                    metadata=metadata,
                )
            )
        return hits

    async def _apply_graph_reranking(
        self,
        query: str,
        tenant_id: str,
        result: VectorSearchResult,
    ) -> VectorSearchResult:
        if not self._graph_reranker or not result.hits:
            return result

        payloads = []
        for hit in result.hits:
            payloads.append({
                "id": hit.chunk_id,
                "document_id": hit.document_id,
                "content": hit.content,
                "score": hit.similarity,
                "metadata": hit.metadata or {},
            })

        try:
            reranked = await self._graph_reranker.rerank(
                query=query,
                results=payloads,
                tenant_id=tenant_id,
            )
        except Exception as exc:  # pragma: no cover - best-effort fallback
            logger.warning("pipeline_graph_rerank_failed", error=str(exc))
            return result

        hits: list[VectorHit] = []
        for item in reranked:
            original = item.original_result
            metadata = dict(original.get("metadata") or {})
            metadata.update({
                "graph_context": item.graph_context.to_dict(),
                "graph_score": item.graph_score,
                "original_score": item.original_score,
            })
            hits.append(
                VectorHit(
                    chunk_id=str(original.get("id", "")),
                    document_id=str(original.get("document_id", "")),
                    content=original.get("content", ""),
                    similarity=item.combined_score,
                    metadata=metadata,
                )
            )

        return VectorSearchResult(
            hits=hits,
            original_hits=result.original_hits,
            reranked=result.reranked,
            reranking_applied=result.reranking_applied,
            reranker_model=result.reranker_model,
        )

    async def graph_traversal(
        self,
        query: str,
        tenant_id: str,
    ) -> Optional[GraphTraversalResult]:
        """Run graph traversal if available."""
        if not self._graph_traversal:
            logger.warning("pipeline_graph_traversal_unavailable")
            return None
        return await self._graph_traversal.traverse(query, tenant_id)

    async def hybrid_retrieve(
        self,
        query: str,
        tenant_id: str,
        num_results: int,
        use_reranking: bool = True,
    ) -> HybridRetrievalResult:
        """Run vector + Graphiti hybrid retrieval in parallel."""
        if not self._graphiti or not getattr(self._graphiti, "is_connected", False):
            raise RuntimeError("Graphiti client not connected")

        vector_task = self.vector_search(
            query=query,
            tenant_id=tenant_id,
            use_reranking=use_reranking,
            top_k=num_results,
            strategy="hybrid",
        )
        graph_task = graphiti_search(
            graphiti_client=self._graphiti,
            query=query,
            tenant_id=tenant_id,
            num_results=num_results,
        )
        vector_result, graph_result = await asyncio.gather(vector_task, graph_task)
        return HybridRetrievalResult(vector=vector_result, graphiti=graph_result)
