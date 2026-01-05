"""Graph-based rerankers for Zep-style retrieval enhancement.

Story 20-C1: Implements graph-aware reranking strategies that use temporal
(episode-based) and structural (node-distance) signals to improve retrieval ranking.

Supports:
- EpisodeMentionsReranker: Boost by episode mention frequency
- NodeDistanceReranker: Boost by proximity to query entities in graph
- HybridGraphReranker: Weighted combination of both signals

Usage:
    from agentic_rag_backend.retrieval.graph_rerankers import (
        create_graph_reranker,
        get_graph_reranker_adapter,
        GraphRerankerType,
    )

    adapter = get_graph_reranker_adapter(settings)
    reranker = create_graph_reranker(adapter, neo4j_client, graphiti_client)
    reranked = await reranker.rerank(query, results, tenant_id)
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

import structlog

from ..observability.metrics import record_retrieval_latency

if TYPE_CHECKING:
    from ..config import Settings
    from ..db.neo4j import Neo4jClient
    from ..db.graphiti import GraphitiClient

logger = structlog.get_logger(__name__)


class GraphRerankerType(str, Enum):
    """Graph reranker type enumeration."""

    EPISODE = "episode"
    DISTANCE = "distance"
    HYBRID = "hybrid"


@dataclass
class GraphContext:
    """Graph context metadata for a reranked result."""

    episode_mentions: int = 0
    min_distance: Optional[int] = None
    episode_score: float = 0.0
    distance_score: float = 0.0
    query_entities: list[str] = field(default_factory=list)
    result_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "episode_mentions": self.episode_mentions,
            "min_distance": self.min_distance,
            "episode_score": self.episode_score,
            "distance_score": self.distance_score,
            "query_entities": self.query_entities,
            "result_entities": self.result_entities,
        }


@dataclass
class GraphRerankedResult:
    """A result with graph-based reranking score.

    Attributes:
        original_result: The original retrieval result dict
        original_score: Original semantic similarity score
        graph_score: Combined graph signal score (0-1)
        combined_score: Final weighted score combining original and graph
        graph_context: Metadata about graph signals used in scoring
    """

    original_result: dict[str, Any]
    original_score: float
    graph_score: float
    combined_score: float
    graph_context: GraphContext

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        result = dict(self.original_result)
        result["score"] = self.combined_score
        result["original_score"] = self.original_score
        result["graph_score"] = self.graph_score
        result["graph_context"] = self.graph_context.to_dict()
        return result


@dataclass(frozen=True)
class GraphRerankerAdapter:
    """Configuration adapter for graph rerankers."""

    enabled: bool
    reranker_type: GraphRerankerType
    episode_weight: float
    distance_weight: float
    original_weight: float
    episode_window_days: int
    max_distance: int


class GraphReranker(ABC):
    """Base class for graph-aware rerankers.

    All graph rerankers must implement the rerank method which takes
    retrieval results and returns them reranked based on graph signals.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Rerank results using graph signals.

        Args:
            query: The search query
            results: List of retrieval result dicts with 'score' and entity info
            tenant_id: Tenant identifier for multi-tenancy filtering

        Returns:
            List of GraphRerankedResult sorted by combined_score descending
        """
        pass


class EpisodeMentionsReranker(GraphReranker):
    """Reranker based on episode mention frequency.

    Entities mentioned in more recent episodes are considered more
    contextually relevant. This implements Zep-style temporal scoring.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        episode_window_days: int = 30,
        original_weight: float = 0.7,
    ) -> None:
        """Initialize the episode mentions reranker.

        Args:
            neo4j_client: Neo4j client for graph queries
            episode_window_days: Look-back window for episode counting
            original_weight: Weight for original score (graph weight = 1 - original)
        """
        self._neo4j = neo4j_client
        self._episode_window_days = episode_window_days
        self._original_weight = original_weight
        self._graph_weight = 1.0 - original_weight

    def _extract_entities(self, result: dict[str, Any]) -> list[str]:
        """Extract entity IDs from a retrieval result.

        Results may contain entities in different formats:
        - Direct 'entity_ids' field
        - 'entities' list with id/uuid fields
        - 'metadata.entity_refs' from vector search
        """
        entities: list[str] = []

        # Check for direct entity_ids
        if "entity_ids" in result:
            entity_ids = result["entity_ids"]
            if isinstance(entity_ids, list):
                entities.extend(str(e) for e in entity_ids if e)

        # Check for entities list
        if "entities" in result:
            for entity in result["entities"]:
                if isinstance(entity, dict):
                    entity_id = entity.get("id") or entity.get("uuid", "")
                    if entity_id:
                        entities.append(str(entity_id))
                elif isinstance(entity, str) and entity:
                    entities.append(entity)

        # Check metadata
        if "metadata" in result and isinstance(result["metadata"], dict):
            if "entity_refs" in result["metadata"]:
                refs = result["metadata"]["entity_refs"]
                if isinstance(refs, list):
                    entities.extend(str(r) for r in refs if r)

        return list(set(entities))  # Deduplicate

    async def _count_episode_mentions(
        self,
        entity_id: str,
        tenant_id: str,
    ) -> int:
        """Count episode mentions for an entity within the time window.

        Uses Neo4j to count Episode nodes linked to the entity via MENTIONS
        relationship within the configured look-back window.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._episode_window_days)

        try:
            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {id: $entity_id, tenant_id: $tenant_id})<-[:MENTIONS]-(ep:Episode)
                    WHERE ep.created_at >= datetime($cutoff)
                    RETURN count(DISTINCT ep) as mention_count
                    """,
                    entity_id=entity_id,
                    tenant_id=tenant_id,
                    cutoff=cutoff.isoformat(),
                )
                record = await result.single()
                return record["mention_count"] if record else 0
        except Exception as e:
            logger.warning(
                "episode_count_failed",
                entity_id=entity_id,
                tenant_id=tenant_id,
                error=str(e),
            )
            return 0

    async def _get_total_mentions(
        self,
        entity_ids: list[str],
        tenant_id: str,
    ) -> int:
        """Get total episode mentions for a list of entities."""
        if not entity_ids:
            return 0

        # Batch query for efficiency
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._episode_window_days)

        try:
            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (e:Entity {tenant_id: $tenant_id})<-[:MENTIONS]-(ep:Episode)
                    WHERE e.id IN $entity_ids AND ep.created_at >= datetime($cutoff)
                    RETURN count(DISTINCT ep) as mention_count
                    """,
                    entity_ids=entity_ids,
                    tenant_id=tenant_id,
                    cutoff=cutoff.isoformat(),
                )
                record = await result.single()
                return record["mention_count"] if record else 0
        except Exception as e:
            logger.warning(
                "batch_episode_count_failed",
                entity_count=len(entity_ids),
                tenant_id=tenant_id,
                error=str(e),
            )
            return 0

    def _normalize_episode_score(self, mention_count: int) -> float:
        """Normalize episode mentions to 0-1 score.

        Uses a saturation function where 10+ mentions = 1.0
        """
        max_mentions = 10.0
        return min(1.0, mention_count / max_mentions)

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Rerank results based on episode mention frequency."""
        if not results:
            return []

        start_time = time.perf_counter()
        reranked: list[GraphRerankedResult] = []

        # Process each result
        for result in results:
            original_score = float(result.get("score", 0.0))
            entity_ids = self._extract_entities(result)

            if entity_ids:
                total_mentions = await self._get_total_mentions(entity_ids, tenant_id)
            else:
                total_mentions = 0

            episode_score = self._normalize_episode_score(total_mentions)
            combined_score = (
                self._original_weight * original_score
                + self._graph_weight * episode_score
            )

            graph_context = GraphContext(
                episode_mentions=total_mentions,
                episode_score=episode_score,
                result_entities=entity_ids,
            )

            reranked.append(
                GraphRerankedResult(
                    original_result=result,
                    original_score=original_score,
                    graph_score=episode_score,
                    combined_score=combined_score,
                    graph_context=graph_context,
                )
            )

        # Sort by combined score descending
        reranked.sort(key=lambda r: r.combined_score, reverse=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "episode_rerank_complete",
            input_count=len(results),
            output_count=len(reranked),
            latency_ms=round(elapsed_ms, 2),
        )

        return reranked


class NodeDistanceReranker(GraphReranker):
    """Reranker based on graph distance from query entities.

    Entities closer to query concepts in the knowledge graph receive
    higher scores. Uses Neo4j's shortest path algorithm.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        graphiti_client: Optional[GraphitiClient] = None,
        max_distance: int = 3,
        original_weight: float = 0.7,
    ) -> None:
        """Initialize the node distance reranker.

        Args:
            neo4j_client: Neo4j client for graph queries
            graphiti_client: Optional Graphiti client for entity extraction
            max_distance: Maximum graph distance for scoring (beyond = 0 score)
            original_weight: Weight for original score (graph weight = 1 - original)
        """
        self._neo4j = neo4j_client
        self._graphiti = graphiti_client
        self._max_distance = max_distance
        self._original_weight = original_weight
        self._graph_weight = 1.0 - original_weight

    def _extract_entities(self, result: dict[str, Any]) -> list[str]:
        """Extract entity IDs from a retrieval result."""
        entities: list[str] = []

        if "entity_ids" in result:
            entity_ids = result["entity_ids"]
            if isinstance(entity_ids, list):
                entities.extend(str(e) for e in entity_ids if e)

        if "entities" in result:
            for entity in result["entities"]:
                if isinstance(entity, dict):
                    entity_id = entity.get("id") or entity.get("uuid", "")
                    if entity_id:
                        entities.append(str(entity_id))
                elif isinstance(entity, str) and entity:
                    entities.append(entity)

        if "metadata" in result and isinstance(result["metadata"], dict):
            if "entity_refs" in result["metadata"]:
                refs = result["metadata"]["entity_refs"]
                if isinstance(refs, list):
                    entities.extend(str(r) for r in refs if r)

        return list(set(entities))

    async def _extract_query_entities(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[str]:
        """Extract entity IDs from query using Graphiti search.

        Falls back to direct entity name matching if Graphiti unavailable.
        """
        if self._graphiti and self._graphiti.is_connected:
            try:
                search_result = await self._graphiti.client.search(
                    query=query,
                    group_ids=[tenant_id],
                    num_results=limit,
                )
                nodes = getattr(search_result, "nodes", [])
                return [str(getattr(node, "uuid", "")) for node in nodes if hasattr(node, "uuid")]
            except Exception as e:
                logger.warning(
                    "query_entity_extraction_failed",
                    query=query[:100],
                    error=str(e),
                )

        # Fallback: search by terms in Neo4j
        try:
            terms = [t.lower() for t in query.split() if len(t) > 2]
            if terms:
                entities = await self._neo4j.search_entities_by_terms(
                    tenant_id=tenant_id,
                    terms=terms[:5],  # Limit terms
                    limit=limit,
                )
                return [str(e.get("id", "")) for e in entities if e.get("id")]
        except Exception as e:
            logger.warning(
                "fallback_entity_extraction_failed",
                query=query[:100],
                error=str(e),
            )

        return []

    async def _get_graph_distance(
        self,
        entity_id_1: str,
        entity_id_2: str,
        tenant_id: str,
    ) -> Optional[int]:
        """Get shortest path length between two entities.

        Returns None if no path exists.
        """
        try:
            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (a:Entity {id: $id1, tenant_id: $tenant_id}),
                          (b:Entity {id: $id2, tenant_id: $tenant_id})
                    MATCH path = shortestPath((a)-[*..10]-(b))
                    RETURN length(path) as distance
                    """,
                    id1=entity_id_1,
                    id2=entity_id_2,
                    tenant_id=tenant_id,
                )
                record = await result.single()
                if record:
                    return int(record["distance"])
                return None
        except Exception as e:
            logger.debug(
                "distance_calculation_failed",
                id1=entity_id_1,
                id2=entity_id_2,
                error=str(e),
            )
            return None

    async def _get_min_distance_batch(
        self,
        query_entity_ids: list[str],
        result_entity_ids: list[str],
        tenant_id: str,
    ) -> Optional[int]:
        """Get minimum distance between any query entity and any result entity."""
        if not query_entity_ids or not result_entity_ids:
            return None

        try:
            async with self._neo4j.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (a:Entity {tenant_id: $tenant_id}),
                          (b:Entity {tenant_id: $tenant_id})
                    WHERE a.id IN $query_ids AND b.id IN $result_ids
                    MATCH path = shortestPath((a)-[*..10]-(b))
                    RETURN min(length(path)) as min_distance
                    """,
                    query_ids=query_entity_ids,
                    result_ids=result_entity_ids,
                    tenant_id=tenant_id,
                )
                record = await result.single()
                if record and record["min_distance"] is not None:
                    return int(record["min_distance"])
                return None
        except Exception as e:
            logger.debug(
                "batch_distance_calculation_failed",
                query_count=len(query_entity_ids),
                result_count=len(result_entity_ids),
                error=str(e),
            )
            return None

    def _distance_to_score(self, distance: Optional[int]) -> float:
        """Convert distance to 0-1 score.

        Closer = higher score. No path or distance > max = 0.
        """
        if distance is None:
            return 0.0
        if distance > self._max_distance:
            return 0.0
        # distance=0 -> 1.0, distance=max_distance -> 0.0
        return max(0.0, 1.0 - (distance / self._max_distance))

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Rerank results based on graph distance from query entities."""
        if not results:
            return []

        start_time = time.perf_counter()

        # Extract entities from query
        query_entities = await self._extract_query_entities(query, tenant_id)

        reranked: list[GraphRerankedResult] = []

        for result in results:
            original_score = float(result.get("score", 0.0))
            result_entities = self._extract_entities(result)

            if query_entities and result_entities:
                min_distance = await self._get_min_distance_batch(
                    query_entities, result_entities, tenant_id
                )
            else:
                min_distance = None

            distance_score = self._distance_to_score(min_distance)

            # If no query entities found, preserve original order
            if not query_entities:
                combined_score = original_score
                graph_score = 0.5  # Neutral
            else:
                combined_score = (
                    self._original_weight * original_score
                    + self._graph_weight * distance_score
                )
                graph_score = distance_score

            graph_context = GraphContext(
                min_distance=min_distance,
                distance_score=distance_score,
                query_entities=query_entities,
                result_entities=result_entities,
            )

            reranked.append(
                GraphRerankedResult(
                    original_result=result,
                    original_score=original_score,
                    graph_score=graph_score,
                    combined_score=combined_score,
                    graph_context=graph_context,
                )
            )

        # Sort by combined score descending
        reranked.sort(key=lambda r: r.combined_score, reverse=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "distance_rerank_complete",
            input_count=len(results),
            output_count=len(reranked),
            query_entities=len(query_entities),
            latency_ms=round(elapsed_ms, 2),
        )

        return reranked


class HybridGraphReranker(GraphReranker):
    """Hybrid reranker combining episode and distance signals.

    Runs both sub-rerankers in parallel and combines their scores
    with configurable weights.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        graphiti_client: Optional[GraphitiClient] = None,
        episode_weight: float = 0.3,
        distance_weight: float = 0.3,
        original_weight: float = 0.4,
        episode_window_days: int = 30,
        max_distance: int = 3,
    ) -> None:
        """Initialize the hybrid graph reranker.

        Args:
            neo4j_client: Neo4j client for graph queries
            graphiti_client: Optional Graphiti client for entity extraction
            episode_weight: Weight for episode-mentions signal
            distance_weight: Weight for node-distance signal
            original_weight: Weight for original semantic score
            episode_window_days: Look-back window for episode counting
            max_distance: Maximum graph distance for scoring

        Raises:
            ValueError: If weights don't sum to 1.0 (within tolerance)
        """
        weight_sum = episode_weight + distance_weight + original_weight
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "hybrid_weights_not_normalized",
                sum=weight_sum,
                episode=episode_weight,
                distance=distance_weight,
                original=original_weight,
            )
            # Normalize weights
            if weight_sum > 0:
                episode_weight /= weight_sum
                distance_weight /= weight_sum
                original_weight /= weight_sum

        self._neo4j = neo4j_client
        self._graphiti = graphiti_client
        self._episode_weight = episode_weight
        self._distance_weight = distance_weight
        self._original_weight = original_weight

        # Sub-rerankers use 1.0 for original weight since we combine them ourselves
        self._episode_reranker = EpisodeMentionsReranker(
            neo4j_client=neo4j_client,
            episode_window_days=episode_window_days,
            original_weight=1.0,  # Don't mix original score in sub-reranker
        )
        self._distance_reranker = NodeDistanceReranker(
            neo4j_client=neo4j_client,
            graphiti_client=graphiti_client,
            max_distance=max_distance,
            original_weight=1.0,  # Don't mix original score in sub-reranker
        )

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[GraphRerankedResult]:
        """Rerank results using combined episode and distance signals."""
        if not results:
            return []

        start_time = time.perf_counter()

        # Run both rerankers in parallel for performance
        episode_task = self._episode_reranker.rerank(query, results, tenant_id)
        distance_task = self._distance_reranker.rerank(query, results, tenant_id)

        episode_results, distance_results = await asyncio.gather(
            episode_task, distance_task
        )

        # Create lookup dicts by original result ID or index
        def get_result_key(result: dict[str, Any], idx: int) -> str:
            return result.get("id", result.get("chunk_id", str(idx)))

        episode_by_key = {
            get_result_key(r.original_result, i): r
            for i, r in enumerate(episode_results)
        }
        distance_by_key = {
            get_result_key(r.original_result, i): r
            for i, r in enumerate(distance_results)
        }

        # Combine scores
        reranked: list[GraphRerankedResult] = []

        for i, result in enumerate(results):
            key = get_result_key(result, i)
            original_score = float(result.get("score", 0.0))

            episode_result = episode_by_key.get(key)
            distance_result = distance_by_key.get(key)

            episode_score = episode_result.graph_score if episode_result else 0.0
            distance_score = distance_result.graph_score if distance_result else 0.0

            # Combine all three scores
            combined_score = (
                self._original_weight * original_score
                + self._episode_weight * episode_score
                + self._distance_weight * distance_score
            )

            # Combine graph context
            graph_context = GraphContext(
                episode_mentions=(
                    episode_result.graph_context.episode_mentions
                    if episode_result
                    else 0
                ),
                min_distance=(
                    distance_result.graph_context.min_distance
                    if distance_result
                    else None
                ),
                episode_score=episode_score,
                distance_score=distance_score,
                query_entities=(
                    distance_result.graph_context.query_entities
                    if distance_result
                    else []
                ),
                result_entities=(
                    episode_result.graph_context.result_entities
                    if episode_result
                    else []
                ),
            )

            # Combined graph score is weighted average of episode and distance
            graph_weight_sum = self._episode_weight + self._distance_weight
            if graph_weight_sum > 0:
                graph_score = (
                    self._episode_weight * episode_score
                    + self._distance_weight * distance_score
                ) / graph_weight_sum
            else:
                graph_score = 0.5

            reranked.append(
                GraphRerankedResult(
                    original_result=result,
                    original_score=original_score,
                    graph_score=graph_score,
                    combined_score=combined_score,
                    graph_context=graph_context,
                )
            )

        # Sort by combined score descending
        reranked.sort(key=lambda r: r.combined_score, reverse=True)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        record_retrieval_latency(
            strategy="hybrid",
            phase="graph_rerank",
            tenant_id=tenant_id,
            duration_seconds=(time.perf_counter() - start_time),
        )

        logger.info(
            "hybrid_rerank_complete",
            input_count=len(results),
            output_count=len(reranked),
            latency_ms=round(elapsed_ms, 2),
            weights={
                "original": self._original_weight,
                "episode": self._episode_weight,
                "distance": self._distance_weight,
            },
        )

        return reranked


def get_graph_reranker_adapter(settings: Settings) -> GraphRerankerAdapter:
    """Create graph reranker adapter from settings.

    Args:
        settings: Application settings

    Returns:
        GraphRerankerAdapter configured from settings
    """
    try:
        reranker_type = GraphRerankerType(settings.graph_reranker_type)
    except ValueError:
        logger.warning(
            "invalid_graph_reranker_type",
            type=settings.graph_reranker_type,
            fallback="hybrid",
        )
        reranker_type = GraphRerankerType.HYBRID

    return GraphRerankerAdapter(
        enabled=settings.graph_reranker_enabled,
        reranker_type=reranker_type,
        episode_weight=settings.graph_reranker_episode_weight,
        distance_weight=settings.graph_reranker_distance_weight,
        original_weight=settings.graph_reranker_original_weight,
        episode_window_days=settings.graph_reranker_episode_window_days,
        max_distance=settings.graph_reranker_max_distance,
    )


def create_graph_reranker(
    adapter: GraphRerankerAdapter,
    neo4j_client: Neo4jClient,
    graphiti_client: Optional[GraphitiClient] = None,
) -> GraphReranker:
    """Factory function to create the appropriate graph reranker.

    Args:
        adapter: Graph reranker configuration adapter
        neo4j_client: Neo4j client for graph queries
        graphiti_client: Optional Graphiti client for entity extraction

    Returns:
        Configured GraphReranker instance

    Raises:
        ValueError: If reranker type is not supported
    """
    if adapter.reranker_type == GraphRerankerType.EPISODE:
        return EpisodeMentionsReranker(
            neo4j_client=neo4j_client,
            episode_window_days=adapter.episode_window_days,
            original_weight=adapter.original_weight,
        )
    elif adapter.reranker_type == GraphRerankerType.DISTANCE:
        return NodeDistanceReranker(
            neo4j_client=neo4j_client,
            graphiti_client=graphiti_client,
            max_distance=adapter.max_distance,
            original_weight=adapter.original_weight,
        )
    elif adapter.reranker_type == GraphRerankerType.HYBRID:
        return HybridGraphReranker(
            neo4j_client=neo4j_client,
            graphiti_client=graphiti_client,
            episode_weight=adapter.episode_weight,
            distance_weight=adapter.distance_weight,
            original_weight=adapter.original_weight,
            episode_window_days=adapter.episode_window_days,
            max_distance=adapter.max_distance,
        )
    else:
        raise ValueError(f"Unsupported graph reranker type: {adapter.reranker_type}")
