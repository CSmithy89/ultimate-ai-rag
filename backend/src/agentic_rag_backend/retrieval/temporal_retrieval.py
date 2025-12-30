"""Temporal query capabilities using Graphiti's bi-temporal tracking.

Provides point-in-time queries and change tracking for knowledge graph data.
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Any

import structlog

from ..config import DEFAULT_SEARCH_RESULTS
from ..core.errors import Neo4jError
from ..db.graphiti import GraphitiClient

logger = structlog.get_logger(__name__)


def _redact_query(query: str, max_length: int = 50) -> str:
    """Redact query for logging - show prefix and hash for traceability."""
    if len(query) <= max_length:
        return f"{query[:20]}..." if len(query) > 20 else query
    query_hash = hashlib.sha256(query.encode()).hexdigest()[:8]
    return f"{query[:20]}...[hash:{query_hash}]"


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize datetime to UTC for consistent comparison.
    
    Args:
        dt: Datetime to normalize (may be naive or aware)
        
    Returns:
        UTC datetime or None if input was None
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class TemporalNode:
    """A node with temporal context."""

    uuid: str
    name: str
    summary: str
    labels: list[str]

    @classmethod
    def from_graphiti_node(cls, node: Any) -> "TemporalNode":
        """Create TemporalNode from Graphiti node object."""
        return cls(
            uuid=str(getattr(node, "uuid", "")),
            name=getattr(node, "name", ""),
            summary=getattr(node, "summary", ""),
            labels=list(getattr(node, "labels", [])),
        )


@dataclass
class TemporalEdge:
    """An edge with temporal validity information."""

    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    name: str
    fact: str
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None

    @classmethod
    def from_graphiti_edge(cls, edge: Any) -> "TemporalEdge":
        """Create TemporalEdge from Graphiti edge object."""
        return cls(
            uuid=str(getattr(edge, "uuid", "")),
            source_node_uuid=str(getattr(edge, "source_node_uuid", "")),
            target_node_uuid=str(getattr(edge, "target_node_uuid", "")),
            name=getattr(edge, "name", ""),
            fact=getattr(edge, "fact", ""),
            valid_at=getattr(edge, "valid_at", None),
            invalid_at=getattr(edge, "invalid_at", None),
        )


@dataclass
class TemporalSearchResult:
    """Result of temporal search query."""

    query: str
    tenant_id: str
    as_of_date: Optional[datetime]
    nodes: list[TemporalNode]
    edges: list[TemporalEdge]
    processing_time_ms: int


@dataclass
class EpisodeChange:
    """A single episode (document ingestion) change."""

    uuid: str
    name: str
    created_at: datetime
    entities_added: int
    edges_added: int


@dataclass
class KnowledgeChangesResult:
    """Result of knowledge changes query."""

    tenant_id: str
    start_date: datetime
    end_date: datetime
    episodes: list[EpisodeChange]
    total_entities_added: int
    total_edges_added: int
    processing_time_ms: int


async def temporal_search(
    graphiti_client: GraphitiClient,
    query: str,
    tenant_id: str,
    as_of_date: Optional[datetime] = None,
    num_results: int = DEFAULT_SEARCH_RESULTS,
) -> TemporalSearchResult:
    """
    Execute temporal search - query knowledge at a specific point in time.

    This function:
    1. Validates the client connection
    2. Executes Graphiti search with temporal context
    3. Filters results by validity at as_of_date if specified

    Args:
        graphiti_client: Connected GraphitiClient instance
        query: Search query string
        tenant_id: Tenant ID for multi-tenancy
        as_of_date: Optional point-in-time for historical query
        num_results: Maximum number of results

    Returns:
        TemporalSearchResult with nodes, edges, and temporal context

    Raises:
        Neo4jError: If Graphiti client is not connected
    """
    start_time = time.perf_counter()

    # Validate client connection
    if not graphiti_client.is_connected:
        raise Neo4jError("temporal_search", "Graphiti client is not connected")

    # Default to current time if no as_of_date specified
    effective_date = as_of_date or datetime.now(timezone.utc)

    logger.info(
        "temporal_search_started",
        query_ref=_redact_query(query),
        tenant_id=tenant_id,
        as_of_date=effective_date.isoformat(),
    )

    try:
        # Execute Graphiti search
        search_result = await graphiti_client.client.search(
            query=query,
            group_ids=[tenant_id],
            num_results=num_results,
        )

        # Extract and filter nodes with tenant validation
        nodes = []
        skipped_node_tenant_mismatch = 0
        for node in getattr(search_result, "nodes", []):
            # Defense-in-depth: Validate tenant even though Graphiti filters
            node_group = getattr(node, "group_id", None)
            if node_group and node_group != tenant_id:
                skipped_node_tenant_mismatch += 1
                logger.warning(
                    "node_tenant_mismatch",
                    node_uuid=str(getattr(node, "uuid", "")),
                    expected_tenant=tenant_id,
                    actual_tenant=node_group,
                )
                continue
            nodes.append(TemporalNode.from_graphiti_node(node))
        
        if skipped_node_tenant_mismatch > 0:
            logger.error(
                "temporal_search_tenant_isolation_violation",
                tenant_id=tenant_id,
                nodes_skipped=skipped_node_tenant_mismatch,
            )

        # Extract and filter edges by temporal validity and tenant
        raw_edges = getattr(search_result, "edges", [])
        edges = []
        skipped_edge_tenant_mismatch = 0
        for edge in raw_edges:
            # Defense-in-depth: Validate tenant even though Graphiti filters
            edge_group = getattr(edge, "group_id", None)
            if edge_group and edge_group != tenant_id:
                skipped_edge_tenant_mismatch += 1
                logger.warning(
                    "edge_tenant_mismatch",
                    edge_uuid=str(getattr(edge, "uuid", "")),
                    expected_tenant=tenant_id,
                    actual_tenant=edge_group,
                )
                continue
                
            temporal_edge = TemporalEdge.from_graphiti_edge(edge)
            
            # Filter by temporal validity if as_of_date specified
            if as_of_date:
                # Normalize all datetimes to UTC for consistent comparison
                as_of_utc = _ensure_utc(as_of_date)
                if as_of_utc is None:
                    continue
                valid_at = _ensure_utc(temporal_edge.valid_at)
                invalid_at = _ensure_utc(temporal_edge.invalid_at)
                
                # Edge validity rules (explicit for maintainability):
                # 1. No valid_at means edge is eternal (valid from beginning)
                # 2. No invalid_at means edge is still valid (not invalidated)
                if valid_at is None:
                    valid_from_start = True  # Eternal edge
                else:
                    valid_from_start = valid_at <= as_of_utc
                
                if invalid_at is None:
                    valid_to_end = True  # Not yet invalidated
                else:
                    valid_to_end = invalid_at > as_of_utc
                
                is_valid = valid_from_start and valid_to_end
                    
                if is_valid:
                    edges.append(temporal_edge)
            else:
                edges.append(temporal_edge)
        
        if skipped_edge_tenant_mismatch > 0:
            logger.error(
                "temporal_search_edge_tenant_violation",
                tenant_id=tenant_id,
                edges_skipped=skipped_edge_tenant_mismatch,
            )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = TemporalSearchResult(
            query=query,
            tenant_id=tenant_id,
            as_of_date=as_of_date,
            nodes=nodes,
            edges=edges,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "temporal_search_completed",
            query_ref=_redact_query(query),
            as_of_date=effective_date.isoformat(),
            nodes_found=len(nodes),
            edges_found=len(edges),
            processing_time_ms=processing_time_ms,
        )

        return result

    except Exception as e:
        logger.error(
            "temporal_search_failed",
            query_ref=_redact_query(query),
            error=str(e),
        )
        raise


async def get_knowledge_changes(
    graphiti_client: GraphitiClient,
    tenant_id: str,
    start_date: datetime,
    end_date: datetime,
) -> KnowledgeChangesResult:
    """
    Get knowledge changes over a time period.

    This function:
    1. Retrieves episodes (document ingestions) in the date range
    2. Validates tenant isolation for each episode
    3. Aggregates changes by entities and edges added
    4. Returns timeline of knowledge evolution

    Args:
        graphiti_client: Connected GraphitiClient instance
        tenant_id: Tenant ID for multi-tenancy
        start_date: Start of time period (timezone-aware recommended)
        end_date: End of time period (timezone-aware recommended)

    Returns:
        KnowledgeChangesResult with change timeline and aggregates

    Raises:
        Neo4jError: If Graphiti client is not connected
    """
    start_time = time.perf_counter()

    # Validate client connection
    if not graphiti_client.is_connected:
        raise Neo4jError("get_knowledge_changes", "Graphiti client is not connected")

    logger.info(
        "get_knowledge_changes_started",
        tenant_id=tenant_id,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    try:
        # Get episodes for the tenant in the date range
        episodes_raw = await graphiti_client.client.get_episodes_by_group_ids(  # type: ignore[attr-defined]
            group_ids=[tenant_id],
        )

        # Filter and transform episodes with tenant validation
        episodes = []
        total_entities = 0
        total_edges = 0
        skipped_tenant_mismatch = 0

        for ep in episodes_raw:
            # Validate tenant isolation - verify episode belongs to tenant
            ep_group_id = getattr(ep, "group_id", None)
            if ep_group_id and ep_group_id != tenant_id:
                skipped_tenant_mismatch += 1
                logger.warning(
                    "episode_tenant_mismatch",
                    episode_uuid=str(getattr(ep, "uuid", "")),
                    expected_tenant=tenant_id,
                    actual_tenant=ep_group_id,
                )
                continue

            created_at = getattr(ep, "created_at", None)
            
            # Filter by date range
            if created_at:
                if created_at < start_date or created_at > end_date:
                    continue

            entity_refs = getattr(ep, "entity_references", [])
            edge_refs = getattr(ep, "edge_references", [])
            
            entities_added = len(entity_refs) if entity_refs else 0
            edges_added = len(edge_refs) if edge_refs else 0

            episode_change = EpisodeChange(
                uuid=str(getattr(ep, "uuid", "")),
                name=getattr(ep, "name", ""),
                created_at=created_at or datetime.now(timezone.utc),
                entities_added=entities_added,
                edges_added=edges_added,
            )
            episodes.append(episode_change)
            total_entities += entities_added
            total_edges += edges_added

        # Log if any tenant mismatches were found (potential security issue)
        if skipped_tenant_mismatch > 0:
            logger.error(
                "tenant_isolation_violation_detected",
                tenant_id=tenant_id,
                episodes_skipped=skipped_tenant_mismatch,
            )

        # Sort by created_at
        episodes.sort(key=lambda e: e.created_at)

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = KnowledgeChangesResult(
            tenant_id=tenant_id,
            start_date=start_date,
            end_date=end_date,
            episodes=episodes,
            total_entities_added=total_entities,
            total_edges_added=total_edges,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "get_knowledge_changes_completed",
            tenant_id=tenant_id,
            episodes_found=len(episodes),
            total_entities=total_entities,
            total_edges=total_edges,
            processing_time_ms=processing_time_ms,
        )

        return result

    except Exception as e:
        logger.error(
            "get_knowledge_changes_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise
