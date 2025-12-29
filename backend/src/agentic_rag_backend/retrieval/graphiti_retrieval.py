"""Graphiti-based hybrid retrieval service.

Provides hybrid search capabilities using Graphiti's integrated
semantic search, BM25, and graph traversal for knowledge retrieval.
"""

import time
from dataclasses import dataclass
from typing import Optional, Any

import structlog

from ..config import DEFAULT_SEARCH_RESULTS
from ..core.errors import Neo4jError
from ..db.graphiti import GraphitiClient

logger = structlog.get_logger(__name__)


@dataclass
class SearchNode:
    """A node returned from Graphiti search."""

    uuid: str
    name: str
    summary: str
    labels: list[str]

    @classmethod
    def from_graphiti_node(cls, node: Any) -> "SearchNode":
        """Create SearchNode from Graphiti node object."""
        return cls(
            uuid=str(getattr(node, "uuid", "")),
            name=getattr(node, "name", ""),
            summary=getattr(node, "summary", ""),
            labels=list(getattr(node, "labels", [])),
        )


@dataclass
class SearchEdge:
    """An edge/relationship returned from Graphiti search."""

    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    name: str
    fact: str

    @classmethod
    def from_graphiti_edge(cls, edge: Any) -> "SearchEdge":
        """Create SearchEdge from Graphiti edge object."""
        return cls(
            uuid=str(getattr(edge, "uuid", "")),
            source_node_uuid=str(getattr(edge, "source_node_uuid", "")),
            target_node_uuid=str(getattr(edge, "target_node_uuid", "")),
            name=getattr(edge, "name", ""),
            fact=getattr(edge, "fact", ""),
        )


@dataclass
class GraphitiSearchResult:
    """Result of Graphiti hybrid search."""

    query: str
    tenant_id: str
    nodes: list[SearchNode]
    edges: list[SearchEdge]
    processing_time_ms: int


async def graphiti_search(
    graphiti_client: GraphitiClient,
    query: str,
    tenant_id: str,
    num_results: int = DEFAULT_SEARCH_RESULTS,
    center_node_uuid: Optional[str] = None,
) -> GraphitiSearchResult:
    """
    Execute hybrid search using Graphiti.

    This function:
    1. Validates the client connection
    2. Executes Graphiti's hybrid search (semantic + BM25 + graph)
    3. Returns structured search results

    Args:
        graphiti_client: Connected GraphitiClient instance
        query: Search query string
        tenant_id: Tenant ID for multi-tenancy filtering
        num_results: Maximum number of results to return
        center_node_uuid: Optional node UUID to center graph search around

    Returns:
        GraphitiSearchResult with nodes, edges, and timing info

    Raises:
        RuntimeError: If Graphiti client is not connected
    """
    start_time = time.perf_counter()

    # Validate client connection
    if not graphiti_client.is_connected:
        raise Neo4jError("graphiti_search", "Graphiti client is not connected")

    logger.info(
        "graphiti_search_started",
        query=query[:100],
        tenant_id=tenant_id,
        num_results=num_results,
    )

    try:
        # Execute Graphiti hybrid search
        search_result = await graphiti_client.client.search(
            query=query,
            group_ids=[tenant_id],  # Multi-tenancy via group_ids
            num_results=num_results,
            center_node_uuid=center_node_uuid,
        )

        # Extract nodes
        nodes = [
            SearchNode.from_graphiti_node(node)
            for node in getattr(search_result, "nodes", [])
        ]

        # Extract edges
        edges = [
            SearchEdge.from_graphiti_edge(edge)
            for edge in getattr(search_result, "edges", [])
        ]

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = GraphitiSearchResult(
            query=query,
            tenant_id=tenant_id,
            nodes=nodes,
            edges=edges,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "graphiti_search_completed",
            query=query[:100],
            nodes_found=len(nodes),
            edges_found=len(edges),
            processing_time_ms=processing_time_ms,
        )

        return result

    except ConnectionError as e:
        logger.error(
            "graphiti_search_connection_failed",
            query=query[:100],
            error=str(e),
        )
        raise Neo4jError("graphiti_search", f"Connection failed: {e}") from e
    except Exception as e:
        logger.error(
            "graphiti_search_failed",
            query=query[:100],
            error=str(e),
        )
        raise


async def search_with_backend_routing(
    query: str,
    tenant_id: str,
    graphiti_client: Optional[GraphitiClient],
    legacy_retriever: Any,  # Legacy retriever type
    retrieval_backend: str,
    num_results: int = DEFAULT_SEARCH_RESULTS,
) -> GraphitiSearchResult:
    """
    Route search to appropriate backend based on feature flag.

    Args:
        query: Search query string
        tenant_id: Tenant ID for multi-tenancy
        graphiti_client: GraphitiClient for Graphiti backend
        legacy_retriever: Legacy retrieval service
        retrieval_backend: "graphiti" or "legacy"
        num_results: Maximum number of results

    Returns:
        GraphitiSearchResult (or equivalent for legacy)

    Raises:
        ValueError: If invalid backend specified
        RuntimeError: If required client not available
    """
    if retrieval_backend == "graphiti":
        if graphiti_client is None or not graphiti_client.is_connected:
            raise Neo4jError(
                "search_with_backend_routing",
                "Graphiti client not available but graphiti backend selected"
            )
        return await graphiti_search(
            graphiti_client=graphiti_client,
            query=query,
            tenant_id=tenant_id,
            num_results=num_results,
        )

    elif retrieval_backend == "legacy":
        if legacy_retriever is None:
            raise ValueError(
                "Legacy retriever not available but legacy backend selected"
            )

        # Use legacy retrieval service
        start_time = time.perf_counter()
        result = await legacy_retriever.search(
            query=query,
            tenant_id=tenant_id,
            limit=num_results,
        )

        # Convert to GraphitiSearchResult for consistency
        nodes = [
            SearchNode(
                uuid=str(node.get("id", "")),
                name=node.get("name", ""),
                summary=node.get("summary", ""),
                labels=node.get("labels", []),
            )
            for node in result.get("nodes", [])
        ]
        edges = [
            SearchEdge(
                uuid=str(edge.get("id", "")),
                source_node_uuid=str(edge.get("source", "")),
                target_node_uuid=str(edge.get("target", "")),
                name=edge.get("type", ""),
                fact=edge.get("fact", ""),
            )
            for edge in result.get("edges", [])
        ]

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        return GraphitiSearchResult(
            query=query,
            tenant_id=tenant_id,
            nodes=nodes,
            edges=edges,
            processing_time_ms=processing_time_ms,
        )

    else:
        raise ValueError(f"Invalid retrieval backend: {retrieval_backend}")
