"""Knowledge Graph API endpoints for visualization."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, Query, Request

from agentic_rag_backend.core.errors import Neo4jError
from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.models.graphs import (
    GraphData,
    GraphEdge,
    GraphNode,
    GraphStats,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


def success_response(data: Any) -> dict[str, Any]:
    """
    Wrap data in standard success response format.

    Args:
        data: Response data

    Returns:
        Dictionary with data and meta fields
    """
    return {
        "data": data,
        "meta": {
            "requestId": str(uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
    }


async def get_neo4j(request: Request) -> Neo4jClient:
    """Get Neo4j client from app.state."""
    return request.app.state.neo4j


@router.get(
    "/graph",
    summary="Get knowledge graph data",
    description="Retrieve nodes and edges for graph visualization with optional filtering.",
)
async def get_graph(
    tenant_id: UUID = Query(..., description="Tenant identifier (required)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum nodes to return"),
    offset: int = Query(0, ge=0, description="Number of nodes to skip"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type"),
    relationship_type: Optional[str] = Query(None, description="Filter by relationship type"),
    neo4j: Neo4jClient = Depends(get_neo4j),
) -> dict[str, Any]:
    """
    Get knowledge graph data for visualization.

    Returns nodes with their properties and edges representing relationships.
    All data is filtered by tenant_id for multi-tenancy.

    Args:
        tenant_id: Tenant identifier (required for multi-tenancy)
        limit: Maximum number of nodes to return (default 100)
        offset: Number of nodes to skip for pagination
        entity_type: Optional filter by entity type (Person, Organization, etc.)
        relationship_type: Optional filter by relationship type (USES, MENTIONS, etc.)
        neo4j: Neo4j client dependency

    Returns:
        Success response with nodes and edges data
    """
    logger.info(
        "getting_graph_data",
        tenant_id=str(tenant_id),
        limit=limit,
        offset=offset,
        entity_type=entity_type,
        relationship_type=relationship_type,
    )

    try:
        graph_data = await neo4j.get_graph_data(
            tenant_id=str(tenant_id),
            limit=limit,
            offset=offset,
            entity_type=entity_type,
            relationship_type=relationship_type,
        )

        # Convert to Pydantic models for validation
        nodes = [GraphNode(**node) for node in graph_data["nodes"]]
        edges = [GraphEdge(**edge) for edge in graph_data["edges"]]
        
        validated_data = GraphData(nodes=nodes, edges=edges)

        logger.info(
            "graph_data_retrieved",
            tenant_id=str(tenant_id),
            node_count=len(nodes),
            edge_count=len(edges),
        )

        return success_response(validated_data.model_dump())

    except Neo4jError:
        raise
    except Exception as e:
        logger.error(
            "get_graph_failed",
            tenant_id=str(tenant_id),
            error=str(e),
        )
        raise Neo4jError("get_graph", str(e)) from e


@router.get(
    "/stats",
    summary="Get knowledge graph statistics",
    description="Retrieve statistics about the knowledge graph including node/edge counts.",
)
async def get_stats(
    tenant_id: UUID = Query(..., description="Tenant identifier (required)"),
    neo4j: Neo4jClient = Depends(get_neo4j),
) -> dict[str, Any]:
    """
    Get knowledge graph statistics.

    Returns counts of nodes, edges, orphan nodes, and breakdowns by type.

    Args:
        tenant_id: Tenant identifier (required for multi-tenancy)
        neo4j: Neo4j client dependency

    Returns:
        Success response with graph statistics
    """
    logger.info(
        "getting_graph_stats",
        tenant_id=str(tenant_id),
    )

    try:
        stats_data = await neo4j.get_visualization_stats(
            tenant_id=str(tenant_id),
        )

        validated_stats = GraphStats(**stats_data)

        logger.info(
            "graph_stats_retrieved",
            tenant_id=str(tenant_id),
            node_count=validated_stats.node_count,
            edge_count=validated_stats.edge_count,
            orphan_count=validated_stats.orphan_count,
        )

        return success_response(validated_stats.model_dump())

    except Neo4jError:
        raise
    except Exception as e:
        logger.error(
            "get_stats_failed",
            tenant_id=str(tenant_id),
            error=str(e),
        )
        raise Neo4jError("get_stats", str(e)) from e


@router.get(
    "/orphans",
    summary="Get orphan nodes",
    description="Retrieve nodes that have no relationships (orphans).",
)
async def get_orphans(
    tenant_id: UUID = Query(..., description="Tenant identifier (required)"),
    limit: int = Query(50, ge=1, le=500, description="Maximum orphan nodes to return"),
    neo4j: Neo4jClient = Depends(get_neo4j),
) -> dict[str, Any]:
    """
    Get orphan nodes (nodes with no relationships).

    Orphan nodes indicate potential data quality issues or incomplete
    entity extraction.

    Args:
        tenant_id: Tenant identifier (required for multi-tenancy)
        limit: Maximum number of orphan nodes to return
        neo4j: Neo4j client dependency

    Returns:
        Success response with list of orphan nodes
    """
    logger.info(
        "getting_orphan_nodes",
        tenant_id=str(tenant_id),
        limit=limit,
    )

    try:
        orphan_data = await neo4j.get_orphan_nodes(
            tenant_id=str(tenant_id),
            limit=limit,
        )

        # Convert to Pydantic models for validation
        orphan_nodes = [GraphNode(**node) for node in orphan_data]

        logger.info(
            "orphan_nodes_retrieved",
            tenant_id=str(tenant_id),
            count=len(orphan_nodes),
        )

        return success_response({
            "orphans": [node.model_dump() for node in orphan_nodes],
            "count": len(orphan_nodes),
        })

    except Neo4jError:
        raise
    except Exception as e:
        logger.error(
            "get_orphans_failed",
            tenant_id=str(tenant_id),
            error=str(e),
        )
        raise Neo4jError("get_orphans", str(e)) from e
