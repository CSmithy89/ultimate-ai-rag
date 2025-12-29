"""Knowledge Graph API endpoints for visualization."""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from agentic_rag_backend.config import DEFAULT_SEARCH_RESULTS, MAX_SEARCH_RESULTS
from agentic_rag_backend.core.errors import Neo4jError
from agentic_rag_backend.db.graphiti import GraphitiClient
from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.models.graphs import (
    GraphData,
    GraphEdge,
    GraphNode,
    GraphStats,
)
from agentic_rag_backend.retrieval.temporal_retrieval import (
    EntityTypeFilterNotImplemented,
    get_knowledge_changes,
    temporal_search,
)

logger = structlog.get_logger(__name__)

# Rate limiter instance - key function extracts client IP
limiter = Limiter(key_func=get_remote_address)

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


async def get_graphiti(request: Request) -> Optional[GraphitiClient]:
    """Get Graphiti client from app.state if available."""
    return getattr(request.app.state, "graphiti", None)


async def get_connected_graphiti(
    graphiti: Optional[GraphitiClient] = Depends(get_graphiti),
) -> GraphitiClient:
    """
    Dependency to get a connected Graphiti client, or raise an error.
    
    Raises:
        Neo4jError: If Graphiti client is not available or not connected
    """
    if not graphiti or not graphiti.is_connected:
        raise Neo4jError("graphiti_dependency", "Graphiti client not available")
    return graphiti


@router.get(
    "/graph",
    summary="Get knowledge graph data",
    description="Retrieve nodes and edges for graph visualization with optional filtering.",
)
@limiter.limit("30/minute")
async def get_graph(
    request: Request,
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
@limiter.limit("60/minute")
async def get_stats(
    request: Request,
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
@limiter.limit("30/minute")
async def get_orphans(
    request: Request,
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


# Epic 5 - Temporal Query Models and Endpoints


class TemporalNodeResponse(BaseModel):
    """A node in temporal search results."""

    uuid: str
    name: str
    summary: str
    labels: list[str]


class TemporalEdgeResponse(BaseModel):
    """An edge in temporal search results with validity period."""

    uuid: str
    source_node_uuid: str
    target_node_uuid: str
    name: str
    fact: str
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None


class TemporalSearchResponse(BaseModel):
    """Response model for temporal search query."""

    query: str
    as_of_date: Optional[str] = None
    nodes: list[TemporalNodeResponse]
    edges: list[TemporalEdgeResponse]
    processing_time_ms: int


class EpisodeResponse(BaseModel):
    """A single episode (document ingestion) in change history."""

    uuid: str
    name: str
    created_at: str
    entities_added: int
    edges_added: int


class KnowledgeChangesResponse(BaseModel):
    """Response model for knowledge changes query."""

    start_date: str
    end_date: str
    episodes: list[EpisodeResponse]
    total_entities_added: int
    total_edges_added: int
    processing_time_ms: int


class TemporalQueryRequest(BaseModel):
    """Request body for temporal knowledge query."""

    query: str = Field(..., min_length=1, description="Search query (non-empty)")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    as_of_date: Optional[datetime] = Field(
        None, description="Point-in-time for historical query (ISO 8601)"
    )
    num_results: int = Field(
        DEFAULT_SEARCH_RESULTS,
        ge=1,
        le=MAX_SEARCH_RESULTS,
        description="Max results to return",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()

    @field_validator("as_of_date")
    @classmethod
    def as_of_date_not_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate as_of_date is not in the future."""
        if v is not None:
            now = datetime.now(timezone.utc)
            # Allow some tolerance for clock skew (5 minutes)
            if v > now.replace(minute=now.minute + 5 if now.minute < 55 else 59):
                raise ValueError("as_of_date cannot be in the future")
        return v


@router.post(
    "/temporal/search",
    summary="Query knowledge at a point in time",
    description="Execute temporal search - query knowledge graph at specific point in time.",
    response_model_exclude_none=True,
)
@limiter.limit("30/minute")
async def post_temporal_query(
    request: Request,
    body: TemporalQueryRequest,
    graphiti: GraphitiClient = Depends(get_connected_graphiti),
) -> dict[str, Any]:
    """
    Query knowledge graph at a specific point in time.

    Returns nodes and edges that were valid at the as_of_date.
    If no as_of_date is provided, returns current state.

    Args:
        body: Request with query, tenant_id, and optional as_of_date
        graphiti: Graphiti client dependency

    Returns:
        Success response with temporal search results
    """
    logger.info(
        "temporal_query_requested",
        tenant_id=str(body.tenant_id),
        as_of_date=body.as_of_date.isoformat() if body.as_of_date else None,
    )

    try:
        result = await temporal_search(
            graphiti_client=graphiti,
            query=body.query,
            tenant_id=str(body.tenant_id),
            as_of_date=body.as_of_date,
            num_results=body.num_results,
        )

        response = TemporalSearchResponse(
            query=result.query,
            as_of_date=result.as_of_date.isoformat() if result.as_of_date else None,
            nodes=[
                TemporalNodeResponse(
                    uuid=n.uuid,
                    name=n.name,
                    summary=n.summary,
                    labels=n.labels,
                )
                for n in result.nodes
            ],
            edges=[
                TemporalEdgeResponse(
                    uuid=e.uuid,
                    source_node_uuid=e.source_node_uuid,
                    target_node_uuid=e.target_node_uuid,
                    name=e.name,
                    fact=e.fact,
                    valid_at=e.valid_at.isoformat() if e.valid_at else None,
                    invalid_at=e.invalid_at.isoformat() if e.invalid_at else None,
                )
                for e in result.edges
            ],
            processing_time_ms=result.processing_time_ms,
        )

        return success_response(response.model_dump())

    except Exception as e:
        logger.error(
            "temporal_query_failed",
            tenant_id=str(body.tenant_id),
            error=str(e),
        )
        raise Neo4jError("temporal_query", str(e)) from e


@router.get(
    "/temporal/changes",
    summary="Get knowledge changes over time",
    description="Retrieve knowledge graph changes (episodes) within a date range.",
    response_model_exclude_none=True,
)
@limiter.limit("30/minute")
async def get_changes(
    request: Request,
    tenant_id: UUID = Query(..., description="Tenant identifier (required)"),
    start_date: datetime = Query(..., description="Start of date range (ISO 8601)"),
    end_date: datetime = Query(..., description="End of date range (ISO 8601)"),
    entity_type: Optional[str] = Query(
        None, 
        description="Filter by entity type (NOT IMPLEMENTED - returns 501)"
    ),
    graphiti: GraphitiClient = Depends(get_connected_graphiti),
) -> dict[str, Any]:
    """
    Get knowledge changes over a time period.

    Returns episodes (document ingestions) and aggregated change statistics.

    Args:
        tenant_id: Tenant identifier
        start_date: Start of time period
        end_date: End of time period (must be after start_date)
        entity_type: Optional filter by entity type (NOT IMPLEMENTED)
        graphiti: Graphiti client dependency

    Returns:
        Success response with change timeline

    Raises:
        HTTPException 400: If end_date <= start_date
        HTTPException 501: If entity_type filter is requested
    """
    # Validate date range
    if end_date <= start_date:
        raise HTTPException(
            status_code=400,
            detail="end_date must be after start_date",
        )

    logger.info(
        "get_changes_requested",
        tenant_id=str(tenant_id),
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    try:
        result = await get_knowledge_changes(
            graphiti_client=graphiti,
            tenant_id=str(tenant_id),
            start_date=start_date,
            end_date=end_date,
            entity_type=entity_type,
        )

        response = KnowledgeChangesResponse(
            start_date=result.start_date.isoformat(),
            end_date=result.end_date.isoformat(),
            episodes=[
                EpisodeResponse(
                    uuid=ep.uuid,
                    name=ep.name,
                    created_at=ep.created_at.isoformat(),
                    entities_added=ep.entities_added,
                    edges_added=ep.edges_added,
                )
                for ep in result.episodes
            ],
            total_entities_added=result.total_entities_added,
            total_edges_added=result.total_edges_added,
            processing_time_ms=result.processing_time_ms,
        )

        return success_response(response.model_dump())

    except EntityTypeFilterNotImplemented as e:
        # Return 501 Not Implemented for entity_type filter
        raise HTTPException(
            status_code=501,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "get_changes_failed",
            tenant_id=str(tenant_id),
            error=str(e),
        )
        raise Neo4jError("get_changes", str(e)) from e
