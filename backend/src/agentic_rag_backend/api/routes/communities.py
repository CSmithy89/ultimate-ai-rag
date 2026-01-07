"""Community Detection API endpoints (Story 20-B1).

This module provides REST endpoints for community detection:
- POST /api/v1/communities/detect - Trigger community detection
- GET /api/v1/communities - List communities
- GET /api/v1/communities/{community_id} - Get single community
- DELETE /api/v1/communities/{community_id} - Delete community
- POST /api/v1/communities/search - Search communities by keyword

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by COMMUNITY_DETECTION_ENABLED configuration flag.
"""

import time
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from agentic_rag_backend.config import Settings
from agentic_rag_backend.graph import (
    CommunityAlgorithm,
    CommunityDetectionError,
    CommunityDetectionRequest,
    CommunityDetectionResponse,
    CommunityDetector,
    CommunityListResponse,
    CommunityNotFoundError,
    CommunitySearchRequest,
    CommunitySearchResponse,
    GraphTooSmallError,
    NETWORKX_AVAILABLE,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/communities", tags=["communities"])


# Response wrapper models (consistent with other route patterns)
class Meta(BaseModel):
    """Response metadata."""

    requestId: str
    timestamp: str


class SuccessResponse(BaseModel):
    """Standard success response wrapper."""

    data: Any
    meta: Meta


def success_response(data: Any) -> dict[str, Any]:
    """Wrap data in standard success response format.

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


# Dependency injection


async def get_settings(request: Request) -> Settings:
    """Get settings from app.state."""
    return request.app.state.settings


async def get_neo4j(request: Request):
    """Get Neo4j client from app.state."""
    return getattr(request.app.state, "neo4j", None)


async def get_llm_client(request: Request):
    """Get LLM client from app.state (optional).

    Returns the orchestrator's LLM adapter for summary generation.
    """
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator:
        # Return None for now - LLM integration will be added in a future story
        # The CommunityDetector can work without summaries
        return None
    return None


async def get_community_detector(request: Request) -> CommunityDetector:
    """Get or create CommunityDetector from app.state."""
    settings: Settings = request.app.state.settings

    # Check if we've already created the detector
    if hasattr(request.app.state, "community_detector"):
        return request.app.state.community_detector

    # Ensure required graph library is available before constructing detector
    check_networkx_available()

    neo4j = getattr(request.app.state, "neo4j", None)
    if not neo4j:
        raise HTTPException(
            status_code=503,
            detail="Neo4j client not available. Cannot perform community detection.",
        )

    llm_client = await get_llm_client(request)

    # Determine algorithm from settings
    algorithm = CommunityAlgorithm.LOUVAIN
    if settings.community_algorithm == "leiden":
        algorithm = CommunityAlgorithm.LEIDEN

    detector = CommunityDetector(
        neo4j_client=neo4j,
        llm_client=llm_client,
        algorithm=algorithm,
        min_community_size=settings.community_min_size,
        max_hierarchy_levels=settings.community_max_levels,
        summary_model=settings.community_summary_model,
    )

    # Cache the detector
    request.app.state.community_detector = detector

    return detector


def check_feature_enabled(settings: Settings) -> None:
    """Check if community detection feature is enabled.

    Raises:
        HTTPException: 404 if feature is disabled
    """
    if not settings.community_detection_enabled:
        raise HTTPException(
            status_code=404,
            detail="Community detection feature is not enabled. Set COMMUNITY_DETECTION_ENABLED=true to enable.",
        )


def check_networkx_available() -> None:
    """Check if NetworkX is available.

    Raises:
        HTTPException: 503 if NetworkX is not installed
    """
    if not NETWORKX_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="NetworkX is not installed. Community detection requires networkx>=3.0.",
        )


# API Endpoints


@router.post(
    "/detect",
    response_model=SuccessResponse,
    summary="Trigger community detection",
    description="Run community detection algorithm on the knowledge graph.",
)
async def detect_communities(
    detection_request: CommunityDetectionRequest,
    settings: Settings = Depends(get_settings),
    detector: CommunityDetector = Depends(get_community_detector),
) -> dict[str, Any]:
    """Trigger community detection on the knowledge graph.

    This endpoint runs the specified community detection algorithm on all
    entities in the tenant's knowledge graph. It creates Community nodes
    with BELONGS_TO relationships from entities.

    The process:
    1. Exports Neo4j graph to NetworkX format
    2. Runs Louvain or Leiden algorithm
    3. Builds hierarchical community structure
    4. Generates LLM summaries (if enabled)
    5. Stores communities to Neo4j

    Args:
        detection_request: Detection parameters
        settings: Application settings
        detector: Community detector instance

    Returns:
        Success response with detection results

    Raises:
        HTTPException: 404 if feature is disabled
        HTTPException: 422 if graph is too small
        HTTPException: 500 if detection fails
    """
    check_feature_enabled(settings)
    check_networkx_available()

    start_time = time.perf_counter()
    tenant_id = str(detection_request.tenant_id)

    logger.info(
        "community_detection_requested",
        tenant_id=tenant_id,
        algorithm=detection_request.algorithm.value,
        generate_summaries=detection_request.generate_summaries,
    )

    try:
        # Delete existing communities first
        deleted_count = await detector.delete_all_communities(tenant_id)
        if deleted_count > 0:
            logger.info(
                "existing_communities_cleared",
                tenant_id=tenant_id,
                count=deleted_count,
            )

        communities = await detector.detect_communities(
            tenant_id=tenant_id,
            generate_summaries=detection_request.generate_summaries,
            algorithm=detection_request.algorithm,
            min_size=detection_request.min_community_size,
            max_levels=detection_request.max_levels,
        )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Determine levels generated
        levels_generated = max((c.level for c in communities), default=0) + 1 if communities else 0

        # Get top-level communities for response
        top_level_communities = [c for c in communities if c.level == levels_generated - 1] if communities else []

        response = CommunityDetectionResponse(
            communities_created=len(communities),
            levels_generated=levels_generated,
            algorithm=detection_request.algorithm,
            processing_time_ms=processing_time_ms,
            tenant_id=tenant_id,
            communities=top_level_communities,
        )

        logger.info(
            "community_detection_completed",
            tenant_id=tenant_id,
            communities_created=len(communities),
            levels=levels_generated,
            processing_time_ms=processing_time_ms,
        )

        return success_response(response.model_dump(mode="json"))

    except GraphTooSmallError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Graph too small for community detection: {e.node_count} nodes "
                   f"(minimum {e.min_required} required)",
        ) from e

    except CommunityDetectionError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Community detection failed: {str(e)}",
        ) from e

    except Exception as e:
        logger.error(
            "community_detection_error",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Community detection failed: {str(e)}",
        ) from e


@router.get(
    "",
    response_model=SuccessResponse,
    summary="List communities",
    description="List communities with optional level filtering.",
)
async def list_communities(
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    level: Optional[int] = Query(default=None, ge=0, description="Filter by hierarchy level"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    settings: Settings = Depends(get_settings),
    detector: CommunityDetector = Depends(get_community_detector),
) -> dict[str, Any]:
    """List communities for a tenant.

    Args:
        tenant_id: Tenant identifier (required)
        level: Optional filter by hierarchy level (0 = most granular)
        limit: Maximum results to return
        offset: Pagination offset
        settings: Application settings
        detector: Community detector instance

    Returns:
        Success response with community list and total count
    """
    check_feature_enabled(settings)
    check_networkx_available()

    communities, total = await detector.list_communities(
        tenant_id=str(tenant_id),
        level=level,
        limit=limit,
        offset=offset,
    )

    response = CommunityListResponse(
        communities=communities,
        total=total,
        limit=limit,
        offset=offset,
        level=level,
    )

    return success_response(response.model_dump(mode="json"))


@router.get(
    "/{community_id}",
    response_model=SuccessResponse,
    summary="Get a community",
    description="Get a specific community by ID.",
)
async def get_community(
    community_id: str,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    include_entities: bool = Query(default=False, description="Include entity details"),
    settings: Settings = Depends(get_settings),
    detector: CommunityDetector = Depends(get_community_detector),
) -> dict[str, Any]:
    """Get a specific community by ID.

    Args:
        community_id: Community UUID
        tenant_id: Tenant identifier for access control
        include_entities: Whether to include full entity details
        settings: Application settings
        detector: Community detector instance

    Returns:
        Success response with community data

    Raises:
        HTTPException: 404 if community not found
    """
    check_feature_enabled(settings)
    check_networkx_available()

    try:
        community = await detector.get_community(
            community_id=community_id,
            tenant_id=str(tenant_id),
            include_entities=include_entities,
        )

        return success_response(community.model_dump(mode="json"))

    except CommunityNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Community '{community_id}' not found",
        ) from e


@router.delete(
    "/{community_id}",
    response_model=SuccessResponse,
    summary="Delete a community",
    description="Delete a specific community by ID.",
)
async def delete_community(
    community_id: str,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    settings: Settings = Depends(get_settings),
    detector: CommunityDetector = Depends(get_community_detector),
) -> dict[str, Any]:
    """Delete a specific community.

    Note: This only deletes the Community node and its relationships.
    The underlying entities are preserved.

    Args:
        community_id: Community UUID
        tenant_id: Tenant identifier for access control
        settings: Application settings
        detector: Community detector instance

    Returns:
        Success response confirming deletion

    Raises:
        HTTPException: 404 if community not found
    """
    check_feature_enabled(settings)
    check_networkx_available()

    deleted = await detector.delete_community(
        community_id=community_id,
        tenant_id=str(tenant_id),
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Community '{community_id}' not found",
        )

    logger.info(
        "community_deleted_via_api",
        community_id=community_id,
        tenant_id=str(tenant_id),
    )

    return success_response({"deleted": True, "community_id": community_id})


@router.post(
    "/search",
    response_model=SuccessResponse,
    summary="Search communities",
    description="Search communities by keyword, name, or summary content.",
)
async def search_communities(
    search_request: CommunitySearchRequest,
    settings: Settings = Depends(get_settings),
    detector: CommunityDetector = Depends(get_community_detector),
) -> dict[str, Any]:
    """Search communities by keyword.

    Performs text search on community names, summaries, and keywords.

    Args:
        search_request: Search parameters
        settings: Application settings
        detector: Community detector instance

    Returns:
        Success response with matching communities
    """
    check_feature_enabled(settings)
    check_networkx_available()

    communities = await detector.search_communities(
        query=search_request.query,
        tenant_id=str(search_request.tenant_id),
        level=search_request.level,
        limit=search_request.limit,
    )

    response = CommunitySearchResponse(
        communities=communities,
        total=len(communities),
        query=search_request.query,
    )

    logger.info(
        "community_search",
        query=search_request.query,
        tenant_id=str(search_request.tenant_id),
        results_count=len(communities),
    )

    return success_response(response.model_dump(mode="json"))


@router.delete(
    "",
    response_model=SuccessResponse,
    summary="Delete all communities",
    description="Delete all communities for a tenant. Use before re-running detection.",
)
async def delete_all_communities(
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    settings: Settings = Depends(get_settings),
    detector: CommunityDetector = Depends(get_community_detector),
) -> dict[str, Any]:
    """Delete all communities for a tenant.

    Useful before re-running community detection to start fresh.

    Args:
        tenant_id: Tenant identifier
        settings: Application settings
        detector: Community detector instance

    Returns:
        Success response with count of deleted communities
    """
    check_feature_enabled(settings)
    check_networkx_available()

    deleted_count = await detector.delete_all_communities(
        tenant_id=str(tenant_id),
    )

    logger.info(
        "all_communities_deleted_via_api",
        tenant_id=str(tenant_id),
        count=deleted_count,
    )

    return success_response({
        "deleted_count": deleted_count,
        "tenant_id": str(tenant_id),
    })
