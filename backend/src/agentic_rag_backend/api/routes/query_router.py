"""Query Router API endpoints (Story 20-B3).

This module provides REST endpoints for query routing:
- POST /api/v1/query-router/route - Route a query to global/local/hybrid
- GET /api/v1/query-router/patterns - List patterns (debug)
- GET /api/v1/query-router/status - Feature status

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by QUERY_ROUTING_ENABLED configuration flag.
"""

import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from agentic_rag_backend.config import Settings
from agentic_rag_backend.retrieval.query_router import QueryRouter
from agentic_rag_backend.retrieval.query_router_models import (
    PatternListResponse,
    QueryRouteRequest,
    RouterStatusResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/query-router", tags=["query-router"])


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


async def get_query_router(request: Request) -> QueryRouter:
    """Get or create QueryRouter from app.state."""
    settings: Settings = request.app.state.settings

    # Check if we've already created the router
    if hasattr(request.app.state, "query_router"):
        return request.app.state.query_router

    # Create new router
    query_router = QueryRouter(settings=settings)

    # Cache the router
    request.app.state.query_router = query_router

    return query_router


def check_feature_enabled(settings: Settings) -> None:
    """Check if query routing feature is enabled.

    Raises:
        HTTPException: 404 if feature is disabled
    """
    if not settings.query_routing_enabled:
        raise HTTPException(
            status_code=404,
            detail="Query routing feature is not enabled. Set QUERY_ROUTING_ENABLED=true to enable.",
        )


# API Endpoints


@router.post(
    "/route",
    response_model=SuccessResponse,
    summary="Route a query",
    description="Route a query to global (community-level), local (entity-level), or hybrid retrieval.",
)
async def route_query(
    route_request: QueryRouteRequest,
    settings: Settings = Depends(get_settings),
    query_router: QueryRouter = Depends(get_query_router),
) -> dict[str, Any]:
    """Route a query to the appropriate retrieval strategy.

    This endpoint analyzes the query and determines whether it should be:
    - GLOBAL: Processed via community-level retrieval (themes, summaries)
    - LOCAL: Processed via entity-level retrieval (specific facts)
    - HYBRID: Processed via weighted combination of both

    The routing decision includes:
    - query_type: The classified type
    - confidence: Confidence score (0.0-1.0)
    - reasoning: Human-readable explanation
    - global_weight/local_weight: Weights for hybrid retrieval

    Args:
        route_request: Request containing query and tenant_id
        settings: Application settings
        query_router: QueryRouter instance

    Returns:
        Success response with routing decision

    Raises:
        HTTPException: 404 if feature is disabled
        HTTPException: 500 if routing fails
    """
    check_feature_enabled(settings)

    start_time = time.perf_counter()
    tenant_id = str(route_request.tenant_id)

    logger.info(
        "query_routing_requested",
        query=route_request.query[:100],
        tenant_id=tenant_id,
        use_llm=route_request.use_llm,
    )

    try:
        decision = await query_router.route(
            query=route_request.query,
            tenant_id=tenant_id,
            use_llm=route_request.use_llm,
        )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        logger.info(
            "query_routing_completed",
            query=route_request.query[:100],
            tenant_id=tenant_id,
            query_type=decision.query_type.value,
            confidence=decision.confidence,
            processing_time_ms=processing_time_ms,
        )

        return success_response(decision.to_response().model_dump(mode="json"))

    except Exception as e:
        logger.error(
            "query_routing_error",
            query=route_request.query[:100],
            tenant_id=tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail="Query routing failed due to an internal server error.",
        ) from e


@router.get(
    "/patterns",
    response_model=SuccessResponse,
    summary="List patterns",
    description="List global and local query patterns (for debugging).",
)
async def get_patterns(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Get list of global and local query patterns.

    This endpoint is useful for debugging and understanding
    which patterns are used for rule-based classification.

    Args:
        settings: Application settings

    Returns:
        Success response with pattern lists

    Raises:
        HTTPException: 404 if feature is disabled
    """
    check_feature_enabled(settings)

    global_patterns = QueryRouter.get_global_patterns()
    local_patterns = QueryRouter.get_local_patterns()

    response = PatternListResponse(
        global_patterns=global_patterns,
        local_patterns=local_patterns,
        global_pattern_count=len(global_patterns),
        local_pattern_count=len(local_patterns),
    )

    logger.debug(
        "patterns_retrieved",
        global_count=len(global_patterns),
        local_count=len(local_patterns),
    )

    return success_response(response.model_dump(mode="json"))


@router.get(
    "/status",
    response_model=SuccessResponse,
    summary="Get router status",
    description="Get query router configuration and availability status.",
)
async def get_status(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Get query router configuration and availability status.

    This endpoint returns:
    - Whether query routing is enabled
    - Whether LLM classification is enabled
    - The LLM model used for classification
    - The confidence threshold for LLM fallback
    - Whether community detection (20-B1) is available
    - Whether LazyRAG (20-B2) is available

    Args:
        request: FastAPI request for app state access
        settings: Application settings

    Returns:
        Success response with router status
    """
    # Check if community detection is available
    community_detector = getattr(request.app.state, "community_detector", None)
    community_detection_available = (
        settings.community_detection_enabled and community_detector is not None
    )

    # Check if LazyRAG is available
    lazy_rag_retriever = getattr(request.app.state, "lazy_rag_retriever", None)
    lazy_rag_available = (
        settings.lazy_rag_enabled and lazy_rag_retriever is not None
    )

    response = RouterStatusResponse(
        enabled=settings.query_routing_enabled,
        use_llm=settings.query_routing_use_llm,
        llm_model=settings.query_routing_llm_model,
        confidence_threshold=settings.query_routing_confidence_threshold,
        community_detection_available=community_detection_available,
        lazy_rag_available=lazy_rag_available,
    )

    return success_response(response.model_dump(mode="json"))
