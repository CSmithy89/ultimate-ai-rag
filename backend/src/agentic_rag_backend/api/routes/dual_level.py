"""Dual-Level Retrieval API endpoints (Story 20-C2).

This module provides REST endpoints for dual-level retrieval combining:
- Low-level: Entity/chunk granular retrieval
- High-level: Community/theme contextual retrieval

Endpoints:
- POST /api/v1/dual-level/retrieve - Execute dual-level retrieval
- GET /api/v1/dual-level/status - Feature status and configuration

The dual-level approach is inspired by LightRAG's architecture for combining
specific entity knowledge with broader thematic context.

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by DUAL_LEVEL_RETRIEVAL_ENABLED configuration flag.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from agentic_rag_backend.config import Settings
from agentic_rag_backend.retrieval.dual_level import DualLevelRetriever
from agentic_rag_backend.retrieval.dual_level_models import (
    DualLevelRetrieveRequest,
    DualLevelStatusResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/dual-level", tags=["dual-level"])


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


async def get_graphiti(request: Request):
    """Get Graphiti client from app.state."""
    return getattr(request.app.state, "graphiti", None)


async def get_neo4j(request: Request):
    """Get Neo4j client from app.state."""
    return getattr(request.app.state, "neo4j", None)


async def get_community_detector(request: Request):
    """Get CommunityDetector from app.state if available."""
    return getattr(request.app.state, "community_detector", None)


async def get_dual_level_retriever(request: Request) -> DualLevelRetriever:
    """Get or create DualLevelRetriever from app.state."""
    settings: Settings = request.app.state.settings

    # Check if we've already created the retriever
    if hasattr(request.app.state, "dual_level_retriever"):
        return request.app.state.dual_level_retriever

    graphiti = await get_graphiti(request)
    neo4j = await get_neo4j(request)
    community_detector = await get_community_detector(request)

    retriever = DualLevelRetriever(
        graphiti_client=graphiti,
        neo4j_client=neo4j,
        settings=settings,
        community_detector=community_detector,
    )

    # Cache the retriever
    request.app.state.dual_level_retriever = retriever

    return retriever


def check_feature_enabled(settings: Settings) -> None:
    """Check if dual-level retrieval feature is enabled.

    Raises:
        HTTPException: 404 if feature is disabled
    """
    if not settings.dual_level_retrieval_enabled:
        raise HTTPException(
            status_code=404,
            detail="Dual-level retrieval feature is not enabled. Set DUAL_LEVEL_RETRIEVAL_ENABLED=true to enable.",
        )


def check_neo4j_available(request: Request) -> None:
    """Check if Neo4j is available.

    Raises:
        HTTPException: 503 if Neo4j is not connected
    """
    neo4j = getattr(request.app.state, "neo4j", None)
    if not neo4j:
        raise HTTPException(
            status_code=503,
            detail="Neo4j client not available. Cannot perform dual-level retrieval.",
        )


# API Endpoints


@router.post(
    "/retrieve",
    response_model=SuccessResponse,
    summary="Execute dual-level retrieval",
    description="Execute dual-level retrieval combining low-level (entities) and high-level (themes).",
)
async def retrieve_dual_level(
    retrieve_request: DualLevelRetrieveRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    retriever: DualLevelRetriever = Depends(get_dual_level_retriever),
) -> dict[str, Any]:
    """Execute dual-level retrieval with optional synthesis.

    This endpoint implements the LightRAG-inspired dual-level pattern:
    1. Low-level: Finds specific entities via Graphiti hybrid search
    2. High-level: Finds relevant communities/themes via CommunityDetector
    3. Synthesis: Optionally combines both perspectives via LLM

    Args:
        retrieve_request: Retrieval parameters
        request: FastAPI request
        settings: Application settings
        retriever: DualLevelRetriever instance

    Returns:
        Success response with low-level, high-level, and synthesized results

    Raises:
        HTTPException: 404 if feature is disabled
        HTTPException: 503 if Neo4j is not available
        HTTPException: 500 if retrieval fails
    """
    check_feature_enabled(settings)
    check_neo4j_available(request)

    tenant_id = str(retrieve_request.tenant_id)

    logger.info(
        "dual_level_retrieve_api_request",
        query=retrieve_request.query[:100],
        tenant_id=tenant_id,
        low_level_limit=retrieve_request.low_level_limit,
        high_level_limit=retrieve_request.high_level_limit,
        include_synthesis=retrieve_request.include_synthesis,
        low_weight=retrieve_request.low_weight,
        high_weight=retrieve_request.high_weight,
    )

    try:
        result = await retriever.retrieve(
            query=retrieve_request.query,
            tenant_id=tenant_id,
            low_level_limit=retrieve_request.low_level_limit,
            high_level_limit=retrieve_request.high_level_limit,
            include_synthesis=retrieve_request.include_synthesis,
            low_weight=retrieve_request.low_weight,
            high_weight=retrieve_request.high_weight,
        )

        response = result.to_response()

        logger.info(
            "dual_level_retrieve_api_completed",
            tenant_id=tenant_id,
            low_level_count=response.low_level_count,
            high_level_count=response.high_level_count,
            confidence=response.confidence,
            fallback_used=response.fallback_used,
            processing_time_ms=response.processing_time_ms,
        )

        return success_response(response.model_dump(mode="json"))

    except Exception as e:
        logger.error(
            "dual_level_retrieve_api_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Dual-level retrieval failed: {str(e)}",
        ) from e


@router.get(
    "/status",
    response_model=SuccessResponse,
    summary="Get dual-level retrieval status",
    description="Get dual-level retrieval feature status and configuration.",
)
async def get_dual_level_status(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Get dual-level retrieval feature status and configuration.

    Returns current configuration values and availability of
    dependent features (Graphiti for low-level, community detection
    for high-level from 20-B1).

    Args:
        request: FastAPI request
        settings: Application settings

    Returns:
        Success response with status and configuration
    """
    # Check if Graphiti is available for low-level retrieval
    graphiti = getattr(request.app.state, "graphiti", None)
    graphiti_available = graphiti is not None and getattr(graphiti, "is_connected", False)

    # Check if community detection is available for high-level retrieval
    community_detector = getattr(request.app.state, "community_detector", None)
    community_available = (
        community_detector is not None and settings.community_detection_enabled
    )

    response = DualLevelStatusResponse(
        enabled=settings.dual_level_retrieval_enabled,
        low_weight=settings.dual_level_low_weight,
        high_weight=settings.dual_level_high_weight,
        low_limit=settings.dual_level_low_limit,
        high_limit=settings.dual_level_high_limit,
        synthesis_model=settings.dual_level_synthesis_model,
        graphiti_available=graphiti_available,
        community_detection_available=community_available,
    )

    logger.debug(
        "dual_level_status_requested",
        enabled=settings.dual_level_retrieval_enabled,
        graphiti_available=graphiti_available,
        community_detection_available=community_available,
    )

    return success_response(response.model_dump(mode="json"))
