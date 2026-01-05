"""LazyRAG API endpoints (Story 20-B2).

This module provides REST endpoints for LazyRAG query-time summarization:
- POST /api/v1/lazy-rag/query - Execute lazy RAG query with summary
- POST /api/v1/lazy-rag/expand - Expand subgraph without summary (debug)
- GET /api/v1/lazy-rag/status - Feature status and configuration

LazyRAG defers graph summarization to query time, achieving up to 99%
reduction in indexing costs compared to MS GraphRAG's eager approach.

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by LAZY_RAG_ENABLED configuration flag.
"""

import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from agentic_rag_backend.config import Settings
from agentic_rag_backend.retrieval.lazy_rag import LazyRAGRetriever
from agentic_rag_backend.retrieval.lazy_rag_models import (
    LazyRAGExpandRequest,
    LazyRAGExpandResponse,
    LazyRAGQueryRequest,
    LazyRAGStatusResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/lazy-rag", tags=["lazy-rag"])


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


async def get_lazy_rag_retriever(request: Request) -> LazyRAGRetriever:
    """Get or create LazyRAGRetriever from app.state."""
    settings: Settings = request.app.state.settings

    # Check if we've already created the retriever
    if hasattr(request.app.state, "lazy_rag_retriever"):
        return request.app.state.lazy_rag_retriever

    graphiti = await get_graphiti(request)
    neo4j = await get_neo4j(request)
    community_detector = await get_community_detector(request)

    retriever = LazyRAGRetriever(
        graphiti_client=graphiti,
        neo4j_client=neo4j,
        settings=settings,
        community_detector=community_detector,
    )

    # Cache the retriever
    request.app.state.lazy_rag_retriever = retriever

    return retriever


def check_feature_enabled(settings: Settings) -> None:
    """Check if LazyRAG feature is enabled.

    Raises:
        HTTPException: 404 if feature is disabled
    """
    if not settings.lazy_rag_enabled:
        raise HTTPException(
            status_code=404,
            detail="LazyRAG feature is not enabled. Set LAZY_RAG_ENABLED=true to enable.",
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
            detail="Neo4j client not available. Cannot perform LazyRAG query.",
        )


# API Endpoints


@router.post(
    "/query",
    response_model=SuccessResponse,
    summary="Execute LazyRAG query",
    description="Execute a LazyRAG query with query-time summarization.",
)
async def query_lazy_rag(
    query_request: LazyRAGQueryRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    retriever: LazyRAGRetriever = Depends(get_lazy_rag_retriever),
) -> dict[str, Any]:
    """Execute a LazyRAG query with query-time summarization.

    This endpoint implements the LazyRAG pattern:
    1. Finds seed entities via Graphiti hybrid search
    2. Expands subgraph via N-hop traversal
    3. Optionally includes community context from 20-B1
    4. Generates LLM summary at query time

    Args:
        query_request: Query parameters
        request: FastAPI request
        settings: Application settings
        retriever: LazyRAG retriever instance

    Returns:
        Success response with query results, summary, and confidence

    Raises:
        HTTPException: 404 if feature is disabled
        HTTPException: 503 if Neo4j is not available
        HTTPException: 500 if query fails
    """
    check_feature_enabled(settings)
    check_neo4j_available(request)

    tenant_id = str(query_request.tenant_id)

    logger.info(
        "lazy_rag_query_api_request",
        query=query_request.query[:100],
        tenant_id=tenant_id,
        max_entities=query_request.max_entities,
        max_hops=query_request.max_hops,
        use_communities=query_request.use_communities,
        include_summary=query_request.include_summary,
    )

    try:
        result = await retriever.query(
            query=query_request.query,
            tenant_id=tenant_id,
            max_entities=query_request.max_entities,
            max_hops=query_request.max_hops,
            use_communities=query_request.use_communities,
            include_summary=query_request.include_summary,
        )

        response = result.to_response()

        logger.info(
            "lazy_rag_query_api_completed",
            tenant_id=tenant_id,
            entities=response.expanded_entity_count,
            confidence=response.confidence,
            processing_time_ms=response.processing_time_ms,
        )

        return success_response(response.model_dump(mode="json"))

    except Exception as e:
        logger.error(
            "lazy_rag_query_api_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"LazyRAG query failed: {str(e)}",
        ) from e


@router.post(
    "/expand",
    response_model=SuccessResponse,
    summary="Expand subgraph only",
    description="Expand subgraph without generating summary (debug endpoint).",
)
async def expand_lazy_rag(
    expand_request: LazyRAGExpandRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    retriever: LazyRAGRetriever = Depends(get_lazy_rag_retriever),
) -> dict[str, Any]:
    """Expand subgraph without generating summary.

    This debug endpoint performs only the subgraph expansion step,
    returning entities and relationships without LLM summarization.

    Useful for:
    - Debugging graph connectivity
    - Testing seed entity quality
    - Understanding subgraph structure

    Args:
        expand_request: Expansion parameters
        request: FastAPI request
        settings: Application settings
        retriever: LazyRAG retriever instance

    Returns:
        Success response with entities and relationships

    Raises:
        HTTPException: 404 if feature is disabled
        HTTPException: 503 if Neo4j is not available
        HTTPException: 500 if expansion fails
    """
    check_feature_enabled(settings)
    check_neo4j_available(request)

    tenant_id = str(expand_request.tenant_id)
    start_time = time.perf_counter()

    logger.info(
        "lazy_rag_expand_api_request",
        query=expand_request.query[:100],
        tenant_id=tenant_id,
        max_entities=expand_request.max_entities,
        max_hops=expand_request.max_hops,
    )

    try:
        result = await retriever.expand_only(
            query=expand_request.query,
            tenant_id=tenant_id,
            max_entities=expand_request.max_entities,
            max_hops=expand_request.max_hops,
        )

        processing_time_ms = int((time.perf_counter() - start_time) * 1000)

        response = LazyRAGExpandResponse(
            query=expand_request.query,
            tenant_id=tenant_id,
            entities=[e.to_response() for e in result.entities],
            relationships=[r.to_response() for r in result.relationships],
            seed_entity_count=result.seed_count,
            expanded_entity_count=result.expanded_count,
            processing_time_ms=processing_time_ms,
        )

        logger.info(
            "lazy_rag_expand_api_completed",
            tenant_id=tenant_id,
            entities=result.expanded_count,
            relationships=len(result.relationships),
            processing_time_ms=processing_time_ms,
        )

        return success_response(response.model_dump(mode="json"))

    except Exception as e:
        logger.error(
            "lazy_rag_expand_api_failed",
            tenant_id=tenant_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"LazyRAG expansion failed: {str(e)}",
        ) from e


@router.get(
    "/status",
    response_model=SuccessResponse,
    summary="Get LazyRAG status",
    description="Get LazyRAG feature status and configuration.",
)
async def get_lazy_rag_status(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Get LazyRAG feature status and configuration.

    Returns current configuration values and availability of
    dependent features (community detection from 20-B1).

    Args:
        request: FastAPI request
        settings: Application settings

    Returns:
        Success response with status and configuration
    """
    # Check if community detection is available
    community_detector = getattr(request.app.state, "community_detector", None)
    community_available = (
        community_detector is not None and settings.community_detection_enabled
    )

    response = LazyRAGStatusResponse(
        enabled=settings.lazy_rag_enabled,
        max_entities=settings.lazy_rag_max_entities,
        max_hops=settings.lazy_rag_max_hops,
        use_communities=settings.lazy_rag_use_communities,
        summary_model=settings.lazy_rag_summary_model,
        community_detection_available=community_available,
    )

    logger.debug(
        "lazy_rag_status_requested",
        enabled=settings.lazy_rag_enabled,
    )

    return success_response(response.model_dump(mode="json"))
