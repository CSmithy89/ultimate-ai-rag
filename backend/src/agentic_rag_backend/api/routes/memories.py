"""Memory API endpoints for Epic 20 Memory Platform.

This module provides REST endpoints for managing scoped memories:
- Create, read, update, delete memories
- Search memories within scope hierarchy
- Delete memories by scope

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by MEMORY_SCOPES_ENABLED configuration flag.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from agentic_rag_backend.config import Settings
from agentic_rag_backend.db.postgres import PostgresClient
from agentic_rag_backend.db.redis import RedisClient
from agentic_rag_backend.memory import (
    MemoryLimitExceededError,
    MemoryScope,
    MemoryScopeError,
    ScopedMemoryCreate,
    ScopedMemoryStore,
    ScopedMemoryUpdate,
)
from agentic_rag_backend.memory.models import (
    DeleteByScopeResponse,
    MemoryListResponse,
    MemorySearchRequest,
    MemorySearchResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/memories", tags=["memories"])


# Response wrapper models (consistent with ingest.py pattern)
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


async def get_postgres(request: Request) -> PostgresClient:
    """Get PostgreSQL client from app.state."""
    return request.app.state.postgres


async def get_redis(request: Request) -> Optional[RedisClient]:
    """Get Redis client from app.state (optional)."""
    return getattr(request.app.state, "redis_client", None)


async def get_memory_store(request: Request) -> ScopedMemoryStore:
    """Get or create memory store from app.state."""
    if not hasattr(request.app.state, "memory_store"):
        settings: Settings = request.app.state.settings
        postgres: PostgresClient = request.app.state.postgres
        redis: Optional[RedisClient] = getattr(request.app.state, "redis_client", None)

        request.app.state.memory_store = ScopedMemoryStore(
            postgres_client=postgres,
            redis_client=redis,
            embedding_provider=settings.embedding_provider,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
            embedding_model=settings.embedding_model,
            cache_ttl_seconds=settings.memory_cache_ttl_seconds,
            max_per_scope=settings.memory_max_per_scope,
        )
    return request.app.state.memory_store


def check_feature_enabled(settings: Settings) -> None:
    """Check if memory scopes feature is enabled.

    Raises:
        HTTPException: 404 if feature is disabled
    """
    if not settings.memory_scopes_enabled:
        raise HTTPException(
            status_code=404,
            detail="Memory scopes feature is not enabled. Set MEMORY_SCOPES_ENABLED=true to enable.",
        )


# API Endpoints


@router.post(
    "",
    response_model=SuccessResponse,
    summary="Create a memory",
    description="Create a new scoped memory entry.",
)
async def create_memory(
    memory_request: ScopedMemoryCreate,
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """Create a new scoped memory.

    Args:
        memory_request: Memory creation request with content and scope
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response with created memory

    Raises:
        HTTPException: 400 if scope context is invalid
        HTTPException: 404 if feature is disabled
        HTTPException: 429 if scope limit exceeded
    """
    check_feature_enabled(settings)

    try:
        memory = await store.add_memory(
            content=memory_request.content,
            scope=memory_request.scope,
            tenant_id=str(memory_request.tenant_id),
            user_id=str(memory_request.user_id) if memory_request.user_id else None,
            session_id=str(memory_request.session_id) if memory_request.session_id else None,
            agent_id=memory_request.agent_id,
            importance=memory_request.importance,
            metadata=memory_request.metadata,
        )

        logger.info(
            "memory_created",
            memory_id=str(memory.id),
            scope=memory.scope.value,
            tenant_id=str(memory_request.tenant_id),
        )

        return success_response(memory.model_dump(mode="json", exclude={"embedding"}))

    except MemoryScopeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except MemoryLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e


@router.get(
    "",
    response_model=SuccessResponse,
    summary="List memories",
    description="List memories with optional scope and context filtering.",
)
async def list_memories(
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    scope: Optional[MemoryScope] = Query(default=None, description="Filter by scope"),
    user_id: Optional[UUID] = Query(default=None, description="Filter by user ID"),
    session_id: Optional[UUID] = Query(default=None, description="Filter by session ID"),
    agent_id: Optional[str] = Query(default=None, description="Filter by agent ID"),
    limit: int = Query(default=50, ge=1, le=100, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """List memories with filtering.

    Args:
        tenant_id: Tenant identifier (required)
        scope: Optional scope filter
        user_id: Optional user filter
        session_id: Optional session filter
        agent_id: Optional agent filter
        limit: Maximum results to return
        offset: Pagination offset
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response with memory list and total count
    """
    check_feature_enabled(settings)

    memories, total = await store.list_memories(
        tenant_id=str(tenant_id),
        scope=scope,
        user_id=str(user_id) if user_id else None,
        session_id=str(session_id) if session_id else None,
        agent_id=agent_id,
        limit=limit,
        offset=offset,
    )

    response = MemoryListResponse(
        memories=memories,
        total=total,
        limit=limit,
        offset=offset,
    )

    return success_response(response.model_dump(mode="json"))


@router.post(
    "/search",
    response_model=SuccessResponse,
    summary="Search memories",
    description="Search memories using semantic similarity within scope hierarchy.",
)
async def search_memories(
    search_request: MemorySearchRequest,
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """Search memories using semantic similarity.

    The search respects scope hierarchy:
    - SESSION scope includes USER and GLOBAL memories
    - USER scope includes GLOBAL memories
    - AGENT scope includes GLOBAL memories

    Args:
        search_request: Search parameters
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response with matching memories and scopes searched
    """
    check_feature_enabled(settings)

    memories, scopes_searched = await store.search_memories(
        query=search_request.query,
        scope=search_request.scope,
        tenant_id=str(search_request.tenant_id),
        user_id=str(search_request.user_id) if search_request.user_id else None,
        session_id=str(search_request.session_id) if search_request.session_id else None,
        agent_id=search_request.agent_id,
        limit=search_request.limit,
        include_parent_scopes=search_request.include_parent_scopes,
    )

    response = MemorySearchResponse(
        memories=memories,
        total=len(memories),
        query=search_request.query,
        scopes_searched=scopes_searched,
    )

    logger.info(
        "memory_search",
        query_length=len(search_request.query),
        scope=search_request.scope.value,
        results_count=len(memories),
        scopes_searched=[s.value for s in scopes_searched],
    )

    return success_response(response.model_dump(mode="json"))


@router.get(
    "/{memory_id}",
    response_model=SuccessResponse,
    summary="Get a memory",
    description="Get a specific memory by ID.",
)
async def get_memory(
    memory_id: UUID,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """Get a specific memory by ID.

    Args:
        memory_id: Memory UUID
        tenant_id: Tenant identifier for access control
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response with memory data

    Raises:
        HTTPException: 404 if memory not found
    """
    check_feature_enabled(settings)

    memory = await store.get_memory(
        memory_id=str(memory_id),
        tenant_id=str(tenant_id),
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail=f"Memory with ID '{memory_id}' not found",
        )

    return success_response(memory.model_dump(mode="json", exclude={"embedding"}))


@router.put(
    "/{memory_id}",
    response_model=SuccessResponse,
    summary="Update a memory",
    description="Update a memory's content, importance, or metadata.",
)
async def update_memory(
    memory_id: UUID,
    update_request: ScopedMemoryUpdate,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """Update a memory.

    Args:
        memory_id: Memory UUID
        update_request: Fields to update
        tenant_id: Tenant identifier for access control
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response with updated memory

    Raises:
        HTTPException: 404 if memory not found
    """
    check_feature_enabled(settings)

    memory = await store.update_memory(
        memory_id=str(memory_id),
        tenant_id=str(tenant_id),
        content=update_request.content,
        importance=update_request.importance,
        metadata=update_request.metadata,
    )

    if not memory:
        raise HTTPException(
            status_code=404,
            detail=f"Memory with ID '{memory_id}' not found",
        )

    logger.info(
        "memory_updated",
        memory_id=str(memory_id),
        tenant_id=str(tenant_id),
    )

    return success_response(memory.model_dump(mode="json", exclude={"embedding"}))


@router.delete(
    "/{memory_id}",
    response_model=SuccessResponse,
    summary="Delete a memory",
    description="Delete a specific memory by ID.",
)
async def delete_memory(
    memory_id: UUID,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """Delete a specific memory.

    Args:
        memory_id: Memory UUID
        tenant_id: Tenant identifier for access control
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response confirming deletion

    Raises:
        HTTPException: 404 if memory not found
    """
    check_feature_enabled(settings)

    deleted = await store.delete_memory(
        memory_id=str(memory_id),
        tenant_id=str(tenant_id),
    )

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Memory with ID '{memory_id}' not found",
        )

    logger.info(
        "memory_deleted",
        memory_id=str(memory_id),
        tenant_id=str(tenant_id),
    )

    return success_response({"deleted": True, "memory_id": str(memory_id)})


@router.delete(
    "/scope/{scope}",
    response_model=SuccessResponse,
    summary="Delete memories by scope",
    description="Delete all memories in a specific scope. Useful for session cleanup.",
)
async def delete_memories_by_scope(
    scope: MemoryScope,
    tenant_id: UUID = Query(..., description="Tenant identifier"),
    user_id: Optional[UUID] = Query(default=None, description="User ID (for USER/SESSION scope)"),
    session_id: Optional[UUID] = Query(default=None, description="Session ID (for SESSION scope)"),
    agent_id: Optional[str] = Query(default=None, description="Agent ID (for AGENT scope)"),
    settings: Settings = Depends(get_settings),
    store: ScopedMemoryStore = Depends(get_memory_store),
) -> dict[str, Any]:
    """Delete all memories in a scope.

    This is useful for:
    - Clearing session memories when a session ends
    - Removing all user memories on account deletion
    - Resetting agent operational memory

    Args:
        scope: Scope to clear
        tenant_id: Tenant identifier
        user_id: User identifier (for USER/SESSION scope)
        session_id: Session identifier (for SESSION scope)
        agent_id: Agent identifier (for AGENT scope)
        settings: Application settings
        store: Memory store instance

    Returns:
        Success response with count of deleted memories
    """
    check_feature_enabled(settings)

    deleted_count = await store.delete_memories_by_scope(
        scope=scope,
        tenant_id=str(tenant_id),
        user_id=str(user_id) if user_id else None,
        session_id=str(session_id) if session_id else None,
        agent_id=agent_id,
    )

    response = DeleteByScopeResponse(
        deleted_count=deleted_count,
        scope=scope,
    )

    logger.info(
        "memories_deleted_by_scope",
        scope=scope.value,
        tenant_id=str(tenant_id),
        deleted_count=deleted_count,
    )

    return success_response(response.model_dump())
