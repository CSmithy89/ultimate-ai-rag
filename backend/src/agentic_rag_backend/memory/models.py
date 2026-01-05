"""Pydantic models for Epic 20 Memory Platform."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class MemoryScope(str, Enum):
    """Hierarchical memory scopes.

    Scope hierarchy for search:
    - SESSION includes USER and GLOBAL memories
    - USER includes GLOBAL memories
    - AGENT includes GLOBAL memories
    - GLOBAL is the root scope
    """

    USER = "user"  # Persists across all sessions for a user
    SESSION = "session"  # Persists within a single conversation session
    AGENT = "agent"  # Persists across agent invocations (operational memory)
    GLOBAL = "global"  # Tenant-wide shared memory


class ScopedMemoryCreate(BaseModel):
    """Request model for creating a scoped memory."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Memory content",
    )
    scope: MemoryScope = Field(..., description="Memory scope level")
    tenant_id: UUID = Field(..., description="Tenant identifier (always required)")
    user_id: Optional[UUID] = Field(
        default=None, description="User ID (required for USER and SESSION scope)"
    )
    session_id: Optional[UUID] = Field(
        default=None, description="Session ID (required for SESSION scope)"
    )
    agent_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Agent ID (required for AGENT scope)",
    )
    importance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Importance score for consolidation (0.0-1.0)",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "User prefers dark mode interface",
                    "scope": "user",
                    "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
                    "user_id": "123e4567-e89b-12d3-a456-426614174002",
                    "importance": 0.8,
                    "metadata": {"source": "preferences", "category": "ui"},
                }
            ]
        }
    }


class ScopedMemoryUpdate(BaseModel):
    """Request model for updating a scoped memory."""

    content: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=10000,
        description="Updated memory content",
    )
    importance: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Updated importance score",
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Updated metadata (replaces existing)"
    )


class ScopedMemory(BaseModel):
    """A memory entry with scope context."""

    id: UUID = Field(..., description="Memory unique identifier")
    content: str = Field(..., description="Memory content")
    scope: MemoryScope = Field(..., description="Memory scope level")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    user_id: Optional[UUID] = Field(default=None, description="User identifier")
    session_id: Optional[UUID] = Field(default=None, description="Session identifier")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier")
    importance: float = Field(default=1.0, description="Importance score 0.0-1.0")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    accessed_at: datetime = Field(..., description="Last access timestamp")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")
    embedding: Optional[list[float]] = Field(
        default=None, description="Embedding vector (1536 dimensions)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "content": "User prefers dark mode",
                    "scope": "user",
                    "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
                    "user_id": "123e4567-e89b-12d3-a456-426614174002",
                    "importance": 0.8,
                    "metadata": {"source": "preferences"},
                    "created_at": "2026-01-05T12:00:00Z",
                    "accessed_at": "2026-01-05T12:00:00Z",
                    "access_count": 0,
                }
            ]
        }
    }


class MemorySearchRequest(BaseModel):
    """Request model for searching memories."""

    query: str = Field(
        ..., min_length=1, max_length=1000, description="Search query"
    )
    scope: MemoryScope = Field(..., description="Search starting scope")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    user_id: Optional[UUID] = Field(
        default=None, description="User ID for USER/SESSION scope"
    )
    session_id: Optional[UUID] = Field(
        default=None, description="Session ID for SESSION scope"
    )
    agent_id: Optional[str] = Field(
        default=None, description="Agent ID for AGENT scope"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    include_parent_scopes: bool = Field(
        default=True, description="Include memories from parent scopes in hierarchy"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "user preferences",
                    "scope": "session",
                    "tenant_id": "123e4567-e89b-12d3-a456-426614174001",
                    "user_id": "123e4567-e89b-12d3-a456-426614174002",
                    "session_id": "123e4567-e89b-12d3-a456-426614174003",
                    "limit": 10,
                    "include_parent_scopes": True,
                }
            ]
        }
    }


class MemorySearchResponse(BaseModel):
    """Response model for memory search."""

    memories: list[ScopedMemory] = Field(..., description="List of matching memories")
    total: int = Field(..., ge=0, description="Total number of matches")
    query: str = Field(..., description="Original search query")
    scopes_searched: list[MemoryScope] = Field(
        ..., description="Scopes that were searched"
    )


class MemoryListResponse(BaseModel):
    """Response model for listing memories."""

    memories: list[ScopedMemory] = Field(..., description="List of memories")
    total: int = Field(..., ge=0, description="Total number of memories matching filters")
    limit: int = Field(..., description="Maximum results returned")
    offset: int = Field(..., description="Offset used for pagination")


class DeleteByScopeResponse(BaseModel):
    """Response model for deleting memories by scope."""

    deleted_count: int = Field(..., ge=0, description="Number of memories deleted")
    scope: MemoryScope = Field(..., description="Scope that was cleared")
