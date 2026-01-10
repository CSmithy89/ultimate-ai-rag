"""Pydantic models for Community Detection (Story 20-B1).

This module defines the data models for community detection in the knowledge graph:
- CommunityAlgorithm: Enum for supported algorithms (Louvain, Leiden)
- Community: Core community data model with hierarchy support
- CommunityDetectionRequest/Response: API request/response models
- CommunitySearchRequest/Response: Search API models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CommunityAlgorithm(str, Enum):
    """Supported community detection algorithms.

    LOUVAIN: General-purpose algorithm with good balance of quality and speed.
             Available via networkx.algorithms.community.louvain_communities.

    LEIDEN: Higher quality community detection, requires optional leidenalg package.
            Falls back to Louvain if leidenalg is not installed.
    """

    LOUVAIN = "louvain"
    LEIDEN = "leiden"


class Community(BaseModel):
    """A community of related entities in the knowledge graph.

    Communities are detected using graph clustering algorithms and represent
    groups of entities that are more densely connected to each other than
    to the rest of the graph. Each community has an LLM-generated summary
    that describes the theme and key entities.

    Attributes:
        id: Unique community identifier (UUID)
        name: LLM-generated community name
        level: Hierarchy level (0 = most granular, higher = more abstract)
        tenant_id: Tenant identifier for multi-tenancy
        entity_ids: List of entity IDs belonging to this community
        entity_count: Number of entities in this community
        summary: LLM-generated summary describing the community theme
        keywords: List of keywords extracted from the community
        parent_id: Parent community ID in hierarchy (if any)
        child_ids: Child community IDs in hierarchy (if any)
        created_at: Timestamp when community was created
        updated_at: Timestamp when community was last updated
    """

    id: str = Field(..., description="Community UUID")
    name: str = Field(..., description="LLM-generated community name")
    level: int = Field(default=0, ge=0, description="Hierarchy level (0 = most granular)")
    tenant_id: str = Field(..., description="Tenant identifier")
    entity_ids: list[str] = Field(default_factory=list, description="Entity IDs in community")
    entity_count: int = Field(default=0, ge=0, description="Number of entities")
    summary: Optional[str] = Field(default=None, description="LLM-generated summary")
    keywords: list[str] = Field(default_factory=list, description="Keywords for this community")
    parent_id: Optional[str] = Field(default=None, description="Parent community ID")
    child_ids: list[str] = Field(default_factory=list, description="Child community IDs")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True)


class CommunityDetectionRequest(BaseModel):
    """Request model for triggering community detection.

    Attributes:
        tenant_id: Tenant identifier (required)
        algorithm: Community detection algorithm to use (default: louvain)
        generate_summaries: Whether to generate LLM summaries (default: true)
        min_community_size: Minimum entities per community (default: from config)
        max_levels: Maximum hierarchy levels to generate (default: from config)
    """

    tenant_id: UUID = Field(..., description="Tenant identifier")
    algorithm: CommunityAlgorithm = Field(
        default=CommunityAlgorithm.LOUVAIN,
        description="Algorithm to use for detection",
    )
    generate_summaries: bool = Field(
        default=True,
        description="Generate LLM summaries for communities",
    )
    min_community_size: Optional[int] = Field(
        default=None,
        ge=2,
        description="Override minimum community size from config",
    )
    max_levels: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Override maximum hierarchy levels from config",
    )


class CommunityDetectionResponse(BaseModel):
    """Response model for community detection.

    Attributes:
        communities_created: Number of communities detected
        levels_generated: Number of hierarchy levels generated
        algorithm: Algorithm used for detection
        processing_time_ms: Time taken for detection in milliseconds
        tenant_id: Tenant identifier
        communities: List of detected communities (top-level summary)
    """

    communities_created: int = Field(..., description="Number of communities created")
    levels_generated: int = Field(..., description="Hierarchy levels generated")
    algorithm: CommunityAlgorithm = Field(..., description="Algorithm used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    tenant_id: str = Field(..., description="Tenant identifier")
    communities: list[Community] = Field(
        default_factory=list,
        description="Top-level communities created",
    )


class CommunitySearchRequest(BaseModel):
    """Request model for searching communities.

    Attributes:
        query: Search query string
        tenant_id: Tenant identifier
        level: Filter by hierarchy level (optional)
        limit: Maximum results to return (default: 10)
    """

    query: str = Field(..., min_length=1, description="Search query")
    tenant_id: UUID = Field(..., description="Tenant identifier")
    level: Optional[int] = Field(default=None, ge=0, description="Filter by level")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")


class CommunitySearchResponse(BaseModel):
    """Response model for community search.

    Attributes:
        communities: List of matching communities
        total: Total matching communities
        query: Original search query
    """

    communities: list[Community] = Field(..., description="Matching communities")
    total: int = Field(..., description="Total matches")
    query: str = Field(..., description="Original query")


class CommunityListResponse(BaseModel):
    """Response model for listing communities.

    Attributes:
        communities: List of communities
        total: Total count
        limit: Requested limit
        offset: Requested offset
        level: Level filter applied (if any)
    """

    communities: list[Community] = Field(..., description="Community list")
    total: int = Field(..., description="Total count")
    limit: int = Field(..., description="Requested limit")
    offset: int = Field(..., description="Requested offset")
    level: Optional[int] = Field(default=None, description="Level filter")


class CommunityWithEntities(Community):
    """Extended community model with full entity details.

    Used when fetching a single community with its entity information.

    Attributes:
        entities: List of entity dictionaries with details
    """

    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entity details",
    )
