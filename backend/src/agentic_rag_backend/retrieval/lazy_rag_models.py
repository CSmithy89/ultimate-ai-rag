"""LazyRAG data models (Story 20-B2).

This module defines the data models for LazyRAG query-time summarization:
- LazyRAGQuery: Input parameters for lazy RAG queries
- LazyRAGEntity: Entity representation in subgraph
- LazyRAGRelationship: Relationship representation in subgraph
- LazyRAGResult: Complete result with summary and confidence
- SubgraphExpansionResult: Intermediate result from subgraph expansion

LazyRAG defers graph summarization to query time, achieving up to 99%
reduction in indexing costs compared to MS GraphRAG's eager approach.
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================
# Pydantic Models for API Request/Response
# ============================================================


class LazyRAGQueryRequest(BaseModel):
    """Request model for POST /api/v1/lazy-rag/query."""

    query: str = Field(..., min_length=1, max_length=10000, description="Query string")
    tenant_id: UUID = Field(..., description="Tenant identifier for multi-tenancy")
    max_entities: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum entities to include in context",
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum relationship hops for subgraph expansion",
    )
    use_communities: bool = Field(
        default=True,
        description="Include community context from Story 20-B1",
    )
    include_summary: bool = Field(
        default=True,
        description="Generate LLM summary (False returns only subgraph)",
    )


class LazyRAGExpandRequest(BaseModel):
    """Request model for POST /api/v1/lazy-rag/expand (debug endpoint)."""

    query: str = Field(..., min_length=1, max_length=10000, description="Query string")
    tenant_id: UUID = Field(..., description="Tenant identifier for multi-tenancy")
    max_entities: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum entities to include",
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum relationship hops",
    )


class LazyRAGEntityResponse(BaseModel):
    """Entity representation in LazyRAG response."""

    id: str = Field(..., description="Entity UUID")
    name: str = Field(..., description="Entity name")
    type: str = Field(default="Entity", description="Entity type/label")
    description: Optional[str] = Field(default=None, description="Entity description")
    summary: Optional[str] = Field(default=None, description="Entity summary")


class LazyRAGRelationshipResponse(BaseModel):
    """Relationship representation in LazyRAG response."""

    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    type: str = Field(..., description="Relationship type")
    fact: Optional[str] = Field(default=None, description="Relationship fact/description")


class LazyRAGCommunityResponse(BaseModel):
    """Community context in LazyRAG response."""

    id: str = Field(..., description="Community ID")
    name: str = Field(..., description="Community name")
    summary: Optional[str] = Field(default=None, description="Community summary")
    keywords: list[str] = Field(default_factory=list, description="Community keywords")
    level: int = Field(default=0, description="Hierarchy level")


class LazyRAGQueryResponse(BaseModel):
    """Response model for POST /api/v1/lazy-rag/query."""

    query: str = Field(..., description="Original query")
    tenant_id: str = Field(..., description="Tenant identifier")
    summary: Optional[str] = Field(default=None, description="LLM-generated summary")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: list[LazyRAGEntityResponse] = Field(
        default_factory=list, description="Entities in subgraph"
    )
    relationships: list[LazyRAGRelationshipResponse] = Field(
        default_factory=list, description="Relationships in subgraph"
    )
    communities: list[LazyRAGCommunityResponse] = Field(
        default_factory=list, description="Relevant communities"
    )
    seed_entity_count: int = Field(default=0, description="Number of seed entities found")
    expanded_entity_count: int = Field(default=0, description="Total entities after expansion")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    missing_info: Optional[str] = Field(
        default=None, description="Description of missing information"
    )


class LazyRAGExpandResponse(BaseModel):
    """Response model for POST /api/v1/lazy-rag/expand."""

    query: str = Field(..., description="Original query")
    tenant_id: str = Field(..., description="Tenant identifier")
    entities: list[LazyRAGEntityResponse] = Field(
        default_factory=list, description="Entities in subgraph"
    )
    relationships: list[LazyRAGRelationshipResponse] = Field(
        default_factory=list, description="Relationships in subgraph"
    )
    seed_entity_count: int = Field(default=0, description="Number of seed entities found")
    expanded_entity_count: int = Field(default=0, description="Total entities after expansion")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")


class LazyRAGStatusResponse(BaseModel):
    """Response model for GET /api/v1/lazy-rag/status."""

    enabled: bool = Field(..., description="Whether LazyRAG feature is enabled")
    max_entities: int = Field(..., description="Configured max entities")
    max_hops: int = Field(..., description="Configured max hops")
    use_communities: bool = Field(..., description="Whether community context is enabled")
    summary_model: str = Field(..., description="LLM model for summaries")
    community_detection_available: bool = Field(
        ..., description="Whether community detection (20-B1) is available"
    )


# ============================================================
# Internal Dataclasses for LazyRAGRetriever
# ============================================================


@dataclass(frozen=True)
class LazyRAGEntity:
    """Internal entity representation for LazyRAG processing."""

    id: str
    name: str
    type: str = "Entity"
    description: Optional[str] = None
    summary: Optional[str] = None
    labels: list[str] = field(default_factory=list)

    def to_response(self) -> LazyRAGEntityResponse:
        """Convert to API response model."""
        return LazyRAGEntityResponse(
            id=self.id,
            name=self.name,
            type=self.type,
            description=self.description,
            summary=self.summary,
        )


@dataclass(frozen=True)
class LazyRAGRelationship:
    """Internal relationship representation for LazyRAG processing."""

    source_id: str
    target_id: str
    type: str
    fact: Optional[str] = None
    confidence: Optional[float] = None

    def to_response(self) -> LazyRAGRelationshipResponse:
        """Convert to API response model."""
        return LazyRAGRelationshipResponse(
            source_id=self.source_id,
            target_id=self.target_id,
            type=self.type,
            fact=self.fact,
        )


@dataclass(frozen=True)
class LazyRAGCommunity:
    """Internal community representation for LazyRAG processing."""

    id: str
    name: str
    summary: Optional[str] = None
    keywords: tuple[str, ...] = ()
    level: int = 0

    def to_response(self) -> LazyRAGCommunityResponse:
        """Convert to API response model."""
        return LazyRAGCommunityResponse(
            id=self.id,
            name=self.name,
            summary=self.summary,
            keywords=list(self.keywords),
            level=self.level,
        )


@dataclass
class SummaryResult:
    """Result of LLM summary generation."""

    text: str
    confidence: float
    missing_info: Optional[str] = None


@dataclass
class SubgraphExpansionResult:
    """Result of subgraph expansion from seed entities."""

    entities: list[LazyRAGEntity]
    relationships: list[LazyRAGRelationship]
    seed_count: int
    expanded_count: int


@dataclass
class LazyRAGResult:
    """Complete result of a LazyRAG query."""

    query: str
    tenant_id: str
    entities: list[LazyRAGEntity]
    relationships: list[LazyRAGRelationship]
    communities: list[LazyRAGCommunity]
    summary: Optional[str]
    confidence: float
    seed_entity_count: int
    expanded_entity_count: int
    processing_time_ms: int
    missing_info: Optional[str] = None

    def to_response(self) -> LazyRAGQueryResponse:
        """Convert to API response model."""
        return LazyRAGQueryResponse(
            query=self.query,
            tenant_id=self.tenant_id,
            summary=self.summary,
            confidence=self.confidence,
            entities=[e.to_response() for e in self.entities],
            relationships=[r.to_response() for r in self.relationships],
            communities=[c.to_response() for c in self.communities],
            seed_entity_count=self.seed_entity_count,
            expanded_entity_count=self.expanded_entity_count,
            processing_time_ms=self.processing_time_ms,
            missing_info=self.missing_info,
        )
