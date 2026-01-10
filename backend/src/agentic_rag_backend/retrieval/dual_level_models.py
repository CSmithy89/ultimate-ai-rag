"""Dual-Level Retrieval data models (Story 20-C2).

This module defines the data models for dual-level retrieval that combines:
- Low-level retrieval: Entity/chunk level results (specific facts, entities)
- High-level retrieval: Community/theme level results (broader context)

The dual-level approach is inspired by LightRAG's architecture for combining
granular knowledge with thematic understanding.

Models:
- LowLevelResult: Entity/chunk level result
- HighLevelResult: Community/theme level result
- DualLevelResult: Combined result with synthesis
- Request/Response models for API
"""

from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================
# Pydantic Models for API Request/Response
# ============================================================


class DualLevelRetrieveRequest(BaseModel):
    """Request model for POST /api/v1/dual-level/retrieve."""

    query: str = Field(..., min_length=1, max_length=10000, description="Query string")
    tenant_id: UUID = Field(..., description="Tenant identifier for multi-tenancy")
    low_level_limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum low-level (entity/chunk) results",
    )
    high_level_limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum high-level (community/theme) results",
    )
    include_synthesis: bool = Field(
        default=True,
        description="Generate LLM synthesis of both levels",
    )
    low_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for low-level results in final ranking",
    )
    high_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for high-level results in final ranking",
    )


class LowLevelResultResponse(BaseModel):
    """Low-level (entity/chunk) result in response."""

    id: str = Field(..., description="Entity or chunk UUID")
    name: str = Field(..., description="Entity name or chunk title")
    type: str = Field(default="Entity", description="Result type (Entity, Chunk)")
    content: Optional[str] = Field(default=None, description="Entity description or chunk content")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    source: Optional[str] = Field(default=None, description="Source document or episode")


class HighLevelResultResponse(BaseModel):
    """High-level (community/theme) result in response."""

    id: str = Field(..., description="Community or theme UUID")
    name: str = Field(..., description="Community or theme name")
    summary: Optional[str] = Field(default=None, description="Community summary")
    keywords: list[str] = Field(default_factory=list, description="Theme keywords")
    level: int = Field(default=0, description="Hierarchy level (0=leaf, higher=broader)")
    entity_count: int = Field(default=0, description="Number of entities in community")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class DualLevelRetrieveResponse(BaseModel):
    """Response model for POST /api/v1/dual-level/retrieve."""

    query: str = Field(..., description="Original query")
    tenant_id: str = Field(..., description="Tenant identifier")
    low_level_results: list[LowLevelResultResponse] = Field(
        default_factory=list, description="Low-level (entity/chunk) results"
    )
    high_level_results: list[HighLevelResultResponse] = Field(
        default_factory=list, description="High-level (community/theme) results"
    )
    synthesis: Optional[str] = Field(
        default=None, description="LLM synthesis of both levels"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    low_level_count: int = Field(default=0, description="Number of low-level results")
    high_level_count: int = Field(default=0, description="Number of high-level results")
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    fallback_used: bool = Field(
        default=False, description="Whether fallback was used (one level empty)"
    )


class DualLevelStatusResponse(BaseModel):
    """Response model for GET /api/v1/dual-level/status."""

    enabled: bool = Field(..., description="Whether dual-level retrieval is enabled")
    low_weight: float = Field(..., description="Default low-level weight")
    high_weight: float = Field(..., description="Default high-level weight")
    low_limit: int = Field(..., description="Default low-level limit")
    high_limit: int = Field(..., description="Default high-level limit")
    synthesis_model: str = Field(..., description="LLM model for synthesis")
    synthesis_temperature: float = Field(..., description="LLM temperature for synthesis")
    graphiti_available: bool = Field(
        ..., description="Whether Graphiti is available for low-level"
    )
    community_detection_available: bool = Field(
        ..., description="Whether community detection is available for high-level"
    )


# ============================================================
# Internal Dataclasses for DualLevelRetriever
# ============================================================


@dataclass(frozen=True)
class LowLevelResult:
    """Internal low-level (entity/chunk) result representation.

    Represents granular knowledge items retrieved from:
    - Graphiti entities via semantic search
    - Vector chunks via pgvector similarity
    - Direct graph node matches
    """

    id: str
    name: str
    type: str = "Entity"
    content: Optional[str] = None
    score: float = 0.0
    source: Optional[str] = None
    labels: list[str] = field(default_factory=list)

    def to_response(self) -> LowLevelResultResponse:
        """Convert to API response model."""
        return LowLevelResultResponse(
            id=self.id,
            name=self.name,
            type=self.type,
            content=self.content,
            score=self.score,
            source=self.source,
        )


@dataclass(frozen=True)
class HighLevelResult:
    """Internal high-level (community/theme) result representation.

    Represents broader thematic context retrieved from:
    - Community summaries from Louvain/Leiden detection (20-B1)
    - Aggregated theme clusters
    - Higher-level graph structures
    """

    id: str
    name: str
    summary: Optional[str] = None
    keywords: tuple[str, ...] = ()
    level: int = 0
    entity_count: int = 0
    score: float = 0.0
    entity_ids: tuple[str, ...] = ()

    def to_response(self) -> HighLevelResultResponse:
        """Convert to API response model."""
        return HighLevelResultResponse(
            id=self.id,
            name=self.name,
            summary=self.summary,
            keywords=list(self.keywords),
            level=self.level,
            entity_count=self.entity_count,
            score=self.score,
        )


@dataclass
class SynthesisResult:
    """Result of LLM synthesis across both levels."""

    text: str
    confidence: float
    reasoning: Optional[str] = None


@dataclass
class DualLevelResult:
    """Complete result of dual-level retrieval.

    Combines low-level and high-level results with optional
    LLM synthesis that weaves together both perspectives.
    """

    query: str
    tenant_id: str
    low_level_results: list[LowLevelResult]
    high_level_results: list[HighLevelResult]
    synthesis: Optional[str]
    confidence: float
    processing_time_ms: int
    fallback_used: bool = False

    def to_response(self) -> DualLevelRetrieveResponse:
        """Convert to API response model."""
        return DualLevelRetrieveResponse(
            query=self.query,
            tenant_id=self.tenant_id,
            low_level_results=[r.to_response() for r in self.low_level_results],
            high_level_results=[r.to_response() for r in self.high_level_results],
            synthesis=self.synthesis,
            confidence=self.confidence,
            low_level_count=len(self.low_level_results),
            high_level_count=len(self.high_level_results),
            processing_time_ms=self.processing_time_ms,
            fallback_used=self.fallback_used,
        )
