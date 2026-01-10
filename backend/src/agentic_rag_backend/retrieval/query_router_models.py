"""Query Router Models for Global/Local Query Routing (Story 20-B3).

This module provides Pydantic models and data classes for the query routing system:
- QueryType enum (GLOBAL, LOCAL, HYBRID)
- RoutingDecision dataclass for internal routing logic
- API request/response models for REST endpoints

The query router determines whether queries should be:
- GLOBAL: Processed via community-level retrieval (20-B1)
- LOCAL: Processed via entity-level retrieval (20-B2 LazyRAG)
- HYBRID: Processed via weighted combination of both approaches
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class QueryType(str, Enum):
    """Query classification types for routing decisions.

    Based on Microsoft GraphRAG's global vs local query distinction:
    - GLOBAL: High-level, abstract understanding (themes, summaries, trends)
    - LOCAL: Specific information about particular entities, facts, or details
    - HYBRID: Both high-level context and specific details
    """

    GLOBAL = "global"
    LOCAL = "local"
    HYBRID = "hybrid"


@dataclass
class RoutingDecision:
    """Internal routing decision container.

    Captures the full context of a routing decision including:
    - The determined query type
    - Confidence score (0.0-1.0)
    - Human-readable reasoning
    - Weights for hybrid retrieval

    Attributes:
        query_type: The classified query type (GLOBAL, LOCAL, or HYBRID)
        confidence: Confidence score between 0.0 and 1.0
        reasoning: Human-readable explanation for the decision
        global_weight: Weight for global (community-level) retrieval (0.0-1.0)
        local_weight: Weight for local (entity-level) retrieval (0.0-1.0)
        classification_method: How the decision was made ("rule_based", "llm", or "combined")
        global_matches: Number of global pattern matches
        local_matches: Number of local pattern matches
        processing_time_ms: Time taken to classify the query
    """

    query_type: QueryType
    confidence: float
    reasoning: str
    global_weight: float = field(default=0.0)
    local_weight: float = field(default=0.0)
    classification_method: str = field(default="rule_based")
    global_matches: int = field(default=0)
    local_matches: int = field(default=0)
    processing_time_ms: int = field(default=0)

    def __post_init__(self) -> None:
        """Validate and normalize weights after initialization."""
        # Ensure confidence is clamped to valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Set default weights based on query type if not explicitly set
        if self.global_weight == 0.0 and self.local_weight == 0.0:
            if self.query_type == QueryType.GLOBAL:
                self.global_weight = 1.0
                self.local_weight = 0.0
            elif self.query_type == QueryType.LOCAL:
                self.global_weight = 0.0
                self.local_weight = 1.0
            else:  # HYBRID
                self.global_weight = 0.5
                self.local_weight = 0.5

    def to_response(self) -> "QueryRouteResponse":
        """Convert to API response model."""
        return QueryRouteResponse(
            query_type=self.query_type,
            confidence=self.confidence,
            reasoning=self.reasoning,
            global_weight=self.global_weight,
            local_weight=self.local_weight,
            classification_method=self.classification_method,
            global_matches=self.global_matches,
            local_matches=self.local_matches,
            processing_time_ms=self.processing_time_ms,
        )


# Pydantic API Models


class QueryRouteRequest(BaseModel):
    """Request model for query routing endpoint.

    Attributes:
        query: The query text to classify
        tenant_id: Tenant identifier for multi-tenancy
        use_llm: Override for LLM classification (None = use config default)
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The query text to classify",
    )
    tenant_id: UUID = Field(
        ...,
        description="Tenant identifier for multi-tenancy",
    )
    use_llm: Optional[bool] = Field(
        default=None,
        description="Override for LLM classification. None uses config default.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the main themes in this document?",
                "tenant_id": "12345678-1234-5678-1234-567812345678",
                "use_llm": None,
            }
        }
    )


class QueryRouteResponse(BaseModel):
    """Response model for query routing endpoint.

    Attributes:
        query_type: The classified query type
        confidence: Confidence score between 0.0 and 1.0
        reasoning: Human-readable explanation for the decision
        global_weight: Weight for global (community-level) retrieval
        local_weight: Weight for local (entity-level) retrieval
        classification_method: How the decision was made
        global_matches: Number of global pattern matches
        local_matches: Number of local pattern matches
        processing_time_ms: Time taken to classify the query
    """

    query_type: QueryType = Field(
        ...,
        description="Classified query type: global, local, or hybrid",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0",
    )
    reasoning: str = Field(
        ...,
        description="Human-readable explanation for the decision",
    )
    global_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight for global (community-level) retrieval",
    )
    local_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Weight for local (entity-level) retrieval",
    )
    classification_method: str = Field(
        default="rule_based",
        description="Classification method used: rule_based, llm, or combined",
    )
    global_matches: int = Field(
        default=0,
        ge=0,
        description="Number of global pattern matches",
    )
    local_matches: int = Field(
        default=0,
        ge=0,
        description="Number of local pattern matches",
    )
    processing_time_ms: int = Field(
        default=0,
        ge=0,
        description="Processing time in milliseconds",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_type": "global",
                "confidence": 0.85,
                "reasoning": "Query asks about main themes, indicating corpus-wide understanding",
                "global_weight": 1.0,
                "local_weight": 0.0,
                "classification_method": "rule_based",
                "global_matches": 2,
                "local_matches": 0,
                "processing_time_ms": 5,
            }
        }
    )


class PatternListResponse(BaseModel):
    """Response model for pattern list endpoint (debugging).

    Attributes:
        global_patterns: List of global query pattern strings
        local_patterns: List of local query pattern strings
        global_pattern_count: Number of global patterns
        local_pattern_count: Number of local patterns
    """

    global_patterns: list[str] = Field(
        ...,
        description="List of global query pattern strings",
    )
    local_patterns: list[str] = Field(
        ...,
        description="List of local query pattern strings",
    )
    global_pattern_count: int = Field(
        ...,
        ge=0,
        description="Number of global patterns",
    )
    local_pattern_count: int = Field(
        ...,
        ge=0,
        description="Number of local patterns",
    )


class RouterStatusResponse(BaseModel):
    """Response model for router status endpoint.

    Attributes:
        enabled: Whether query routing is enabled
        use_llm: Whether LLM classification is enabled for uncertain queries
        llm_model: Model used for LLM classification
        confidence_threshold: Threshold below which LLM/hybrid fallback is used
        community_detection_available: Whether community detection (20-B1) is available
        lazy_rag_available: Whether LazyRAG (20-B2) is available
    """

    enabled: bool = Field(
        ...,
        description="Whether query routing is enabled",
    )
    use_llm: bool = Field(
        ...,
        description="Whether LLM classification is enabled for uncertain queries",
    )
    llm_model: str = Field(
        ...,
        description="Model used for LLM classification",
    )
    confidence_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Threshold below which LLM/hybrid fallback is used",
    )
    community_detection_available: bool = Field(
        ...,
        description="Whether community detection (20-B1) is available for global queries",
    )
    lazy_rag_available: bool = Field(
        ...,
        description="Whether LazyRAG (20-B2) is available for local queries",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "enabled": True,
                "use_llm": False,
                "llm_model": "gpt-4o-mini",
                "confidence_threshold": 0.7,
                "community_detection_available": True,
                "lazy_rag_available": True,
            }
        }
    )
