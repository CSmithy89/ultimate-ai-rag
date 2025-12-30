from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .validation import TENANT_ID_PATTERN


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(..., min_length=1, max_length=10000)
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        pattern=TENANT_ID_PATTERN,
    )
    session_id: str | None = Field(
        None,
        max_length=255,
        pattern=TENANT_ID_PATTERN,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Validate query input."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query must not be blank")
        if "\x00" in trimmed:
            raise ValueError("query contains invalid characters")
        return trimmed

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: str) -> str:
        """Validate tenant identifier input."""
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("tenant_id must not be blank")
        return trimmed


class PlanStep(BaseModel):
    step: str = Field(..., min_length=1)
    status: Literal["pending", "in_progress", "completed"]


class VectorCitation(BaseModel):
    chunk_id: str
    document_id: str
    similarity: float
    source: str | None = None
    content_preview: str
    metadata: dict[str, Any] | None = None


class GraphNodeEvidence(BaseModel):
    id: str
    name: str
    type: str
    description: str | None = None
    source_chunks: list[str] = Field(default_factory=list)


class GraphEdgeEvidence(BaseModel):
    source_id: str
    target_id: str
    type: str
    confidence: float | None = None
    source_chunk: str | None = None


class GraphPathEvidence(BaseModel):
    node_ids: list[str]
    edge_types: list[str]


class GraphEvidence(BaseModel):
    nodes: list[GraphNodeEvidence] = Field(default_factory=list)
    edges: list[GraphEdgeEvidence] = Field(default_factory=list)
    paths: list[GraphPathEvidence] = Field(default_factory=list)
    explanation: str | None = None


class RetrievalEvidence(BaseModel):
    vector: list[VectorCitation] = Field(default_factory=list)
    graph: GraphEvidence | None = None


class QueryResponse(BaseModel):
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
    retrieval_strategy: str
    trajectory_id: str | None = None
    evidence: RetrievalEvidence | None = None


class ResponseMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    request_id: str = Field(alias="requestId")
    timestamp: datetime


class QueryEnvelope(BaseModel):
    data: QueryResponse
    meta: ResponseMeta
