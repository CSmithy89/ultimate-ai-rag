from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9._:-]{1,255}$",
    )
    session_id: str | None = Field(
        None,
        max_length=255,
        pattern=r"^[A-Za-z0-9._:-]{1,255}$",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("query must not be blank")
        if "\x00" in trimmed:
            raise ValueError("query contains invalid characters")
        return trimmed

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("tenant_id must not be blank")
        return trimmed


class PlanStep(BaseModel):
    step: str
    status: str


class QueryResponse(BaseModel):
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
    retrieval_strategy: str
    trajectory_id: str | None = None


class ResponseMeta(BaseModel):
    request_id: str
    timestamp: datetime


class QueryEnvelope(BaseModel):
    data: QueryResponse
    meta: ResponseMeta
