from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
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
    step: str = Field(..., min_length=1)
    status: Literal["pending", "in_progress", "completed"]


class QueryResponse(BaseModel):
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
    retrieval_strategy: str
    trajectory_id: str | None = None


class ResponseMeta(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    request_id: str = Field(alias="requestId")
    timestamp: datetime


class QueryEnvelope(BaseModel):
    data: QueryResponse
    meta: ResponseMeta
