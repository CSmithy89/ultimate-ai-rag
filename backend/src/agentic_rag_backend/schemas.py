from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    tenant_id: str = Field(..., min_length=1, max_length=255)
    session_id: str | None = Field(None, max_length=255)


class PlanStep(BaseModel):
    step: str
    status: str


class QueryResponse(BaseModel):
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
    retrieval_strategy: str
    trajectory_id: str | None = None
