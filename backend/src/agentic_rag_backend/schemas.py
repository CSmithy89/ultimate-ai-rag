from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


class PlanStep(BaseModel):
    step: str
    status: str


class QueryResponse(BaseModel):
    answer: str
    plan: list[PlanStep]
    thoughts: list[str]
    retrieval_strategy: str
