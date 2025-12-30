import pytest
from pydantic import ValidationError

from agentic_rag_backend.schemas import PlanStep, QueryRequest


def test_query_request_trims_tenant_id() -> None:
    model = QueryRequest(
        query="hello",
        tenant_id=" 11111111-1111-1111-1111-111111111111 ",
        session_id="s1",
    )
    assert model.tenant_id == "11111111-1111-1111-1111-111111111111"


def test_query_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(
            query="   ",
            tenant_id="11111111-1111-1111-1111-111111111111",
            session_id="s1",
        )


def test_plan_step_status_validation() -> None:
    with pytest.raises(ValidationError):
        PlanStep(step="Step", status="unknown")
