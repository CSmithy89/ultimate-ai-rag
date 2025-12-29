import pytest
from fastapi import HTTPException

from agentic_rag_backend.main import run_query
from agentic_rag_backend.agents.orchestrator import OrchestratorResult
from agentic_rag_backend.retrieval_router import RetrievalStrategy
from agentic_rag_backend.schemas import PlanStep, QueryRequest
import psycopg


class DummyOrchestrator:
    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        return OrchestratorResult(
            answer="ok",
            plan=[PlanStep(step="Step", status="completed")],
            thoughts=["Plan step: Step"],
            retrieval_strategy=RetrievalStrategy.VECTOR,
            trajectory_id=None,
        )


class ErrorOrchestrator:
    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        raise RuntimeError("boom")


class DbErrorOrchestrator:
    async def run(
        self, query: str, tenant_id: str, session_id: str | None = None
    ) -> OrchestratorResult:
        raise psycopg.OperationalError("db down")


class AllowLimiter:
    async def allow(self, key: str) -> bool:
        return True


class DenyLimiter:
    async def allow(self, key: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_query_endpoint_envelope() -> None:
    payload = QueryRequest(
        query="hello",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    response = await run_query(
        payload,
        orchestrator=DummyOrchestrator(),
        limiter=AllowLimiter(),
    )

    assert response.data.answer == "ok"
    assert response.data.retrieval_strategy == "vector"
    assert response.meta.request_id
    assert response.meta.timestamp


@pytest.mark.asyncio
async def test_query_endpoint_rate_limit() -> None:
    payload = QueryRequest(
        query="hello",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    with pytest.raises(HTTPException) as exc_info:
        await run_query(
            payload,
            orchestrator=DummyOrchestrator(),
            limiter=DenyLimiter(),
        )
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_query_endpoint_internal_error() -> None:
    payload = QueryRequest(
        query="hello",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    with pytest.raises(HTTPException) as exc_info:
        await run_query(
            payload,
            orchestrator=ErrorOrchestrator(),
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_query_endpoint_db_error() -> None:
    payload = QueryRequest(
        query="hello",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    with pytest.raises(HTTPException) as exc_info:
        await run_query(
            payload,
            orchestrator=DbErrorOrchestrator(),
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 503
