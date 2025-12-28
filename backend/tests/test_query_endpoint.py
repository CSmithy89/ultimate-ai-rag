from fastapi.testclient import TestClient

from agentic_rag_backend.main import create_app, get_orchestrator
from agentic_rag_backend.agents.orchestrator import OrchestratorResult
from agentic_rag_backend.retrieval_router import RetrievalStrategy
from agentic_rag_backend.schemas import PlanStep
import psycopg


class DummyOrchestrator:
    def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
        return OrchestratorResult(
            answer="ok",
            plan=[PlanStep(step="Step", status="completed")],
            thoughts=["Plan step: Step"],
            retrieval_strategy=RetrievalStrategy.VECTOR,
            trajectory_id=None,
        )


class ErrorOrchestrator:
    def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
        raise RuntimeError("boom")


class DbErrorOrchestrator:
    def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
        raise psycopg.OperationalError("db down")


def test_query_endpoint_envelope(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test"
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SKIP_DB_POOL", "1")

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: DummyOrchestrator()

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "hello",
                "tenant_id": "tenant-1",
                "session_id": "session-1",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert "data" in body
    assert "meta" in body
    assert body["data"]["answer"] == "ok"
    assert body["data"]["retrieval_strategy"] == "vector"
    assert body["meta"]["requestId"]
    assert body["meta"]["timestamp"]


def test_query_endpoint_rate_limit(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test"
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SKIP_DB_POOL", "1")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: DummyOrchestrator()

    with TestClient(app) as client:
        first = client.post(
            "/query",
            json={"query": "hello", "tenant_id": "tenant-1", "session_id": "session-1"},
        )
        second = client.post(
            "/query",
            json={"query": "hello", "tenant_id": "tenant-1", "session_id": "session-1"},
        )

    assert first.status_code == 200
    assert second.status_code == 429


def test_query_endpoint_internal_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test"
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SKIP_DB_POOL", "1")

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: ErrorOrchestrator()

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "hello", "tenant_id": "tenant-1", "session_id": "session-1"},
        )

    assert response.status_code == 500


def test_query_endpoint_db_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg://test:test@localhost:5432/test"
    )
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("SKIP_DB_POOL", "1")

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: DbErrorOrchestrator()

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "hello", "tenant_id": "tenant-1", "session_id": "session-1"},
        )

    assert response.status_code == 503
