from fastapi.testclient import TestClient

from agentic_rag_backend.main import create_app, get_orchestrator
from agentic_rag_backend.agents.orchestrator import OrchestratorResult
from agentic_rag_backend.retrieval_router import RetrievalStrategy
from agentic_rag_backend.schemas import PlanStep


class DummyOrchestrator:
    def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
        return OrchestratorResult(
            answer="ok",
            plan=[PlanStep(step="Step", status="completed")],
            thoughts=["Plan step: Step"],
            retrieval_strategy=RetrievalStrategy.VECTOR,
            trajectory_id=None,
        )


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
