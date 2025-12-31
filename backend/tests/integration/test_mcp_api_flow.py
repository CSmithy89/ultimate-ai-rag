"""Integration tests for MCP API flow."""

import os
from uuid import uuid4

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SKIP_DB_POOL", "1")
os.environ.setdefault("SKIP_GRAPHITI", "1")

from fastapi.testclient import TestClient

from agentic_rag_backend.agents.orchestrator import OrchestratorResult
from agentic_rag_backend.api.routes.mcp import get_mcp_registry
from agentic_rag_backend.protocols.mcp import MCPToolRegistry
from agentic_rag_backend.retrieval_router import RetrievalStrategy
from agentic_rag_backend.schemas import PlanStep


class DummyOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    async def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
        self.calls.append((query, tenant_id, session_id))
        return OrchestratorResult(
            answer="ok",
            plan=[PlanStep(step="Step", status="completed")],
            thoughts=["Plan step: Step"],
            retrieval_strategy=RetrievalStrategy.VECTOR,
            trajectory_id=uuid4(),
        )


class AllowLimiter:
    async def allow(self, key: str) -> bool:
        return True


def test_mcp_call_invokes_orchestrator() -> None:
    from agentic_rag_backend.main import app

    orchestrator = DummyOrchestrator()
    registry = MCPToolRegistry(orchestrator=orchestrator, neo4j=None)
    app.dependency_overrides[get_mcp_registry] = lambda: registry

    with TestClient(app) as client:
        app.state.rate_limiter = AllowLimiter()
        response = client.post(
            "/api/v1/mcp/call",
            json={
                "tool": "knowledge.query",
                "arguments": {"query": "hello", "tenant_id": "11111111-1111-1111-1111-111111111111"},
            },
        )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["answer"] == "ok"
    assert orchestrator.calls == [("hello", "11111111-1111-1111-1111-111111111111", None)]
