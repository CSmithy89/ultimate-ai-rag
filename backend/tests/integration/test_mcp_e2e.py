"""Integration tests for MCP tool registry."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentic_rag_backend.api.utils import rate_limit_exceeded
from agentic_rag_backend.protocols.mcp import MCPToolRegistry
from agentic_rag_backend.schemas import PlanStep

pytestmark = pytest.mark.integration

TENANT_ID = "00000000-0000-0000-0000-000000000001"


class DummyOrchestrator:
    async def run(self, query: str, tenant_id: str, session_id: str | None = None):
        return SimpleNamespace(
            answer=f"ok:{query}",
            plan=[PlanStep(step="retrieve", status="completed")],
            thoughts=[],
            retrieval_strategy=SimpleNamespace(value="vector"),
            trajectory_id=None,
            evidence=None,
        )


class DummyNeo4j:
    async def get_visualization_stats(self, tenant_id: str):
        return {"nodes": 0, "edges": 0, "documents": 0}


@pytest.mark.asyncio
async def test_mcp_tool_invocation_returns_response() -> None:
    registry = MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j())

    tools = registry.list_tools()
    assert any(tool["name"] == "knowledge.query" for tool in tools)

    response = await registry.call_tool(
        "knowledge.query",
        {"query": "hello", "tenant_id": TENANT_ID},
    )
    assert response["answer"] == "ok:hello"
    assert response["retrieval_strategy"] == "vector"


@pytest.mark.asyncio
async def test_mcp_graph_stats_tool() -> None:
    registry = MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j())

    response = await registry.call_tool(
        "knowledge.graph_stats",
        {"tenant_id": TENANT_ID},
    )
    assert response["nodes"] == 0


def test_rate_limit_exceeded_includes_retry_after() -> None:
    exc = rate_limit_exceeded(retry_after_seconds=15)
    assert exc.status_code == 429
    assert exc.headers["Retry-After"] == "15"
