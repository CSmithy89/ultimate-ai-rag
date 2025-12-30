"""Tests for MCP tool endpoints."""

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

import pytest
from fastapi import HTTPException

from agentic_rag_backend.agents.orchestrator import OrchestratorResult
from agentic_rag_backend.api.routes.mcp import ToolCallRequest, call_tool, list_tools
from agentic_rag_backend.protocols.mcp import MCPToolRegistry
from agentic_rag_backend.retrieval_router import RetrievalStrategy
from agentic_rag_backend.schemas import PlanStep


class DummyOrchestrator:
    async def run(self, query: str, tenant_id: str, session_id: str | None = None) -> OrchestratorResult:
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


class DenyLimiter:
    async def allow(self, key: str) -> bool:
        return False


class DummyNeo4j:
    async def get_visualization_stats(self, tenant_id: str) -> dict:
        return {
            "node_count": 1,
            "edge_count": 2,
            "orphan_count": 0,
            "node_type_breakdown": {"Doc": 1},
            "edge_type_breakdown": {"RELATES_TO": 2},
        }


@pytest.mark.asyncio
async def test_list_tools_includes_defaults() -> None:
    registry = MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j())
    response = await list_tools(registry=registry)
    tool_names = {tool.name for tool in response.tools}
    assert "knowledge.query" in tool_names
    assert "knowledge.graph_stats" in tool_names


@pytest.mark.asyncio
async def test_call_tool_requires_tenant() -> None:
    request = ToolCallRequest(tool="knowledge.query", arguments={"query": "hello"})
    with pytest.raises(HTTPException) as exc_info:
        await call_tool(
            request_body=request,
            registry=MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j()),
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_call_tool_rate_limited() -> None:
    request = ToolCallRequest(
        tool="knowledge.query",
        arguments={"query": "hello", "tenant_id": "tenant-1"},
    )
    with pytest.raises(HTTPException) as exc_info:
        await call_tool(
            request_body=request,
            registry=MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j()),
            limiter=DenyLimiter(),
        )
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_call_tool_unknown_tool() -> None:
    request = ToolCallRequest(
        tool="unknown.tool",
        arguments={"tenant_id": "tenant-1"},
    )
    with pytest.raises(HTTPException) as exc_info:
        await call_tool(
            request_body=request,
            registry=MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j()),
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_call_tool_query_success() -> None:
    request = ToolCallRequest(
        tool="knowledge.query",
        arguments={"query": "hello", "tenant_id": "tenant-1"},
    )
    response = await call_tool(
        request_body=request,
        registry=MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j()),
        limiter=AllowLimiter(),
    )

    assert response.tool == "knowledge.query"
    assert response.result["answer"] == "ok"
    assert response.result["retrieval_strategy"] == "vector"


@pytest.mark.asyncio
async def test_call_tool_graph_stats_success() -> None:
    request = ToolCallRequest(
        tool="knowledge.graph_stats",
        arguments={"tenant_id": "tenant-1"},
    )
    response = await call_tool(
        request_body=request,
        registry=MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=DummyNeo4j()),
        limiter=AllowLimiter(),
    )

    assert response.tool == "knowledge.graph_stats"
    assert response.result["node_count"] == 1
    assert response.result["edge_count"] == 2


@pytest.mark.asyncio
async def test_call_tool_graph_stats_requires_neo4j() -> None:
    request = ToolCallRequest(
        tool="knowledge.graph_stats",
        arguments={"tenant_id": "tenant-1"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await call_tool(
            request_body=request,
            registry=MCPToolRegistry(orchestrator=DummyOrchestrator(), neo4j=None),
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 422
