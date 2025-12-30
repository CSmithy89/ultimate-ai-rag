"""MCP-style tool registry and invocation helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field

from ..agents.orchestrator import OrchestratorAgent
from ..db.neo4j import Neo4jClient
from ..schemas import QueryRequest


class MCPToolNotFoundError(KeyError):
    """Raised when a requested tool is not registered."""


class GraphStatsRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
    timeout_seconds: float | None = None


class MCPToolRegistry:
    """Registry for MCP-style tool definitions and execution."""

    def __init__(
        self,
        orchestrator: OrchestratorAgent,
        neo4j: Neo4jClient | None,
        timeout_seconds: float | None = None,
        tool_timeouts: dict[str, float] | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._neo4j = neo4j
        self._default_timeout_seconds = timeout_seconds
        self._tool_timeouts = tool_timeouts or {}
        self._tools = {
            "knowledge.query": MCPTool(
                name="knowledge.query",
                description="Run a RAG query against the orchestrator.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "tenant_id": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                    "required": ["query", "tenant_id"],
                },
                handler=self._run_query,
                timeout_seconds=self._tool_timeouts.get("knowledge.query"),
            ),
            "knowledge.graph_stats": MCPTool(
                name="knowledge.graph_stats",
                description="Fetch knowledge graph statistics for a tenant.",
                input_schema={
                    "type": "object",
                    "properties": {"tenant_id": {"type": "string"}},
                    "required": ["tenant_id"],
                },
                handler=self._graph_stats,
                timeout_seconds=self._tool_timeouts.get("knowledge.graph_stats"),
            ),
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return tool descriptors for discovery."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool by name with arguments."""
        tool = self._tools.get(name)
        if not tool:
            raise MCPToolNotFoundError(name)
        timeout = (
            tool.timeout_seconds
            if tool.timeout_seconds is not None
            else self._default_timeout_seconds
        )
        if timeout and timeout > 0:
            return await asyncio.wait_for(tool.handler(arguments), timeout=timeout)
        return await tool.handler(arguments)

    async def _run_query(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        tenant_id = arguments.get("tenant_id")
        session_id = arguments.get("session_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required")
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("tenant_id is required")
        payload = QueryRequest(
            query=query,
            tenant_id=tenant_id,
            session_id=session_id if isinstance(session_id, str) else None,
        )
        result = await self._orchestrator.run(
            payload.query,
            payload.tenant_id,
            payload.session_id,
        )
        return {
            "answer": result.answer,
            "plan": [step.model_dump() for step in result.plan],
            "thoughts": result.thoughts,
            "retrieval_strategy": result.retrieval_strategy.value,
            "trajectory_id": str(result.trajectory_id) if result.trajectory_id else None,
            "evidence": result.evidence.model_dump() if result.evidence else None,
        }

    async def _graph_stats(self, arguments: dict[str, Any]) -> dict[str, Any]:
        payload = GraphStatsRequest(**arguments)
        tenant_id = payload.tenant_id
        if self._neo4j is None:
            raise ValueError("neo4j client not available")
        return await self._neo4j.get_visualization_stats(tenant_id=str(tenant_id))
