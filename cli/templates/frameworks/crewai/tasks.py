"""CrewAI tasks that call MCP tools via HTTP."""

from __future__ import annotations

from crewai import Task
import httpx


def create_research_task(agent) -> Task:
    return Task(
        description="Research GraphRAG and return key points.",
        expected_output="Key points with citations",
        agent=agent,
    )


def call_mcp_tool(base_url: str, query: str) -> str:
    payload = {"tool": "knowledge.query", "params": {"query": query, "tenant_id": "default"}}
    response = httpx.post(f"{base_url}/api/v1/mcp/call", json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data.get("result", "")
