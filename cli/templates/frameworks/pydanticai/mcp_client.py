"""Minimal MCP client helper for querying knowledge."""

from __future__ import annotations

import httpx


async def query_knowledge(base_url: str, query: str, tenant_id: str = "default") -> str:
    payload = {"tool": "knowledge.query", "arguments": {"query": query, "tenant_id": tenant_id}}
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/api/v1/mcp/call", json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data.get("result", "")
