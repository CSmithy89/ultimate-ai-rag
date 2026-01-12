"""LangGraph nodes for retrieval and response generation."""

from __future__ import annotations

import httpx


def query_rag(base_url: str, query: str) -> str:
    payload = {"tool": "knowledge.query", "params": {"query": query, "tenant_id": "default"}}
    response = httpx.post(f"{base_url}/api/v1/mcp/call", json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()
    return data.get("result", "")
