"""Async SDK client for Agentic RAG protocol endpoints."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from .models import A2ASessionEnvelope, MCPToolCallResult, MCPToolList


class AgenticRagClient:
    """Async SDK client for MCP and A2A APIs."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if http_client is None:
            self._client = httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
            )
            self._owns_client = True
        else:
            self._client = http_client
            self._owns_client = False

    async def __aenter__(self) -> "AgenticRagClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def list_tools(self) -> MCPToolList:
        response = await self._client.get("/api/v1/mcp/tools")
        response.raise_for_status()
        return MCPToolList(**response.json())

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        response = await self._client.post(
            "/api/v1/mcp/call",
            json={"tool": tool, "arguments": arguments},
        )
        response.raise_for_status()
        return MCPToolCallResult(**response.json())

    async def create_a2a_session(self, tenant_id: str) -> A2ASessionEnvelope:
        response = await self._client.post(
            "/api/v1/a2a/sessions",
            json={"tenant_id": tenant_id},
        )
        response.raise_for_status()
        return A2ASessionEnvelope(**response.json())

    async def add_a2a_message(
        self,
        session_id: str,
        tenant_id: str,
        sender: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> A2ASessionEnvelope:
        response = await self._client.post(
            f"/api/v1/a2a/sessions/{session_id}/messages",
            json={
                "tenant_id": tenant_id,
                "sender": sender,
                "content": content,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return A2ASessionEnvelope(**response.json())

    async def get_a2a_session(self, session_id: str, tenant_id: str) -> A2ASessionEnvelope:
        response = await self._client.get(
            f"/api/v1/a2a/sessions/{session_id}",
            params={"tenant_id": tenant_id},
        )
        response.raise_for_status()
        return A2ASessionEnvelope(**response.json())
