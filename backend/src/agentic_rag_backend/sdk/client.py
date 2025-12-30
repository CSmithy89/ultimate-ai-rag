"""Async SDK client for Agentic RAG protocol endpoints."""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Iterable

import httpx

from .models import A2ASessionEnvelope, MCPToolCallResult, MCPToolList


class AgenticRagClientError(Exception):
    """Base exception for SDK errors."""


class AgenticRagHTTPError(AgenticRagClientError):
    """Raised for non-success HTTP responses."""

    def __init__(self, status_code: int, message: str | None = None) -> None:
        self.status_code = status_code
        super().__init__(message or f"Request failed with status {status_code}")


class AgenticRagNetworkError(AgenticRagClientError):
    """Raised for network or transport errors."""


class AgenticRagClient:
    """Async SDK client for MCP and A2A APIs."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
        backoff_factor: float = 0.5,
        retry_statuses: Iterable[int] = (429, 503),
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
        self._max_retries = max(0, max_retries)
        self._backoff_factor = max(0.0, backoff_factor)
        self._retry_statuses = tuple(retry_statuses)

    def _get_retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        delay = self._backoff_factor * (2**attempt)
        if response is None:
            return delay
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                retry_after_seconds = float(retry_after)
            except ValueError:
                return delay
            return max(delay, retry_after_seconds)
        return delay

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, url, **kwargs)
            except httpx.RequestError as exc:
                if attempt >= self._max_retries:
                    raise AgenticRagNetworkError(str(exc)) from exc
                delay = self._get_retry_delay(None, attempt)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            if response.status_code in self._retry_statuses and attempt < self._max_retries:
                delay = self._get_retry_delay(response, attempt)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise AgenticRagHTTPError(
                    status_code=exc.response.status_code,
                    message=f"{exc.response.text} (original: {exc!r})",
                ) from exc
            return response

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
        response = await self._request("GET", "/api/v1/mcp/tools")
        return MCPToolList(**response.json())

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        response = await self._request(
            "POST",
            "/api/v1/mcp/call",
            json={"tool": tool, "arguments": arguments},
        )
        return MCPToolCallResult(**response.json())

    async def create_a2a_session(self, tenant_id: str) -> A2ASessionEnvelope:
        response = await self._request(
            "POST",
            "/api/v1/a2a/sessions",
            json={"tenant_id": tenant_id},
        )
        return A2ASessionEnvelope(**response.json())

    async def add_a2a_message(
        self,
        session_id: str,
        tenant_id: str,
        sender: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> A2ASessionEnvelope:
        response = await self._request(
            "POST",
            f"/api/v1/a2a/sessions/{session_id}/messages",
            json={
                "tenant_id": tenant_id,
                "sender": sender,
                "content": content,
                "metadata": metadata,
            },
        )
        return A2ASessionEnvelope(**response.json())

    async def get_a2a_session(self, session_id: str, tenant_id: str) -> A2ASessionEnvelope:
        response = await self._request(
            "GET",
            f"/api/v1/a2a/sessions/{session_id}",
            params={"tenant_id": tenant_id},
        )
        return A2ASessionEnvelope(**response.json())
