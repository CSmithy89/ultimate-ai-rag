"""Tests for the Python SDK client."""

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("SKIP_DB_POOL", "1")
os.environ.setdefault("SKIP_GRAPHITI", "1")

import httpx
import pytest

from agentic_rag_backend.sdk.client import (
    AgenticRagClient,
    AgenticRagHTTPError,
)


@pytest.mark.asyncio
async def test_sdk_list_tools_and_call() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/mcp/tools":
            return httpx.Response(
                200,
                json={
                    "tools": [
                        {
                            "name": "knowledge.query",
                            "description": "Run a RAG query",
                            "input_schema": {"type": "object"},
                        }
                    ],
                    "meta": {"requestId": "1"},
                },
            )
        if request.url.path == "/api/v1/mcp/call":
            return httpx.Response(
                200,
                json={
                    "tool": "knowledge.query",
                    "result": {"answer": "ok"},
                    "meta": {"requestId": "2"},
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = AgenticRagClient(base_url="http://test", http_client=http_client)
        tools = await client.list_tools()
        assert tools.tools[0].name == "knowledge.query"

        result = await client.call_tool("knowledge.query", {"tenant_id": "t", "query": "q"})
        assert result.result["answer"] == "ok"


@pytest.mark.asyncio
async def test_sdk_a2a_session_flow() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/a2a/sessions":
            return httpx.Response(
                200,
                json={
                    "session": {
                        "session_id": "session-1",
                        "tenant_id": "tenant-1",
                        "created_at": "2025-01-01T00:00:00Z",
                        "messages": [],
                    },
                    "meta": {"requestId": "1"},
                },
            )
        if request.url.path == "/api/v1/a2a/sessions/session-1/messages":
            return httpx.Response(
                200,
                json={
                    "session": {
                        "session_id": "session-1",
                        "tenant_id": "tenant-1",
                        "created_at": "2025-01-01T00:00:00Z",
                        "messages": [
                            {
                                "sender": "agent",
                                "content": "hello",
                                "timestamp": "2025-01-01T00:00:01Z",
                                "metadata": {},
                            }
                        ],
                    },
                    "meta": {"requestId": "2"},
                },
            )
        if request.url.path == "/api/v1/a2a/sessions/session-1":
            return httpx.Response(
                200,
                json={
                    "session": {
                        "session_id": "session-1",
                        "tenant_id": "tenant-1",
                        "created_at": "2025-01-01T00:00:00Z",
                        "messages": [],
                    },
                    "meta": {"requestId": "3"},
                },
            )
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = AgenticRagClient(base_url="http://test", http_client=http_client)

        session = await client.create_a2a_session("tenant-1")
        assert session.session.session_id == "session-1"

        session = await client.add_a2a_message(
            "session-1",
            tenant_id="tenant-1",
            sender="agent",
            content="hello",
        )
        assert session.session.messages[0].content == "hello"

        session = await client.get_a2a_session("session-1", tenant_id="tenant-1")
        assert session.session.session_id == "session-1"


@pytest.mark.asyncio
async def test_sdk_raises_for_error_responses() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = AgenticRagClient(base_url="http://test", http_client=http_client)
        with pytest.raises(AgenticRagHTTPError):
            await client.list_tools()


@pytest.mark.asyncio
async def test_sdk_retries_on_transient_error() -> None:
    calls = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(503, json={"detail": "unavailable"})
        return httpx.Response(
            200,
            json={
                "tools": [
                    {
                        "name": "knowledge.query",
                        "description": "Run a RAG query",
                        "input_schema": {"type": "object"},
                    }
                ],
                "meta": {"requestId": "1"},
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http_client:
        client = AgenticRagClient(base_url="http://test", http_client=http_client, max_retries=1)
        tools = await client.list_tools()
        assert tools.tools[0].name == "knowledge.query"
        assert calls["count"] == 2
