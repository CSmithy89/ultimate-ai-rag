"""Tests for A2A collaboration endpoints."""

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

import pytest
from fastapi import HTTPException

from agentic_rag_backend.api.routes.a2a import (
    CreateSessionRequest,
    MessageRequest,
    add_message,
    create_session,
    get_session,
)
from agentic_rag_backend.protocols.a2a import A2ASessionManager


class AllowLimiter:
    async def allow(self, key: str) -> bool:
        return True


class DenyLimiter:
    async def allow(self, key: str) -> bool:
        return False


@pytest.mark.asyncio
async def test_create_session_success() -> None:
    manager = A2ASessionManager()
    response = await create_session(
        request_body=CreateSessionRequest(tenant_id="tenant-1"),
        manager=manager,
        limiter=AllowLimiter(),
    )

    assert response.session["tenant_id"] == "tenant-1"
    assert response.session["session_id"]
    assert response.session["created_at"]


@pytest.mark.asyncio
async def test_add_message_success() -> None:
    manager = A2ASessionManager()
    session = await manager.create_session("tenant-1")

    response = await add_message(
        session_id=session.session_id,
        request_body=MessageRequest(
            tenant_id="tenant-1",
            sender="agent",
            content="hello",
        ),
        manager=manager,
        limiter=AllowLimiter(),
    )

    assert response.session["session_id"] == session.session_id
    assert response.session["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_get_session_tenant_mismatch() -> None:
    manager = A2ASessionManager()
    session = await manager.create_session("tenant-1")

    with pytest.raises(HTTPException) as exc_info:
        await get_session(
            session_id=session.session_id,
            tenant_id="tenant-2",
            manager=manager,
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_rate_limit_applies() -> None:
    manager = A2ASessionManager()
    with pytest.raises(HTTPException) as exc_info:
        await create_session(
            request_body=CreateSessionRequest(tenant_id="tenant-1"),
            manager=manager,
            limiter=DenyLimiter(),
        )
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_session_limit_enforced() -> None:
    manager = A2ASessionManager(max_sessions_per_tenant=1, max_sessions_total=1)
    await create_session(
        request_body=CreateSessionRequest(tenant_id="tenant-1"),
        manager=manager,
        limiter=AllowLimiter(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_session(
            request_body=CreateSessionRequest(tenant_id="tenant-1"),
            manager=manager,
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_message_limit_enforced() -> None:
    manager = A2ASessionManager(max_messages_per_session=1)
    session = await manager.create_session("tenant-1")

    await add_message(
        session_id=session.session_id,
        request_body=MessageRequest(
            tenant_id="tenant-1",
            sender="agent",
            content="hello",
        ),
        manager=manager,
        limiter=AllowLimiter(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await add_message(
            session_id=session.session_id,
            request_body=MessageRequest(
                tenant_id="tenant-1",
                sender="agent",
                content="second",
            ),
            manager=manager,
            limiter=AllowLimiter(),
        )
    assert exc_info.value.status_code == 409
