"""Tests for A2A session manager concurrency and TTL behavior."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agentic_rag_backend.protocols.a2a import A2ASessionManager


@pytest.mark.asyncio
async def test_a2a_prunes_expired_sessions() -> None:
    manager = A2ASessionManager(session_ttl_seconds=1)
    session = await manager.create_session("tenant-1")
    manager._sessions[session.session_id].last_activity = datetime.now(timezone.utc) - timedelta(seconds=5)

    fetched = await manager.get_session(session.session_id)

    assert fetched is None


@pytest.mark.asyncio
async def test_a2a_concurrent_session_creation() -> None:
    manager = A2ASessionManager()

    sessions = await asyncio.gather(
        *[manager.create_session("tenant-1") for _ in range(5)]
    )

    session_ids = {session.session_id for session in sessions}
    assert len(session_ids) == 5


@pytest.mark.asyncio
async def test_a2a_concurrent_message_add() -> None:
    manager = A2ASessionManager()
    session = await manager.create_session("tenant-1")

    await asyncio.gather(
        *[
            manager.add_message(
                session_id=session.session_id,
                tenant_id="tenant-1",
                sender="agent",
                content=f"message-{idx}",
            )
            for idx in range(5)
        ]
    )

    fetched = await manager.get_session(session.session_id)
    assert fetched is not None
    assert len(fetched.messages) == 5
