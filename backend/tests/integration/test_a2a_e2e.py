"""Integration tests for A2A session lifecycle."""

from __future__ import annotations

import pytest

from agentic_rag_backend.protocols.a2a import A2ASessionManager

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_a2a_session_message_flow() -> None:
    manager = A2ASessionManager(max_sessions_total=10)
    session = await manager.create_session("tenant-a")

    session_id = session["session_id"]
    updated = await manager.add_message(
        session_id=session_id,
        tenant_id="tenant-a",
        sender="agent",
        content="hello",
    )

    assert updated["messages"][0]["content"] == "hello"

    fetched = await manager.get_session(session_id)
    assert fetched is not None
    assert fetched["session_id"] == session_id
    assert len(fetched["messages"]) == 1
