"""Tests for A2A session manager concurrency and TTL behavior."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agentic_rag_backend.protocols.a2a import A2ASessionManager


@pytest.mark.asyncio
async def test_a2a_prunes_expired_sessions() -> None:
    manager = A2ASessionManager(session_ttl_seconds=1)
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")
    manager._sessions[session["session_id"]].last_activity = datetime.now(timezone.utc) - timedelta(seconds=5)
    manager._prune_interval_seconds = 0

    fetched = await manager.get_session(session["session_id"])

    assert fetched is None


@pytest.mark.asyncio
async def test_a2a_cleanup_task_prunes_expired_sessions() -> None:
    manager = A2ASessionManager(session_ttl_seconds=1)
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")
    manager._sessions[session["session_id"]].last_activity = datetime.now(timezone.utc) - timedelta(seconds=5)

    await manager.start_cleanup_task(0.01)
    try:
        await asyncio.sleep(0.02)
        assert session["session_id"] not in manager._sessions
    finally:
        await manager.stop_cleanup_task()


@pytest.mark.asyncio
async def test_a2a_concurrent_session_creation() -> None:
    manager = A2ASessionManager()

    sessions = await asyncio.gather(
        *[manager.create_session("11111111-1111-1111-1111-111111111111") for _ in range(5)]
    )

    session_ids = {session["session_id"] for session in sessions}
    assert len(session_ids) == 5


@pytest.mark.asyncio
async def test_a2a_concurrent_message_add() -> None:
    manager = A2ASessionManager()
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")

    await asyncio.gather(
        *[
            manager.add_message(
                session_id=session["session_id"],
                tenant_id="11111111-1111-1111-1111-111111111111",
                sender="agent",
                content=f"message-{idx}",
            )
            for idx in range(5)
        ]
    )

    fetched = await manager.get_session(session["session_id"])
    assert fetched is not None
    assert len(fetched["messages"]) == 5


@pytest.mark.asyncio
async def test_a2a_persists_sessions_to_redis() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self.store[key] = value

        async def get(self, key: str) -> str | None:
            return self.store.get(key)

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")

    manager = A2ASessionManager(redis_client=redis_wrapper)
    fetched = await manager.get_session(session["session_id"])

    assert fetched is not None
    assert fetched["session_id"] == session["session_id"]


@pytest.mark.asyncio
async def test_a2a_redis_load_handles_corrupt_payload() -> None:
    class FakeRedis:
        async def get(self, key: str) -> str:
            return "{not-json}"

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    fetched = await manager.get_session("corrupt-session")

    assert fetched is None


@pytest.mark.asyncio
async def test_a2a_redis_load_handles_failure() -> None:
    class FakeRedis:
        async def get(self, key: str) -> str:
            raise RuntimeError("redis down")

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    fetched = await manager.get_session("missing-session")

    assert fetched is None


@pytest.mark.asyncio
async def test_a2a_redis_ttl_expiry() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}
            self.expirations: dict[str, float] = {}
            self.now = 0.0

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self.store[key] = value
            if ex is not None:
                self.expirations[key] = self.now + float(ex)

        async def get(self, key: str) -> str | None:
            expires_at = self.expirations.get(key)
            if expires_at is not None and self.now >= expires_at:
                self.store.pop(key, None)
                self.expirations.pop(key, None)
                return None
            return self.store.get(key)

        def advance(self, seconds: float) -> None:
            self.now += seconds

    redis = FakeRedis()
    redis_wrapper = MagicMock()
    redis_wrapper.client = redis

    manager = A2ASessionManager(redis_client=redis_wrapper, session_ttl_seconds=1)
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")
    redis.advance(2.0)

    manager = A2ASessionManager(redis_client=redis_wrapper, session_ttl_seconds=1)
    fetched = await manager.get_session(session["session_id"])

    assert fetched is None


@pytest.mark.asyncio
async def test_a2a_persist_failure_is_nonfatal() -> None:
    class FakeRedis:
        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            raise RuntimeError("cannot persist")

        async def get(self, key: str) -> str | None:
            return None

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")
    fetched = await manager.get_session(session["session_id"])

    assert fetched is not None


@pytest.mark.asyncio
async def test_a2a_session_reconstructs_messages() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self.store[key] = value

        async def get(self, key: str) -> str | None:
            return self.store.get(key)

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    session = await manager.create_session("11111111-1111-1111-1111-111111111111")
    await manager.add_message(
        session_id=session["session_id"],
        tenant_id="11111111-1111-1111-1111-111111111111",
        sender="agent",
        content="hello",
    )

    manager = A2ASessionManager(redis_client=redis_wrapper)
    fetched = await manager.get_session(session["session_id"])

    assert fetched is not None
    assert fetched["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_a2a_invalid_timestamps_are_discarded() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self.store[key] = value

        async def get(self, key: str) -> str | None:
            return self.store.get(key)

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    session_id = "invalid-session"
    key = manager._session_key(session_id)
    redis_wrapper.client.store[key] = (
        '{"session_id":"invalid-session","tenant_id":"t1","created_at":"bad","last_activity":"bad","messages":[]}'
    )

    fetched = await manager.get_session(session_id)

    assert fetched is None


@pytest.mark.asyncio
async def test_a2a_invalid_message_timestamp_discards_session() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self.store[key] = value

        async def get(self, key: str) -> str | None:
            return self.store.get(key)

    redis_wrapper = MagicMock()
    redis_wrapper.client = FakeRedis()

    manager = A2ASessionManager(redis_client=redis_wrapper)
    session_id = "invalid-message-ts"
    key = manager._session_key(session_id)
    redis_wrapper.client.store[key] = (
        '{"session_id":"invalid-message-ts","tenant_id":"t1","created_at":"2024-01-01T00:00:00Z",'
        '"last_activity":"2024-01-01T00:00:00Z","messages":[{"sender":"agent","content":"hi","timestamp":"bad"}]}'
    )

    fetched = await manager.get_session(session_id)

    assert fetched is None
