"""A2A (agent-to-agent) collaboration session manager."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
import time
from dataclasses import dataclass, field
from asyncio import Lock
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import structlog

from agentic_rag_backend.db.redis import RedisClient

logger = structlog.get_logger(__name__)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

@dataclass
class A2AMessage:
    sender: str
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender": self.sender,
            "content": self.content,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "metadata": self.metadata,
        }


@dataclass
class A2ASession:
    session_id: str
    tenant_id: str
    created_at: datetime
    last_activity: datetime
    messages: list[A2AMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "last_activity": self.last_activity.isoformat().replace("+00:00", "Z"),
            "messages": [message.to_dict() for message in self.messages],
        }


class A2ASessionManager:
    """In-memory session manager for agent-to-agent collaboration.

    Sessions are persisted to Redis when available for restart recovery.
    """

    def __init__(
        self,
        session_ttl_seconds: int = 21600,
        max_sessions_per_tenant: int = 100,
        max_sessions_total: int = 1000,
        max_messages_per_session: int = 1000,
        redis_client: Optional[RedisClient] = None,
        redis_prefix: str = "a2a:sessions",
    ) -> None:
        # Defaults align with .env.example; override via settings in production.
        self._sessions: dict[str, A2ASession] = {}
        self._lock = Lock()
        self._session_ttl_seconds = session_ttl_seconds
        self._max_sessions_per_tenant = max_sessions_per_tenant
        self._max_sessions_total = max_sessions_total
        self._max_messages_per_session = max_messages_per_session
        self._cleanup_task: asyncio.Task[None] | None = None
        self._last_prune = 0.0
        self._prune_interval_seconds = 60.0
        self._redis = redis_client.client if redis_client else None
        self._redis_prefix = redis_prefix

    def _session_key(self, session_id: str) -> str:
        return f"{self._redis_prefix}:{session_id}"

    def _parse_timestamp(self, value: str, *, session_id: str, field: str) -> datetime:
        if not value:
            logger.warning(
                "a2a_timestamp_missing",
                session_id=session_id,
                field=field,
            )
            return _EPOCH
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            logger.warning(
                "a2a_timestamp_invalid",
                session_id=session_id,
                field=field,
                value=value,
            )
            return _EPOCH

    def _session_from_payload(self, payload: dict[str, Any]) -> A2ASession | None:
        messages = []
        session_id = payload.get("session_id", "")
        for message in payload.get("messages", []):
            timestamp = self._parse_timestamp(
                message.get("timestamp", ""),
                session_id=session_id,
                field="message_timestamp",
            )
            messages.append(
                A2AMessage(
                    sender=message.get("sender", ""),
                    content=message.get("content", ""),
                    timestamp=timestamp,
                    metadata=message.get("metadata") or {},
                )
            )
        created_at = self._parse_timestamp(
            payload.get("created_at", ""),
            session_id=session_id,
            field="created_at",
        )
        last_activity = self._parse_timestamp(
            payload.get("last_activity", ""),
            session_id=session_id,
            field="last_activity",
        )
        if created_at == _EPOCH or last_activity == _EPOCH:
            logger.warning(
                "a2a_session_discarded_invalid_timestamp",
                session_id=session_id,
            )
            return None
        return A2ASession(
            session_id=session_id,
            tenant_id=payload.get("tenant_id", ""),
            created_at=created_at,
            last_activity=last_activity,
            messages=messages,
        )

    async def _persist_session(self, session: A2ASession) -> None:
        if not self._redis:
            return
        try:
            payload = json.dumps(session.to_dict())
            key = self._session_key(session.session_id)
            if self._session_ttl_seconds > 0:
                await self._redis.set(key, payload, ex=self._session_ttl_seconds)
            else:
                await self._redis.set(key, payload)
        except Exception as exc:  # pragma: no cover - non-critical persistence
            logger.warning("a2a_session_persist_failed", error=str(exc))

    async def _load_session(self, session_id: str) -> A2ASession | None:
        if not self._redis:
            return None
        try:
            payload = await self._redis.get(self._session_key(session_id))
            if not payload:
                return None
            raw = payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload
            data = json.loads(raw)
            return self._session_from_payload(data)
        except Exception as exc:  # pragma: no cover - non-critical persistence
            logger.warning("a2a_session_load_failed", error=str(exc))
            return None

    def _prune_expired_locked(self) -> None:
        if self._session_ttl_seconds <= 0:
            return
        now = datetime.now(timezone.utc)
        for session_id, session in list(self._sessions.items()):
            if (now - session.last_activity).total_seconds() > self._session_ttl_seconds:
                self._sessions.pop(session_id, None)

    def _maybe_prune_locked(self) -> None:
        if self._session_ttl_seconds <= 0:
            return
        now = time.monotonic()
        if now - self._last_prune < self._prune_interval_seconds:
            return
        self._last_prune = now
        self._prune_expired_locked()

    def _tenant_session_count_locked(self, tenant_id: str) -> int:
        return sum(1 for session in self._sessions.values() if session.tenant_id == tenant_id)

    async def _periodic_cleanup_task(self, interval_seconds: int) -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                async with self._lock:
                    self._prune_expired_locked()
        except asyncio.CancelledError:
            return

    async def start_cleanup_task(self, interval_seconds: int) -> None:
        if interval_seconds <= 0:
            return
        if self._cleanup_task and not self._cleanup_task.done():
            return
        self._prune_interval_seconds = max(1.0, min(self._prune_interval_seconds, float(interval_seconds)))
        self._cleanup_task = asyncio.create_task(
            self._periodic_cleanup_task(interval_seconds)
        )

    async def stop_cleanup_task(self) -> None:
        if not self._cleanup_task:
            return
        self._cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._cleanup_task
        self._cleanup_task = None

    def _snapshot_session(self, session: A2ASession) -> dict[str, Any]:
        return session.to_dict()

    async def create_session(self, tenant_id: str) -> dict[str, Any]:
        """Create a new A2A session snapshot.

        Raises:
            ValueError: If session or tenant limits are exceeded.
        """
        async with self._lock:
            self._maybe_prune_locked()
            if self._max_sessions_total and len(self._sessions) >= self._max_sessions_total:
                raise ValueError("Session limit reached")
            if (
                self._max_sessions_per_tenant
                and self._tenant_session_count_locked(tenant_id) >= self._max_sessions_per_tenant
            ):
                raise ValueError("Tenant session limit reached")

            session_id = str(uuid4())
            now = datetime.now(timezone.utc)
            session = A2ASession(
                session_id=session_id,
                tenant_id=tenant_id,
                created_at=now,
                last_activity=now,
            )
            self._sessions[session_id] = session
            await self._persist_session(session)
            return self._snapshot_session(session)

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self._lock:
            self._maybe_prune_locked()
            session = self._sessions.get(session_id)
            if session is None:
                session = await self._load_session(session_id)
                if session:
                    self._sessions[session_id] = session
            if session:
                session.last_activity = datetime.now(timezone.utc)
                await self._persist_session(session)
            return self._snapshot_session(session) if session else None

    async def add_message(
        self,
        session_id: str,
        tenant_id: str,
        sender: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a message and return a session snapshot."""
        async with self._lock:
            self._maybe_prune_locked()
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("session not found")
            if session.tenant_id != tenant_id:
                raise PermissionError("tenant mismatch")
            if (
                self._max_messages_per_session
                and len(session.messages) >= self._max_messages_per_session
            ):
                raise ValueError("Session message limit reached")
            message = A2AMessage(
                sender=sender,
                content=content,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            session.messages.append(message)
            session.last_activity = datetime.now(timezone.utc)
            await self._persist_session(session)
            return self._snapshot_session(session)
