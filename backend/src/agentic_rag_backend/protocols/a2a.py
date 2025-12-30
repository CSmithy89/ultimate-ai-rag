"""A2A (agent-to-agent) collaboration session manager."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from asyncio import Lock
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


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

    def copy(self) -> "A2AMessage":
        return A2AMessage(
            sender=self.sender,
            content=self.content,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
        )


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

    def copy(self) -> "A2ASession":
        return A2ASession(
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            created_at=self.created_at,
            last_activity=self.last_activity,
            messages=[message.copy() for message in self.messages],
        )


class A2ASessionManager:
    """In-memory session manager for agent-to-agent collaboration.

    Sessions are ephemeral and cleared on service restart.
    """

    def __init__(
        self,
        session_ttl_seconds: int = 21600,
        max_sessions_per_tenant: int = 100,
        max_sessions_total: int = 1000,
        max_messages_per_session: int = 1000,
    ) -> None:
        # Defaults align with .env.example; override via settings in production.
        self._sessions: dict[str, A2ASession] = {}
        self._lock = Lock()
        self._session_ttl_seconds = session_ttl_seconds
        self._max_sessions_per_tenant = max_sessions_per_tenant
        self._max_sessions_total = max_sessions_total
        self._max_messages_per_session = max_messages_per_session
        self._cleanup_task: asyncio.Task[None] | None = None

    def _prune_expired_locked(self) -> None:
        if self._session_ttl_seconds <= 0:
            return
        now = datetime.now(timezone.utc)
        for session_id, session in list(self._sessions.items()):
            if (now - session.last_activity).total_seconds() > self._session_ttl_seconds:
                self._sessions.pop(session_id, None)

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

    def _clone_session(self, session: A2ASession) -> A2ASession:
        return session.copy()

    async def create_session(self, tenant_id: str) -> A2ASession:
        """Create a new A2A session.

        Raises:
            ValueError: If session or tenant limits are exceeded.
        """
        async with self._lock:
            self._prune_expired_locked()
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
            return self._clone_session(session)

    async def get_session(self, session_id: str) -> A2ASession | None:
        async with self._lock:
            self._prune_expired_locked()
            session = self._sessions.get(session_id)
            if session:
                session.last_activity = datetime.now(timezone.utc)
            return self._clone_session(session) if session else None

    async def add_message(
        self,
        session_id: str,
        tenant_id: str,
        sender: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> A2ASession:
        async with self._lock:
            self._prune_expired_locked()
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
            return self._clone_session(session)
