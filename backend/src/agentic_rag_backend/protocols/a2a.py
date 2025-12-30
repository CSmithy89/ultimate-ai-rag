"""A2A (agent-to-agent) collaboration session manager."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class A2ASession:
    session_id: str
    tenant_id: str
    created_at: datetime
    messages: list[A2AMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "messages": [message.to_dict() for message in self.messages],
        }


class A2ASessionManager:
    """In-memory session manager for agent-to-agent collaboration."""

    def __init__(self) -> None:
        self._sessions: dict[str, A2ASession] = {}

    def create_session(self, tenant_id: str) -> A2ASession:
        session_id = str(uuid4())
        session = A2ASession(
            session_id=session_id,
            tenant_id=tenant_id,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> A2ASession | None:
        return self._sessions.get(session_id)

    def add_message(
        self,
        session_id: str,
        tenant_id: str,
        sender: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> A2ASession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError("session not found")
        if session.tenant_id != tenant_id:
            raise PermissionError("tenant mismatch")
        message = A2AMessage(
            sender=sender,
            content=content,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        session.messages.append(message)
        return session
