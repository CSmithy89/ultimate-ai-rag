"""A2A collaboration endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
import structlog

from ...protocols.a2a import A2ASessionManager
from ...rate_limit import RateLimiter

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/a2a", tags=["a2a"])


class CreateSessionRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=255)


class CreateSessionResponse(BaseModel):
    session: dict[str, Any]
    meta: dict[str, Any]


class MessageRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=255)
    sender: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    session: dict[str, Any]
    meta: dict[str, Any]


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


def get_a2a_manager(request: Request) -> A2ASessionManager:
    """Get or initialize the A2A session manager."""
    manager = getattr(request.app.state, "a2a_manager", None)
    if manager is None:
        manager = A2ASessionManager()
        request.app.state.a2a_manager = manager
    return manager


def _meta() -> dict[str, Any]:
    return {
        "requestId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request_body: CreateSessionRequest,
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> CreateSessionResponse:
    """Create a new A2A collaboration session."""
    if not await limiter.allow(request_body.tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    session = manager.create_session(request_body.tenant_id)
    logger.info("a2a_session_created", session_id=session.session_id)

    return CreateSessionResponse(session=session.to_dict(), meta=_meta())


@router.post("/sessions/{session_id}/messages", response_model=SessionResponse)
async def add_message(
    session_id: str,
    request_body: MessageRequest,
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> SessionResponse:
    """Add a message to an existing A2A session."""
    if not await limiter.allow(request_body.tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        session = manager.add_message(
            session_id=session_id,
            tenant_id=request_body.tenant_id,
            sender=request_body.sender,
            content=request_body.content,
            metadata=request_body.metadata,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Tenant not authorized") from exc
    except Exception as exc:  # pragma: no cover - safeguard
        logger.exception("a2a_add_message_failed", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to add message") from exc

    return SessionResponse(session=session.to_dict(), meta=_meta())


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255),
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> SessionResponse:
    """Fetch session transcript for a tenant."""
    if not await limiter.allow(tenant_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    session = manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Tenant not authorized")

    return SessionResponse(session=session.to_dict(), meta=_meta())
