"""A2A collaboration endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
import structlog

from ...api.utils import build_meta, rate_limit_exceeded
from ...protocols.a2a import A2ASessionManager
from ...rate_limit import RateLimiter
from ...validation import TENANT_ID_PATTERN

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/a2a", tags=["a2a"])


class CreateSessionRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)


class CreateSessionResponse(BaseModel):
    session: dict[str, Any]
    meta: dict[str, Any]


class MessageRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)
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
    """Get the A2A session manager from app state."""
    manager = getattr(request.app.state, "a2a_manager", None)
    if manager is None:
        raise RuntimeError("A2A session manager not initialized")
    return manager


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request_body: CreateSessionRequest,
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> CreateSessionResponse:
    """Create a new A2A collaboration session."""
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    try:
        session = await manager.create_session(request_body.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info("a2a_session_created", session_id=session["session_id"])

    return CreateSessionResponse(session=session, meta=build_meta())


@router.post("/sessions/{session_id}/messages", response_model=SessionResponse)
async def add_message(
    session_id: str,
    request_body: MessageRequest,
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> SessionResponse:
    """Add a message to an existing A2A session."""
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    try:
        session = await manager.add_message(
            session_id=session_id,
            tenant_id=request_body.tenant_id,
            sender=request_body.sender,
            content=request_body.content,
            metadata=request_body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Tenant not authorized") from exc

    return SessionResponse(session=session, meta=build_meta())


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> SessionResponse:
    """Fetch session transcript for a tenant."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    session = await manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Tenant not authorized")

    return SessionResponse(session=session, meta=build_meta())
