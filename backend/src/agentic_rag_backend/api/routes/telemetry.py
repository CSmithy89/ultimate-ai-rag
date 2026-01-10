from typing import Any
from datetime import datetime, timezone
import hashlib
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..utils import rate_limit_exceeded
from ...rate_limit import RateLimiter

router = APIRouter()


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state.

    Defined locally to avoid circular import with main.py.
    """
    return request.app.state.rate_limiter

class TelemetryEvent(BaseModel):
    event: str = Field(..., description="Name of the event")
    properties: dict[str, Any] | None = Field(default=None, description="Event properties")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TelemetryResponse(BaseModel):
    status: str
    received_at: datetime

def _hash_pii(text: str) -> str:
    """Hash sensitive content like message text."""
    if not text:
        return ""
    # Hash first 100 chars to avoid processing huge payloads
    preview = text[:100].encode("utf-8")
    return hashlib.sha256(preview).hexdigest()

def _sanitize_properties(props: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize telemetry properties to remove PII."""
    if not props:
        return {}
    
    sanitized = {}
    sensitive_keys = {"password", "secret", "token", "key", "auth", "credential", "api_key"}
    
    for key, value in props.items():
        # Redact sensitive keys
        if any(s in key.lower() for s in sensitive_keys):
            sanitized[key] = "[REDACTED]"
            continue
            
        # Special handling for message content
        if key == "message" and isinstance(value, str):
            sanitized["message_hash"] = _hash_pii(value)
            sanitized["message_length"] = len(value)
            continue
            
        # Pass through other values
        sanitized[key] = value
        
    return sanitized

@router.post("/telemetry", response_model=TelemetryResponse)
async def track_telemetry(
    payload: TelemetryEvent,
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
):
    """
    Track frontend telemetry events.
    
    Events are sanitized to remove PII before logging/storage.
    Currently logs structured events; can be extended to store in DB/Timescale.
    """
    # Simple rate limiting using IP since this is a public-facing endpoint (auth optional for tracking)
    client_ip = request.client.host if request.client else "unknown"
    if not await limiter.allow(f"telemetry:{client_ip}"):
        raise rate_limit_exceeded()

    # Sanitize payload
    sanitized_props = _sanitize_properties(payload.properties)
    
    # Log the event (structured logging will handle JSON formatting)
    # In a real system, this would go to a specialized telemetry store (e.g. PostHog, Mixpanel, ClickHouse)
    # For now, we use our structured logger which flows to stdout/logs
    import structlog
    logger = structlog.get_logger("telemetry")
    
    logger.info(
        "frontend_telemetry",
        event=payload.event,
        properties=sanitized_props,
        client_timestamp=payload.timestamp.isoformat(),
        received_at=datetime.now(timezone.utc).isoformat(),
        source="frontend"
    )

    return TelemetryResponse(
        status="accepted",
        received_at=datetime.now(timezone.utc)
    )
