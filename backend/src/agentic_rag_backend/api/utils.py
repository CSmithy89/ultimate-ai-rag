"""Shared helpers for API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException


def build_meta() -> dict[str, Any]:
    """Build standard response metadata."""
    return {
        "requestId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def rate_limit_exceeded(retry_after_seconds: int | None = None) -> HTTPException:
    """Create a rate limit exceeded exception with Retry-After header."""
    if retry_after_seconds is None:
        from ..config import get_settings

        retry_after_seconds = get_settings().rate_limit_retry_after_seconds
    return HTTPException(
        status_code=429,
        detail="Rate limit exceeded",
        headers={"Retry-After": str(retry_after_seconds)},
    )
