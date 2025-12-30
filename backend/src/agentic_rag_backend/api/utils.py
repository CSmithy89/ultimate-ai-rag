"""Shared helpers for API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException

RATE_LIMIT_RETRY_AFTER_SECONDS = 60


def build_meta() -> dict[str, Any]:
    """Build standard response metadata."""
    return {
        "requestId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def rate_limit_exceeded() -> HTTPException:
    """Create a rate limit exceeded exception with Retry-After header."""
    return HTTPException(
        status_code=429,
        detail="Rate limit exceeded",
        headers={"Retry-After": str(RATE_LIMIT_RETRY_AFTER_SECONDS)},
    )
