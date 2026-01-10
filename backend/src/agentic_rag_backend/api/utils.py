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


def rfc7807_error(
    status: int,
    title: str,
    detail: str,
    error_type: str = "about:blank",
    instance: str | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPException:
    """Create an RFC 7807 Problem Details compliant HTTPException.

    RFC 7807 specifies a standard format for error responses:
    - type: URI reference for the error type (default: about:blank)
    - title: Short human-readable summary
    - status: HTTP status code
    - detail: Human-readable explanation
    - instance: URI reference for the specific occurrence

    Args:
        status: HTTP status code
        title: Short summary of the error
        detail: Detailed explanation
        error_type: URI reference for error type (default: about:blank)
        instance: URI reference for this specific error occurrence
        headers: Optional HTTP headers to include

    Returns:
        HTTPException with RFC 7807 compliant detail body
    """
    body: dict[str, Any] = {
        "type": error_type,
        "title": title,
        "status": status,
        "detail": detail,
    }
    if instance:
        body["instance"] = instance
    return HTTPException(status_code=status, detail=body, headers=headers)
