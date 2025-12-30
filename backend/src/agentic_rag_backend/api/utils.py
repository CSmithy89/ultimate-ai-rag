"""Shared helpers for API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def build_meta() -> dict[str, Any]:
    """Build standard response metadata."""
    return {
        "requestId": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
