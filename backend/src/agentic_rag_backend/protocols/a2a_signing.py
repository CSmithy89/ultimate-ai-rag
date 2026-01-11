"""Shared helpers for A2A request signing."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any


def canonicalize_payload(payload: dict[str, Any]) -> str:
    """Serialize payload in a stable form for signing."""
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def sign_a2a_payload(
    secret: str,
    payload: dict[str, Any],
    *,
    timestamp: int | None = None,
) -> tuple[str, int]:
    """Return signature + timestamp for an A2A payload."""
    ts = timestamp or int(time.time())
    body = canonicalize_payload(payload)
    message = f"{ts}.{body}".encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return signature, ts


def verify_a2a_signature(
    secret: str,
    payload: dict[str, Any],
    timestamp: str | None,
    signature: str | None,
    *,
    ttl_seconds: int,
) -> bool:
    """Verify the signature for an incoming A2A request."""
    if not timestamp or not signature:
        return False
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False
    now = int(time.time())
    if abs(now - ts) > ttl_seconds:
        return False
    expected_signature, _ = sign_a2a_payload(secret, payload, timestamp=ts)
    return hmac.compare_digest(signature, expected_signature)
