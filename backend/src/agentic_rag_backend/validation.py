"""Shared validation helpers."""

from __future__ import annotations

import re

TENANT_ID_PATTERN = (
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)
SESSION_ID_PATTERN = r"^[A-Za-z0-9]+(?:[._:-][A-Za-z0-9]+)*$"

TENANT_ID_REGEX = re.compile(TENANT_ID_PATTERN)
SESSION_ID_REGEX = re.compile(SESSION_ID_PATTERN)


def is_valid_tenant_id(value: str) -> bool:
    """Return True when tenant_id matches the allowed pattern."""
    return bool(TENANT_ID_REGEX.fullmatch(value))


def is_valid_session_id(value: str) -> bool:
    """Return True when session_id matches the allowed pattern."""
    return bool(SESSION_ID_REGEX.fullmatch(value))
