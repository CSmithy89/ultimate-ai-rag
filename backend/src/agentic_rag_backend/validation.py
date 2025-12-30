"""Shared validation helpers."""

from __future__ import annotations

import re

TENANT_ID_PATTERN = r"^[A-Za-z0-9]+(?:[._:-][A-Za-z0-9]+)*$"
TENANT_ID_REGEX = re.compile(TENANT_ID_PATTERN)


def is_valid_tenant_id(value: str) -> bool:
    """Return True when tenant_id matches the allowed pattern."""
    return bool(TENANT_ID_REGEX.fullmatch(value))
