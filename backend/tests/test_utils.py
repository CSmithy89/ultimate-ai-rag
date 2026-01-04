"""Shared test helpers."""

from __future__ import annotations

import pytest


def set_core_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required environment variables for settings tests."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j_password")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
