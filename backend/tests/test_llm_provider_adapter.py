"""Tests for LLM provider adapter selection."""

from __future__ import annotations

import pytest

from agentic_rag_backend.config import load_settings
from agentic_rag_backend.llm.providers import (
    UnsupportedLLMProviderError,
    get_llm_adapter,
)


def _set_core_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j_password")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")


def test_get_llm_adapter_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = load_settings()
    adapter = get_llm_adapter(settings)

    assert adapter.provider == "openrouter"
    assert adapter.openai_kwargs()["api_key"] == "router-key"
    assert adapter.openai_kwargs()["base_url"] == "https://openrouter.ai/api/v1"


def test_get_llm_adapter_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    settings = load_settings()
    adapter = get_llm_adapter(settings)

    assert adapter.provider == "openai"
    assert adapter.openai_kwargs()["base_url"] == "https://example.com/v1"


def test_get_llm_adapter_rejects_unsupported_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = load_settings()
    with pytest.raises(UnsupportedLLMProviderError):
        get_llm_adapter(settings)
