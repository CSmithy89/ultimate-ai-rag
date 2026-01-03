"""Tests for configuration loading and provider selection."""

from __future__ import annotations

import pytest

from agentic_rag_backend.config import load_settings


def _set_core_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j_password")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")


def test_load_settings_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    settings = load_settings()

    assert settings.llm_provider == "openai"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_base_url is None
    assert settings.llm_model_id == settings.openai_model_id


def test_load_settings_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = load_settings()

    assert settings.llm_provider == "openrouter"
    assert settings.llm_api_key == "router-key"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"


def test_load_settings_requires_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    with pytest.raises(ValueError) as excinfo:
        load_settings()

    assert "ANTHROPIC_API_KEY" in str(excinfo.value)


def test_llm_model_id_overrides_openai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_core_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL_ID", "gpt-4o-mini")
    monkeypatch.setenv("LLM_MODEL_ID", "gpt-4o")

    settings = load_settings()

    assert settings.llm_model_id == "gpt-4o"
    assert settings.openai_model_id == "gpt-4o-mini"
