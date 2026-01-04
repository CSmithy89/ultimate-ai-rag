"""Tests for LLM provider adapter selection."""

from __future__ import annotations

import importlib.util

import pytest

from agentic_rag_backend.config import load_settings
from agentic_rag_backend.llm.providers import get_llm_adapter
from tests.test_utils import set_core_env


def test_get_llm_adapter_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    settings = load_settings()
    adapter = get_llm_adapter(settings)

    assert adapter.provider == "openrouter"
    assert adapter.openai_kwargs()["api_key"] == "router-key"
    assert adapter.openai_kwargs()["base_url"] == "https://openrouter.ai/api/v1"


def test_get_llm_adapter_openai_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    set_core_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    settings = load_settings()
    adapter = get_llm_adapter(settings)

    assert adapter.provider == "openai"
    assert adapter.openai_kwargs()["base_url"] == "https://example.com/v1"


def test_get_llm_adapter_anthropic_requires_openai_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anthropic LLM requires OpenAI embeddings since Anthropic has no native embeddings.

    When EMBEDDING_PROVIDER defaults to openai (because anthropic is not in EMBEDDING_PROVIDERS),
    the config should require OPENAI_API_KEY.
    """
    set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    # Now fails during config loading since embedding provider validation happens there
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        load_settings()


def test_get_llm_adapter_anthropic_with_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("anthropic") is None or importlib.util.find_spec(
        "agno.models.anthropic"
    ) is None:
        pytest.skip("Anthropic dependencies not installed")
    set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    settings = load_settings()
    adapter = get_llm_adapter(settings)

    assert adapter.provider == "anthropic"
    assert adapter.embedding_api_key == "openai-key"


def test_get_llm_adapter_gemini_with_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini LLM now uses Gemini embeddings by default (since gemini is in EMBEDDING_PROVIDERS)."""
    if (
        importlib.util.find_spec("agno.models.google") is None
        or (
            importlib.util.find_spec("google.genai") is None
            and importlib.util.find_spec("google.generativeai") is None
        )
    ):
        pytest.skip("Gemini dependencies not installed")
    set_core_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    settings = load_settings()
    adapter = get_llm_adapter(settings)

    assert adapter.provider == "gemini"
    # Now defaults to gemini for embeddings (no longer requires OpenAI)
    assert adapter.embedding_api_key == "gemini-key"
    assert settings.embedding_provider == "gemini"
