from __future__ import annotations

import os

from agentic_rag_backend.config import ConfigLoader, _apply_profile_defaults


def test_profile_defaults_applied(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)

    config = ConfigLoader("minimal").load()
    _apply_profile_defaults(config)

    assert os.getenv("LLM_PROVIDER") == "openai"
    assert os.getenv("EMBEDDING_PROVIDER") == "openai"


def test_env_overrides_profile(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    config = ConfigLoader("standard").load()
    _apply_profile_defaults(config)

    assert os.getenv("LLM_PROVIDER") == "anthropic"
