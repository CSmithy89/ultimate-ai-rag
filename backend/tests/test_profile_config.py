from __future__ import annotations

import os

import pytest

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


def test_profile_validation_rejects_invalid(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    (profile_dir / "bad.yaml").write_text("llm:\n  provider: openai\n", encoding="utf-8")
    monkeypatch.setattr(ConfigLoader, "PROFILE_DIR", profile_dir)

    with pytest.raises(ValueError, match="Profile validation failed"):
        ConfigLoader("bad").load()
