from __future__ import annotations

import errno
from pathlib import Path

import pytest

from cli import profile as profile_module


def _write_profile(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_load_profile_rejects_invalid_yaml(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile_path = profile_dir / "broken.yaml"
    _write_profile(profile_path, "llm: [")

    monkeypatch.setattr(profile_module, "PROFILE_DIR", profile_dir)
    with pytest.raises(ValueError, match="invalid YAML"):
        profile_module.load_profile("broken")


def test_load_profile_rejects_missing_required_fields(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile_path = profile_dir / "minimal.yaml"
    _write_profile(profile_path, "llm:\n  provider: openai\n  model: gpt-4o\n")

    monkeypatch.setattr(profile_module, "PROFILE_DIR", profile_dir)
    with pytest.raises(ValueError, match="failed validation"):
        profile_module.load_profile("minimal")


def test_load_custom_profile_rejects_invalid_yaml(tmp_path, monkeypatch) -> None:
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    custom_path = profile_dir / "custom.yaml"
    _write_profile(custom_path, "llm: [")

    monkeypatch.setattr(profile_module, "PROFILE_DIR", profile_dir)
    with pytest.raises(ValueError, match="invalid YAML"):
        profile_module.load_custom_profile()


def test_write_custom_profile_propagates_permission_error(monkeypatch) -> None:
    def raise_permission(*args, **kwargs) -> None:
        raise PermissionError(errno.EACCES, "permission denied")

    monkeypatch.setattr(Path, "mkdir", raise_permission)

    with pytest.raises(PermissionError):
        profile_module.write_custom_profile({"llm": {"provider": "openai", "model": "gpt-4o"}})
