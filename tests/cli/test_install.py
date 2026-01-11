from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.main import app
from cli.commands import install as install_module


def _write_env_example(root: Path) -> None:
    (root / ".env.example").write_text(
        "LLM_PROVIDER=openai\nOPENAI_API_KEY=\nEMBEDDING_PROVIDER=\nRERANKER_ENABLED=false\n"
        "CONTEXTUAL_RETRIEVAL_ENABLED=false\nVOICE_IO_ENABLED=false\n"
        "DATABASE_URL=postgresql://user:pass@localhost:5432/db\n"
        "NEO4J_URI=bolt://localhost:7687\n",
        encoding="utf-8",
    )


def test_non_interactive_install_writes_env_and_template() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        _write_env_example(root)
        result = runner.invoke(
            app,
            [
                "rag-install",
                "--profile",
                "standard",
                "--llm",
                "openai",
                "--api-key",
                "sk-test-1234567890",
                "--framework",
                "pydanticai",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        env_output = (root / ".env").read_text(encoding="utf-8")
        assert "LLM_PROVIDER=openai" in env_output
        assert "OPENAI_API_KEY=sk-test-1234567890" in env_output
        assert (root / "examples" / "pydanticai" / "README.md").exists()


def test_profile_mapping_from_ram(monkeypatch) -> None:
    monkeypatch.setattr(install_module, "_detect_gpu", lambda: "not detected")
    monkeypatch.setattr(install_module, "_read_total_memory_gb", lambda: 8)
    profile, _ = install_module._recommend_profile()
    assert profile == "minimal"

    monkeypatch.setattr(install_module, "_read_total_memory_gb", lambda: 24)
    profile, _ = install_module._recommend_profile()
    assert profile == "standard"


def test_env_backup_created() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        _write_env_example(root)
        (root / ".env").write_text("LLM_PROVIDER=openai\n", encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "rag-install",
                "--profile",
                "standard",
                "--llm",
                "openai",
                "--api-key",
                "sk-test-1234567890",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert (root / ".env.bak").exists()


def test_invalid_neo4j_uri_fails() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        (root / ".env.example").write_text(
            "LLM_PROVIDER=openai\nOPENAI_API_KEY=\nDATABASE_URL=postgresql://user:pass@localhost:5432/db\n"
            "NEO4J_URI=http://localhost:7687\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "rag-install",
                "--profile",
                "standard",
                "--llm",
                "openai",
                "--api-key",
                "sk-test-1234567890",
                "--yes",
            ],
        )
        assert result.exit_code != 0

    monkeypatch.setattr(install_module, "_read_total_memory_gb", lambda: 64)
    profile, _ = install_module._recommend_profile()
    assert profile == "enterprise"

    monkeypatch.setattr(install_module, "_read_total_memory_gb", lambda: None)
    profile, _ = install_module._recommend_profile()
    assert profile == "standard"
