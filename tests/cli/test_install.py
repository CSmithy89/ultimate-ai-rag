from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.main import app


def _write_env_example(root: Path) -> None:
    (root / ".env.example").write_text(
        "LLM_PROVIDER=openai\nOPENAI_API_KEY=\nEMBEDDING_PROVIDER=\nRERANKER_ENABLED=false\n"
        "CONTEXTUAL_RETRIEVAL_ENABLED=false\nVOICE_IO_ENABLED=false\n",
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
