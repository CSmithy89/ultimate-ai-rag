from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.main import app


def test_migrate_execute_writes_overrides() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        (root / ".env").write_text("LLM_PROVIDER=anthropic\n", encoding="utf-8")
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "standard.yaml").write_text(
            "llm:\n  provider: openai\n  model: gpt-4o\n",
            encoding="utf-8",
        )

        result = runner.invoke(app, ["migrate", "execute", "--profile", "standard"])
        assert result.exit_code == 0
        custom_path = profile_dir / "custom.yaml"
        assert custom_path.exists()
        contents = custom_path.read_text(encoding="utf-8")
        assert "anthropic" in contents
