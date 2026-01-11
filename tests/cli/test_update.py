from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app


def test_update_check() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["update", "check"])
    assert result.exit_code == 0
    assert "Up to date" in result.output


def test_update_apply() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["update", "apply"])
    assert result.exit_code == 0
    assert "Update complete" in result.output
