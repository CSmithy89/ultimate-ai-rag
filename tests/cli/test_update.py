from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app


def test_update_check() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["update", "check"], env={"RAG_CLI_UPDATE_NO_FETCH": "1"})
    assert result.exit_code == 0
    assert "Update" in result.output


def test_update_apply() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["update", "apply"], env={"RAG_CLI_UPDATE_DRY_RUN": "1"})
    assert result.exit_code == 0
    assert "Update complete" in result.output
