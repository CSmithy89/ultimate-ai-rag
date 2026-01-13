from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.main import app
from cli.commands import doctor as doctor_module


def test_doctor_quick_json_ok() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        (root / ".env").write_text("CONFIG_PROFILE=standard\n", encoding="utf-8")
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "standard.yaml").write_text("{}\n", encoding="utf-8")

        result = runner.invoke(app, ["doctor", "--quick", "--json"])
        assert result.exit_code == 0
        assert '"status": "ok"' in result.output


def test_doctor_missing_env_fails() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["doctor", "--quick"])
        assert result.exit_code != 0


def test_doctor_health_check_timeout(monkeypatch) -> None:
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("timeout")

    monkeypatch.setattr(doctor_module, "urlopen", raise_timeout)
    assert doctor_module._check_url("http://localhost:8000/health") is False
