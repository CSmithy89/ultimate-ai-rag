from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import typer
from rich.console import Console

from cli.profile import parse_env_file


def _check_url(url: str) -> bool:
    try:
        with urlopen(url, timeout=2.0) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError):
        return False


def run_doctor(
    quick: bool = typer.Option(False, "--quick"),
    json_output: bool = typer.Option(False, "--json"),
    service: str | None = typer.Option(None, "--service"),
    fix: bool = typer.Option(False, "--fix"),
) -> None:
    """Run CLI health checks for environment and services."""
    console = Console()
    checks: list[dict[str, Any]] = []

    env_path = Path(".env")
    env_example = Path(".env.example")
    if not env_path.exists():
        if fix and env_example.exists():
            env_path.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
            checks.append({"check": "env", "status": "fixed"})
        else:
            checks.append({"check": "env", "status": "fail", "message": ".env missing"})
    else:
        checks.append({"check": "env", "status": "ok"})

    profile_name = "standard"
    if env_path.exists():
        env_values = parse_env_file(env_path)
        profile_name = env_values.get("CONFIG_PROFILE", "").strip() or "standard"
    profile_path = Path("config/profiles") / f"{profile_name}.yaml"
    if profile_path.exists():
        checks.append({"check": "profile", "status": "ok", "profile": profile_name})
    else:
        checks.append(
            {
                "check": "profile",
                "status": "fail",
                "message": f"Profile not found: {profile_name}",
            }
        )

    if not quick:
        if service in {None, "backend"}:
            backend_ok = _check_url("http://localhost:8000/health")
            checks.append(
                {"check": "backend", "status": "ok" if backend_ok else "fail"}
            )
        if service in {None, "frontend"}:
            frontend_ok = _check_url("http://localhost:3000")
            checks.append(
                {"check": "frontend", "status": "ok" if frontend_ok else "fail"}
            )

    has_failures = any(check["status"] == "fail" for check in checks)
    payload = {"status": "fail" if has_failures else "ok", "checks": checks}

    if json_output:
        console.print(json.dumps(payload, indent=2))
    else:
        for check in checks:
            status = check["status"]
            name = check["check"]
            detail = check.get("message", "")
            suffix = f" - {detail}" if detail else ""
            console.print(f"{name}: {status}{suffix}")

    if has_failures:
        raise typer.Exit(code=1)
