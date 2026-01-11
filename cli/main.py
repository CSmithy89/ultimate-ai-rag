from __future__ import annotations

import typer

from cli.commands.install import run_install
from cli.commands.setup import run_setup
from cli.commands.doctor import run_doctor
from cli.commands.migrate import run_analyze, run_execute
from cli.commands.update import run_update_check, run_update_apply

app = typer.Typer(add_completion=False)
migrate_app = typer.Typer()
update_app = typer.Typer()


@app.command("rag-install")
def rag_install(
    profile: str | None = typer.Option(None, "--profile"),
    llm: str | None = typer.Option(None, "--llm"),
    api_key: str | None = typer.Option(None, "--api-key"),
    framework: str | None = typer.Option(None, "--framework"),
    customize: bool = typer.Option(False, "--customize"),
    yes: bool = typer.Option(False, "--yes"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    with_skills: bool = typer.Option(False, "--with-skills"),
) -> None:
    run_install(
        profile=profile,
        llm=llm,
        api_key=api_key,
        framework=framework,
        customize=customize,
        yes=yes,
        dry_run=dry_run,
        with_skills=with_skills,
    )


@app.command("setup")
def setup(
    category: str | None = typer.Option(None, "--category"),
    profile: str | None = typer.Option(None, "--profile"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    run_setup(category=category, profile=profile, yes=yes)


@app.command("doctor")
def doctor(
    quick: bool = typer.Option(False, "--quick"),
    json_output: bool = typer.Option(False, "--json"),
    service: str | None = typer.Option(None, "--service"),
    fix: bool = typer.Option(False, "--fix"),
) -> None:
    run_doctor(quick=quick, json_output=json_output, service=service, fix=fix)


@migrate_app.command("analyze")
def migrate_analyze(profile: str | None = typer.Option(None, "--profile")) -> None:
    run_analyze(profile=profile)


@migrate_app.command("execute")
def migrate_execute(profile: str | None = typer.Option(None, "--profile")) -> None:
    run_execute(profile=profile)


app.add_typer(migrate_app, name="migrate")


@update_app.command("check")
def update_check() -> None:
    run_update_check()


@update_app.command("apply")
def update_apply() -> None:
    run_update_apply()


app.add_typer(update_app, name="update")


if __name__ == "__main__":
    app()
