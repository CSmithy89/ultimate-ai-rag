from __future__ import annotations

import typer

from cli.commands.install import run_install
from cli.commands.setup import run_setup

app = typer.Typer(add_completion=False)


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


if __name__ == "__main__":
    app()
