from __future__ import annotations

import typer

from cli.commands.install import run_install

app = typer.Typer(add_completion=False)


@app.command("rag-install")
def rag_install(
    profile: str | None = typer.Option(None, "--profile"),
    llm: str | None = typer.Option(None, "--llm"),
    api_key: str | None = typer.Option(None, "--api-key"),
    framework: str | None = typer.Option(None, "--framework"),
    customize: bool = typer.Option(False, "--customize"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    run_install(
        profile=profile,
        llm=llm,
        api_key=api_key,
        framework=framework,
        customize=customize,
        yes=yes,
    )


if __name__ == "__main__":
    app()
