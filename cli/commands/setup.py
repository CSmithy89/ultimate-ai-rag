from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from cli.profile import load_custom_profile, load_profile, write_custom_profile


def _prompt_bool(console: Console, label: str, default: bool) -> bool:
    return Confirm.ask(label, default=default, console=console)


def run_setup(
    category: str | None = typer.Option(None, "--category"),
    profile: str | None = typer.Option(None, "--profile"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    console = Console()
    profile_name = (profile or os.getenv("CONFIG_PROFILE", "standard")).strip().lower()
    try:
        base_config = load_profile(profile_name)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    custom_config = load_custom_profile()
    target_category = category or "all"

    if target_category in {"ingestion", "all"}:
        ingestion_defaults = base_config.get("ingestion", {})
        crawl_profile = ingestion_defaults.get("crawl_profile", "thorough")
        fallback_enabled = bool(ingestion_defaults.get("fallback_enabled", False))
        pdf_enabled = bool(ingestion_defaults.get("pdf_enabled", True))
        youtube_enabled = bool(ingestion_defaults.get("youtube_enabled", True))
        codebase_enabled = bool(ingestion_defaults.get("codebase_enabled", False))
        external_sync_enabled = bool(ingestion_defaults.get("external_sync_enabled", False))

        if not yes:
            crawl_profile = Prompt.ask(
                "Select crawl profile",
                choices=["fast", "thorough", "stealth"],
                default=crawl_profile,
                console=console,
            )
            fallback_enabled = _prompt_bool(console, "Enable crawl fallback?", fallback_enabled)
            pdf_enabled = _prompt_bool(console, "Enable PDF ingestion?", pdf_enabled)
            youtube_enabled = _prompt_bool(console, "Enable YouTube ingestion?", youtube_enabled)

            if profile_name == "enterprise":
                codebase_enabled = _prompt_bool(
                    console, "Enable codebase ingestion?", codebase_enabled
                )
                external_sync_enabled = _prompt_bool(
                    console, "Enable external sync ingestion?", external_sync_enabled
                )

        custom_config["ingestion"] = {
            "crawl_profile": crawl_profile,
            "fallback_enabled": fallback_enabled,
            "pdf_enabled": pdf_enabled,
            "youtube_enabled": youtube_enabled,
            "codebase_enabled": codebase_enabled,
            "external_sync_enabled": external_sync_enabled,
        }

    write_custom_profile(custom_config)
    console.print("Configuration saved to config/profiles/custom.yaml")
