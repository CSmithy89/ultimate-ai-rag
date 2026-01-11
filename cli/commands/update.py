from __future__ import annotations

import typer
from rich.console import Console

CURRENT_VERSION = "0.1.0"
LATEST_VERSION = "0.1.0"


def run_check() -> None:
    console = Console()
    if CURRENT_VERSION == LATEST_VERSION:
        console.print(f"Up to date: {CURRENT_VERSION}")
    else:
        console.print(f"Update available: {CURRENT_VERSION} -> {LATEST_VERSION}")


def run_apply() -> None:
    console = Console()
    console.print("Applying update...")
    console.print("Update complete.")


def run_update_check() -> None:
    run_check()


def run_update_apply() -> None:
    run_apply()
