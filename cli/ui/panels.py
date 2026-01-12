from __future__ import annotations

from typing import Iterable

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def header_panel() -> Panel:
    title = Text("RAG SYSTEM INSTALLER", style="bold")
    return Panel(title, expand=False)


def summary_panel(summary: dict[str, str]) -> Panel:
    table = Table(show_header=False, box=None)
    for key, value in summary.items():
        table.add_row(f"{key}:", value)
    return Panel(table, title="Ready to install", expand=False)


def success_panel(lines: Iterable[str]) -> Panel:
    body = "\n".join(lines)
    return Panel(body, title="SUCCESS", expand=False)
