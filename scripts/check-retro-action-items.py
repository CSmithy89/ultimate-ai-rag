#!/usr/bin/env python3
"""List overdue retrospective action items.

This script intentionally avoids external YAML dependencies by parsing a
restricted schema from docs/retrospectives/action-items.yaml.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

DEFAULT_PATH = Path("docs/retrospectives/action-items.yaml")


def parse_action_items(path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("action_items:"):
            continue

        if line.startswith("-"):
            if current:
                items.append(current)
            current = {}
            line = line[1:].lstrip()
            if line:
                key, value = _split_kv(line)
                if key:
                    current[key] = value
            continue

        if current is None:
            continue

        key, value = _split_kv(line)
        if key:
            current[key] = value

    if current:
        items.append(current)

    return items


def _split_kv(line: str) -> tuple[str | None, str]:
    if ":" not in line:
        return None, ""
    key, value = line.split(":", 1)
    return key.strip(), value.strip().strip("\"\'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List overdue retro action items.")
    parser.add_argument(
        "--file",
        default=str(DEFAULT_PATH),
        help="Path to action-items.yaml",
    )
    parser.add_argument(
        "--today",
        default=None,
        help="Override today's date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fail-on-overdue",
        action="store_true",
        help="Exit with status 1 if overdue items are found",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        print(f"Action items file not found: {path}", file=sys.stderr)
        return 2

    today = date.today()
    if args.today:
        try:
            today = datetime.strptime(args.today, "%Y-%m-%d").date()
        except ValueError:
            print("Invalid --today format. Use YYYY-MM-DD.", file=sys.stderr)
            return 2

    items = parse_action_items(path)
    if not items:
        print("No action items found.")
        return 0

    overdue: list[dict[str, str]] = []
    invalid_dates: list[dict[str, str]] = []

    for item in items:
        status = item.get("status", "").lower()
        if status == "done":
            continue
        due_raw = item.get("due_date", "")
        if not due_raw:
            invalid_dates.append(item)
            continue
        try:
            due = datetime.strptime(due_raw, "%Y-%m-%d").date()
        except ValueError:
            invalid_dates.append(item)
            continue
        if due < today:
            overdue.append(item)

    if overdue:
        print("Overdue action items:")
        for item in overdue:
            print(
                f"- {item.get('id', 'unknown')}: {item.get('title', 'Untitled')} "
                f"(owner: {item.get('owner', 'unassigned')}, due: {item.get('due_date', 'n/a')}, "
                f"status: {item.get('status', 'n/a')})"
            )
    else:
        print("No overdue action items.")

    if invalid_dates:
        print("\nItems with missing/invalid due dates:")
        for item in invalid_dates:
            print(
                f"- {item.get('id', 'unknown')}: {item.get('title', 'Untitled')} "
                f"(owner: {item.get('owner', 'unassigned')}, due: {item.get('due_date', 'n/a')})"
            )

    if overdue and args.fail_on_overdue:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
