#!/usr/bin/env python3
"""Validate that story file statuses match sprint-status.yaml."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import sys

SPRINT_STATUS_PATH = Path("_bmad-output/implementation-artifacts/sprint-status.yaml")
STORY_DIR = Path("_bmad-output/implementation-artifacts/stories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate story status alignment.")
    parser.add_argument(
        "--sprint-status",
        default=str(SPRINT_STATUS_PATH),
        help="Path to sprint-status.yaml",
    )
    parser.add_argument(
        "--story-dir",
        default=str(STORY_DIR),
        help="Directory containing story files",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail if a story file has no entry in sprint-status.yaml",
    )
    return parser.parse_args()


def load_sprint_status(path: Path) -> dict[str, str]:
    status_map: dict[str, str] = {}
    pattern = re.compile(r"^\s{2}([0-9]+-[0-9]+-[a-z0-9-]+):\s+([a-z-]+)\s*$")

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            key, status = match.groups()
            status_map[key] = status

    return status_map


def read_story_status(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Status:"):
            return line.split(":", 1)[1].strip()
    return None


def should_skip(path: Path) -> bool:
    name = path.name
    return name.startswith("_") or name.lower() == "readme.md"


def main() -> int:
    args = parse_args()
    sprint_path = Path(args.sprint_status)
    story_dir = Path(args.story_dir)

    if not sprint_path.exists():
        print(f"sprint-status.yaml not found at {sprint_path}", file=sys.stderr)
        return 2
    if not story_dir.exists():
        print(f"Story directory not found at {story_dir}", file=sys.stderr)
        return 2

    sprint_map = load_sprint_status(sprint_path)

    mismatches: list[str] = []
    missing: list[str] = []
    missing_status: list[str] = []

    for story_file in sorted(story_dir.glob("*.md")):
        if should_skip(story_file):
            continue
        key = story_file.stem
        story_status = read_story_status(story_file)
        if story_status is None:
            missing_status.append(str(story_file))
            continue
        sprint_status = sprint_map.get(key)
        if sprint_status is None:
            missing.append(key)
            continue
        if story_status != sprint_status:
            mismatches.append(
                f"{key}: sprint-status={sprint_status}, story={story_status} ({story_file})"
            )

    if mismatches:
        print("Status mismatches:")
        for item in mismatches:
            print(f"- {item}")

    if missing_status:
        print("\nStories missing Status line:")
        for item in missing_status:
            print(f"- {item}")

    if missing:
        print("\nStories missing from sprint-status.yaml:")
        for item in missing:
            print(f"- {item}")

    if mismatches:
        return 1
    if missing and args.fail_on_missing:
        return 1
    if missing_status:
        return 1

    print("Story status validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
