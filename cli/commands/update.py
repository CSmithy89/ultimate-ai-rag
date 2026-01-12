from __future__ import annotations

import os
import subprocess
from pathlib import Path

import typer
from rich.console import Console

DEFAULT_SUBPROCESS_TIMEOUT_S = 5.0


def _get_timeout(env_key: str, default: float) -> float:
    raw_value = os.getenv(env_key)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_git_repo(repo_root: Path) -> bool:
    return (repo_root / ".git").exists()


def _run_git(repo_root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=_get_timeout("RAG_CLI_SUBPROCESS_TIMEOUT", DEFAULT_SUBPROCESS_TIMEOUT_S),
    )


def _skip_fetch() -> bool:
    return os.getenv("RAG_CLI_UPDATE_NO_FETCH", "").strip().lower() in {"1", "true", "yes"}


def _dry_run() -> bool:
    return os.getenv("RAG_CLI_UPDATE_DRY_RUN", "").strip().lower() in {"1", "true", "yes"}


def run_check() -> None:
    console = Console()
    repo_root = _repo_root()
    if not _is_git_repo(repo_root):
        console.print("Update check unavailable (not a git checkout).")
        return

    try:
        if not _skip_fetch():
            _run_git(repo_root, ["git", "fetch", "--quiet", "origin", "main"])
        current = _run_git(repo_root, ["git", "rev-parse", "--short", "HEAD"]).stdout.strip()
        try:
            behind = _run_git(
                repo_root, ["git", "rev-list", "--count", "HEAD..origin/main"]
            ).stdout.strip()
        except subprocess.SubprocessError:
            console.print(f"Up to date: {current}")
            return
    except (OSError, subprocess.SubprocessError) as exc:
        console.print(f"Update check unavailable ({exc}).")
        return

    if behind == "0":
        console.print(f"Up to date: {current}")
    else:
        console.print(f"Update available: {current} -> origin/main (+{behind} commits)")


def run_apply() -> None:
    console = Console()
    repo_root = _repo_root()
    if not _is_git_repo(repo_root):
        console.print("Update apply unavailable (not a git checkout).")
        raise typer.Exit(code=1)

    if _dry_run():
        console.print("Applying update (dry run)...")
        console.print("Update complete.")
        return

    try:
        status = _run_git(repo_root, ["git", "status", "--porcelain"]).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        console.print(f"Unable to check git status ({exc}).")
        raise typer.Exit(code=1) from exc

    if status:
        console.print("Working tree is dirty. Commit or stash changes first.")
        raise typer.Exit(code=1)

    console.print("Applying update...")
    try:
        _run_git(repo_root, ["git", "pull", "--rebase", "origin", "main"])
    except (OSError, subprocess.SubprocessError) as exc:
        console.print(f"Update failed ({exc}).")
        raise typer.Exit(code=1) from exc
    console.print("Update complete.")


def run_update_check() -> None:
    run_check()


def run_update_apply() -> None:
    run_apply()
