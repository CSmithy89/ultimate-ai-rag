"""File scanning utilities for codebase indexing."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterable, Optional

import structlog

from ..types import get_language_for_file

logger = structlog.get_logger(__name__)


class FileScanner:
    """Scan a repository for source files respecting ignore patterns."""

    def __init__(
        self,
        repo_path: Path,
        exclude_patterns: Optional[list[str]] = None,
        languages: Optional[Iterable[str]] = None,
    ) -> None:
        self.repo_path = repo_path
        self.exclude_patterns = exclude_patterns or []
        self.languages = {lang.lower() for lang in (languages or [])} or None
        self._gitignore_matcher = None

        gitignore_path = repo_path / ".gitignore"
        if gitignore_path.exists():
            try:
                from gitignore_parser import parse_gitignore

                self._gitignore_matcher = parse_gitignore(str(gitignore_path))
            except Exception as exc:
                logger.warning("gitignore_parse_failed", error=str(exc), path=str(gitignore_path))

    def scan(self) -> list[str]:
        """Return a list of source file paths to index."""
        files: list[str] = []
        repo_root = self.repo_path.resolve()

        for root, dirs, filenames in os.walk(repo_root):
            root_path = Path(root)
            # Filter directories in-place to avoid descending into ignored paths
            dirs[:] = [
                d for d in dirs
                if not self._should_ignore(Path(root_path, d), is_dir=True)
            ]

            for filename in filenames:
                file_path = Path(root_path, filename)
                if self._should_ignore(file_path, is_dir=False):
                    continue

                language = get_language_for_file(filename)
                if language is None:
                    continue
                if self.languages and language.value not in self.languages:
                    continue

                files.append(str(file_path))

        return files

    def _should_ignore(self, path: Path, is_dir: bool) -> bool:
        rel_path = path.relative_to(self.repo_path).as_posix()

        if self._gitignore_matcher and self._gitignore_matcher(str(path)):
            return True

        for pattern in self.exclude_patterns:
            if Path(rel_path).match(pattern):
                return True
            collapsed = pattern.replace("**/", "").replace("/**", "").strip("/")
            if collapsed and not any(ch in collapsed for ch in ("*", "?", "[")):
                if f"/{collapsed}/" in f"/{rel_path}/" or rel_path.startswith(f"{collapsed}/"):
                    return True
            if pattern.endswith("/"):
                prefix = pattern[:-1]
                if rel_path.startswith(prefix) or f"/{prefix}/" in f"/{rel_path}/":
                    return True
            else:
                if fnmatch.fnmatch(rel_path, pattern):
                    return True
                if is_dir and fnmatch.fnmatch(f"{rel_path}/", pattern):
                    return True
                if fnmatch.fnmatch(path.name, pattern):
                    return True

        return False
