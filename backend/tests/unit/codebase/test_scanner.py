"""Tests for codebase file scanner."""

from pathlib import Path

from agentic_rag_backend.codebase.indexing.scanner import FileScanner


def test_scanner_respects_gitignore(tmp_path: Path):
    repo = tmp_path
    (repo / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
    ignored_dir = repo / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "skip.py").write_text("print('skip')\n", encoding="utf-8")

    scanner = FileScanner(repo_path=repo, exclude_patterns=[])
    files = scanner.scan()

    assert any(file.endswith("main.py") for file in files)
    assert not any("skip.py" in file for file in files)


def test_scanner_respects_exclude_patterns(tmp_path: Path):
    repo = tmp_path
    (repo / "main.py").write_text("print('ok')\n", encoding="utf-8")
    build_dir = repo / "build"
    build_dir.mkdir()
    (build_dir / "artifact.py").write_text("print('build')\n", encoding="utf-8")

    scanner = FileScanner(repo_path=repo, exclude_patterns=["**/build/**"])
    files = scanner.scan()

    assert any(file.endswith("main.py") for file in files)
    assert not any("artifact.py" in file for file in files)
