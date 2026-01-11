from __future__ import annotations

from pathlib import Path


def test_framework_template_files_exist() -> None:
    base = Path("cli/templates/frameworks")
    required = {
        "pydanticai": ["README.md", "pyproject.toml", "agent.py", "mcp_client.py"],
        "crewai": ["README.md", "pyproject.toml", "crew.py", "tasks.py"],
        "langgraph": ["README.md", "pyproject.toml", "graph.py", "nodes.py"],
        "anthropic": ["README.md", "pyproject.toml", "agent.py"],
    }

    for framework, files in required.items():
        framework_dir = base / framework
        assert framework_dir.exists()
        for filename in files:
            assert (framework_dir / filename).exists()


def test_template_readme_has_setup() -> None:
    base = Path("cli/templates/frameworks")
    for framework_dir in base.iterdir():
        if not framework_dir.is_dir():
            continue
        readme = framework_dir / "README.md"
        contents = readme.read_text(encoding="utf-8")
        assert "Setup" in contents
