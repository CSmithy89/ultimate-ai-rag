from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli.main import app


def test_setup_ingestion_writes_custom_profile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "standard.yaml").write_text(
            "ingestion:\n"
            "  crawl_profile: thorough\n"
            "  fallback_enabled: false\n"
            "  pdf_enabled: true\n"
            "  youtube_enabled: true\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["setup", "--category", "ingestion", "--profile", "standard", "--yes"],
        )
        assert result.exit_code == 0
        custom_path = profile_dir / "custom.yaml"
        assert custom_path.exists()
        contents = custom_path.read_text(encoding="utf-8")
        assert "ingestion" in contents
        assert "crawl_profile" in contents


def test_setup_memory_graph_writes_custom_profile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "standard.yaml").write_text(
            "memory:\n"
            "  scopes_enabled: true\n"
            "  default_scope: session\n"
            "  consolidation_enabled: false\n"
            "community:\n"
            "  detection_enabled: false\n"
            "graph_intelligence:\n"
            "  lazy_rag_enabled: false\n"
            "  query_routing_enabled: true\n"
            "  graph_reranker_enabled: false\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["setup", "--category", "memory-graph", "--profile", "standard", "--yes"],
        )
        assert result.exit_code == 0
        custom_path = profile_dir / "custom.yaml"
        assert custom_path.exists()
        contents = custom_path.read_text(encoding="utf-8")
        assert "memory" in contents
        assert "graph_intelligence" in contents
