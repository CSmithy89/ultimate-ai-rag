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


def test_setup_voice_writes_custom_profile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "enterprise.yaml").write_text(
            "voice:\n"
            "  enabled: true\n"
            "  whisper_model: base\n"
            "  tts_provider: openai\n"
            "  tts_voice: alloy\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["setup", "--category", "voice", "--profile", "enterprise", "--yes"],
        )
        assert result.exit_code == 0
        custom_path = profile_dir / "custom.yaml"
        assert custom_path.exists()
        contents = custom_path.read_text(encoding="utf-8")
        assert "voice" in contents


def test_setup_observability_writes_custom_profile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "standard.yaml").write_text(
            "observability:\n"
            "  prometheus_enabled: true\n"
            "  cost_tracking_enabled: true\n"
            "  trajectory_debugging_enabled: false\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["setup", "--category", "observability", "--profile", "standard", "--yes"],
        )
        assert result.exit_code == 0
        custom_path = profile_dir / "custom.yaml"
        assert custom_path.exists()
        contents = custom_path.read_text(encoding="utf-8")
        assert "observability" in contents


def test_setup_codebase_writes_custom_profile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem() as root_str:
        root = Path(root_str)
        profile_dir = root / "config" / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "enterprise.yaml").write_text(
            "ingestion:\n"
            "  codebase_enabled: true\n"
            "codebase_intelligence:\n"
            "  hallucination_detection_enabled: true\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["setup", "--category", "codebase", "--profile", "enterprise", "--yes"],
        )
        assert result.exit_code == 0
        custom_path = profile_dir / "custom.yaml"
        assert custom_path.exists()
        contents = custom_path.read_text(encoding="utf-8")
        assert "codebase_intelligence" in contents
