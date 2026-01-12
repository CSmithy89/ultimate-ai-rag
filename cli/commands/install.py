from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

import typer
from rich.console import Console

from cli.prompts.customize import run_customize
from cli.prompts.fast_path import run_fast_path
from cli.prompts.shared import requires_api_key, validate_api_key
from cli.ui.panels import header_panel, success_panel, summary_panel

DEFAULT_SUBPROCESS_TIMEOUT_S = 5.0
DEFAULT_DOCKER_TIMEOUT_S = 300.0


@dataclass
class InstallSelections:
    profile: str
    llm_provider: str
    api_key: str | None
    framework: str
    embedding_provider: str
    enable_reranking: bool
    enable_contextual_retrieval: bool
    enable_voice: bool


def _get_timeout(env_key: str, default: float) -> float:
    raw_value = os.getenv(env_key)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _read_total_memory_gb() -> float | None:
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        kb = int(parts[1])
                        return kb / 1024 / 1024
        except (OSError, ValueError):
            return None
    if sys.platform == "darwin":
        try:
            import subprocess

            output = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                timeout=_get_timeout("RAG_CLI_SUBPROCESS_TIMEOUT", DEFAULT_SUBPROCESS_TIMEOUT_S),
            )
            bytes_total = int(output.strip())
            return bytes_total / 1024 / 1024 / 1024
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def _detect_gpu() -> str:
    if sys.platform.startswith("linux"):
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=_get_timeout("RAG_CLI_SUBPROCESS_TIMEOUT", DEFAULT_SUBPROCESS_TIMEOUT_S),
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"NVIDIA ({result.stdout.strip()})"
        except (OSError, subprocess.SubprocessError):
            pass
    if sys.platform == "darwin":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "Apple MPS"
        except (ImportError, AttributeError):
            pass
    return "not detected"


def _recommend_profile() -> tuple[str, list[str]]:
    cpu_count = os.cpu_count() or 1
    ram_gb = _read_total_memory_gb()
    gpu_info = _detect_gpu()
    profile = "standard"
    if ram_gb is not None:
        if ram_gb < 16:
            profile = "minimal"
        elif ram_gb >= 32:
            profile = "enterprise"
    summary_lines = [f"CPU: {cpu_count} cores"]
    if ram_gb is None:
        summary_lines.append("RAM: unknown")
    else:
        summary_lines.append(f"RAM: {ram_gb:.1f} GB")
    summary_lines.append(f"GPU: {gpu_info}")
    return profile, summary_lines


def _update_env_lines(lines: list[str], key: str, value: str) -> list[str]:
    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key}="):
            before, sep, after = line.partition(" #")
            suffix = f" #" + after if sep else ""
            new_lines.append(f"{key}={value}{suffix}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"{key}={value}")
    return new_lines


def _extract_env_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].split(" #", 1)[0].strip()
    return None


def _validate_database_url(value: str) -> None:
    if not value.startswith("postgresql"):
        raise typer.BadParameter("Invalid PostgreSQL connection string")


def _validate_neo4j_uri(value: str) -> None:
    if not value.startswith("bolt://"):
        raise typer.BadParameter("Neo4j URI must start with 'bolt://'")


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    tail = value[-4:]
    return f"{'*' * max(0, len(value) - 4)}{tail}"


def _run_docker_compose(console: Console, dry_run: bool) -> None:
    command = ["docker", "compose", "up", "-d"]
    if dry_run:
        console.print(f"[dry-run] {' '.join(command)}")
        return
    try:
        import subprocess

        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_get_timeout("RAG_CLI_DOCKER_TIMEOUT", DEFAULT_DOCKER_TIMEOUT_S),
        )
        if result.returncode != 0:
            output = result.stderr.strip() or result.stdout.strip()
            message = output or "Docker compose failed"
            if "Cannot connect to the Docker daemon" in message:
                raise typer.BadParameter("Docker daemon not running. Start Docker Desktop")
            raise typer.BadParameter(message)
    except FileNotFoundError:
        raise typer.BadParameter("Docker not installed. Install Docker Desktop or engine")
    except subprocess.TimeoutExpired:
        raise typer.BadParameter("Docker compose timed out after 300s. Check docker logs")


def _check_url(url: str, timeout: float) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (URLError, TimeoutError):
        return False


def _wait_for_service(console: Console, name: str, url: str, timeout_s: float = 30.0) -> None:
    start = monotonic()
    while monotonic() - start < timeout_s:
        if _check_url(url, timeout=2.0):
            elapsed = monotonic() - start
            console.print(f"  ✓ {name} - healthy ({elapsed:.1f}s)")
            return
        sleep(1.0)
    raise typer.BadParameter(
        f"{name} failed to become healthy. Check for port conflicts or docker logs.")


def _write_env(selections: InstallSelections, template_path: Path, output_path: Path) -> None:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    lines = _update_env_lines(lines, "LLM_PROVIDER", selections.llm_provider)
    if selections.api_key:
        if selections.llm_provider == "openai":
            if not validate_api_key("openai", selections.api_key):
                raise typer.BadParameter("OpenAI keys start with 'sk-'")
            lines = _update_env_lines(lines, "OPENAI_API_KEY", selections.api_key)
        elif selections.llm_provider == "anthropic":
            if not validate_api_key("anthropic", selections.api_key):
                raise typer.BadParameter("Anthropic keys start with 'sk-ant-'")
            lines = _update_env_lines(lines, "ANTHROPIC_API_KEY", selections.api_key)
        elif selections.llm_provider == "openrouter":
            lines = _update_env_lines(lines, "OPENROUTER_API_KEY", selections.api_key)
        elif selections.llm_provider == "gemini":
            lines = _update_env_lines(lines, "GEMINI_API_KEY", selections.api_key)

    lines = _update_env_lines(lines, "EMBEDDING_PROVIDER", selections.embedding_provider)
    lines = _update_env_lines(
        lines,
        "RERANKER_ENABLED",
        "true" if selections.enable_reranking else "false",
    )
    if selections.enable_reranking:
        lines = _update_env_lines(lines, "RERANKER_PROVIDER", "flashrank")
    lines = _update_env_lines(
        lines,
        "CONTEXTUAL_RETRIEVAL_ENABLED",
        "true" if selections.enable_contextual_retrieval else "false",
    )
    lines = _update_env_lines(
        lines,
        "VOICE_IO_ENABLED",
        "true" if selections.enable_voice else "false",
    )

    db_url = _extract_env_value(lines, "DATABASE_URL")
    if db_url:
        _validate_database_url(db_url)
    neo4j_uri = _extract_env_value(lines, "NEO4J_URI")
    if neo4j_uri:
        _validate_neo4j_uri(neo4j_uri)

    header_lines = [
        "# ═══════════════════════════════════════════════════════════════",
        "# AGENTIC RAG CONFIGURATION",
        f"# Generated by rag-install on {datetime.utcnow().isoformat()}",
        f"# Profile: {selections.profile}",
        "# ═══════════════════════════════════════════════════════════════",
    ]
    if output_path.exists():
        backup_path = output_path.parent / (output_path.name + ".bak")
        output_path.replace(backup_path)
    output_path.write_text("\n".join([*header_lines, *lines]) + "\n", encoding="utf-8")


def _generate_framework_template(framework: str, target_root: Path) -> Path:
    import shutil

    source = Path("cli") / "templates" / "frameworks" / framework
    if not source.exists():
        raise typer.BadParameter(f"Unknown framework template: {framework}")
    target = target_root / framework
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target


def _generate_skills(target_root: Path) -> Path:
    import shutil

    source = Path("cli") / "templates" / "skills"
    if not source.exists():
        raise typer.BadParameter("Skills templates not found")
    target = target_root
    shutil.copytree(source, target, dirs_exist_ok=True)
    return target

def _gather_selections(
    console: Console,
    profile: str | None,
    llm_provider: str | None,
    api_key: str | None,
    framework: str | None,
    customize: bool,
    yes: bool,
) -> InstallSelections:
    llm_providers = ["openai", "anthropic", "openrouter", "ollama", "gemini"]
    frameworks = ["none", "pydanticai", "crewai", "langgraph", "anthropic"]

    if yes:
        if not profile or not llm_provider:
            raise typer.BadParameter("--profile and --llm are required with --yes")
        if requires_api_key(llm_provider) and not api_key:
            raise typer.BadParameter("--api-key is required for the selected provider")
        return InstallSelections(
            profile=profile,
            llm_provider=llm_provider,
            api_key=api_key,
            framework=framework or "none",
            embedding_provider=llm_provider,
            enable_reranking=False,
            enable_contextual_retrieval=False,
            enable_voice=False,
        )

    recommended_profile, _ = _recommend_profile()
    selections = run_fast_path(
        console=console,
        recommended_profile=recommended_profile,
        llm_providers=llm_providers,
        frameworks=frameworks,
    )
    embedding_provider = selections.llm_provider
    enable_reranking = False
    enable_contextual_retrieval = False
    enable_voice = False

    if customize or selections.customize:
        customize_selections = run_customize(console, embedding_provider)
        embedding_provider = customize_selections.embedding_provider
        enable_reranking = customize_selections.enable_reranking
        enable_contextual_retrieval = customize_selections.enable_contextual_retrieval
        enable_voice = customize_selections.enable_stt or customize_selections.enable_tts

    return InstallSelections(
        profile=selections.profile,
        llm_provider=selections.llm_provider,
        api_key=selections.api_key,
        framework=selections.framework,
        embedding_provider=embedding_provider,
        enable_reranking=enable_reranking,
        enable_contextual_retrieval=enable_contextual_retrieval,
        enable_voice=enable_voice,
    )


def run_install(
    profile: str | None = typer.Option(None, "--profile"),
    llm: str | None = typer.Option(None, "--llm"),
    api_key: str | None = typer.Option(None, "--api-key"),
    framework: str | None = typer.Option(None, "--framework"),
    customize: bool = typer.Option(False, "--customize"),
    yes: bool = typer.Option(False, "--yes"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    with_skills: bool = typer.Option(False, "--with-skills"),
) -> None:
    console = Console()
    recommended_profile, hardware_lines = _recommend_profile()

    console.print(header_panel())
    for line in hardware_lines:
        console.print(f"  ✓ {line}")
    console.print(f"  ✓ Recommended: {recommended_profile.title()} profile")

    selections = _gather_selections(
        console=console,
        profile=profile,
        llm_provider=llm,
        api_key=api_key,
        framework=framework,
        customize=customize,
        yes=yes,
    )

    summary = {
        "Profile": selections.profile,
        "LLM": selections.llm_provider,
        "Embedding": selections.embedding_provider,
        "Framework": selections.framework,
    }
    if selections.api_key:
        summary["API Key"] = _mask_secret(selections.api_key)
    console.print(summary_panel(summary))

    template_path = Path(".env.example")
    output_path = Path(".env")
    if not template_path.exists():
        console.print("Missing .env.example template. Run from the repo root.")
        raise typer.Exit(code=1)

    _write_env(selections, template_path, output_path)

    if selections.framework != "none":
        _generate_framework_template(selections.framework, Path("examples"))
    if with_skills:
        _generate_skills(Path(".skills"))

    _run_docker_compose(console, dry_run)
    if not dry_run:
        console.print("Starting services...")
        _wait_for_service(console, "Backend", "http://localhost:8000/health")
        _wait_for_service(console, "Frontend", "http://localhost:3000")

    success_lines = [
        "Your RAG system configuration is ready.",
        "Frontend: http://localhost:3000",
        "API Docs: http://localhost:8000/docs",
        "Run 'rag-cli setup' to customize advanced features.",
    ]
    console.print(success_panel(success_lines))
