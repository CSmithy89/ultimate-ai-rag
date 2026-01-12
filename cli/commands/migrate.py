from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from cli.profile import load_profile, parse_env_file, write_custom_profile

ENV_TO_PROFILE_PATH = {
    "LLM_PROVIDER": ("llm", "provider"),
    "LLM_MODEL_ID": ("llm", "model"),
    "EMBEDDING_PROVIDER": ("embedding", "provider"),
    "EMBEDDING_MODEL": ("embedding", "model"),
    "EMBEDDING_DIMENSION": ("embedding", "dimension"),
    "RERANKER_ENABLED": ("retrieval", "reranker", "enabled"),
    "RERANKER_PROVIDER": ("retrieval", "reranker", "provider"),
    "CONTEXTUAL_RETRIEVAL_ENABLED": ("retrieval", "contextual_retrieval", "enabled"),
    "GRADER_ENABLED": ("retrieval", "grader", "enabled"),
    "MEMORY_SCOPES_ENABLED": ("memory", "scopes_enabled"),
    "MEMORY_DEFAULT_SCOPE": ("memory", "default_scope"),
    "MEMORY_CONSOLIDATION_ENABLED": ("memory", "consolidation_enabled"),
    "COMMUNITY_DETECTION_ENABLED": ("community", "detection_enabled"),
    "CRAWL4AI_PROFILE": ("ingestion", "crawl_profile"),
    "CRAWL_FALLBACK_ENABLED": ("ingestion", "fallback_enabled"),
    "CODEBASE_RAG_ENABLED": ("ingestion", "codebase_enabled"),
    "EXTERNAL_SYNC_ENABLED": ("ingestion", "external_sync_enabled"),
    "VOICE_IO_ENABLED": ("voice", "enabled"),
    "PROMETHEUS_ENABLED": ("observability", "prometheus_enabled"),
    "A2A_ENABLED": ("protocols", "a2a", "enabled"),
    "A2A_MAX_SESSIONS_PER_TENANT": ("protocols", "a2a", "max_sessions_per_tenant"),
    "A2A_MAX_MESSAGES_PER_SESSION": ("protocols", "a2a", "max_messages_per_session"),
}


def _get_nested(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _set_nested(config: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def _diff_overrides(base: dict[str, Any], env_values: dict[str, str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for env_key, path in ENV_TO_PROFILE_PATH.items():
        if env_key not in env_values:
            continue
        base_value = _get_nested(base, path)
        env_value = env_values[env_key]
        if base_value is None:
            _set_nested(overrides, path, env_value)
            continue
        if str(base_value).lower() != env_value.lower():
            _set_nested(overrides, path, env_value)
    return overrides


def analyze(profile: str = "standard", env_path: Path = Path(".env")) -> dict[str, Any]:
    base = load_profile(profile)
    env_values = parse_env_file(env_path)
    overrides = _diff_overrides(base, env_values)
    return {"profile": profile, "override_count": len(overrides), "overrides": overrides}


def run_analyze(profile: str | None = typer.Option(None, "--profile")) -> None:
    console = Console()
    env_path = Path(".env")
    if not env_path.exists():
        raise typer.BadParameter(".env not found")
    profile_name = profile or "standard"
    try:
        result = analyze(profile_name, env_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Base profile: {result['profile']}")
    console.print(f"Overrides: {result['override_count']}")


def run_execute(profile: str | None = typer.Option(None, "--profile")) -> None:
    console = Console()
    env_path = Path(".env")
    if not env_path.exists():
        raise typer.BadParameter(".env not found")
    profile_name = profile or "standard"
    try:
        result = analyze(profile_name, env_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not result["overrides"]:
        console.print("No overrides detected - custom profile not written")
        return
    write_custom_profile(result["overrides"])
    console.print("Custom profile written to config/profiles/custom.yaml")
