from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from cli.profile import load_custom_profile, load_profile, write_custom_profile


def _prompt_bool(console: Console, label: str, default: bool) -> bool:
    return Confirm.ask(label, default=default, console=console)


def run_setup(
    category: str | None = typer.Option(None, "--category"),
    profile: str | None = typer.Option(None, "--profile"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    console = Console()
    profile_name = (profile or os.getenv("CONFIG_PROFILE", "standard")).strip().lower()
    try:
        base_config = load_profile(profile_name)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    custom_config = load_custom_profile()
    target_category = category or "all"

    if target_category in {"ingestion", "all"}:
        ingestion_defaults = base_config.get("ingestion", {})
        crawl_profile = ingestion_defaults.get("crawl_profile", "thorough")
        fallback_enabled = bool(ingestion_defaults.get("fallback_enabled", False))
        pdf_enabled = bool(ingestion_defaults.get("pdf_enabled", True))
        youtube_enabled = bool(ingestion_defaults.get("youtube_enabled", True))
        codebase_enabled = bool(ingestion_defaults.get("codebase_enabled", False))
        external_sync_enabled = bool(ingestion_defaults.get("external_sync_enabled", False))

        if not yes:
            crawl_profile = Prompt.ask(
                "Select crawl profile",
                choices=["fast", "thorough", "stealth"],
                default=crawl_profile,
                console=console,
            )
            fallback_enabled = _prompt_bool(console, "Enable crawl fallback?", fallback_enabled)
            pdf_enabled = _prompt_bool(console, "Enable PDF ingestion?", pdf_enabled)
            youtube_enabled = _prompt_bool(console, "Enable YouTube ingestion?", youtube_enabled)

            if profile_name == "enterprise":
                codebase_enabled = _prompt_bool(
                    console, "Enable codebase ingestion?", codebase_enabled
                )
                external_sync_enabled = _prompt_bool(
                    console, "Enable external sync ingestion?", external_sync_enabled
                )

        custom_config["ingestion"] = {
            "crawl_profile": crawl_profile,
            "fallback_enabled": fallback_enabled,
            "pdf_enabled": pdf_enabled,
            "youtube_enabled": youtube_enabled,
            "codebase_enabled": codebase_enabled,
            "external_sync_enabled": external_sync_enabled,
        }

    if target_category in {"memory", "graph", "memory-graph", "all"}:
        memory_defaults = base_config.get("memory", {})
        community_defaults = base_config.get("community", {})
        graph_defaults = base_config.get("graph_intelligence", {})

        scopes_enabled = bool(memory_defaults.get("scopes_enabled", False))
        default_scope = memory_defaults.get("default_scope", "session")
        consolidation_enabled = bool(memory_defaults.get("consolidation_enabled", False))

        community_detection_enabled = bool(community_defaults.get("detection_enabled", False))
        lazy_rag_enabled = bool(graph_defaults.get("lazy_rag_enabled", False))
        query_routing_enabled = bool(graph_defaults.get("query_routing_enabled", False))
        graph_reranker_enabled = bool(graph_defaults.get("graph_reranker_enabled", False))

        if not yes and profile_name != "minimal":
            scopes_enabled = _prompt_bool(console, "Enable memory scopes?", scopes_enabled)
            if scopes_enabled:
                default_scope = Prompt.ask(
                    "Default memory scope",
                    choices=["session", "user", "agent"],
                    default=default_scope,
                    console=console,
                )
            consolidation_enabled = _prompt_bool(
                console, "Enable memory consolidation?", consolidation_enabled
            )
            community_detection_enabled = _prompt_bool(
                console, "Enable community detection?", community_detection_enabled
            )
            lazy_rag_enabled = _prompt_bool(console, "Enable LazyRAG?", lazy_rag_enabled)
            query_routing_enabled = _prompt_bool(
                console, "Enable query routing?", query_routing_enabled
            )
            graph_reranker_enabled = _prompt_bool(
                console, "Enable graph reranker?", graph_reranker_enabled
            )

        custom_config["memory"] = {
            "scopes_enabled": scopes_enabled,
            "default_scope": default_scope,
            "consolidation_enabled": consolidation_enabled,
        }
        custom_config["community"] = {
            "detection_enabled": community_detection_enabled,
        }
        custom_config["graph_intelligence"] = {
            "lazy_rag_enabled": lazy_rag_enabled,
            "query_routing_enabled": query_routing_enabled,
            "graph_reranker_enabled": graph_reranker_enabled,
        }

    if target_category in {"voice", "all"}:
        voice_defaults = base_config.get("voice", {})
        voice_enabled = bool(voice_defaults.get("enabled", False))
        whisper_model = voice_defaults.get("whisper_model", "base")
        tts_provider = voice_defaults.get("tts_provider", "openai")
        tts_voice = voice_defaults.get("tts_voice", "alloy")

        if not yes:
            voice_enabled = _prompt_bool(console, "Enable voice I/O?", voice_enabled)
            if voice_enabled:
                whisper_model = Prompt.ask(
                    "Whisper model",
                    choices=["tiny", "base", "small", "medium", "large"],
                    default=whisper_model,
                    console=console,
                )
                tts_provider = Prompt.ask(
                    "TTS provider",
                    choices=["openai", "elevenlabs", "pyttsx3"],
                    default=tts_provider,
                    console=console,
                )
                if tts_provider == "openai":
                    tts_voice = Prompt.ask(
                        "OpenAI TTS voice",
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        default=tts_voice,
                        console=console,
                    )

        custom_config["voice"] = {
            "enabled": voice_enabled,
            "whisper_model": whisper_model,
            "tts_provider": tts_provider,
            "tts_voice": tts_voice,
        }

    if target_category in {"observability", "all"}:
        observability_defaults = base_config.get("observability", {})
        prometheus_enabled = bool(observability_defaults.get("prometheus_enabled", False))
        cost_tracking_enabled = bool(observability_defaults.get("cost_tracking_enabled", False))
        trajectory_debugging_enabled = bool(
            observability_defaults.get("trajectory_debugging_enabled", False)
        )

        if not yes and profile_name != "minimal":
            prometheus_enabled = _prompt_bool(
                console, "Enable Prometheus metrics?", prometheus_enabled
            )
            cost_tracking_enabled = _prompt_bool(
                console, "Enable cost tracking?", cost_tracking_enabled
            )
            trajectory_debugging_enabled = _prompt_bool(
                console, "Enable trajectory debugging?", trajectory_debugging_enabled
            )

        custom_config["observability"] = {
            "prometheus_enabled": prometheus_enabled,
            "cost_tracking_enabled": cost_tracking_enabled,
            "trajectory_debugging_enabled": trajectory_debugging_enabled,
        }

    if target_category in {"codebase", "all"}:
        codebase_defaults = base_config.get("codebase_intelligence", {})
        ingestion_defaults = base_config.get("ingestion", {})
        codebase_enabled = bool(
            codebase_defaults.get("codebase_enabled", ingestion_defaults.get("codebase_enabled", False))
        )
        hallucination_detection_enabled = bool(
            codebase_defaults.get("hallucination_detection_enabled", False)
        )

        if not yes and profile_name == "enterprise":
            codebase_enabled = _prompt_bool(
                console, "Enable codebase indexing?", codebase_enabled
            )
            hallucination_detection_enabled = _prompt_bool(
                console,
                "Enable hallucination detection?",
                hallucination_detection_enabled,
            )

        custom_config["codebase_intelligence"] = {
            "codebase_enabled": codebase_enabled,
            "hallucination_detection_enabled": hallucination_detection_enabled,
        }

    if target_category in {"protocols", "all"}:
        protocols_defaults = base_config.get("protocols", {})
        a2a_defaults = protocols_defaults.get("a2a", {})
        mcp_defaults = protocols_defaults.get("mcp", {})

        a2a_enabled = bool(a2a_defaults.get("enabled", True))
        max_sessions = int(a2a_defaults.get("max_sessions_per_tenant", 100))
        max_messages = int(a2a_defaults.get("max_messages_per_session", 1000))
        mcp_enabled = bool(mcp_defaults.get("enabled", True))

        if not yes:
            a2a_enabled = _prompt_bool(console, "Enable A2A protocol?", a2a_enabled)
            max_sessions = int(
                Prompt.ask(
                    "A2A max sessions per tenant",
                    default=str(max_sessions),
                    console=console,
                )
            )
            max_messages = int(
                Prompt.ask(
                    "A2A max messages per session",
                    default=str(max_messages),
                    console=console,
                )
            )
            mcp_enabled = _prompt_bool(console, "Enable MCP protocol?", mcp_enabled)

        custom_config["protocols"] = {
            "a2a": {
                "enabled": a2a_enabled,
                "max_sessions_per_tenant": max_sessions,
                "max_messages_per_session": max_messages,
            },
            "mcp": {
                "enabled": mcp_enabled,
            },
        }

    write_custom_profile(custom_config)
    console.print("Configuration saved to config/profiles/custom.yaml")
