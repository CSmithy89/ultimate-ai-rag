from __future__ import annotations

from dataclasses import dataclass

from rich.prompt import Confirm, Prompt


@dataclass
class CustomizeSelections:
    embedding_provider: str
    enable_reranking: bool
    enable_contextual_retrieval: bool
    enable_stt: bool
    enable_tts: bool


def run_customize(console, embedding_default: str) -> CustomizeSelections:
    embedding_provider = Prompt.ask(
        "Embedding provider",
        choices=["openai", "voyage", "gemini", "ollama"],
        default=embedding_default,
        console=console,
    )

    enable_reranking = Confirm.ask(
        "Enable cross-encoder reranking?",
        default=False,
        console=console,
    )
    enable_contextual = Confirm.ask(
        "Enable contextual retrieval?",
        default=False,
        console=console,
    )
    enable_stt = Confirm.ask(
        "Enable speech-to-text (Whisper)?",
        default=False,
        console=console,
    )
    enable_tts = Confirm.ask(
        "Enable text-to-speech?",
        default=False,
        console=console,
    )

    return CustomizeSelections(
        embedding_provider=embedding_provider,
        enable_reranking=enable_reranking,
        enable_contextual_retrieval=enable_contextual,
        enable_stt=enable_stt,
        enable_tts=enable_tts,
    )
