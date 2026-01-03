"""Provider adapter definitions for LLM clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..config import Settings

OPENAI_COMPATIBLE_PROVIDERS = {"openai", "openrouter", "ollama"}


class UnsupportedLLMProviderError(RuntimeError):
    """Raised when an unsupported LLM provider is selected."""


@dataclass(frozen=True)
class LLMProviderAdapter:
    """Adapter for OpenAI-compatible providers."""

    provider: str
    api_key: Optional[str]
    base_url: Optional[str]

    def openai_kwargs(self) -> dict[str, Any]:
        """Build kwargs for OpenAI-compatible clients."""
        kwargs: dict[str, Any] = {"api_key": self.api_key or ""}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


def get_llm_adapter(settings: Settings) -> LLMProviderAdapter:
    """Resolve provider adapter for the configured LLM provider."""
    if settings.llm_provider not in OPENAI_COMPATIBLE_PROVIDERS:
        raise UnsupportedLLMProviderError(
            "LLM_PROVIDER must be openai/openrouter/ollama until adapters "
            f"for {settings.llm_provider!r} are implemented."
        )
    return LLMProviderAdapter(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
