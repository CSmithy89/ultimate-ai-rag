"""Provider adapter definitions for LLM and embedding clients."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..config import Settings

OPENAI_COMPATIBLE_LLM_PROVIDERS = {"openai", "openrouter", "ollama"}
OPENAI_COMPATIBLE_EMBEDDING_PROVIDERS = {"openai", "openrouter", "ollama"}


class UnsupportedLLMProviderError(RuntimeError):
    """Raised when an unsupported LLM provider is selected."""


class UnsupportedEmbeddingProviderError(RuntimeError):
    """Raised when an unsupported embedding provider is selected."""


class EmbeddingProviderType(str, Enum):
    """Embedding provider types."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    VOYAGE = "voyage"


@dataclass(frozen=True)
class LLMProviderAdapter:
    """Adapter for OpenAI-compatible providers."""

    provider: str
    api_key: Optional[str]
    base_url: Optional[str]
    embedding_api_key: Optional[str]
    embedding_base_url: Optional[str]

    def openai_kwargs(self) -> dict[str, Any]:
        """Build kwargs for OpenAI-compatible clients."""
        if self.provider not in OPENAI_COMPATIBLE_LLM_PROVIDERS:
            raise UnsupportedLLMProviderError(
                f"Provider {self.provider!r} is not OpenAI-compatible."
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key or ""}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs

    def embedding_kwargs(self) -> dict[str, Any]:
        """Build kwargs for OpenAI-compatible embedding clients."""
        kwargs: dict[str, Any] = {"api_key": self.embedding_api_key or ""}
        if self.embedding_base_url:
            kwargs["base_url"] = self.embedding_base_url
        return kwargs


@dataclass(frozen=True)
class EmbeddingProviderAdapter:
    """Adapter for embedding providers (supports OpenAI, Gemini, Voyage, etc.)."""

    provider: EmbeddingProviderType
    api_key: Optional[str]
    base_url: Optional[str]
    model: str

    def is_openai_compatible(self) -> bool:
        """Check if provider uses OpenAI-compatible API."""
        return self.provider.value in OPENAI_COMPATIBLE_EMBEDDING_PROVIDERS

    def openai_kwargs(self) -> dict[str, Any]:
        """Build kwargs for OpenAI-compatible embedding clients."""
        if not self.is_openai_compatible():
            raise UnsupportedEmbeddingProviderError(
                f"Provider {self.provider.value!r} is not OpenAI-compatible. "
                "Use provider-specific client instead."
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key or ""}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _ensure_llm_provider_runtime(provider: str) -> None:
    """Ensure required LLM provider packages are installed."""
    if provider == "anthropic":
        if not _module_available("anthropic"):
            raise UnsupportedLLMProviderError(
                "Anthropic support requires the `anthropic` package. "
                "Install with `uv add anthropic`."
            )
        if not _module_available("agno.models.anthropic"):
            raise UnsupportedLLMProviderError(
                "Anthropic support requires agno anthropic models. "
                "Ensure `agno` is installed with anthropic extras."
            )
    if provider == "gemini":
        if not (_module_available("google.genai") or _module_available("google.generativeai")):
            raise UnsupportedLLMProviderError(
                "Gemini support requires the `google-genai` package. "
                "Install with `uv add google-genai`."
            )
        if not _module_available("agno.models.google"):
            raise UnsupportedLLMProviderError(
                "Gemini support requires agno google models. "
                "Ensure `agno` is installed with google extras."
            )


def _ensure_embedding_provider_runtime(provider: str) -> None:
    """Ensure required embedding provider packages are installed."""
    if provider == "gemini":
        if not (_module_available("google.genai") or _module_available("google.generativeai")):
            raise UnsupportedEmbeddingProviderError(
                "Gemini embedding support requires the `google-genai` package. "
                "Install with `uv add google-genai`."
            )
    elif provider == "voyage":
        if not _module_available("voyageai"):
            raise UnsupportedEmbeddingProviderError(
                "Voyage AI embedding support requires the `voyageai` package. "
                "Install with `uv add voyageai`."
            )


def get_llm_adapter(settings: Settings) -> LLMProviderAdapter:
    """Resolve provider adapter for the configured LLM provider.

    Now uses the separate embedding_provider config for embeddings.
    """
    if settings.llm_provider in {"anthropic", "gemini"}:
        _ensure_llm_provider_runtime(settings.llm_provider)
        return LLMProviderAdapter(
            provider=settings.llm_provider,
            api_key=settings.llm_api_key,
            base_url=None,
            embedding_api_key=settings.embedding_api_key,
            embedding_base_url=settings.embedding_base_url,
        )
    if settings.llm_provider not in OPENAI_COMPATIBLE_LLM_PROVIDERS:
        raise UnsupportedLLMProviderError(
            "LLM_PROVIDER must be openai/openrouter/ollama/anthropic/gemini. "
            f"Got {settings.llm_provider!r}."
        )
    return LLMProviderAdapter(
        provider=settings.llm_provider,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        embedding_api_key=settings.embedding_api_key,
        embedding_base_url=settings.embedding_base_url,
    )


def get_embedding_adapter(settings: Settings) -> EmbeddingProviderAdapter:
    """Resolve provider adapter for the configured embedding provider.

    Supports:
    - openai: OpenAI embeddings (text-embedding-ada-002, text-embedding-3-small/large)
    - openrouter: OpenRouter embeddings (via OpenAI-compatible API)
    - ollama: Local Ollama embeddings (via OpenAI-compatible API)
    - gemini: Google Gemini embeddings (text-embedding-004)
    - voyage: Voyage AI embeddings (voyage-3, voyage-3-lite, voyage-code-3)
    """
    provider_str = settings.embedding_provider
    _ensure_embedding_provider_runtime(provider_str)

    try:
        provider = EmbeddingProviderType(provider_str)
    except ValueError:
        raise UnsupportedEmbeddingProviderError(
            f"EMBEDDING_PROVIDER must be openai/openrouter/ollama/gemini/voyage. "
            f"Got {provider_str!r}."
        )

    return EmbeddingProviderAdapter(
        provider=provider,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
    )
