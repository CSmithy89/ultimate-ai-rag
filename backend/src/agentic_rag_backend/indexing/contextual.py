"""Contextual retrieval chunk enrichment.

Implements Anthropic's contextual retrieval approach where each chunk is
enriched with document context (title, summary) before embedding. This
improves retrieval accuracy by 35-67% by preserving document context.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import structlog
import tiktoken

from .chunker import ChunkData

logger = structlog.get_logger(__name__)

# Configuration constants
MAX_DOCUMENT_CONTEXT_TOKENS = 2000  # Maximum tokens for document context (more accurate than chars)
TIKTOKEN_ENCODING = "cl100k_base"  # Encoding used by Claude and GPT-4

# Lazy-loaded tiktoken encoder
_tiktoken_encoder: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder (singleton pattern)."""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        _tiktoken_encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _tiktoken_encoder


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text with "... [truncated]" suffix if truncation occurred
    """
    encoder = _get_encoder()
    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Truncate tokens and decode back to text
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoder.decode(truncated_tokens)

    return truncated_text + "... [truncated]"


# Default context generation prompt with cache-friendly structure
CONTEXT_GENERATION_PROMPT = """<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


@dataclass
class EnrichedChunk:
    """A chunk enriched with contextual information."""

    original_content: str
    enriched_content: str
    context: str
    chunk_index: int
    token_count: int
    start_char: int
    end_char: int
    context_generation_ms: float


@dataclass
class DocumentContext:
    """Document-level context for chunk enrichment."""

    title: Optional[str]
    summary: Optional[str]
    full_content: str


class ContextualChunkEnricher:
    """Enriches chunks with document context for improved retrieval.

    Uses a cost-effective LLM (claude-3-haiku, gpt-4o-mini) to generate
    chunk-specific context that preserves document meaning.

    Supported providers:
        - anthropic: Direct Anthropic API with prompt caching (~90% cost reduction)
        - openai: Direct OpenAI API
        - openrouter: OpenRouter gateway for multiple model providers (uses OpenAI-compatible API)
    """

    def __init__(
        self,
        model: str = "claude-3-haiku-20240307",
        use_prompt_caching: bool = True,
        api_key: Optional[str] = None,
        provider: str = "anthropic",
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the contextual enricher.

        Args:
            model: LLM model to use for context generation
            use_prompt_caching: Enable prompt caching (Anthropic only)
            api_key: API key for the LLM provider
            provider: LLM provider (anthropic, openai, openrouter)
            base_url: Custom base URL for OpenAI-compatible providers (e.g., OpenRouter)
        """
        self._model = model
        self._use_prompt_caching = use_prompt_caching
        self._api_key = api_key
        self._provider = provider
        self._base_url = base_url
        self._client = None
        self._initialized = False

        logger.info(
            "contextual_enricher_created",
            model=model,
            provider=provider,
            prompt_caching=use_prompt_caching,
            base_url=base_url,
        )

    async def _ensure_client(self) -> None:
        """Initialize the LLM client lazily."""
        if self._initialized:
            return

        if self._provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
                self._initialized = True
                logger.info("anthropic_client_initialized", model=self._model)
            except ImportError:
                raise RuntimeError(
                    "anthropic package required for contextual retrieval. "
                    "Install with: uv add anthropic"
                )
        elif self._provider in {"openai", "openrouter"}:
            try:
                import openai

                # Use custom base_url for OpenRouter or other OpenAI-compatible providers
                client_kwargs = {"api_key": self._api_key}
                if self._base_url:
                    client_kwargs["base_url"] = self._base_url

                self._client = openai.AsyncOpenAI(**client_kwargs)
                self._initialized = True
                logger.info(
                    "openai_client_initialized",
                    model=self._model,
                    base_url=self._base_url,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package required for contextual retrieval. "
                    "Install with: uv add openai"
                )
        else:
            raise ValueError(f"Unsupported provider for contextual retrieval: {self._provider}")

    async def generate_context(
        self,
        chunk_content: str,
        document_context: DocumentContext,
    ) -> tuple[str, float]:
        """Generate context for a single chunk.

        Args:
            chunk_content: The chunk text to contextualize
            document_context: Document-level context (title, summary, full content)

        Returns:
            Tuple of (generated context, latency in ms)
        """
        await self._ensure_client()

        # Build document header for context
        doc_header = ""
        if document_context.title:
            doc_header += f"Document Title: {document_context.title}\n"
        if document_context.summary:
            doc_header += f"Summary: {document_context.summary}\n"

        # Truncate document content using token-aware truncation
        doc_content = truncate_to_tokens(
            document_context.full_content,
            MAX_DOCUMENT_CONTEXT_TOKENS,
        )

        full_doc = doc_header + doc_content if doc_header else doc_content

        prompt = CONTEXT_GENERATION_PROMPT.format(
            document_content=full_doc,
            chunk_content=chunk_content,
        )

        start_time = time.perf_counter()

        try:
            if self._provider == "anthropic":
                context = await self._generate_anthropic(prompt)
            else:
                context = await self._generate_openai(prompt)
        except Exception as e:
            logger.warning(
                "context_generation_failed",
                error=str(e),
                chunk_preview=chunk_content[:100],
            )
            # Return empty context on failure - graceful degradation
            context = ""

        latency_ms = (time.perf_counter() - start_time) * 1000

        return context, latency_ms

    async def _generate_anthropic(self, prompt: str) -> str:
        """Generate context using Anthropic API with optional caching."""
        messages = [{"role": "user", "content": prompt}]

        # Use prompt caching if enabled (cache the document part)
        if self._use_prompt_caching:
            # Split prompt to cache the document section
            doc_start = prompt.find("<document>")
            doc_end = prompt.find("</document>") + len("</document>")

            if doc_start != -1 and doc_end != -1:
                doc_section = prompt[doc_start:doc_end]
                rest_of_prompt = prompt[doc_end:]

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": doc_section,
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": rest_of_prompt,
                            },
                        ],
                    }
                ]

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=150,  # Context should be succinct
            messages=messages,
        )

        # Defensive check for empty response
        if not response.content:
            logger.warning("anthropic_empty_response", model=self._model)
            return ""

        return response.content[0].text.strip()

    async def _generate_openai(self, prompt: str) -> str:
        """Generate context using OpenAI API."""
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        # Defensive check for empty response
        if not response.choices or not response.choices[0].message.content:
            logger.warning("openai_empty_response", model=self._model)
            return ""

        return response.choices[0].message.content.strip()

    async def enrich_chunk(
        self,
        chunk: ChunkData,
        document_context: DocumentContext,
    ) -> EnrichedChunk:
        """Enrich a single chunk with contextual information.

        Args:
            chunk: The chunk to enrich
            document_context: Document-level context

        Returns:
            EnrichedChunk with context prepended
        """
        context, latency_ms = await self.generate_context(
            chunk.content,
            document_context,
        )

        # Build enriched content with context prepended
        enriched_parts = []
        if document_context.title:
            enriched_parts.append(f"Document: {document_context.title}")
        if context:
            enriched_parts.append(f"Context: {context}")
        enriched_parts.append(chunk.content)

        enriched_content = "\n\n".join(enriched_parts)

        return EnrichedChunk(
            original_content=chunk.content,
            enriched_content=enriched_content,
            context=context,
            chunk_index=chunk.chunk_index,
            token_count=chunk.token_count,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            context_generation_ms=latency_ms,
        )

    async def enrich_chunks(
        self,
        chunks: list[ChunkData],
        document_context: DocumentContext,
        max_concurrency: int = 5,
    ) -> list[EnrichedChunk]:
        """Enrich multiple chunks with contextual information.

        Uses asyncio.gather() for parallel processing with rate limiting
        to improve throughput while respecting API limits.

        Args:
            chunks: List of chunks to enrich
            document_context: Document-level context
            max_concurrency: Maximum concurrent API calls (default: 5)

        Returns:
            List of EnrichedChunk objects in the same order as input
        """
        if not chunks:
            return []

        start_time = time.perf_counter()

        # Use semaphore to limit concurrency and avoid overwhelming API
        semaphore = asyncio.Semaphore(max_concurrency)

        async def enrich_with_limit(chunk: ChunkData) -> EnrichedChunk:
            async with semaphore:
                return await self.enrich_chunk(chunk, document_context)

        # Process all chunks in parallel with rate limiting
        enriched = await asyncio.gather(
            *[enrich_with_limit(chunk) for chunk in chunks]
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000
        avg_context_gen_ms = (
            sum(e.context_generation_ms for e in enriched) / len(enriched)
            if enriched else 0
        )

        logger.info(
            "chunks_enriched",
            chunk_count=len(chunks),
            total_latency_ms=round(total_latency_ms, 2),
            avg_context_gen_ms=round(avg_context_gen_ms, 2),
            max_concurrency=max_concurrency,
            model=self._model,
        )

        return list(enriched)

    def get_model(self) -> str:
        """Get the model name used for context generation."""
        return self._model


def create_contextual_enricher(
    settings: "Settings",
) -> Optional[ContextualChunkEnricher]:
    """Create a contextual enricher from settings if enabled.

    Args:
        settings: Application settings

    Returns:
        ContextualChunkEnricher if enabled, None otherwise
    """
    from ..config import Settings  # Avoid circular import

    if not settings.contextual_retrieval_enabled:
        return None

    # Determine provider from model name or explicit setting
    model = settings.contextual_model
    base_url: Optional[str] = None

    # Check if OpenRouter is explicitly configured
    openrouter_key = getattr(settings, "openrouter_api_key", None)
    openrouter_base = getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")

    if openrouter_key and (
        "openrouter" in model.lower()
        or "/" in model  # OpenRouter uses "org/model" format
    ):
        provider = "openrouter"
        api_key = openrouter_key
        base_url = openrouter_base
    elif "claude" in model.lower():
        provider = "anthropic"
        api_key = settings.anthropic_api_key
    elif "gpt" in model.lower():
        provider = "openai"
        api_key = settings.openai_api_key
    else:
        # Default to anthropic for unknown models
        provider = "anthropic"
        api_key = settings.anthropic_api_key

    if not api_key:
        logger.warning(
            "contextual_enricher_no_api_key",
            provider=provider,
            model=model,
        )
        return None

    return ContextualChunkEnricher(
        model=model,
        use_prompt_caching=settings.contextual_prompt_caching,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
    )
