"""Contextual retrieval chunk enrichment.

Implements Anthropic's contextual retrieval approach where each chunk is
enriched with document context (title, summary) before embedding. This
improves retrieval accuracy by 35-67% by preserving document context.

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional, TYPE_CHECKING

import structlog
import tiktoken

from .chunker import ChunkData

if TYPE_CHECKING:
    from ..config import Settings

logger = structlog.get_logger(__name__)


# =============================================================================
# Model Pricing Configuration (USD per 1K tokens)
# =============================================================================
# Default pricing for contextual retrieval models. Can be overridden via config.
# Anthropic pricing includes cache read/write costs for prompt caching.
DEFAULT_CONTEXTUAL_PRICING: dict[str, dict[str, Decimal]] = {
    # Anthropic Claude 3 Haiku (default for contextual retrieval)
    "claude-3-haiku-20240307": {
        "input_per_1k": Decimal("0.00025"),
        "output_per_1k": Decimal("0.00125"),
        "cache_write_per_1k": Decimal("0.0003"),  # 20% premium for cache write
        "cache_read_per_1k": Decimal("0.00003"),  # 90% discount for cache read
    },
    # Anthropic Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {
        "input_per_1k": Decimal("0.001"),
        "output_per_1k": Decimal("0.005"),
        "cache_write_per_1k": Decimal("0.00125"),
        "cache_read_per_1k": Decimal("0.0001"),
    },
    # OpenAI GPT-4o-mini
    "gpt-4o-mini": {
        "input_per_1k": Decimal("0.00015"),
        "output_per_1k": Decimal("0.0006"),
        "cache_write_per_1k": Decimal("0"),  # No cache pricing for OpenAI
        "cache_read_per_1k": Decimal("0"),
    },
    # OpenAI GPT-4o
    "gpt-4o": {
        "input_per_1k": Decimal("0.0025"),
        "output_per_1k": Decimal("0.01"),
        "cache_write_per_1k": Decimal("0"),
        "cache_read_per_1k": Decimal("0"),
    },
}

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
class ContextualUsageStats:
    """Token usage and cost statistics for a single context generation call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0  # Anthropic: tokens written to cache
    cache_read_input_tokens: int = 0  # Anthropic: tokens read from cache
    is_cache_hit: bool = False  # True if cache was used for this request

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens


@dataclass
class ContextualCostEstimate:
    """Cost estimate for contextual retrieval operations."""

    input_cost_usd: Decimal = Decimal("0")
    output_cost_usd: Decimal = Decimal("0")
    cache_write_cost_usd: Decimal = Decimal("0")
    cache_read_cost_usd: Decimal = Decimal("0")

    @property
    def total_cost_usd(self) -> Decimal:
        """Total estimated cost in USD."""
        return (
            self.input_cost_usd
            + self.output_cost_usd
            + self.cache_write_cost_usd
            + self.cache_read_cost_usd
        )


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
    usage_stats: Optional[ContextualUsageStats] = None


@dataclass
class AggregatedContextualStats:
    """Aggregated statistics for a batch of contextual enrichment calls."""

    chunks_enriched: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    estimated_cost_usd: Decimal = Decimal("0")
    total_latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as a percentage (0.0 to 1.0)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


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
        self._client: Any | None = None
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

                api_key = self._api_key or ""
                if self._base_url:
                    self._client = openai.AsyncOpenAI(api_key=api_key, base_url=self._base_url)
                else:
                    self._client = openai.AsyncOpenAI(api_key=api_key)
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
    ) -> tuple[str, float, ContextualUsageStats]:
        """Generate context for a single chunk.

        Args:
            chunk_content: The chunk text to contextualize
            document_context: Document-level context (title, summary, full content)

        Returns:
            Tuple of (generated context, latency in ms, usage stats)
        """
        await self._ensure_client()
        client = self._client
        if client is None:
            raise RuntimeError("Contextual enricher client not initialized.")

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
        usage_stats = ContextualUsageStats()

        try:
            if self._provider == "anthropic":
                context, usage_stats = await self._generate_anthropic(prompt)
            else:
                context, usage_stats = await self._generate_openai(prompt)
        except Exception as e:
            logger.warning(
                "context_generation_failed",
                error=str(e),
                chunk_preview=chunk_content[:100],
            )
            # Return empty context on failure - graceful degradation
            context = ""

        latency_ms = (time.perf_counter() - start_time) * 1000

        return context, latency_ms, usage_stats

    async def _generate_anthropic(self, prompt: str) -> tuple[str, ContextualUsageStats]:
        """Generate context using Anthropic API with optional caching.

        Returns:
            Tuple of (generated context, usage stats with cache info)
        """
        client = self._client
        if client is None:
            raise RuntimeError("Contextual enricher client not initialized.")
        messages: list[dict[str, object]] = [{"role": "user", "content": prompt}]

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

        response = await client.messages.create(
            model=self._model,
            max_tokens=150,  # Context should be succinct
            messages=messages,
        )

        # Extract usage statistics from response
        usage = getattr(response, "usage", None)
        usage_stats = ContextualUsageStats()

        if usage:
            usage_stats.input_tokens = getattr(usage, "input_tokens", 0)
            usage_stats.output_tokens = getattr(usage, "output_tokens", 0)
            # Anthropic cache-specific fields
            usage_stats.cache_creation_input_tokens = getattr(
                usage, "cache_creation_input_tokens", 0
            )
            usage_stats.cache_read_input_tokens = getattr(
                usage, "cache_read_input_tokens", 0
            )
            # Consider it a cache hit if any tokens were read from cache
            usage_stats.is_cache_hit = usage_stats.cache_read_input_tokens > 0

        # Defensive check for empty response
        if not response.content:
            logger.warning("anthropic_empty_response", model=self._model)
            return "", usage_stats

        return response.content[0].text.strip(), usage_stats

    async def _generate_openai(self, prompt: str) -> tuple[str, ContextualUsageStats]:
        """Generate context using OpenAI API.

        Returns:
            Tuple of (generated context, usage stats)
        """
        client = self._client
        if client is None:
            raise RuntimeError("Contextual enricher client not initialized.")
        response = await client.chat.completions.create(
            model=self._model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract usage statistics from response
        usage = getattr(response, "usage", None)
        usage_stats = ContextualUsageStats()

        if usage:
            usage_stats.input_tokens = getattr(usage, "prompt_tokens", 0)
            usage_stats.output_tokens = getattr(usage, "completion_tokens", 0)
            # OpenAI doesn't have cache fields, but set them to 0 for consistency
            usage_stats.cache_creation_input_tokens = 0
            usage_stats.cache_read_input_tokens = 0
            usage_stats.is_cache_hit = False

        # Defensive check for empty response
        if not response.choices or not response.choices[0].message.content:
            logger.warning("openai_empty_response", model=self._model)
            return "", usage_stats

        return response.choices[0].message.content.strip(), usage_stats

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
            EnrichedChunk with context prepended and usage stats
        """
        context, latency_ms, usage_stats = await self.generate_context(
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
            usage_stats=usage_stats,
        )

    async def enrich_chunks(
        self,
        chunks: list[ChunkData],
        document_context: DocumentContext,
        max_concurrency: int = 5,
        tenant_id: Optional[str] = None,
    ) -> tuple[list[EnrichedChunk], AggregatedContextualStats]:
        """Enrich multiple chunks with contextual information.

        Uses asyncio.gather() for parallel processing with rate limiting
        to improve throughput while respecting API limits.

        Args:
            chunks: List of chunks to enrich
            document_context: Document-level context
            max_concurrency: Maximum concurrent API calls (default: 5)
            tenant_id: Optional tenant identifier for multi-tenant metrics

        Returns:
            Tuple of (list of EnrichedChunk, aggregated stats)
        """
        if not chunks:
            return [], AggregatedContextualStats(model=self._model)

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

        # Aggregate usage statistics across all chunks
        aggregated = self._aggregate_stats(list(enriched), total_latency_ms)

        # Calculate cost estimate
        cost_estimate = self._calculate_cost(aggregated)

        # Log with the required format for contextual_retrieval
        logger.info(
            "contextual_retrieval",
            chunks_enriched=aggregated.chunks_enriched,
            model=aggregated.model,
            input_tokens=aggregated.input_tokens,
            output_tokens=aggregated.output_tokens,
            estimated_cost_usd=float(cost_estimate.total_cost_usd),
            cache_hits=aggregated.cache_hits,
            cache_misses=aggregated.cache_misses,
            cache_hit_rate=round(aggregated.cache_hit_rate, 3),
            total_latency_ms=round(total_latency_ms, 2),
            tenant_id=tenant_id,
        )

        # Emit Prometheus metrics if available
        self._emit_prometheus_metrics(aggregated, cost_estimate, tenant_id)

        return list(enriched), aggregated

    def _aggregate_stats(
        self,
        enriched_chunks: list[EnrichedChunk],
        total_latency_ms: float,
    ) -> AggregatedContextualStats:
        """Aggregate usage statistics from enriched chunks.

        Args:
            enriched_chunks: List of enriched chunks with usage stats
            total_latency_ms: Total latency for the batch

        Returns:
            Aggregated statistics for the batch
        """
        stats = AggregatedContextualStats(
            chunks_enriched=len(enriched_chunks),
            model=self._model,
            total_latency_ms=total_latency_ms,
        )

        for chunk in enriched_chunks:
            if chunk.usage_stats:
                stats.input_tokens += chunk.usage_stats.input_tokens
                stats.output_tokens += chunk.usage_stats.output_tokens
                stats.cache_creation_input_tokens += chunk.usage_stats.cache_creation_input_tokens
                stats.cache_read_input_tokens += chunk.usage_stats.cache_read_input_tokens

                if chunk.usage_stats.is_cache_hit:
                    stats.cache_hits += 1
                else:
                    stats.cache_misses += 1

        return stats

    def _calculate_cost(
        self,
        stats: AggregatedContextualStats,
    ) -> ContextualCostEstimate:
        """Calculate cost estimate for aggregated stats.

        Args:
            stats: Aggregated usage statistics

        Returns:
            Cost estimate breakdown
        """
        pricing = DEFAULT_CONTEXTUAL_PRICING.get(self._model)
        if not pricing:
            # Use a generic fallback pricing if model not found
            logger.debug(
                "contextual_pricing_not_found",
                model=self._model,
                using_fallback=True,
            )
            pricing = {
                "input_per_1k": Decimal("0.001"),
                "output_per_1k": Decimal("0.002"),
                "cache_write_per_1k": Decimal("0"),
                "cache_read_per_1k": Decimal("0"),
            }

        # Calculate costs
        input_cost = (Decimal(stats.input_tokens) / Decimal(1000)) * pricing["input_per_1k"]
        output_cost = (Decimal(stats.output_tokens) / Decimal(1000)) * pricing["output_per_1k"]
        cache_write_cost = (
            Decimal(stats.cache_creation_input_tokens) / Decimal(1000)
        ) * pricing["cache_write_per_1k"]
        cache_read_cost = (
            Decimal(stats.cache_read_input_tokens) / Decimal(1000)
        ) * pricing["cache_read_per_1k"]

        estimate = ContextualCostEstimate(
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            cache_write_cost_usd=cache_write_cost,
            cache_read_cost_usd=cache_read_cost,
        )

        # Update the aggregated stats with the total cost
        stats.estimated_cost_usd = estimate.total_cost_usd

        return estimate

    def _emit_prometheus_metrics(
        self,
        stats: AggregatedContextualStats,
        cost: ContextualCostEstimate,
        tenant_id: Optional[str],
    ) -> None:
        """Emit Prometheus metrics for contextual retrieval.

        Args:
            stats: Aggregated usage statistics
            cost: Cost estimate
            tenant_id: Optional tenant identifier
        """
        try:
            from ..observability.metrics import (
                CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL,
                CONTEXTUAL_ENRICHMENT_COST_USD_TOTAL,
                CONTEXTUAL_ENRICHMENT_CACHE_HITS_TOTAL,
                CONTEXTUAL_ENRICHMENT_CACHE_MISSES_TOTAL,
                CONTEXTUAL_ENRICHMENT_CHUNKS_TOTAL,
            )

            tenant = tenant_id or "default"

            # Token counters
            CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL.labels(
                type="input",
                model=self._model,
                tenant_id=tenant,
            ).inc(stats.input_tokens)

            CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL.labels(
                type="output",
                model=self._model,
                tenant_id=tenant,
            ).inc(stats.output_tokens)

            # Cost counter
            CONTEXTUAL_ENRICHMENT_COST_USD_TOTAL.labels(
                model=self._model,
                tenant_id=tenant,
            ).inc(float(cost.total_cost_usd))

            # Cache counters
            CONTEXTUAL_ENRICHMENT_CACHE_HITS_TOTAL.labels(
                model=self._model,
                tenant_id=tenant,
            ).inc(stats.cache_hits)

            CONTEXTUAL_ENRICHMENT_CACHE_MISSES_TOTAL.labels(
                model=self._model,
                tenant_id=tenant,
            ).inc(stats.cache_misses)

            # Chunks counter
            CONTEXTUAL_ENRICHMENT_CHUNKS_TOTAL.labels(
                model=self._model,
                tenant_id=tenant,
            ).inc(stats.chunks_enriched)

        except ImportError:
            # Prometheus metrics not available, skip silently
            pass
        except Exception as e:
            logger.debug(
                "contextual_prometheus_metrics_error",
                error=str(e),
            )

    def get_model(self) -> str:
        """Get the model name used for context generation."""
        return self._model


def create_contextual_enricher(
    settings: Settings,
) -> Optional[ContextualChunkEnricher]:
    """Create a contextual enricher from settings if enabled.

    Args:
        settings: Application settings

    Returns:
        ContextualChunkEnricher if enabled, None otherwise
    """
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
