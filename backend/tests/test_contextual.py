"""Tests for contextual retrieval chunk enrichment.

Tests the ContextualChunkEnricher class and integration with the indexing pipeline.
"""

import sys
from decimal import Decimal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.indexing.chunker import ChunkData
from agentic_rag_backend.indexing.contextual import (
    ContextualChunkEnricher,
    DocumentContext,
    EnrichedChunk,
    ContextualUsageStats,
    ContextualCostEstimate,
    AggregatedContextualStats,
    DEFAULT_CONTEXTUAL_PRICING,
    create_contextual_enricher,
    CONTEXT_GENERATION_PROMPT,
)


class TestDocumentContext:
    """Tests for DocumentContext dataclass."""

    def test_document_context_with_all_fields(self):
        """Test creating DocumentContext with all fields populated."""
        ctx = DocumentContext(
            title="Test Document",
            summary="A test document for unit testing",
            full_content="This is the full content of the document.",
        )
        assert ctx.title == "Test Document"
        assert ctx.summary == "A test document for unit testing"
        assert ctx.full_content == "This is the full content of the document."

    def test_document_context_with_optional_fields_none(self):
        """Test DocumentContext with optional fields as None."""
        ctx = DocumentContext(
            title=None,
            summary=None,
            full_content="Content only",
        )
        assert ctx.title is None
        assert ctx.summary is None
        assert ctx.full_content == "Content only"


class TestEnrichedChunk:
    """Tests for EnrichedChunk dataclass."""

    def test_enriched_chunk_creation(self):
        """Test creating EnrichedChunk with all required fields."""
        chunk = EnrichedChunk(
            original_content="Original text",
            enriched_content="Context: About APIs\n\nOriginal text",
            context="About APIs",
            chunk_index=0,
            token_count=50,
            start_char=0,
            end_char=100,
            context_generation_ms=150.5,
        )
        assert chunk.original_content == "Original text"
        assert chunk.enriched_content == "Context: About APIs\n\nOriginal text"
        assert chunk.context == "About APIs"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 50
        assert chunk.context_generation_ms == 150.5


class TestContextualChunkEnricher:
    """Tests for ContextualChunkEnricher class."""

    def test_enricher_init_anthropic(self):
        """Test initializing enricher with Anthropic provider."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            use_prompt_caching=True,
            api_key="test-key",
            provider="anthropic",
        )
        assert enricher._model == "claude-3-haiku-20240307"
        assert enricher._use_prompt_caching is True
        assert enricher._provider == "anthropic"
        assert enricher._initialized is False

    def test_enricher_init_openai(self):
        """Test initializing enricher with OpenAI provider."""
        enricher = ContextualChunkEnricher(
            model="gpt-4o-mini",
            use_prompt_caching=False,
            api_key="test-key",
            provider="openai",
        )
        assert enricher._model == "gpt-4o-mini"
        assert enricher._use_prompt_caching is False
        assert enricher._provider == "openai"

    def test_get_model(self):
        """Test get_model returns the configured model."""
        enricher = ContextualChunkEnricher(model="claude-3-haiku-20240307")
        assert enricher.get_model() == "claude-3-haiku-20240307"

    @pytest.mark.asyncio
    async def test_ensure_client_anthropic(self):
        """Test lazy initialization of Anthropic client."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
        )

        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            await enricher._ensure_client()

            assert enricher._initialized is True
            assert enricher._client == mock_client
            mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_ensure_client_openai(self):
        """Test lazy initialization of OpenAI client."""
        enricher = ContextualChunkEnricher(
            model="gpt-4o-mini",
            api_key="test-key",
            provider="openai",
        )

        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            await enricher._ensure_client()

            assert enricher._initialized is True
            assert enricher._client == mock_client
            mock_openai.AsyncOpenAI.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_ensure_client_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        enricher = ContextualChunkEnricher(
            model="some-model",
            api_key="test-key",
            provider="unsupported",
        )

        with pytest.raises(ValueError, match="Unsupported provider"):
            await enricher._ensure_client()

    @pytest.mark.asyncio
    async def test_generate_context_anthropic(self):
        """Test context generation with Anthropic API."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
            use_prompt_caching=False,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This chunk discusses API authentication.")]
        # Add usage stats to mock response
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 25
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0
        mock_response.usage = mock_usage

        mock_anthropic = MagicMock()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            doc_context = DocumentContext(
                title="API Guide",
                summary="Guide to REST APIs",
                full_content="REST APIs are...",
            )

            context, latency, usage_stats = await enricher.generate_context(
                "Authentication uses OAuth 2.0",
                doc_context,
            )

            assert context == "This chunk discusses API authentication."
            assert latency > 0
            assert usage_stats.input_tokens == 100
            assert usage_stats.output_tokens == 25
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_context_with_prompt_caching(self):
        """Test context generation uses prompt caching structure."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
            use_prompt_caching=True,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Cached context response")]
        # Add usage stats with cache hit
        mock_usage = MagicMock()
        mock_usage.input_tokens = 50
        mock_usage.output_tokens = 20
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 80  # Cache hit
        mock_response.usage = mock_usage

        mock_anthropic = MagicMock()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            doc_context = DocumentContext(
                title=None,
                summary=None,
                full_content="Test doc content",
            )

            context, latency, usage_stats = await enricher.generate_context("Test chunk", doc_context)

            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            # With caching, content should be a list with cache_control
            assert isinstance(messages[0]["content"], list)
            assert len(messages[0]["content"]) == 2
            assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
            # Verify cache hit was detected
            assert usage_stats.is_cache_hit is True
            assert usage_stats.cache_read_input_tokens == 80

    @pytest.mark.asyncio
    async def test_generate_context_openai(self):
        """Test context generation with OpenAI API."""
        enricher = ContextualChunkEnricher(
            model="gpt-4o-mini",
            api_key="test-key",
            provider="openai",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OpenAI context"))]
        # Add usage stats for OpenAI
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 120
        mock_usage.completion_tokens = 30
        mock_response.usage = mock_usage

        mock_openai = MagicMock()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.AsyncOpenAI.return_value = mock_client

        with patch.dict(sys.modules, {"openai": mock_openai}):
            doc_context = DocumentContext(
                title="Test Doc",
                summary=None,
                full_content="Content here",
            )

            context, latency, usage_stats = await enricher.generate_context("Chunk text", doc_context)

            assert context == "OpenAI context"
            assert latency > 0
            assert usage_stats.input_tokens == 120
            assert usage_stats.output_tokens == 30
            assert usage_stats.is_cache_hit is False  # OpenAI doesn't have cache

    @pytest.mark.asyncio
    async def test_generate_context_graceful_degradation(self):
        """Test graceful degradation when API call fails."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
        )

        mock_anthropic = MagicMock()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            doc_context = DocumentContext(
                title="Test",
                summary=None,
                full_content="Content",
            )

            context, latency, usage_stats = await enricher.generate_context("Chunk", doc_context)

            # Should return empty string on failure, not raise
            assert context == ""
            assert latency > 0
            # Usage stats should be empty on failure
            assert usage_stats.input_tokens == 0
            assert usage_stats.output_tokens == 0

    @pytest.mark.asyncio
    async def test_enrich_chunk(self):
        """Test enriching a single chunk."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This describes OAuth flow")]
        mock_usage = MagicMock()
        mock_usage.input_tokens = 150
        mock_usage.output_tokens = 20
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0
        mock_response.usage = mock_usage

        mock_anthropic = MagicMock()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            chunk = ChunkData(
                content="OAuth 2.0 uses tokens for authentication.",
                chunk_index=0,
                token_count=10,
                start_char=0,
                end_char=50,
            )

            doc_context = DocumentContext(
                title="Auth Guide",
                summary="Authentication documentation",
                full_content="Full document content...",
            )

            enriched = await enricher.enrich_chunk(chunk, doc_context)

            assert isinstance(enriched, EnrichedChunk)
            assert enriched.original_content == chunk.content
            assert "Auth Guide" in enriched.enriched_content
            assert "This describes OAuth flow" in enriched.enriched_content
            assert enriched.context == "This describes OAuth flow"
            assert enriched.chunk_index == 0
            # Verify usage stats are included
            assert enriched.usage_stats is not None
            assert enriched.usage_stats.input_tokens == 150
            assert enriched.usage_stats.output_tokens == 20

    @pytest.mark.asyncio
    async def test_enrich_chunks_multiple(self):
        """Test enriching multiple chunks with aggregated stats."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated context")]
        # Add usage stats per call
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 25
        mock_usage.cache_creation_input_tokens = 50
        mock_usage.cache_read_input_tokens = 0  # First call is cache miss
        mock_response.usage = mock_usage

        mock_anthropic = MagicMock()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            chunks = [
                ChunkData(
                    content=f"Chunk {i} content",
                    chunk_index=i,
                    token_count=10,
                    start_char=i * 50,
                    end_char=(i + 1) * 50,
                )
                for i in range(3)
            ]

            doc_context = DocumentContext(
                title="Test Doc",
                summary=None,
                full_content="Full content",
            )

            enriched_chunks, aggregated_stats = await enricher.enrich_chunks(
                chunks, doc_context, tenant_id="test-tenant"
            )

            assert len(enriched_chunks) == 3
            for i, enriched in enumerate(enriched_chunks):
                assert enriched.chunk_index == i
                assert enriched.context == "Generated context"

            # Verify aggregated stats
            assert aggregated_stats.chunks_enriched == 3
            assert aggregated_stats.model == "claude-3-haiku-20240307"
            assert aggregated_stats.input_tokens == 300  # 100 * 3
            assert aggregated_stats.output_tokens == 75  # 25 * 3
            assert aggregated_stats.cache_creation_input_tokens == 150  # 50 * 3

    @pytest.mark.asyncio
    async def test_document_content_truncation(self):
        """Test that long document content is truncated using token-aware truncation."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
            use_prompt_caching=False,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Context")]
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 10
        mock_usage.cache_creation_input_tokens = 0
        mock_usage.cache_read_input_tokens = 0
        mock_response.usage = mock_usage

        mock_anthropic = MagicMock()
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.AsyncAnthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            # Create very long document content with unique words (each word = ~1 token)
            # MAX_DOCUMENT_CONTEXT_TOKENS is 2000, so 3000 unique words should trigger truncation
            long_content = " ".join(f"word{i}" for i in range(3000))

            doc_context = DocumentContext(
                title="Long Doc",
                summary=None,
                full_content=long_content,
            )

            await enricher.generate_context("Chunk", doc_context)

            # Verify the prompt was called with truncated content
            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            prompt_text = messages[0]["content"]
            assert "... [truncated]" in prompt_text


class TestCreateContextualEnricher:
    """Tests for the create_contextual_enricher factory function."""

    def test_create_enricher_disabled(self):
        """Test that None is returned when feature is disabled."""
        mock_settings = MagicMock()
        mock_settings.contextual_retrieval_enabled = False

        result = create_contextual_enricher(mock_settings)
        assert result is None

    def test_create_enricher_anthropic_model(self):
        """Test creating enricher with Claude model."""
        mock_settings = MagicMock()
        mock_settings.contextual_retrieval_enabled = True
        mock_settings.contextual_model = "claude-3-haiku-20240307"
        mock_settings.contextual_prompt_caching = True
        mock_settings.anthropic_api_key = "sk-ant-test"
        mock_settings.openai_api_key = None

        result = create_contextual_enricher(mock_settings)

        assert result is not None
        assert result._model == "claude-3-haiku-20240307"
        assert result._provider == "anthropic"
        assert result._use_prompt_caching is True

    def test_create_enricher_openai_model(self):
        """Test creating enricher with GPT model."""
        mock_settings = MagicMock()
        mock_settings.contextual_retrieval_enabled = True
        mock_settings.contextual_model = "gpt-4o-mini"
        mock_settings.contextual_prompt_caching = False
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = "sk-openai-test"

        result = create_contextual_enricher(mock_settings)

        assert result is not None
        assert result._model == "gpt-4o-mini"
        assert result._provider == "openai"

    def test_create_enricher_no_api_key(self):
        """Test that None is returned when API key is missing."""
        mock_settings = MagicMock()
        mock_settings.contextual_retrieval_enabled = True
        mock_settings.contextual_model = "claude-3-haiku-20240307"
        mock_settings.anthropic_api_key = None
        mock_settings.openai_api_key = None

        result = create_contextual_enricher(mock_settings)
        assert result is None


class TestContextGenerationPrompt:
    """Tests for the context generation prompt template."""

    def test_prompt_template_format(self):
        """Test that the prompt template can be formatted correctly."""
        prompt = CONTEXT_GENERATION_PROMPT.format(
            document_content="This is the document",
            chunk_content="This is a chunk",
        )

        assert "<document>" in prompt
        assert "This is the document" in prompt
        assert "</document>" in prompt
        assert "<chunk>" in prompt
        assert "This is a chunk" in prompt
        assert "</chunk>" in prompt
        assert "situate this chunk" in prompt

    def test_prompt_has_required_sections(self):
        """Test that prompt has all required sections."""
        assert "<document>" in CONTEXT_GENERATION_PROMPT
        assert "</document>" in CONTEXT_GENERATION_PROMPT
        assert "<chunk>" in CONTEXT_GENERATION_PROMPT
        assert "</chunk>" in CONTEXT_GENERATION_PROMPT
        assert "{document_content}" in CONTEXT_GENERATION_PROMPT
        assert "{chunk_content}" in CONTEXT_GENERATION_PROMPT


# =============================================================================
# Story 19-F5: Contextual Retrieval Cost Logging Tests
# =============================================================================


class TestContextualUsageStats:
    """Tests for ContextualUsageStats dataclass."""

    def test_usage_stats_defaults(self):
        """Test default values for usage stats."""
        stats = ContextualUsageStats()
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.cache_creation_input_tokens == 0
        assert stats.cache_read_input_tokens == 0
        assert stats.is_cache_hit is False

    def test_usage_stats_total_tokens(self):
        """Test total tokens calculation."""
        stats = ContextualUsageStats(
            input_tokens=100,
            output_tokens=25,
        )
        assert stats.total_tokens == 125

    def test_usage_stats_cache_hit_detection(self):
        """Test cache hit detection based on cache read tokens."""
        # Cache miss (no cache read tokens)
        stats_miss = ContextualUsageStats(
            input_tokens=100,
            output_tokens=25,
            cache_read_input_tokens=0,
        )
        assert stats_miss.is_cache_hit is False

        # Cache hit (has cache read tokens)
        stats_hit = ContextualUsageStats(
            input_tokens=100,
            output_tokens=25,
            cache_read_input_tokens=80,
            is_cache_hit=True,
        )
        assert stats_hit.is_cache_hit is True


class TestContextualCostEstimate:
    """Tests for ContextualCostEstimate dataclass."""

    def test_cost_estimate_defaults(self):
        """Test default values for cost estimate."""
        cost = ContextualCostEstimate()
        assert cost.input_cost_usd == Decimal("0")
        assert cost.output_cost_usd == Decimal("0")
        assert cost.cache_write_cost_usd == Decimal("0")
        assert cost.cache_read_cost_usd == Decimal("0")
        assert cost.total_cost_usd == Decimal("0")

    def test_cost_estimate_total_calculation(self):
        """Test total cost calculation."""
        cost = ContextualCostEstimate(
            input_cost_usd=Decimal("0.002"),
            output_cost_usd=Decimal("0.001"),
            cache_write_cost_usd=Decimal("0.0003"),
            cache_read_cost_usd=Decimal("0.00005"),
        )
        assert cost.total_cost_usd == Decimal("0.00335")


class TestAggregatedContextualStats:
    """Tests for AggregatedContextualStats dataclass."""

    def test_aggregated_stats_defaults(self):
        """Test default values for aggregated stats."""
        stats = AggregatedContextualStats()
        assert stats.chunks_enriched == 0
        assert stats.model == ""
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.estimated_cost_usd == Decimal("0")

    def test_aggregated_stats_total_tokens(self):
        """Test total tokens calculation."""
        stats = AggregatedContextualStats(
            input_tokens=1000,
            output_tokens=250,
        )
        assert stats.total_tokens == 1250

    def test_aggregated_stats_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        # No cache operations
        stats_empty = AggregatedContextualStats()
        assert stats_empty.cache_hit_rate == 0.0

        # 50% hit rate
        stats_50 = AggregatedContextualStats(
            cache_hits=5,
            cache_misses=5,
        )
        assert stats_50.cache_hit_rate == 0.5

        # 100% hit rate
        stats_100 = AggregatedContextualStats(
            cache_hits=10,
            cache_misses=0,
        )
        assert stats_100.cache_hit_rate == 1.0


class TestCostCalculation:
    """Tests for cost calculation functionality."""

    def test_cost_calculation_with_known_model(self):
        """Test cost calculation with a model in the pricing table."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
        )

        stats = AggregatedContextualStats(
            chunks_enriched=10,
            model="claude-3-haiku-20240307",
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=300,
        )

        cost = enricher._calculate_cost(stats)

        # Verify cost breakdown using known pricing
        # claude-3-haiku: input=$0.00025/1k, output=$0.00125/1k,
        # cache_write=$0.0003/1k, cache_read=$0.00003/1k
        expected_input = Decimal("1000") / Decimal("1000") * Decimal("0.00025")
        expected_output = Decimal("200") / Decimal("1000") * Decimal("0.00125")
        expected_cache_write = Decimal("500") / Decimal("1000") * Decimal("0.0003")
        expected_cache_read = Decimal("300") / Decimal("1000") * Decimal("0.00003")

        assert cost.input_cost_usd == expected_input
        assert cost.output_cost_usd == expected_output
        assert cost.cache_write_cost_usd == expected_cache_write
        assert cost.cache_read_cost_usd == expected_cache_read

    def test_cost_calculation_with_unknown_model(self):
        """Test cost calculation with a model not in the pricing table."""
        enricher = ContextualChunkEnricher(
            model="unknown-model-xyz",
            api_key="test-key",
            provider="anthropic",
        )

        stats = AggregatedContextualStats(
            chunks_enriched=5,
            model="unknown-model-xyz",
            input_tokens=500,
            output_tokens=100,
        )

        # Should not raise, uses fallback pricing
        cost = enricher._calculate_cost(stats)
        assert cost.total_cost_usd > Decimal("0")


class TestDefaultContextualPricing:
    """Tests for default pricing configuration."""

    def test_haiku_pricing_exists(self):
        """Test that Claude 3 Haiku pricing is configured."""
        assert "claude-3-haiku-20240307" in DEFAULT_CONTEXTUAL_PRICING
        pricing = DEFAULT_CONTEXTUAL_PRICING["claude-3-haiku-20240307"]
        assert "input_per_1k" in pricing
        assert "output_per_1k" in pricing
        assert "cache_write_per_1k" in pricing
        assert "cache_read_per_1k" in pricing

    def test_gpt4o_mini_pricing_exists(self):
        """Test that GPT-4o-mini pricing is configured."""
        assert "gpt-4o-mini" in DEFAULT_CONTEXTUAL_PRICING
        pricing = DEFAULT_CONTEXTUAL_PRICING["gpt-4o-mini"]
        assert pricing["input_per_1k"] > Decimal("0")
        assert pricing["output_per_1k"] > Decimal("0")

    def test_cache_pricing_for_openai(self):
        """Test that OpenAI models have zero cache pricing (not supported)."""
        pricing = DEFAULT_CONTEXTUAL_PRICING["gpt-4o-mini"]
        assert pricing["cache_write_per_1k"] == Decimal("0")
        assert pricing["cache_read_per_1k"] == Decimal("0")
