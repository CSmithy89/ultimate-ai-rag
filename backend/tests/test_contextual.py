"""Tests for contextual retrieval chunk enrichment.

Tests the ContextualChunkEnricher class and integration with the indexing pipeline.
"""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.indexing.chunker import ChunkData
from agentic_rag_backend.indexing.contextual import (
    ContextualChunkEnricher,
    DocumentContext,
    EnrichedChunk,
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

            context, latency = await enricher.generate_context(
                "Authentication uses OAuth 2.0",
                doc_context,
            )

            assert context == "This chunk discusses API authentication."
            assert latency > 0
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

            await enricher.generate_context("Test chunk", doc_context)

            call_args = mock_client.messages.create.call_args
            messages = call_args.kwargs["messages"]
            # With caching, content should be a list with cache_control
            assert isinstance(messages[0]["content"], list)
            assert len(messages[0]["content"]) == 2
            assert messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

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

            context, latency = await enricher.generate_context("Chunk text", doc_context)

            assert context == "OpenAI context"
            assert latency > 0

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

            context, latency = await enricher.generate_context("Chunk", doc_context)

            # Should return empty string on failure, not raise
            assert context == ""
            assert latency > 0

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

    @pytest.mark.asyncio
    async def test_enrich_chunks_multiple(self):
        """Test enriching multiple chunks."""
        enricher = ContextualChunkEnricher(
            model="claude-3-haiku-20240307",
            api_key="test-key",
            provider="anthropic",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated context")]

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

            enriched_chunks = await enricher.enrich_chunks(chunks, doc_context)

            assert len(enriched_chunks) == 3
            for i, enriched in enumerate(enriched_chunks):
                assert enriched.chunk_index == i
                assert enriched.context == "Generated context"

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
