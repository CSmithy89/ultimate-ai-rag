"""Tests for the embedding generator module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.embeddings import (
    EmbeddingGenerator,
    cosine_similarity,
    EMBEDDING_DIMENSION,
)


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity of 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity of 0.0."""
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity of -1.0."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [0.0, 0.0, 0.0]
        assert cosine_similarity(vec1, vec2) == 0.0

    def test_dimension_mismatch_raises_error(self):
        """Vectors of different dimensions should raise ValueError."""
        vec1 = [1.0, 2.0]
        vec2 = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError):
            cosine_similarity(vec1, vec2)


class TestEmbeddingGenerator:
    """Tests for EmbeddingGenerator class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock = AsyncMock()
        return mock

    @pytest.fixture
    def generator(self, mock_openai_client):
        """Create an EmbeddingGenerator with mocked client."""
        with patch("agentic_rag_backend.embeddings.AsyncOpenAI") as mock_class:
            mock_class.return_value = mock_openai_client
            gen = EmbeddingGenerator(api_key="test-key", model="text-embedding-ada-002")
            gen.client = mock_openai_client
            return gen

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, generator, mock_openai_client):
        """Test generating embedding for a single text."""
        # Setup mock response
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1] * EMBEDDING_DIMENSION
        mock_response.data = [mock_item]
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        result = await generator.generate_embedding("Hello world")

        assert len(result) == EMBEDDING_DIMENSION
        mock_openai_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_multiple_embeddings(self, generator, mock_openai_client):
        """Test generating embeddings for multiple texts."""
        # Setup mock response
        mock_response = MagicMock()
        mock_items = []
        for _ in range(3):
            item = MagicMock()
            item.embedding = [0.1] * EMBEDDING_DIMENSION
            mock_items.append(item)
        mock_response.data = mock_items
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        texts = ["Text one", "Text two", "Text three"]
        results = await generator.generate_embeddings(texts)

        assert len(results) == 3
        assert all(len(r) == EMBEDDING_DIMENSION for r in results)

    @pytest.mark.asyncio
    async def test_generate_embeddings_records_usage(self, mock_openai_client):
        """Test cost tracking is recorded when enabled."""
        mock_response = MagicMock()
        mock_items = []
        for _ in range(2):
            item = MagicMock()
            item.embedding = [0.1] * EMBEDDING_DIMENSION
            mock_items.append(item)
        mock_response.data = mock_items
        mock_openai_client.embeddings.create = AsyncMock(return_value=mock_response)

        cost_tracker = MagicMock()
        cost_tracker.record_usage = AsyncMock()

        with patch("agentic_rag_backend.embeddings.AsyncOpenAI") as mock_class:
            mock_class.return_value = mock_openai_client
            generator = EmbeddingGenerator(
                api_key="test-key",
                model="text-embedding-ada-002",
                cost_tracker=cost_tracker,
            )
            generator.client = mock_openai_client

            await generator.generate_embeddings(
                ["Text one", "Text two"],
                tenant_id="11111111-1111-1111-1111-111111111111",
            )

        cost_tracker.record_usage.assert_called()

    @pytest.mark.asyncio
    async def test_empty_text_list(self, generator):
        """Empty text list should return empty result."""
        result = await generator.generate_embeddings([])
        assert result == []
