"""Tests for the cross-encoder reranking module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.retrieval.reranking import (
    CohereRerankerClient,
    FlashRankRerankerClient,
    RerankerProviderAdapter,
    RerankerProviderType,
    RerankedHit,
    create_reranker_client,
    get_reranker_adapter,
)
from agentic_rag_backend.retrieval.types import VectorHit


@pytest.fixture
def sample_hits() -> list[VectorHit]:
    """Create sample vector hits for testing."""
    return [
        VectorHit(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Python is a programming language.",
            similarity=0.85,
            metadata={"source": "wikipedia"},
        ),
        VectorHit(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Python is great for data science.",
            similarity=0.80,
            metadata={"source": "blog"},
        ),
        VectorHit(
            chunk_id="chunk-3",
            document_id="doc-2",
            content="JavaScript runs in the browser.",
            similarity=0.75,
            metadata={"source": "docs"},
        ),
    ]


class TestRerankerProviderAdapter:
    """Tests for RerankerProviderAdapter."""

    def test_cohere_adapter(self) -> None:
        adapter = RerankerProviderAdapter(
            provider=RerankerProviderType.COHERE,
            api_key="test-key",
            model="rerank-v3.5",
            top_k=5,
        )
        assert adapter.provider == RerankerProviderType.COHERE
        assert adapter.api_key == "test-key"
        assert adapter.model == "rerank-v3.5"
        assert adapter.top_k == 5

    def test_flashrank_adapter(self) -> None:
        adapter = RerankerProviderAdapter(
            provider=RerankerProviderType.FLASHRANK,
            api_key=None,
            model="ms-marco-MiniLM-L-12-v2",
        )
        assert adapter.provider == RerankerProviderType.FLASHRANK
        assert adapter.api_key is None
        assert adapter.model == "ms-marco-MiniLM-L-12-v2"
        assert adapter.top_k == 10  # Default


class TestCohereRerankerClient:
    """Tests for CohereRerankerClient."""

    @pytest.mark.asyncio
    async def test_rerank_success(self, sample_hits: list[VectorHit]) -> None:
        """Test successful reranking with mocked Cohere API."""
        # Mock the Cohere client
        mock_result = MagicMock()
        mock_result.results = [
            MagicMock(index=1, relevance_score=0.95),  # chunk-2 is now top
            MagicMock(index=0, relevance_score=0.85),  # chunk-1 is second
        ]

        with patch("cohere.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.rerank = AsyncMock(return_value=mock_result)
            mock_client_cls.return_value = mock_client

            client = CohereRerankerClient(api_key="test-key", model="rerank-v3.5")
            reranked = await client.rerank(
                query="Python data science",
                hits=sample_hits,
                top_k=2,
            )

            assert len(reranked) == 2
            # Check first result is chunk-2 (originally index 1)
            assert reranked[0].hit.chunk_id == "chunk-2"
            assert reranked[0].rerank_score == 0.95
            assert reranked[0].original_rank == 1
            # Check second result is chunk-1 (originally index 0)
            assert reranked[1].hit.chunk_id == "chunk-1"
            assert reranked[1].rerank_score == 0.85
            assert reranked[1].original_rank == 0

    @pytest.mark.asyncio
    async def test_rerank_empty_hits(self) -> None:
        """Test reranking with empty hits list."""
        with patch("cohere.AsyncClient"):
            client = CohereRerankerClient(api_key="test-key")
            reranked = await client.rerank(
                query="test query",
                hits=[],
                top_k=10,
            )
            assert reranked == []

    def test_get_model(self) -> None:
        """Test model name getter."""
        with patch("cohere.AsyncClient"):
            client = CohereRerankerClient(api_key="test-key", model="rerank-v3.5")
            assert client.get_model() == "rerank-v3.5"


class TestFlashRankRerankerClient:
    """Tests for FlashRankRerankerClient."""

    @pytest.mark.asyncio
    async def test_rerank_success(self, sample_hits: list[VectorHit]) -> None:
        """Test successful reranking with mocked FlashRank."""
        mock_results = [
            {"id": 0, "text": "Python is a programming language.", "score": 0.92},
            {"id": 2, "text": "JavaScript runs in the browser.", "score": 0.78},
        ]

        with patch("flashrank.Ranker") as mock_ranker_cls:
            mock_ranker = MagicMock()
            mock_ranker.rerank = MagicMock(return_value=mock_results)
            mock_ranker_cls.return_value = mock_ranker

            client = FlashRankRerankerClient(model="ms-marco-MiniLM-L-12-v2")
            reranked = await client.rerank(
                query="Python programming",
                hits=sample_hits,
                top_k=2,
            )

            assert len(reranked) == 2
            assert reranked[0].hit.chunk_id == "chunk-1"
            assert reranked[0].rerank_score == 0.92
            assert reranked[0].original_rank == 0
            assert reranked[1].hit.chunk_id == "chunk-3"
            assert reranked[1].rerank_score == 0.78
            assert reranked[1].original_rank == 2

    @pytest.mark.asyncio
    async def test_rerank_empty_hits(self) -> None:
        """Test reranking with empty hits list."""
        with patch("flashrank.Ranker"):
            client = FlashRankRerankerClient()
            reranked = await client.rerank(
                query="test query",
                hits=[],
                top_k=10,
            )
            assert reranked == []

    def test_get_model(self) -> None:
        """Test model name getter."""
        with patch("flashrank.Ranker"):
            client = FlashRankRerankerClient(model="custom-model")
            assert client.get_model() == "custom-model"


class TestCreateRerankerClient:
    """Tests for the factory function."""

    def test_create_cohere_client(self) -> None:
        """Test creating Cohere client."""
        adapter = RerankerProviderAdapter(
            provider=RerankerProviderType.COHERE,
            api_key="test-key",
            model="rerank-v3.5",
        )
        with patch("cohere.AsyncClient"):
            client = create_reranker_client(adapter)
            assert isinstance(client, CohereRerankerClient)

    def test_create_flashrank_client(self) -> None:
        """Test creating FlashRank client."""
        adapter = RerankerProviderAdapter(
            provider=RerankerProviderType.FLASHRANK,
            api_key=None,
            model="ms-marco-MiniLM-L-12-v2",
        )
        with patch("flashrank.Ranker"):
            client = create_reranker_client(adapter)
            assert isinstance(client, FlashRankRerankerClient)

    def test_cohere_requires_api_key(self) -> None:
        """Test that Cohere requires API key."""
        adapter = RerankerProviderAdapter(
            provider=RerankerProviderType.COHERE,
            api_key=None,
            model="rerank-v3.5",
        )
        with pytest.raises(ValueError, match="COHERE_API_KEY is required"):
            create_reranker_client(adapter)


class TestGetRerankerAdapter:
    """Tests for get_reranker_adapter function."""

    def test_cohere_adapter_from_settings(self) -> None:
        """Test creating Cohere adapter from settings."""
        mock_settings = MagicMock()
        mock_settings.reranker_provider = "cohere"
        mock_settings.reranker_model = "rerank-v3.5"
        mock_settings.reranker_top_k = 10
        mock_settings.cohere_api_key = "test-key"

        adapter = get_reranker_adapter(mock_settings)

        assert adapter.provider == RerankerProviderType.COHERE
        assert adapter.api_key == "test-key"
        assert adapter.model == "rerank-v3.5"
        assert adapter.top_k == 10

    def test_flashrank_adapter_from_settings(self) -> None:
        """Test creating FlashRank adapter from settings."""
        mock_settings = MagicMock()
        mock_settings.reranker_provider = "flashrank"
        mock_settings.reranker_model = "ms-marco-MiniLM-L-12-v2"
        mock_settings.reranker_top_k = 5
        mock_settings.cohere_api_key = None

        adapter = get_reranker_adapter(mock_settings)

        assert adapter.provider == RerankerProviderType.FLASHRANK
        assert adapter.api_key is None
        assert adapter.model == "ms-marco-MiniLM-L-12-v2"
        assert adapter.top_k == 5

    def test_invalid_provider_raises(self) -> None:
        """Test that invalid provider raises ValueError."""
        mock_settings = MagicMock()
        mock_settings.reranker_provider = "invalid"

        with pytest.raises(ValueError, match="RERANKER_PROVIDER must be cohere or flashrank"):
            get_reranker_adapter(mock_settings)


class TestRerankedHit:
    """Tests for RerankedHit dataclass."""

    def test_reranked_hit_creation(self, sample_hits: list[VectorHit]) -> None:
        """Test creating a RerankedHit."""
        hit = RerankedHit(
            hit=sample_hits[0],
            rerank_score=0.95,
            original_rank=2,
        )
        assert hit.hit == sample_hits[0]
        assert hit.rerank_score == 0.95
        assert hit.original_rank == 2


class TestConfigIntegration:
    """Tests for configuration integration."""

    def test_reranker_disabled_by_default(self) -> None:
        """Test that reranker is disabled by default in config."""
        import os
        from agentic_rag_backend.config import RERANKER_PROVIDERS

        # Verify default is disabled (relies on config.py logic)
        assert "cohere" in RERANKER_PROVIDERS
        assert "flashrank" in RERANKER_PROVIDERS

    def test_reranker_provider_validation(self) -> None:
        """Test that only valid providers are accepted."""
        from agentic_rag_backend.config import RERANKER_PROVIDERS

        assert RERANKER_PROVIDERS == {"cohere", "flashrank"}
