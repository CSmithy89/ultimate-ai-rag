"""Unit tests for ColBERT reranking module (Story 20-H5).

Tests cover:
- ColBERTResult and TokenEmbeddings dataclasses
- ColBERTEncoder with mocked models
- MaxSimScorer score computation
- ColBERTReranker feature flag behavior
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_rag_backend.retrieval import (
    ColBERTEncoder,
    ColBERTReranker,
    ColBERTResult,
    DEFAULT_COLBERT_ENABLED,
    DEFAULT_COLBERT_MAX_LENGTH,
    DEFAULT_COLBERT_MODEL,
    MaxSimScorer,
    TokenEmbeddings,
    create_colbert_reranker,
)


# ============================================================================
# TokenEmbeddings Tests
# ============================================================================


class TestTokenEmbeddings:
    """Tests for TokenEmbeddings dataclass."""

    def test_create_embeddings(self):
        """Test creating token embeddings."""
        embeddings = TokenEmbeddings(
            tokens=["hello", "world"],
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        )
        assert embeddings.tokens == ["hello", "world"]
        assert len(embeddings.embeddings) == 2

    def test_num_tokens_property(self):
        """Test num_tokens property."""
        embeddings = TokenEmbeddings(
            tokens=["a", "b", "c"],
            embeddings=[[0.1], [0.2], [0.3]],
        )
        assert embeddings.num_tokens == 3

    def test_empty_embeddings(self):
        """Test empty embeddings."""
        embeddings = TokenEmbeddings()
        assert embeddings.tokens == []
        assert embeddings.embeddings == []
        assert embeddings.num_tokens == 0


# ============================================================================
# ColBERTResult Tests
# ============================================================================


class TestColBERTResult:
    """Tests for ColBERTResult dataclass."""

    def test_create_result(self):
        """Test creating a ColBERT result."""
        result = ColBERTResult(
            doc_id="doc-1",
            content="Some document content",
            score=0.85,
            original_score=0.7,
            original_rank=2,
        )
        assert result.doc_id == "doc-1"
        assert result.content == "Some document content"
        assert result.score == 0.85
        assert result.original_score == 0.7
        assert result.original_rank == 2

    def test_result_with_metadata(self):
        """Test result with metadata."""
        result = ColBERTResult(
            doc_id="doc-2",
            content="Content",
            score=0.9,
            metadata={"source": "test", "page": 1},
        )
        assert result.metadata["source"] == "test"
        assert result.metadata["page"] == 1

    def test_result_defaults(self):
        """Test result default values."""
        result = ColBERTResult(
            doc_id="doc-3",
            content="Content",
            score=0.5,
        )
        assert result.original_score == 0.0
        assert result.original_rank == 0
        assert result.metadata == {}


# ============================================================================
# MaxSimScorer Tests
# ============================================================================


class TestMaxSimScorer:
    """Tests for MaxSimScorer class."""

    def test_compute_score_basic(self):
        """Test basic MaxSim score computation."""
        query_emb = TokenEmbeddings(
            tokens=["query"],
            embeddings=[[1.0, 0.0, 0.0]],
        )
        doc_emb = TokenEmbeddings(
            tokens=["doc"],
            embeddings=[[1.0, 0.0, 0.0]],  # Perfect match
        )

        score = MaxSimScorer.compute_score(query_emb, doc_emb)
        # Perfect cosine similarity should be 1.0
        assert score > 0.99

    def test_compute_score_multiple_tokens(self):
        """Test MaxSim with multiple query tokens."""
        # Query with 2 tokens
        query_emb = TokenEmbeddings(
            tokens=["hello", "world"],
            embeddings=[
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        )
        # Document with matching tokens
        doc_emb = TokenEmbeddings(
            tokens=["world", "hello"],
            embeddings=[
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        )

        score = MaxSimScorer.compute_score(query_emb, doc_emb)
        # Both tokens should find perfect matches, score ~= 2.0
        assert score > 1.9

    def test_compute_score_no_match(self):
        """Test MaxSim with orthogonal vectors."""
        query_emb = TokenEmbeddings(
            tokens=["q"],
            embeddings=[[1.0, 0.0, 0.0]],
        )
        doc_emb = TokenEmbeddings(
            tokens=["d"],
            embeddings=[[0.0, 1.0, 0.0]],  # Orthogonal
        )

        score = MaxSimScorer.compute_score(query_emb, doc_emb)
        # Orthogonal vectors have zero cosine similarity
        assert abs(score) < 0.01

    def test_compute_score_empty_embeddings(self):
        """Test MaxSim with empty embeddings."""
        empty = TokenEmbeddings()
        non_empty = TokenEmbeddings(
            tokens=["a"],
            embeddings=[[1.0, 0.0]],
        )

        score1 = MaxSimScorer.compute_score(empty, non_empty)
        score2 = MaxSimScorer.compute_score(non_empty, empty)
        score3 = MaxSimScorer.compute_score(empty, empty)

        assert score1 == 0.0
        assert score2 == 0.0
        assert score3 == 0.0

    def test_compute_scores_batch(self):
        """Test batch score computation."""
        query_emb = TokenEmbeddings(
            tokens=["query"],
            embeddings=[[1.0, 0.0, 0.0]],
        )
        doc_embs = [
            TokenEmbeddings(tokens=["a"], embeddings=[[1.0, 0.0, 0.0]]),  # Match
            TokenEmbeddings(tokens=["b"], embeddings=[[0.0, 1.0, 0.0]]),  # No match
            TokenEmbeddings(tokens=["c"], embeddings=[[0.7, 0.7, 0.0]]),  # Partial
        ]

        scores = MaxSimScorer.compute_scores_batch(query_emb, doc_embs)

        assert len(scores) == 3
        assert scores[0] > 0.9  # High similarity
        assert scores[1] < 0.1  # Low similarity
        assert 0.5 < scores[2] < 0.9  # Medium similarity


# ============================================================================
# ColBERTEncoder Tests
# ============================================================================


class TestColBERTEncoder:
    """Tests for ColBERTEncoder class."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ColBERTEncoder(
            model_name="test-model",
            max_length=256,
        )
        assert encoder._model_name == "test-model"
        assert encoder._max_length == 256
        assert encoder._model is None

    def test_default_initialization(self):
        """Test encoder with default values."""
        encoder = ColBERTEncoder()
        assert encoder._model_name == DEFAULT_COLBERT_MODEL
        assert encoder._max_length == DEFAULT_COLBERT_MAX_LENGTH

    @pytest.mark.asyncio
    async def test_encode_query_with_mock_transformers(self):
        """Test query encoding with mocked transformers."""
        encoder = ColBERTEncoder()

        # Create mock outputs
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        # Return a 2D tensor-like object
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value.numpy.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        mock_output.last_hidden_state.__getitem__ = lambda self, idx: mock_tensor

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(__getitem__=lambda self, idx: MagicMock(tolist=lambda: [1, 2])),
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = ["hello", "world"]

        # Mock model
        mock_model = MagicMock()
        mock_model.return_value = mock_output

        encoder._tokenizer = mock_tokenizer
        encoder._model = mock_model

        result = await encoder.encode_query("Hello world")

        assert isinstance(result, TokenEmbeddings)
        assert result.tokens == ["hello", "world"]
        assert len(result.embeddings) == 2

    @pytest.mark.asyncio
    async def test_encode_document_with_mock_sentence_transformers(self):
        """Test document encoding with sentence-transformers fallback."""
        encoder = ColBERTEncoder()

        # Mock sentence-transformers model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

        encoder._model = mock_model
        encoder._tokenizer = None  # Triggers sentence-transformers path

        result = await encoder.encode_document("Test document")

        assert isinstance(result, TokenEmbeddings)
        assert len(result.embeddings) == 2

    @pytest.mark.asyncio
    async def test_encode_documents_batch_empty(self):
        """Test batch encoding with empty list."""
        encoder = ColBERTEncoder()
        result = await encoder.encode_documents_batch([])
        assert result == []


# ============================================================================
# ColBERTReranker Tests
# ============================================================================


class TestColBERTReranker:
    """Tests for ColBERTReranker class."""

    def test_reranker_disabled_by_default(self):
        """Test reranker is disabled by default."""
        reranker = ColBERTReranker()
        assert not reranker.enabled

    def test_reranker_enabled(self):
        """Test reranker can be enabled."""
        reranker = ColBERTReranker(enabled=True)
        assert reranker.enabled

    def test_reranker_config(self):
        """Test reranker configuration."""
        reranker = ColBERTReranker(
            enabled=True,
            model_name="custom-model",
            max_length=256,
            top_k=5,
        )
        assert reranker.enabled
        assert reranker._model_name == "custom-model"
        assert reranker._max_length == 256
        assert reranker._top_k == 5

    @pytest.mark.asyncio
    async def test_rerank_when_disabled(self):
        """Test rerank passes through when disabled."""
        reranker = ColBERTReranker(enabled=False)

        documents = [
            {"id": "doc-1", "content": "First document", "score": 0.9},
            {"id": "doc-2", "content": "Second document", "score": 0.8},
        ]

        results = await reranker.rerank("test query", documents)

        assert len(results) == 2
        assert results[0].doc_id == "doc-1"
        assert results[0].score == 0.9  # Original score preserved
        assert results[1].doc_id == "doc-2"

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        """Test rerank with empty document list."""
        reranker = ColBERTReranker(enabled=True)
        results = await reranker.rerank("test query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_with_mock_encoder(self):
        """Test reranking with mocked encoder."""
        reranker = ColBERTReranker(enabled=True, top_k=2)

        # Create mock encoder
        mock_encoder = MagicMock()

        # Query embeddings
        query_emb = TokenEmbeddings(
            tokens=["query"],
            embeddings=[[1.0, 0.0, 0.0]],
        )

        # Document embeddings - doc2 matches better
        doc_embs = [
            TokenEmbeddings(tokens=["d1"], embeddings=[[0.0, 1.0, 0.0]]),  # Low score
            TokenEmbeddings(tokens=["d2"], embeddings=[[0.9, 0.1, 0.0]]),  # High score
            TokenEmbeddings(tokens=["d3"], embeddings=[[0.5, 0.5, 0.0]]),  # Medium score
        ]

        async def mock_encode_query(q):
            return query_emb

        async def mock_encode_batch(docs):
            return doc_embs

        mock_encoder.encode_query = mock_encode_query
        mock_encoder.encode_documents_batch = mock_encode_batch

        reranker._encoder = mock_encoder

        documents = [
            {"id": "doc-1", "content": "First", "score": 0.5},
            {"id": "doc-2", "content": "Second", "score": 0.4},
            {"id": "doc-3", "content": "Third", "score": 0.3},
        ]

        results = await reranker.rerank("query", documents)

        # Should return top 2 sorted by ColBERT score
        assert len(results) == 2
        # doc-2 should rank higher due to better embedding match
        assert results[0].doc_id == "doc-2"
        assert results[0].original_rank == 1

    @pytest.mark.asyncio
    async def test_rerank_preserves_original_info(self):
        """Test that original rank and score are preserved."""
        reranker = ColBERTReranker(enabled=False)

        documents = [
            {"id": "doc-1", "content": "First", "score": 0.9, "metadata": {"page": 1}},
            {"id": "doc-2", "content": "Second", "score": 0.8, "metadata": {"page": 2}},
        ]

        results = await reranker.rerank("test", documents)

        assert results[0].original_rank == 0
        assert results[0].original_score == 0.9
        assert results[0].metadata["page"] == 1
        assert results[1].original_rank == 1
        assert results[1].original_score == 0.8

    @pytest.mark.asyncio
    async def test_rerank_error_fallback(self):
        """Test graceful fallback on encoding error."""
        reranker = ColBERTReranker(enabled=True)

        # Mock encoder that raises an error
        mock_encoder = MagicMock()
        mock_encoder.encode_query = MagicMock(side_effect=RuntimeError("Model error"))
        reranker._encoder = mock_encoder

        documents = [
            {"id": "doc-1", "content": "First", "score": 0.9},
            {"id": "doc-2", "content": "Second", "score": 0.8},
        ]

        # Should fall back to original order
        results = await reranker.rerank("test", documents)

        assert len(results) == 2
        assert results[0].doc_id == "doc-1"
        assert results[1].doc_id == "doc-2"


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateColBERTReranker:
    """Tests for create_colbert_reranker factory function."""

    def test_create_disabled(self):
        """Test creating disabled reranker."""
        reranker = create_colbert_reranker(enabled=False)
        assert not reranker.enabled

    def test_create_enabled(self):
        """Test creating enabled reranker."""
        reranker = create_colbert_reranker(enabled=True)
        assert reranker.enabled

    def test_create_with_config(self):
        """Test creating reranker with custom config."""
        reranker = create_colbert_reranker(
            enabled=True,
            model_name="custom-model",
            max_length=256,
            top_k=5,
        )
        assert reranker.enabled
        assert reranker._model_name == "custom-model"
        assert reranker._max_length == 256
        assert reranker._top_k == 5


# ============================================================================
# Default Constants Tests
# ============================================================================


class TestDefaultConstants:
    """Tests for default configuration constants."""

    def test_default_colbert_disabled(self):
        """Test ColBERT is disabled by default."""
        assert DEFAULT_COLBERT_ENABLED is False

    def test_default_model(self):
        """Test default ColBERT model."""
        assert DEFAULT_COLBERT_MODEL == "colbert-ir/colbertv2.0"

    def test_default_max_length(self):
        """Test default max sequence length."""
        assert DEFAULT_COLBERT_MAX_LENGTH == 512
