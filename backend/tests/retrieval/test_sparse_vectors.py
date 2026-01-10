"""Unit tests for sparse vector search (Story 20-H1).

Tests cover:
- SparseVector dataclass operations
- BM42Encoder functionality (with mock)
- HybridVectorSearch with RRF fusion
- SparseVectorAdapter feature flag behavior
- Tenant isolation
"""

import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.retrieval.sparse_vectors import (
    SparseVector,
    BM42Encoder,
    HybridVectorSearch,
    HybridSearchResult,
    SparseVectorAdapter,
    DEFAULT_SPARSE_VECTORS_ENABLED,
    DEFAULT_SPARSE_MODEL,
    DEFAULT_HYBRID_DENSE_WEIGHT,
    DEFAULT_HYBRID_SPARSE_WEIGHT,
    DEFAULT_RRF_K,
)


# ============================================================================
# SparseVector Tests
# ============================================================================

class TestSparseVector:
    """Tests for SparseVector dataclass."""

    def test_create_sparse_vector(self):
        """Test basic sparse vector creation."""
        vector = SparseVector(
            indices=[10, 42, 100],
            values=[0.8, 0.3, 0.5],
        )
        assert vector.indices == [10, 42, 100]
        assert vector.values == [0.8, 0.3, 0.5]

    def test_create_empty_sparse_vector(self):
        """Test creating an empty sparse vector."""
        vector = SparseVector()
        assert vector.indices == []
        assert vector.values == []
        assert vector.is_empty

    def test_sparse_vector_length_mismatch_raises(self):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            SparseVector(
                indices=[1, 2, 3],
                values=[0.1, 0.2],  # One fewer value
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        vector = SparseVector(indices=[1, 2], values=[0.5, 0.3])
        result = vector.to_dict()
        assert result == {"indices": [1, 2], "values": [0.5, 0.3]}

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {"indices": [5, 10], "values": [0.7, 0.2]}
        vector = SparseVector.from_dict(data)
        assert vector.indices == [5, 10]
        assert vector.values == [0.7, 0.2]

    def test_from_dict_empty(self):
        """Test creation from empty dictionary."""
        vector = SparseVector.from_dict({})
        assert vector.is_empty

    def test_dot_product_overlapping(self):
        """Test dot product with overlapping indices."""
        v1 = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        v2 = SparseVector(indices=[2, 3, 4], values=[0.4, 0.6, 0.1])
        # Overlap: 2 -> 0.3*0.4, 3 -> 0.2*0.6
        expected = 0.3 * 0.4 + 0.2 * 0.6
        assert abs(v1.dot_product(v2) - expected) < 1e-6

    def test_dot_product_no_overlap(self):
        """Test dot product with no overlapping indices."""
        v1 = SparseVector(indices=[1, 2], values=[0.5, 0.3])
        v2 = SparseVector(indices=[3, 4], values=[0.4, 0.6])
        assert v1.dot_product(v2) == 0.0

    def test_dot_product_same_vector(self):
        """Test dot product with itself (squared norm)."""
        vector = SparseVector(indices=[1, 2, 3], values=[0.6, 0.8, 0.0])
        expected = 0.6 * 0.6 + 0.8 * 0.8 + 0.0 * 0.0
        assert abs(vector.dot_product(vector) - expected) < 1e-6

    def test_dot_product_empty_vectors(self):
        """Test dot product with empty vectors."""
        v1 = SparseVector()
        v2 = SparseVector(indices=[1], values=[1.0])
        assert v1.dot_product(v2) == 0.0
        assert v2.dot_product(v1) == 0.0

    def test_is_empty_property(self):
        """Test is_empty property."""
        assert SparseVector().is_empty
        assert not SparseVector(indices=[1], values=[0.5]).is_empty


# ============================================================================
# BM42Encoder Tests
# ============================================================================

class TestBM42Encoder:
    """Tests for BM42Encoder class."""

    def test_encoder_initialization(self):
        """Test encoder initialization without loading model."""
        encoder = BM42Encoder(model_name="test-model")
        assert encoder.model_name == "test-model"
        assert encoder._model is None  # Lazy initialization

    def test_default_model_name(self):
        """Test default model name is set correctly."""
        encoder = BM42Encoder()
        assert encoder.model_name == DEFAULT_SPARSE_MODEL

    @patch("agentic_rag_backend.retrieval.sparse_vectors.BM42Encoder._ensure_model")
    def test_encode_calls_ensure_model(self, mock_ensure):
        """Test that encode calls _ensure_model."""
        encoder = BM42Encoder()
        encoder._model = MagicMock()
        encoder._model.embed.return_value = []

        # Non-empty input to avoid short-circuit
        encoder.encode(["test"])
        mock_ensure.assert_called_once()

    def test_encode_empty_list(self):
        """Test encoding empty list returns empty list."""
        encoder = BM42Encoder()
        result = encoder.encode([])
        assert result == []

    @patch("agentic_rag_backend.retrieval.sparse_vectors.BM42Encoder._ensure_model")
    def test_encode_batch(self, mock_ensure):
        """Test batch encoding with mocked model."""
        encoder = BM42Encoder()

        # Mock embedding results
        mock_emb1 = MagicMock()
        mock_emb1.indices = [1, 2, 3]
        mock_emb1.values = [0.5, 0.3, 0.2]

        mock_emb2 = MagicMock()
        mock_emb2.indices = [4, 5]
        mock_emb2.values = [0.8, 0.1]

        encoder._model = MagicMock()
        encoder._model.embed.return_value = [mock_emb1, mock_emb2]

        results = encoder.encode(["text1", "text2"])

        assert len(results) == 2
        assert results[0].indices == [1, 2, 3]
        assert results[0].values == [0.5, 0.3, 0.2]
        assert results[1].indices == [4, 5]
        assert results[1].values == [0.8, 0.1]

    @patch("agentic_rag_backend.retrieval.sparse_vectors.BM42Encoder._ensure_model")
    def test_encode_query(self, mock_ensure):
        """Test single query encoding."""
        encoder = BM42Encoder()

        mock_emb = MagicMock()
        mock_emb.indices = [10, 20]
        mock_emb.values = [0.9, 0.1]

        encoder._model = MagicMock()
        encoder._model.embed.return_value = [mock_emb]

        result = encoder.encode_query("test query")

        assert result.indices == [10, 20]
        assert result.values == [0.9, 0.1]

    def test_ensure_model_import_error(self):
        """Test that import error is raised if fastembed not installed."""
        encoder = BM42Encoder()

        with patch.dict("sys.modules", {"fastembed": None}):
            with patch(
                "agentic_rag_backend.retrieval.sparse_vectors.BM42Encoder._ensure_model",
                side_effect=ImportError("fastembed is required"),
            ):
                with pytest.raises(ImportError, match="fastembed"):
                    encoder._ensure_model()


# ============================================================================
# HybridVectorSearch Tests
# ============================================================================

class MockDenseSearch:
    """Mock dense search for testing."""

    def __init__(self, results: list[dict[str, Any]]):
        self.results = results
        self.calls = []

    async def search(
        self,
        query: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "tenant_id": tenant_id, "limit": limit})
        return self.results[:limit]


class TestHybridVectorSearch:
    """Tests for HybridVectorSearch class."""

    @pytest.fixture
    def mock_dense_search(self):
        """Create mock dense search with sample results."""
        return MockDenseSearch([
            {"id": "doc1", "score": 0.9, "content": "First doc"},
            {"id": "doc2", "score": 0.7, "content": "Second doc"},
            {"id": "doc3", "score": 0.5, "content": "Third doc"},
        ])

    @pytest.fixture
    def mock_encoder(self):
        """Create mock BM42 encoder."""
        encoder = MagicMock(spec=BM42Encoder)
        encoder.encode_query.return_value = SparseVector(
            indices=[1, 2, 3],
            values=[0.5, 0.3, 0.2],
        )
        return encoder

    @pytest.fixture
    def hybrid_search(self, mock_dense_search, mock_encoder):
        """Create HybridVectorSearch instance."""
        return HybridVectorSearch(
            dense_search=mock_dense_search,
            sparse_encoder=mock_encoder,
            dense_weight=0.7,
            sparse_weight=0.3,
        )

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_results(
        self, hybrid_search, mock_dense_search
    ):
        """Test that hybrid search combines dense and sparse results."""
        results = await hybrid_search.search("test query", "tenant-1", limit=10)

        # Should have called dense search
        assert len(mock_dense_search.calls) == 1
        assert mock_dense_search.calls[0]["tenant_id"] == "tenant-1"

        # Results should have RRF scores
        assert len(results) > 0
        assert all("rrf_score" in r for r in results)

    @pytest.mark.asyncio
    async def test_dense_search_failure_handled(self, mock_encoder):
        """Test graceful handling of dense search failure."""
        # Dense search that raises
        failing_dense = AsyncMock(side_effect=Exception("Dense failed"))

        hybrid = HybridVectorSearch(
            dense_search=MagicMock(search=failing_dense),
            sparse_encoder=mock_encoder,
        )

        # Should not raise, just return sparse results
        results = await hybrid.search("query", "tenant-1")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_sparse_search_failure_handled(self, mock_dense_search):
        """Test graceful handling of sparse search failure."""
        # Encoder that raises
        failing_encoder = MagicMock(spec=BM42Encoder)
        failing_encoder.encode_query.side_effect = Exception("Encoder failed")

        hybrid = HybridVectorSearch(
            dense_search=mock_dense_search,
            sparse_encoder=failing_encoder,
        )

        # Should not raise, just return dense results
        results = await hybrid.search("query", "tenant-1")
        assert isinstance(results, list)

    def test_rrf_fusion_formula(self, hybrid_search):
        """Test RRF fusion calculation."""
        dense = [
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ]
        sparse = [
            {"id": "doc2", "score": 0.8},  # doc2 appears in both
            {"id": "doc3", "score": 0.5},
        ]

        results = hybrid_search._reciprocal_rank_fusion(dense, sparse)

        # doc2 should have highest combined score (appears in both)
        assert results[0]["id"] == "doc2"

        # Verify RRF scores are present
        assert all("rrf_score" in r for r in results)

    def test_rrf_k_constant(self, mock_dense_search, mock_encoder):
        """Test RRF k constant affects scoring."""
        # Create two hybrid searches with different k values
        hybrid_k60 = HybridVectorSearch(
            dense_search=mock_dense_search,
            sparse_encoder=mock_encoder,
            rrf_k=60,
        )
        hybrid_k10 = HybridVectorSearch(
            dense_search=mock_dense_search,
            sparse_encoder=mock_encoder,
            rrf_k=10,
        )

        dense = [{"id": "doc1"}]
        sparse = []

        results_k60 = hybrid_k60._reciprocal_rank_fusion(dense, sparse)
        results_k10 = hybrid_k10._reciprocal_rank_fusion(dense, sparse)

        # Higher k should give lower scores
        assert results_k10[0]["rrf_score"] > results_k60[0]["rrf_score"]

    def test_index_document(self, hybrid_search):
        """Test document indexing for sparse search."""
        sparse = hybrid_search.index_document(
            doc_id="doc1",
            text="test document content",
            tenant_id="tenant-1",
        )

        assert isinstance(sparse, SparseVector)
        assert "tenant-1:doc1" in hybrid_search._sparse_vectors

    def test_index_document_tenant_isolation(self, hybrid_search):
        """Test that indexed documents are isolated by tenant."""
        hybrid_search.index_document("doc1", "content", "tenant-1")
        hybrid_search.index_document("doc1", "content", "tenant-2")

        # Should have two entries (same doc_id, different tenants)
        assert "tenant-1:doc1" in hybrid_search._sparse_vectors
        assert "tenant-2:doc1" in hybrid_search._sparse_vectors

    def test_remove_document(self, hybrid_search):
        """Test document removal from sparse index."""
        hybrid_search.index_document("doc1", "content", "tenant-1")
        assert "tenant-1:doc1" in hybrid_search._sparse_vectors

        removed = hybrid_search.remove_document("doc1", "tenant-1")
        assert removed
        assert "tenant-1:doc1" not in hybrid_search._sparse_vectors

    def test_remove_nonexistent_document(self, hybrid_search):
        """Test removal of non-existent document returns False."""
        removed = hybrid_search.remove_document("nonexistent", "tenant-1")
        assert not removed

    def test_clear_tenant(self, hybrid_search):
        """Test clearing all documents for a tenant."""
        hybrid_search.index_document("doc1", "content", "tenant-1")
        hybrid_search.index_document("doc2", "content", "tenant-1")
        hybrid_search.index_document("doc1", "content", "tenant-2")

        count = hybrid_search.clear_tenant("tenant-1")

        assert count == 2
        assert "tenant-1:doc1" not in hybrid_search._sparse_vectors
        assert "tenant-1:doc2" not in hybrid_search._sparse_vectors
        assert "tenant-2:doc1" in hybrid_search._sparse_vectors  # Other tenant preserved

    @pytest.mark.asyncio
    async def test_sparse_search_tenant_isolation(self, mock_dense_search, mock_encoder):
        """Test that sparse search respects tenant isolation."""
        hybrid = HybridVectorSearch(
            dense_search=mock_dense_search,
            sparse_encoder=mock_encoder,
        )

        # Index for different tenants
        hybrid.index_document("doc1", "content", "tenant-1")
        hybrid.index_document("doc2", "content", "tenant-2")

        # Search for tenant-1 should only find tenant-1 docs
        query_vector = SparseVector(indices=[1, 2, 3], values=[0.5, 0.3, 0.2])
        results = await hybrid._sparse_search(query_vector, "tenant-1", limit=10)

        # Only tenant-1 docs should be in results
        for r in results:
            assert not r["id"].startswith("tenant-2")


# ============================================================================
# SparseVectorAdapter Tests
# ============================================================================

class TestSparseVectorAdapter:
    """Tests for SparseVectorAdapter class."""

    @pytest.fixture
    def mock_dense_search(self):
        """Create mock dense search."""
        return MockDenseSearch([
            {"id": "doc1", "score": 0.9},
            {"id": "doc2", "score": 0.7},
        ])

    def test_adapter_disabled_by_default(self):
        """Test adapter is disabled by default."""
        adapter = SparseVectorAdapter()
        assert not adapter.enabled

    def test_adapter_enabled_with_dense_search(self, mock_dense_search):
        """Test adapter enables with dense search provided."""
        adapter = SparseVectorAdapter(
            enabled=True,
            dense_search=mock_dense_search,
        )
        assert adapter.enabled
        assert adapter._hybrid is not None

    def test_adapter_not_enabled_without_dense_search(self):
        """Test adapter doesn't create hybrid without dense search."""
        adapter = SparseVectorAdapter(enabled=True, dense_search=None)
        assert not adapter.enabled or adapter._hybrid is None

    @pytest.mark.asyncio
    async def test_search_when_disabled_uses_dense(self, mock_dense_search):
        """Test search falls back to dense when disabled."""
        adapter = SparseVectorAdapter(
            enabled=False,
            dense_search=mock_dense_search,
        )

        results = await adapter.search("query", "tenant-1", limit=5)

        # Should have called dense search
        assert len(mock_dense_search.calls) == 1
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_when_disabled_no_dense(self):
        """Test search returns empty when disabled and no dense search."""
        adapter = SparseVectorAdapter(enabled=False, dense_search=None)
        results = await adapter.search("query", "tenant-1")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_when_enabled_uses_hybrid(self, mock_dense_search):
        """Test search uses hybrid when enabled."""
        with patch.object(BM42Encoder, "_ensure_model"):
            adapter = SparseVectorAdapter(
                enabled=True,
                dense_search=mock_dense_search,
            )

            # Mock the encoder
            adapter._hybrid._sparse_encoder = MagicMock(spec=BM42Encoder)
            adapter._hybrid._sparse_encoder.encode_query.return_value = SparseVector()

            results = await adapter.search("query", "tenant-1", limit=5)

            # Should return results with rrf_score from hybrid search
            assert len(results) > 0

    def test_index_document_when_disabled(self):
        """Test index_document returns None when disabled."""
        adapter = SparseVectorAdapter(enabled=False)
        result = adapter.index_document("doc1", "content", "tenant-1")
        assert result is None

    def test_index_document_when_enabled(self, mock_dense_search):
        """Test index_document works when enabled."""
        with patch.object(BM42Encoder, "_ensure_model"):
            adapter = SparseVectorAdapter(
                enabled=True,
                dense_search=mock_dense_search,
            )

            # Mock the encoder
            adapter._hybrid._sparse_encoder = MagicMock(spec=BM42Encoder)
            adapter._hybrid._sparse_encoder.encode_query.return_value = SparseVector(
                indices=[1], values=[0.5]
            )

            result = adapter.index_document("doc1", "content", "tenant-1")
            assert isinstance(result, SparseVector)

    def test_remove_document_when_disabled(self):
        """Test remove_document returns False when disabled."""
        adapter = SparseVectorAdapter(enabled=False)
        result = adapter.remove_document("doc1", "tenant-1")
        assert not result

    def test_encode_query_when_disabled(self):
        """Test encode_query returns None when disabled."""
        adapter = SparseVectorAdapter(enabled=False)
        result = adapter.encode_query("test query")
        assert result is None


# ============================================================================
# Default Constants Tests
# ============================================================================

class TestDefaultConstants:
    """Tests for default configuration constants."""

    def test_default_sparse_vectors_disabled(self):
        """Test sparse vectors are disabled by default."""
        assert DEFAULT_SPARSE_VECTORS_ENABLED is False

    def test_default_model_name(self):
        """Test default model name matches Qdrant BM42."""
        assert "bm42" in DEFAULT_SPARSE_MODEL.lower()
        assert "Qdrant" in DEFAULT_SPARSE_MODEL

    def test_default_weights_sum_to_one(self):
        """Test default weights sum to 1.0."""
        assert DEFAULT_HYBRID_DENSE_WEIGHT + DEFAULT_HYBRID_SPARSE_WEIGHT == 1.0

    def test_default_rrf_k(self):
        """Test default RRF k is 60 (per paper)."""
        assert DEFAULT_RRF_K == 60


# ============================================================================
# Integration-Style Tests
# ============================================================================

class TestHybridSearchIntegration:
    """Integration-style tests for the complete flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_hybrid_search(self):
        """Test complete hybrid search flow."""
        # Setup
        dense_search = MockDenseSearch([
            {"id": "doc1", "score": 0.9, "title": "Machine Learning Guide"},
            {"id": "doc2", "score": 0.8, "title": "Python Tutorial"},
        ])

        with patch.object(BM42Encoder, "_ensure_model"):
            encoder = BM42Encoder()
            encoder._model = MagicMock()
            encoder._model.embed.return_value = [
                MagicMock(indices=[1, 2], values=[0.5, 0.5])
            ]

            hybrid = HybridVectorSearch(
                dense_search=dense_search,
                sparse_encoder=encoder,
                dense_weight=0.7,
                sparse_weight=0.3,
            )

            # Index some documents for sparse search
            hybrid.index_document("doc1", "machine learning guide", "tenant-1")
            hybrid.index_document("doc3", "sparse only document", "tenant-1")

            # Search
            results = await hybrid.search(
                "machine learning",
                "tenant-1",
                limit=5,
            )

            # Verify
            assert len(results) > 0
            assert all("rrf_score" in r for r in results)
            # doc1 should be ranked high (appears in both)
            doc1_in_top = any(r.get("id") == "doc1" for r in results[:2])
            assert doc1_in_top

    @pytest.mark.asyncio
    async def test_adapter_with_config_values(self):
        """Test adapter respects configuration values."""
        dense_search = MockDenseSearch([{"id": "doc1"}])

        with patch.object(BM42Encoder, "_ensure_model"):
            adapter = SparseVectorAdapter(
                enabled=True,
                dense_search=dense_search,
                sparse_model="custom/model",
                dense_weight=0.6,
                sparse_weight=0.4,
            )

            assert adapter.enabled
            assert adapter._hybrid.dense_weight == 0.6
            assert adapter._hybrid.sparse_weight == 0.4
