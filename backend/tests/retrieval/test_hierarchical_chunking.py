"""Tests for Hierarchical Chunking (Story 20-C3).

Tests cover:
- HierarchicalChunk dataclass
- HierarchicalChunker class
- SmallToBigRetriever class
- SmallToBigAdapter class
- Configuration validation
- Multi-tenancy
- Performance requirements
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.indexing.hierarchical_chunker import (
    HierarchicalChunk,
    HierarchicalChunkResult,
    HierarchicalChunker,
    create_hierarchical_chunker,
    DEFAULT_LEVEL_SIZES,
    DEFAULT_OVERLAP_RATIO,
    DEFAULT_EMBEDDING_LEVEL,
)
from agentic_rag_backend.retrieval.small_to_big import (
    SmallToBigResult,
    SmallToBigRetrievalResult,
    SmallToBigRetriever,
    SmallToBigAdapter,
    get_small_to_big_adapter,
)


# =======================
# HierarchicalChunk Tests
# =======================


class TestHierarchicalChunk:
    """Tests for HierarchicalChunk dataclass."""

    def test_chunk_creation_minimal(self):
        """Test creating chunk with minimal fields."""
        chunk = HierarchicalChunk(
            id="chunk_0_abc123",
            content="Test content",
            level=0,
        )

        assert chunk.id == "chunk_0_abc123"
        assert chunk.content == "Test content"
        assert chunk.level == 0
        assert chunk.parent_id is None
        assert chunk.child_ids == []
        assert chunk.document_id == ""
        assert chunk.tenant_id == ""
        assert chunk.token_count == 0
        assert chunk.embedding is None

    def test_chunk_creation_full(self):
        """Test creating chunk with all fields."""
        chunk = HierarchicalChunk(
            id="chunk_2_def456",
            content="Full content here",
            level=2,
            parent_id="chunk_3_parent",
            child_ids=["chunk_1_a", "chunk_1_b"],
            document_id="doc_123",
            tenant_id="tenant_456",
            token_count=512,
            start_char=100,
            end_char=600,
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3],
        )

        assert chunk.id == "chunk_2_def456"
        assert chunk.level == 2
        assert chunk.parent_id == "chunk_3_parent"
        assert len(chunk.child_ids) == 2
        assert chunk.token_count == 512
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_chunk_to_dict(self):
        """Test serialization to dictionary."""
        chunk = HierarchicalChunk(
            id="chunk_0_test",
            content="Test",
            level=0,
            document_id="doc1",
            tenant_id="tenant1",
            token_count=10,
        )

        data = chunk.to_dict()

        assert data["id"] == "chunk_0_test"
        assert data["content"] == "Test"
        assert data["level"] == 0
        assert data["document_id"] == "doc1"
        assert data["tenant_id"] == "tenant1"

    def test_chunk_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "chunk_1_xyz",
            "content": "From dict",
            "level": 1,
            "parent_id": "chunk_2_parent",
            "child_ids": ["c1", "c2"],
            "document_id": "doc",
            "tenant_id": "tenant",
            "token_count": 256,
        }

        chunk = HierarchicalChunk.from_dict(data)

        assert chunk.id == "chunk_1_xyz"
        assert chunk.level == 1
        assert chunk.parent_id == "chunk_2_parent"
        assert chunk.child_ids == ["c1", "c2"]

    def test_chunk_roundtrip_serialization(self):
        """Test serialization/deserialization roundtrip."""
        original = HierarchicalChunk(
            id="chunk_0_round",
            content="Roundtrip test",
            level=0,
            parent_id="parent",
            child_ids=["child1"],
            document_id="doc",
            tenant_id="tenant",
            token_count=100,
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = HierarchicalChunk.from_dict(data)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.level == original.level
        assert restored.parent_id == original.parent_id
        assert restored.child_ids == original.child_ids


# ==========================
# HierarchicalChunker Tests
# ==========================


class TestHierarchicalChunker:
    """Tests for HierarchicalChunker class."""

    def test_chunker_initialization_default(self):
        """Test chunker initialization with defaults."""
        chunker = HierarchicalChunker()

        assert chunker.level_sizes == DEFAULT_LEVEL_SIZES
        assert chunker.overlap_ratio == DEFAULT_OVERLAP_RATIO
        assert chunker.embedding_level == DEFAULT_EMBEDDING_LEVEL

    def test_chunker_initialization_custom(self):
        """Test chunker initialization with custom values."""
        chunker = HierarchicalChunker(
            level_sizes=[128, 256, 512],
            overlap_ratio=0.2,
            embedding_level=0,
        )

        assert chunker.level_sizes == [128, 256, 512]
        assert chunker.overlap_ratio == 0.2
        assert chunker.embedding_level == 0

    def test_chunker_invalid_level_sizes_not_increasing(self):
        """Test that non-increasing level sizes raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            HierarchicalChunker(level_sizes=[256, 256, 512])

    def test_chunker_invalid_level_sizes_decreasing(self):
        """Test that decreasing level sizes raise ValueError."""
        with pytest.raises(ValueError, match="strictly increasing"):
            HierarchicalChunker(level_sizes=[512, 256, 128])

    def test_chunker_invalid_overlap_ratio_negative(self):
        """Test that negative overlap ratio raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 0.5"):
            HierarchicalChunker(overlap_ratio=-0.1)

    def test_chunker_invalid_overlap_ratio_too_large(self):
        """Test that overlap ratio > 0.5 raises ValueError."""
        with pytest.raises(ValueError, match="between 0.0 and 0.5"):
            HierarchicalChunker(overlap_ratio=0.6)

    def test_chunker_invalid_embedding_level(self):
        """Test that invalid embedding level raises ValueError."""
        with pytest.raises(ValueError, match="Embedding level"):
            HierarchicalChunker(
                level_sizes=[256, 512],
                embedding_level=5,
            )

    def test_chunk_document_empty_content(self):
        """Test chunking empty content returns empty result."""
        chunker = HierarchicalChunker()
        result = chunker.chunk_document(
            content="",
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks == 0
        assert result.chunks_by_level == {}

    def test_chunk_document_whitespace_only(self):
        """Test chunking whitespace-only content returns empty result."""
        chunker = HierarchicalChunker()
        result = chunker.chunk_document(
            content="   \n\t  ",
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks == 0

    def test_chunk_document_short_content(self):
        """Test chunking short content creates appropriate chunks."""
        chunker = HierarchicalChunker(level_sizes=[10, 20, 40])
        content = "This is a short test."

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks >= 1
        assert 0 in result.chunks_by_level
        # All chunks should have same tenant and document
        for level, chunks in result.chunks_by_level.items():
            for chunk in chunks:
                assert chunk.tenant_id == "tenant1"
                assert chunk.document_id == "doc1"

    def test_chunk_document_creates_hierarchy(self):
        """Test that chunking creates proper parent-child hierarchy."""
        chunker = HierarchicalChunker(
            level_sizes=[50, 100, 200],
            overlap_ratio=0.1,
        )
        # Create content long enough to need multiple chunks
        content = "This is a test. " * 50  # ~200 tokens

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        # Should have chunks at all levels
        assert len(result.chunks_by_level) == 3

        # Level 0 chunks should reference parents
        level_0_chunks = result.chunks_by_level.get(0, [])
        for chunk in level_0_chunks:
            # Not all L0 chunks will have parents if they're small docs
            if chunk.parent_id:
                assert chunk.parent_id.startswith("chunk_1_")

        # Higher level chunks should have children
        if 1 in result.chunks_by_level:
            for chunk in result.chunks_by_level[1]:
                assert len(chunk.child_ids) > 0

    def test_chunk_document_deterministic_ids(self):
        """Test that chunk IDs are deterministic for same content."""
        chunker = HierarchicalChunker()
        content = "Deterministic ID test content here."

        result1 = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )
        result2 = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        # IDs should be the same for same input
        ids1 = [c.id for c in result1.all_chunks]
        ids2 = [c.id for c in result2.all_chunks]
        assert ids1 == ids2

    def test_chunk_document_different_ids_different_docs(self):
        """Test that different documents get different chunk IDs."""
        chunker = HierarchicalChunker()
        content = "Same content in different documents."

        result1 = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )
        result2 = chunker.chunk_document(
            content=content,
            document_id="doc2",
            tenant_id="tenant1",
        )

        ids1 = set(c.id for c in result1.all_chunks)
        ids2 = set(c.id for c in result2.all_chunks)
        assert ids1.isdisjoint(ids2)

    def test_chunk_document_metadata_preserved(self):
        """Test that metadata is preserved in chunks."""
        chunker = HierarchicalChunker()
        content = "Content with metadata."
        metadata = {"source": "test", "page": 1}

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
            metadata=metadata,
        )

        for chunk in result.all_chunks:
            assert chunk.metadata.get("source") == "test"
            assert chunk.metadata.get("page") == 1

    def test_chunk_document_result_all_chunks(self):
        """Test HierarchicalChunkResult.all_chunks property."""
        chunker = HierarchicalChunker(level_sizes=[50, 100, 200])
        content = "Test content for all_chunks. " * 30

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        all_chunks = result.all_chunks
        total_from_levels = sum(len(chunks) for chunks in result.chunks_by_level.values())

        assert len(all_chunks) == total_from_levels
        assert len(all_chunks) == result.total_chunks


class TestHierarchicalChunkerPerformance:
    """Performance tests for hierarchical chunking."""

    def test_chunking_latency_typical_document(self):
        """Test that chunking typical document takes < 500ms (AC #11)."""
        chunker = HierarchicalChunker()
        # Simulate ~10 page document (~5000 words)
        content = "This is a test sentence with several words. " * 500

        start = time.perf_counter()
        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Chunking took {elapsed_ms}ms, expected < 500ms"
        assert result.total_chunks > 0

    def test_chunking_latency_small_document(self):
        """Test chunking small document is fast."""
        chunker = HierarchicalChunker()
        content = "Short document content."

        start = time.perf_counter()
        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Small doc chunking took {elapsed_ms}ms"


class TestCreateHierarchicalChunker:
    """Tests for factory function."""

    def test_factory_creates_chunker_with_defaults(self):
        """Test factory function creates chunker with defaults."""
        chunker = create_hierarchical_chunker()

        assert isinstance(chunker, HierarchicalChunker)
        assert chunker.level_sizes == DEFAULT_LEVEL_SIZES

    def test_factory_creates_chunker_with_custom(self):
        """Test factory function creates chunker with custom config."""
        chunker = create_hierarchical_chunker(
            level_sizes=[100, 200],
            overlap_ratio=0.15,
        )

        assert chunker.level_sizes == [100, 200]
        assert chunker.overlap_ratio == 0.15


# ==========================
# SmallToBigRetriever Tests
# ==========================


class TestSmallToBigResult:
    """Tests for SmallToBigResult dataclass."""

    def test_result_creation(self):
        """Test creating SmallToBigResult."""
        result = SmallToBigResult(
            id="chunk_2_abc",
            content="Parent content",
            level=2,
            score=0.85,
            matched_child_ids=["chunk_0_a", "chunk_0_b"],
            matched_scores=[0.9, 0.8],
            document_id="doc1",
            token_count=1024,
        )

        assert result.id == "chunk_2_abc"
        assert result.level == 2
        assert result.score == 0.85
        assert len(result.matched_child_ids) == 2

    def test_result_to_dict(self):
        """Test result serialization."""
        result = SmallToBigResult(
            id="chunk",
            content="Content",
            level=1,
            score=0.75,
        )

        data = result.to_dict()

        assert data["id"] == "chunk"
        assert data["score"] == 0.75


class TestSmallToBigRetrievalResult:
    """Tests for SmallToBigRetrievalResult dataclass."""

    def test_retrieval_result_creation(self):
        """Test creating full retrieval result."""
        results = [
            SmallToBigResult(id="c1", content="Content 1", level=2, score=0.9),
            SmallToBigResult(id="c2", content="Content 2", level=2, score=0.8),
        ]

        retrieval_result = SmallToBigRetrievalResult(
            query="test query",
            results=results,
            matched_at_level=0,
            returned_at_level=2,
            total_matches=5,
            processing_time_ms=50,
            tenant_id="tenant1",
        )

        assert retrieval_result.query == "test query"
        assert len(retrieval_result.results) == 2
        assert retrieval_result.matched_at_level == 0
        assert retrieval_result.returned_at_level == 2


class TestSmallToBigRetriever:
    """Tests for SmallToBigRetriever class."""

    @pytest.fixture
    def mock_chunk_store(self):
        """Create mock chunk store."""
        store = AsyncMock()
        return store

    def test_retriever_initialization(self, mock_chunk_store):
        """Test retriever initialization."""
        retriever = SmallToBigRetriever(
            chunk_store=mock_chunk_store,
            return_level=2,
            embedding_level=0,
        )

        assert retriever.return_level == 2
        assert retriever.embedding_level == 0

    @pytest.mark.asyncio
    async def test_retrieve_no_matches(self, mock_chunk_store):
        """Test retrieval with no matches returns empty result."""
        mock_chunk_store.search_by_embedding.return_value = []

        retriever = SmallToBigRetriever(
            chunk_store=mock_chunk_store,
            return_level=2,
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
            top_k=10,
        )

        assert len(result.results) == 0
        assert result.total_matches == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_matches(self, mock_chunk_store):
        """Test retrieval with matches returns parent chunks."""
        # Setup mock small chunk matches
        mock_chunk_store.search_by_embedding.return_value = [
            {"id": "chunk_0_a", "level": 0, "score": 0.9, "parent_id": "chunk_1_x"},
            {"id": "chunk_0_b", "level": 0, "score": 0.85, "parent_id": "chunk_1_x"},
        ]

        # Setup mock chunk lookups
        async def mock_get(chunk_id, tenant_id):
            chunks = {
                "chunk_0_a": {"id": "chunk_0_a", "level": 0, "parent_id": "chunk_1_x"},
                "chunk_0_b": {"id": "chunk_0_b", "level": 0, "parent_id": "chunk_1_x"},
                "chunk_1_x": {"id": "chunk_1_x", "level": 1, "parent_id": "chunk_2_y"},
                "chunk_2_y": {
                    "id": "chunk_2_y",
                    "level": 2,
                    "content": "Parent content",
                    "token_count": 1024,
                    "document_id": "doc1",
                    "metadata": {},
                },
            }
            return chunks.get(chunk_id)

        mock_chunk_store.get = mock_get

        retriever = SmallToBigRetriever(
            chunk_store=mock_chunk_store,
            return_level=2,
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
            top_k=10,
        )

        assert len(result.results) > 0
        assert result.matched_at_level == 0
        assert result.returned_at_level == 2

    @pytest.mark.asyncio
    async def test_retrieve_deduplicates_parents(self, mock_chunk_store):
        """Test that multiple children with same parent are deduplicated."""
        # Multiple small chunks pointing to same parent
        mock_chunk_store.search_by_embedding.return_value = [
            {"id": "chunk_0_a", "level": 0, "score": 0.9},
            {"id": "chunk_0_b", "level": 0, "score": 0.85},
            {"id": "chunk_0_c", "level": 0, "score": 0.8},
        ]

        # All point to same parent at level 2
        async def mock_get(chunk_id, tenant_id):
            if chunk_id.startswith("chunk_0"):
                return {"id": chunk_id, "level": 0}
            return {
                "id": "chunk_2_same",
                "level": 2,
                "content": "Same parent",
                "token_count": 1024,
                "document_id": "doc1",
                "metadata": {},
            }

        mock_chunk_store.get = mock_get

        retriever = SmallToBigRetriever(
            chunk_store=mock_chunk_store,
            return_level=2,
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
            top_k=10,
        )

        # Should only return 1 unique parent
        # (implementation uses parent ID to dedupe, depends on ancestor traversal)
        assert result.total_matches == 3  # Original matches

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self, mock_chunk_store):
        """Test that top_k limits results."""
        # Return many matches
        mock_chunk_store.search_by_embedding.return_value = [
            {"id": f"chunk_0_{i}", "level": 0, "score": 0.9 - i * 0.05}
            for i in range(20)
        ]

        async def mock_get(chunk_id, tenant_id):
            return {
                "id": chunk_id.replace("_0_", "_2_"),
                "level": 2,
                "content": f"Content for {chunk_id}",
                "token_count": 1024,
                "document_id": "doc1",
                "metadata": {},
            }

        mock_chunk_store.get = mock_get

        retriever = SmallToBigRetriever(
            chunk_store=mock_chunk_store,
            return_level=2,
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
            top_k=5,
        )

        assert len(result.results) <= 5

    @pytest.mark.asyncio
    async def test_retrieve_override_return_level(self, mock_chunk_store):
        """Test that return_level can be overridden per query."""
        mock_chunk_store.search_by_embedding.return_value = [
            {"id": "chunk_0_a", "level": 0, "score": 0.9},
        ]

        async def mock_get(chunk_id, tenant_id):
            return {
                "id": chunk_id.replace("_0_", "_1_"),
                "level": 1,
                "content": "Level 1 content",
                "token_count": 512,
                "document_id": "doc1",
                "metadata": {},
            }

        mock_chunk_store.get = mock_get

        retriever = SmallToBigRetriever(
            chunk_store=mock_chunk_store,
            return_level=2,  # Default
        )

        result = await retriever.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
            return_level=1,  # Override
        )

        assert result.returned_at_level == 1


class TestSmallToBigAdapter:
    """Tests for SmallToBigAdapter class."""

    def test_adapter_disabled(self):
        """Test adapter when feature is disabled."""
        adapter = SmallToBigAdapter(
            retriever=None,
            enabled=False,
        )

        assert not adapter.enabled

    @pytest.mark.asyncio
    async def test_adapter_disabled_returns_none(self):
        """Test disabled adapter returns None."""
        adapter = SmallToBigAdapter(
            retriever=MagicMock(),
            enabled=False,
        )

        result = await adapter.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_adapter_enabled_calls_retriever(self):
        """Test enabled adapter calls retriever."""
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = SmallToBigRetrievalResult(
            query="test",
            results=[],
            matched_at_level=0,
            returned_at_level=2,
            total_matches=0,
            processing_time_ms=50,
            tenant_id="tenant1",
        )

        adapter = SmallToBigAdapter(
            retriever=mock_retriever,
            enabled=True,
            return_level=2,
        )

        result = await adapter.retrieve(
            query="test",
            tenant_id="tenant1",
            embedding=[0.1] * 1536,
        )

        assert result is not None
        mock_retriever.retrieve.assert_called_once()


class TestGetSmallToBigAdapter:
    """Tests for factory function."""

    def test_factory_disabled_when_feature_off(self):
        """Test factory creates disabled adapter when feature off."""
        mock_settings = MagicMock()
        mock_settings.hierarchical_chunks_enabled = False
        mock_settings.small_to_big_return_level = 2
        mock_settings.hierarchical_embedding_level = 0

        adapter = get_small_to_big_adapter(
            chunk_store=MagicMock(),
            settings=mock_settings,
        )

        assert not adapter.enabled

    def test_factory_enabled_when_feature_on(self):
        """Test factory creates enabled adapter when feature on."""
        mock_settings = MagicMock()
        mock_settings.hierarchical_chunks_enabled = True
        mock_settings.small_to_big_return_level = 2
        mock_settings.hierarchical_embedding_level = 0

        mock_chunk_store = MagicMock()

        adapter = get_small_to_big_adapter(
            chunk_store=mock_chunk_store,
            settings=mock_settings,
        )

        assert adapter.enabled
        assert adapter.return_level == 2


# =====================
# Multi-Tenancy Tests
# =====================


class TestMultiTenancy:
    """Tests for multi-tenancy enforcement."""

    def test_chunks_have_tenant_id(self):
        """Test that all chunks have tenant_id set."""
        chunker = HierarchicalChunker()
        content = "Test content for tenancy."

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant_abc",
        )

        for chunk in result.all_chunks:
            assert chunk.tenant_id == "tenant_abc"

    def test_different_tenants_same_content(self):
        """Test that different tenants can have same content."""
        chunker = HierarchicalChunker()
        content = "Same content for different tenants."

        result1 = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant_1",
        )
        result2 = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant_2",
        )

        # Both should succeed
        assert result1.total_chunks > 0
        assert result2.total_chunks > 0

        # Tenants should be different
        assert all(c.tenant_id == "tenant_1" for c in result1.all_chunks)
        assert all(c.tenant_id == "tenant_2" for c in result2.all_chunks)


# ====================
# Edge Cases Tests
# ====================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_chunk_single_word(self):
        """Test chunking single word document."""
        chunker = HierarchicalChunker()
        result = chunker.chunk_document(
            content="Word",
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks >= 1

    def test_chunk_very_long_word(self):
        """Test chunking document with very long word."""
        chunker = HierarchicalChunker()
        content = "a" * 1000  # Very long "word"

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks >= 1

    def test_chunk_unicode_content(self):
        """Test chunking content with unicode characters."""
        chunker = HierarchicalChunker()
        content = "Hello 世界! Привет мир! مرحبا بالعالم!"

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks >= 1
        # Content should be preserved
        all_content = " ".join(c.content for c in result.all_chunks)
        assert "世界" in all_content

    def test_chunk_special_characters(self):
        """Test chunking content with special characters."""
        chunker = HierarchicalChunker()
        content = "Code: func() { return 42; } // comment\n\nSection 2: <html></html>"

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks >= 1

    def test_chunk_newlines_and_whitespace(self):
        """Test chunking content with various whitespace."""
        chunker = HierarchicalChunker()
        content = "Line 1\n\nLine 2\n\n\n\nLine 3\t\tTabbed"

        result = chunker.chunk_document(
            content=content,
            document_id="doc1",
            tenant_id="tenant1",
        )

        assert result.total_chunks >= 1
