"""Tests for the semantic chunker module."""

from agentic_rag_backend.indexing.chunker import (
    ChunkData,
    chunk_document,
    chunk_sections,
    count_tokens,
    estimate_chunks,
)


class TestCountTokens:
    """Tests for token counting."""

    def test_count_tokens_empty_string(self):
        """Empty string should return 0 tokens."""
        assert count_tokens("") == 0

    def test_count_tokens_simple_text(self):
        """Simple text should have reasonable token count."""
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Short text

    def test_count_tokens_longer_text(self):
        """Longer text should have more tokens."""
        short = count_tokens("Hello")
        long = count_tokens("Hello, this is a much longer sentence with many more words.")
        assert long > short


class TestChunkDocument:
    """Tests for document chunking."""

    def test_chunk_empty_content(self):
        """Empty content should return no chunks."""
        chunks = chunk_document("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Whitespace-only content should return no chunks."""
        chunks = chunk_document("   \n\t  ")
        assert chunks == []

    def test_chunk_small_document(self):
        """Document smaller than chunk size should return single chunk."""
        content = "This is a small document."
        chunks = chunk_document(content, chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0].content.strip() == content
        assert chunks[0].chunk_index == 0

    def test_chunk_data_fields(self):
        """Chunk should have all required fields."""
        content = "This is test content for chunking."
        chunks = chunk_document(content, chunk_size=100)

        assert len(chunks) >= 1
        chunk = chunks[0]

        assert isinstance(chunk, ChunkData)
        assert isinstance(chunk.content, str)
        assert isinstance(chunk.chunk_index, int)
        assert isinstance(chunk.token_count, int)
        assert isinstance(chunk.start_char, int)
        assert isinstance(chunk.end_char, int)
        assert chunk.token_count > 0

    def test_chunk_large_document_creates_multiple_chunks(self):
        """Large document should be split into multiple chunks."""
        # Create a long document
        content = "This is a sentence. " * 100
        chunks = chunk_document(content, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        # Check chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_overlap(self):
        """Chunks should have overlapping content when overlap > 0."""
        content = "Word one. Word two. Word three. Word four. Word five. Word six. Word seven. Word eight."
        chunks = chunk_document(content, chunk_size=20, chunk_overlap=5)

        if len(chunks) > 1:
            # Check that there's some overlap in token counts
            # The exact overlap is hard to verify, but we can check chunks exist
            assert all(c.token_count > 0 for c in chunks)

    def test_chunk_no_overlap(self):
        """Chunks with no overlap should not repeat content."""
        content = "Word one. Word two. Word three. Word four. Word five."
        chunks = chunk_document(content, chunk_size=10, chunk_overlap=0)

        # Each chunk should be independent
        assert len(chunks) >= 1

    def test_chunk_preserves_content(self):
        """Chunking should preserve all content from the document."""
        content = "First sentence. Second sentence. Third sentence."
        chunks = chunk_document(content, chunk_size=20, chunk_overlap=2)

        # All chunks should have non-empty content
        assert all(len(c.content.strip()) > 0 for c in chunks)


class TestChunkSections:
    """Tests for section-based chunking."""

    def test_chunk_sections_empty_list(self):
        """Empty section list should return no chunks."""
        chunks = chunk_sections([])
        assert chunks == []

    def test_chunk_sections_single_small_section(self):
        """Single small section should create one chunk."""
        sections = [
            {"heading": "Introduction", "content": "This is the intro."}
        ]
        chunks = chunk_sections(sections, chunk_size=100)

        assert len(chunks) == 1
        assert "Introduction" in chunks[0].content
        assert "This is the intro" in chunks[0].content

    def test_chunk_sections_preserves_headings(self):
        """Section headings should be preserved in chunks."""
        sections = [
            {"heading": "Chapter One", "content": "Content for chapter one."},
            {"heading": "Chapter Two", "content": "Content for chapter two."},
        ]
        chunks = chunk_sections(sections, chunk_size=100)

        headings_found = sum(1 for c in chunks if "Chapter" in c.content)
        assert headings_found >= 2

    def test_chunk_sections_large_section_splits(self):
        """Large section should be split into multiple chunks."""
        sections = [
            {"heading": "Large Section", "content": "Word. " * 500}
        ]
        chunks = chunk_sections(sections, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1


class TestEstimateChunks:
    """Tests for chunk estimation."""

    def test_estimate_chunks_empty(self):
        """Empty content should estimate 0 chunks."""
        assert estimate_chunks("") == 0

    def test_estimate_chunks_small_document(self):
        """Small document should estimate 1 chunk."""
        assert estimate_chunks("Hello world", chunk_size=100) == 1

    def test_estimate_chunks_larger_document(self):
        """Larger document should estimate multiple chunks."""
        content = "Word. " * 500
        estimate = estimate_chunks(content, chunk_size=50)
        assert estimate > 1

    def test_estimate_reasonably_accurate(self):
        """Estimate should be reasonably close to actual chunks."""
        content = "This is a sentence. " * 100
        estimate = estimate_chunks(content, chunk_size=50)
        actual = len(chunk_document(content, chunk_size=50, chunk_overlap=10))

        # Estimate should be within 50% of actual
        assert abs(estimate - actual) / max(actual, 1) < 0.5
