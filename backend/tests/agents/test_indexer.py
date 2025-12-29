"""Tests for the IndexerAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from agentic_rag_backend.agents.indexer import IndexerAgent, TrajectoryEntry
from agentic_rag_backend.models.graphs import ExtractionResult, ExtractedEntity
from agentic_rag_backend.indexing.embeddings import EMBEDDING_DIMENSION


class TestIndexerAgent:
    """Tests for IndexerAgent class."""

    @pytest.fixture
    def mock_postgres(self):
        """Create a mock PostgreSQL client."""
        client = MagicMock()
        client.create_chunk = AsyncMock(return_value=uuid4())
        client.update_job_status = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def mock_neo4j(self):
        """Create a mock Neo4j client."""
        client = MagicMock()
        client.find_similar_entity = AsyncMock(return_value=None)
        client.create_entity = AsyncMock(return_value={"id": "test-id"})
        client.create_relationship = AsyncMock(return_value=True)
        client.create_document_node = AsyncMock(return_value={})
        client.create_chunk_node = AsyncMock(return_value={})
        client.link_chunk_to_entity = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def mock_embedding_generator(self):
        """Create a mock embedding generator."""
        generator = MagicMock()
        # Return enough embeddings for any number of chunks
        def mock_embeddings(texts):
            return [[0.1] * EMBEDDING_DIMENSION for _ in texts]
        generator.generate_embeddings = AsyncMock(side_effect=mock_embeddings)
        return generator

    @pytest.fixture
    def mock_entity_extractor(self):
        """Create a mock entity extractor."""
        extractor = MagicMock()
        extractor.extract_from_chunk = AsyncMock(return_value=ExtractionResult(
            chunk_id="test-chunk",
            entities=[ExtractedEntity(name="TestEntity", type="Concept")],
            relationships=[],
            processing_time_ms=100,
        ))
        return extractor

    @pytest.fixture
    def agent(self, mock_postgres, mock_neo4j, mock_embedding_generator, mock_entity_extractor):
        """Create an IndexerAgent with mocked dependencies."""
        return IndexerAgent(
            postgres=mock_postgres,
            neo4j=mock_neo4j,
            embedding_generator=mock_embedding_generator,
            entity_extractor=mock_entity_extractor,
            chunk_size=512,
            chunk_overlap=64,
        )

    def test_normalize_entity_name(self, agent):
        """Test entity name normalization."""
        assert agent._normalize_entity_name("OpenAI") == "openai"
        assert agent._normalize_entity_name("  GPT-4  ") == "gpt-4"
        assert agent._normalize_entity_name("Multiple   Spaces") == "multiple spaces"

    def test_log_thought(self, agent):
        """Test logging a thought."""
        agent._start_trajectory("doc-1", "tenant-1")
        agent.log_thought("Test thought", key="value")

        trajectory = agent.get_trajectory()
        assert len(trajectory.entries) == 1
        assert trajectory.entries[0].entry_type == "thought"
        assert "Test thought" in trajectory.entries[0].content

    def test_log_action(self, agent):
        """Test logging an action."""
        agent._start_trajectory("doc-1", "tenant-1")
        agent.log_action("test_action", {"param": "value"})

        trajectory = agent.get_trajectory()
        assert len(trajectory.entries) == 1
        assert trajectory.entries[0].entry_type == "action"
        assert trajectory.entries[0].content == "test_action"

    def test_log_observation(self, agent):
        """Test logging an observation."""
        agent._start_trajectory("doc-1", "tenant-1")
        agent.log_observation("Test observation", result="success")

        trajectory = agent.get_trajectory()
        assert len(trajectory.entries) == 1
        assert trajectory.entries[0].entry_type == "observation"

    @pytest.mark.asyncio
    async def test_deduplicate_entity_new(self, agent, mock_neo4j):
        """Test deduplication creates new entity when none exists."""
        agent._start_trajectory("doc-1", "tenant-1")
        entity = ExtractedEntity(name="NewEntity", type="Concept")

        entity_id, is_new = await agent._deduplicate_entity(entity, "tenant-1")

        assert is_new is True
        assert entity_id is not None

    @pytest.mark.asyncio
    async def test_deduplicate_entity_existing(self, agent, mock_neo4j):
        """Test deduplication finds existing entity."""
        mock_neo4j.find_similar_entity = AsyncMock(return_value={"id": "existing-id"})
        agent._start_trajectory("doc-1", "tenant-1")
        entity = ExtractedEntity(name="ExistingEntity", type="Concept")

        entity_id, is_new = await agent._deduplicate_entity(entity, "tenant-1")

        assert is_new is False
        assert entity_id == "existing-id"

    @pytest.mark.asyncio
    async def test_deduplicate_entity_cached(self, agent, mock_neo4j):
        """Test deduplication uses cache for repeated entities."""
        agent._start_trajectory("doc-1", "tenant-1")
        entity = ExtractedEntity(name="CachedEntity", type="Concept")

        # First call
        entity_id1, is_new1 = await agent._deduplicate_entity(entity, "tenant-1")

        # Second call should use cache
        entity_id2, is_new2 = await agent._deduplicate_entity(entity, "tenant-1")

        assert entity_id1 == entity_id2
        assert is_new1 is True
        assert is_new2 is False

    @pytest.mark.asyncio
    async def test_index_document_empty_content(self, agent):
        """Test indexing with empty content returns empty result."""
        result = await agent.index_document(
            document_id=uuid4(),
            tenant_id=uuid4(),
            content="",
        )

        assert result.chunks_created == 0
        assert result.entities_extracted == 0

    @pytest.mark.asyncio
    async def test_index_document_creates_chunks(self, agent, mock_postgres, mock_neo4j):
        """Test that indexing creates chunks."""
        # Create content
        content = "This is a test document with some content for chunking."

        result = await agent.index_document(
            document_id=uuid4(),
            tenant_id=uuid4(),
            content=content,
            metadata={"title": "Test Document"},
        )

        # Should create at least one chunk
        assert result.chunks_created >= 1
        assert result.processing_time_ms >= 0
        mock_postgres.create_chunk.assert_called()
        mock_neo4j.create_document_node.assert_called()


class TestTrajectoryEntry:
    """Tests for TrajectoryEntry dataclass."""

    def test_create_trajectory_entry(self):
        """Test creating a trajectory entry."""
        from datetime import datetime, timezone

        entry = TrajectoryEntry(
            timestamp=datetime.now(timezone.utc),
            entry_type="thought",
            content="Test content",
            metadata={"key": "value"},
        )

        assert entry.entry_type == "thought"
        assert entry.content == "Test content"
        assert entry.metadata == {"key": "value"}
