"""Tests for the Neo4j client module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag_backend.db.neo4j import Neo4jClient
from agentic_rag_backend.core.errors import Neo4jError


class TestNeo4jClient:
    """Tests for Neo4jClient class."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = MagicMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.run = AsyncMock()
        driver.session.return_value = session
        return driver

    @pytest.fixture
    def client(self, mock_driver):
        """Create a Neo4jClient with mocked driver."""
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        client._driver = mock_driver
        return client

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connecting to Neo4j."""
        with patch("agentic_rag_backend.db.neo4j.AsyncGraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver

            client = Neo4jClient(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test",
            )
            await client.connect()

            mock_db.driver.assert_called_once()
            assert client._driver is not None

    @pytest.mark.asyncio
    async def test_disconnect(self, client, mock_driver):
        """Test disconnecting from Neo4j."""
        mock_driver.close = AsyncMock()
        await client.disconnect()
        mock_driver.close.assert_called_once()
        assert client._driver is None

    def test_driver_not_connected_raises_error(self):
        """Test that accessing driver when not connected raises error."""
        client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
        )
        with pytest.raises(Neo4jError):
            _ = client.driver

    @pytest.mark.asyncio
    async def test_create_entity(self, client, mock_driver):
        """Test creating an entity."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value={"id": "test-id", "name": "TestEntity"})
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        result = await client.create_entity(
            entity_id="test-id",
            tenant_id="tenant-1",
            name="TestEntity",
            entity_type="Concept",
            description="A test entity",
        )

        assert result is not None
        session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_valid_type(self, client, mock_driver):
        """Test creating a relationship with valid type."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value=MagicMock())
        session.run = AsyncMock(return_value=mock_result)

        result = await client.create_relationship(
            source_id="source-id",
            target_id="target-id",
            relationship_type="USES",
            tenant_id="tenant-1",
            confidence=0.9,
        )

        assert result is True
        session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_invalid_type(self, client, mock_driver):
        """Test that invalid relationship type is rejected."""
        result = await client.create_relationship(
            source_id="source-id",
            target_id="target-id",
            relationship_type="INVALID_TYPE",
            tenant_id="tenant-1",
            confidence=0.9,
        )

        assert result is False
        mock_driver.session.return_value.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_find_similar_entity(self, client, mock_driver):
        """Test finding a similar entity."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value={"id": "existing-id", "name": "Entity"})
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        result = await client.find_similar_entity(
            tenant_id="tenant-1",
            name="Entity",
            entity_type="Concept",
        )

        assert result is not None
        assert result["id"] == "existing-id"

    @pytest.mark.asyncio
    async def test_find_similar_entity_not_found(self, client, mock_driver):
        """Test finding a similar entity that doesn't exist."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value=None)
        session.run = AsyncMock(return_value=mock_result)

        result = await client.find_similar_entity(
            tenant_id="tenant-1",
            name="NonExistent",
            entity_type="Concept",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_graph_stats(self, client, mock_driver):
        """Test getting graph statistics."""
        session = mock_driver.session.return_value

        # Mock multiple query results
        mock_counts = [10, 5, 20, 15]
        call_count = [0]

        async def mock_single():
            idx = call_count[0]
            call_count[0] += 1
            record = MagicMock()
            record.__getitem__ = MagicMock(return_value=mock_counts[idx] if idx < len(mock_counts) else 0)
            return record

        mock_result = MagicMock()
        mock_result.single = mock_single
        session.run = AsyncMock(return_value=mock_result)

        result = await client.get_graph_stats(tenant_id="tenant-1")

        assert "entity_count" in result
        assert "document_count" in result
        assert "chunk_count" in result
        assert "relationship_count" in result

    @pytest.mark.asyncio
    async def test_create_document_node(self, client, mock_driver):
        """Test creating a document node."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value={"id": "doc-id", "title": "Test Doc"})
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        result = await client.create_document_node(
            document_id="doc-id",
            tenant_id="tenant-1",
            title="Test Document",
            source_type="pdf",
        )

        assert result is not None
        session.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_chunk_node(self, client, mock_driver):
        """Test creating a chunk node."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_record = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value={"id": "chunk-id"})
        mock_result.single = AsyncMock(return_value=mock_record)
        session.run = AsyncMock(return_value=mock_result)

        result = await client.create_chunk_node(
            chunk_id="chunk-id",
            tenant_id="tenant-1",
            document_id="doc-id",
            chunk_index=0,
            preview="This is the preview...",
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_link_chunk_to_entity(self, client, mock_driver):
        """Test linking a chunk to an entity."""
        session = mock_driver.session.return_value
        mock_result = MagicMock()
        mock_result.single = AsyncMock(return_value=MagicMock())
        session.run = AsyncMock(return_value=mock_result)

        result = await client.link_chunk_to_entity(
            chunk_id="chunk-id",
            entity_id="entity-id",
            tenant_id="tenant-1",
        )

        assert result is True
