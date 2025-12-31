"""Tests for the graph builder module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag_backend.indexing.graph_builder import (
    GraphBuilder,
    create_graph_from_extractions,
)
from agentic_rag_backend.models.graphs import ExtractedEntity, ExtractedRelationship


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create a mock Neo4j client."""
        client = MagicMock()
        client.find_similar_entity = AsyncMock(return_value=None)
        client.create_entity = AsyncMock(return_value={"id": "test-id"})
        client.create_relationship = AsyncMock(return_value=True)
        return client

    @pytest.fixture
    def builder(self, mock_neo4j_client):
        """Create a GraphBuilder with mocked Neo4j client."""
        return GraphBuilder(neo4j=mock_neo4j_client)

    def test_normalize_name(self, builder):
        """Test entity name normalization."""
        assert builder._normalize_name("OpenAI") == "openai"
        assert builder._normalize_name("  GPT-4  ") == "gpt-4"
        assert builder._normalize_name("Multiple   Spaces") == "multiple spaces"

    @pytest.mark.asyncio
    async def test_find_or_create_entity_new(self, builder, mock_neo4j_client):
        """Test creating a new entity."""
        entity = ExtractedEntity(name="TestEntity", type="Concept", description="A test")

        entity_id, is_new = await builder.find_or_create_entity(
            entity=entity,
            tenant_id="11111111-1111-1111-1111-111111111111",
            source_chunk_id="chunk-1",
        )

        assert is_new is True
        assert entity_id is not None
        mock_neo4j_client.create_entity.assert_called()

    @pytest.mark.asyncio
    async def test_find_or_create_entity_existing(self, builder, mock_neo4j_client):
        """Test finding an existing entity."""
        mock_neo4j_client.find_similar_entity = AsyncMock(return_value={"id": "existing-id"})
        entity = ExtractedEntity(name="ExistingEntity", type="Concept")

        entity_id, is_new = await builder.find_or_create_entity(
            entity=entity,
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        assert is_new is False
        assert entity_id == "existing-id"

    @pytest.mark.asyncio
    async def test_find_or_create_entity_cached(self, builder, mock_neo4j_client):
        """Test that entities are cached locally."""
        entity = ExtractedEntity(name="CachedEntity", type="Concept")

        # First call
        entity_id1, is_new1 = await builder.find_or_create_entity(
            entity=entity,
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        # Second call should use cache
        entity_id2, is_new2 = await builder.find_or_create_entity(
            entity=entity,
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        assert entity_id1 == entity_id2
        assert is_new1 is True
        assert is_new2 is False  # Second call finds it in cache

    @pytest.mark.asyncio
    async def test_create_relationship_success(self, builder, mock_neo4j_client):
        """Test creating a relationship."""
        relationship = ExtractedRelationship(
            source="A",
            target="B",
            type="RELATED_TO",
            confidence=0.9,
        )
        entity_name_to_id = {"a": "id-a", "b": "id-b"}

        result = await builder.create_relationship(
            relationship=relationship,
            tenant_id="11111111-1111-1111-1111-111111111111",
            entity_name_to_id=entity_name_to_id,
        )

        assert result is True
        mock_neo4j_client.create_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_relationship_missing_entity(self, builder, mock_neo4j_client):
        """Test that relationship creation fails if entities are missing."""
        relationship = ExtractedRelationship(
            source="A",
            target="Missing",
            type="RELATED_TO",
            confidence=0.9,
        )
        entity_name_to_id = {"a": "id-a"}

        result = await builder.create_relationship(
            relationship=relationship,
            tenant_id="11111111-1111-1111-1111-111111111111",
            entity_name_to_id=entity_name_to_id,
        )

        assert result is False
        mock_neo4j_client.create_relationship.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_graph(self, builder, mock_neo4j_client):
        """Test building a graph from entities and relationships."""
        entities = [
            ExtractedEntity(name="EntityA", type="Concept"),
            ExtractedEntity(name="EntityB", type="Technology"),
        ]
        relationships = [
            ExtractedRelationship(
                source="EntityA",
                target="EntityB",
                type="USES",
                confidence=0.85,
            )
        ]

        result = await builder.build_graph(
            entities=entities,
            relationships=relationships,
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        assert result.entities_created == 2
        assert result.entities_deduplicated == 0
        assert result.relationships_created == 1
        assert result.relationships_skipped == 0

    @pytest.mark.asyncio
    async def test_build_graph_with_duplicates(self, builder, mock_neo4j_client):
        """Test building a graph with duplicate entities."""
        entities = [
            ExtractedEntity(name="SameEntity", type="Concept"),
            ExtractedEntity(name="SameEntity", type="Concept"),  # Duplicate
            ExtractedEntity(name="Different", type="Technology"),
        ]

        result = await builder.build_graph(
            entities=entities,
            relationships=[],
            tenant_id="11111111-1111-1111-1111-111111111111",
        )

        # Should only create 2 unique entities
        assert result.entities_created == 2

    def test_clear_cache(self, builder):
        """Test clearing the entity cache."""
        builder._entity_cache["test:Concept"] = "id-1"
        assert len(builder._entity_cache) == 1

        builder.clear_cache()

        assert len(builder._entity_cache) == 0


@pytest.mark.asyncio
async def test_create_graph_from_extractions():
    """Test batch graph creation from multiple extractions."""
    mock_neo4j = MagicMock()
    mock_neo4j.find_similar_entity = AsyncMock(return_value=None)
    mock_neo4j.create_entity = AsyncMock(return_value={"id": "test-id"})
    mock_neo4j.create_relationship = AsyncMock(return_value=True)

    extractions = [
        {
            "chunk_id": "chunk-1",
            "entities": [
                ExtractedEntity(name="Entity1", type="Concept"),
            ],
            "relationships": [],
        },
        {
            "chunk_id": "chunk-2",
            "entities": [
                ExtractedEntity(name="Entity2", type="Technology"),
            ],
            "relationships": [],
        },
    ]

    result = await create_graph_from_extractions(
        neo4j=mock_neo4j,
        extractions=extractions,
        tenant_id="11111111-1111-1111-1111-111111111111",
    )

    # Should create entities from both extractions
    assert result.entities_created >= 2
