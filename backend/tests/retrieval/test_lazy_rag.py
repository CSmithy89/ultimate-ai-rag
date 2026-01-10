"""Tests for LazyRAG Query-Time Summarization (Story 20-B2).

This module tests the LazyRAGRetriever class and its components:
- Seed entity finding via Graphiti
- Subgraph expansion via Neo4j
- Community context integration (20-B1)
- LLM summary generation
- Confidence estimation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from agentic_rag_backend.retrieval.lazy_rag import LazyRAGRetriever
from agentic_rag_backend.retrieval.lazy_rag_models import (
    LazyRAGEntity,
    LazyRAGRelationship,
    LazyRAGCommunity,
    LazyRAGResult,
    SubgraphExpansionResult,
    SummaryResult,
)


def _make_mock_node(uuid: str, name: str, summary: str, labels: list, type_str: str = "Entity"):
    """Create a mock Graphiti node."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)
    node.summary = summary
    node.labels = labels
    node.type = type_str
    return node


def _make_mock_settings(
    lazy_rag_enabled: bool = True,
    lazy_rag_max_entities: int = 50,
    lazy_rag_max_hops: int = 2,
    lazy_rag_use_communities: bool = True,
    lazy_rag_summary_model: str = "gpt-4o-mini",
    llm_provider: str = "openai",
    llm_api_key: str = "test-key",
    llm_base_url: str = None,
    embedding_provider: str = "openai",
    embedding_api_key: str = "test-key",
    embedding_base_url: str = None,
    embedding_model: str = "text-embedding-3-small",
):
    """Create mock settings for LazyRAG."""
    settings = MagicMock()
    settings.lazy_rag_enabled = lazy_rag_enabled
    settings.lazy_rag_max_entities = lazy_rag_max_entities
    settings.lazy_rag_max_hops = lazy_rag_max_hops
    settings.lazy_rag_use_communities = lazy_rag_use_communities
    settings.lazy_rag_summary_model = lazy_rag_summary_model
    settings.llm_provider = llm_provider
    settings.llm_api_key = llm_api_key
    settings.llm_base_url = llm_base_url
    settings.embedding_provider = embedding_provider
    settings.embedding_api_key = embedding_api_key
    settings.embedding_base_url = embedding_base_url
    settings.embedding_model = embedding_model
    return settings


class TestLazyRAGEntity:
    """Tests for LazyRAGEntity dataclass."""

    def test_entity_creation(self):
        """Should create entity with all fields."""
        entity = LazyRAGEntity(
            id="entity-1",
            name="FastAPI",
            type="TechnicalConcept",
            description="Modern Python web framework",
            summary="FastAPI summary",
            labels=["TechnicalConcept", "Framework"],
        )

        assert entity.id == "entity-1"
        assert entity.name == "FastAPI"
        assert entity.type == "TechnicalConcept"
        assert entity.description == "Modern Python web framework"
        assert entity.labels == ["TechnicalConcept", "Framework"]

    def test_entity_to_response(self):
        """Should convert to API response model."""
        entity = LazyRAGEntity(
            id="entity-1",
            name="FastAPI",
            type="Framework",
            description="Web framework",
        )

        response = entity.to_response()

        assert response.id == "entity-1"
        assert response.name == "FastAPI"
        assert response.type == "Framework"


class TestLazyRAGRelationship:
    """Tests for LazyRAGRelationship dataclass."""

    def test_relationship_creation(self):
        """Should create relationship with all fields."""
        rel = LazyRAGRelationship(
            source_id="entity-1",
            target_id="entity-2",
            type="USES",
            fact="FastAPI uses Starlette",
            confidence=0.95,
        )

        assert rel.source_id == "entity-1"
        assert rel.target_id == "entity-2"
        assert rel.type == "USES"
        assert rel.fact == "FastAPI uses Starlette"
        assert rel.confidence == 0.95

    def test_relationship_to_response(self):
        """Should convert to API response model."""
        rel = LazyRAGRelationship(
            source_id="entity-1",
            target_id="entity-2",
            type="DEPENDS_ON",
            fact="Test fact",
        )

        response = rel.to_response()

        assert response.source_id == "entity-1"
        assert response.target_id == "entity-2"
        assert response.type == "DEPENDS_ON"


class TestLazyRAGCommunity:
    """Tests for LazyRAGCommunity dataclass."""

    def test_community_creation(self):
        """Should create community with all fields."""
        community = LazyRAGCommunity(
            id="comm-1",
            name="Web Frameworks",
            summary="Collection of web framework concepts",
            keywords=("web", "framework", "api"),
            level=1,
        )

        assert community.id == "comm-1"
        assert community.name == "Web Frameworks"
        assert community.summary == "Collection of web framework concepts"
        assert "web" in community.keywords
        assert community.level == 1

    def test_community_to_response(self):
        """Should convert to API response model."""
        community = LazyRAGCommunity(
            id="comm-1",
            name="Test Community",
            keywords=("test", "example"),
            level=0,
        )

        response = community.to_response()

        assert response.id == "comm-1"
        assert response.name == "Test Community"
        assert isinstance(response.keywords, list)


class TestSubgraphExpansionResult:
    """Tests for SubgraphExpansionResult dataclass."""

    def test_expansion_result_creation(self):
        """Should create expansion result with entities and relationships."""
        entities = [
            LazyRAGEntity(id="e1", name="Entity1"),
            LazyRAGEntity(id="e2", name="Entity2"),
        ]
        relationships = [
            LazyRAGRelationship(source_id="e1", target_id="e2", type="RELATES")
        ]

        result = SubgraphExpansionResult(
            entities=entities,
            relationships=relationships,
            seed_count=1,
            expanded_count=2,
        )

        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.seed_count == 1
        assert result.expanded_count == 2


class TestLazyRAGRetriever:
    """Tests for LazyRAGRetriever class."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient."""
        client = MagicMock()
        client.client = MagicMock()
        client.is_connected = True

        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("node-1", "FastAPI", "Python web framework", ["Entity"]),
            _make_mock_node("node-2", "Python", "Programming language", ["Entity"]),
        ]
        search_result.edges = []

        client.client.search = AsyncMock(return_value=search_result)
        return client

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create a mock Neo4j client."""
        client = MagicMock()
        driver = MagicMock()

        # Mock session context manager
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        # Mock run returns
        expansion_records = [
            {"id": "node-3", "name": "Starlette", "type": "Framework", "description": None, "summary": None},
        ]
        relationship_records = [
            {"source_id": "node-1", "target_id": "node-3", "rel_type": "USES", "fact": "FastAPI uses Starlette"},
        ]

        session.run = AsyncMock()
        session.run.return_value.data = AsyncMock(side_effect=[expansion_records, relationship_records])

        driver.session = MagicMock(return_value=session)
        client.driver = driver

        return client

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        return _make_mock_settings()

    @pytest.fixture
    def retriever(self, mock_graphiti_client, mock_neo4j_client, mock_settings):
        """Create LazyRAGRetriever with mocked dependencies."""
        return LazyRAGRetriever(
            graphiti_client=mock_graphiti_client,
            neo4j_client=mock_neo4j_client,
            settings=mock_settings,
            community_detector=None,
        )

    def test_retriever_initialization(self, retriever, mock_settings):
        """Should initialize with correct settings."""
        assert retriever.max_entities == mock_settings.lazy_rag_max_entities
        assert retriever.max_hops == mock_settings.lazy_rag_max_hops
        assert retriever.use_communities == mock_settings.lazy_rag_use_communities
        assert retriever.summary_model == mock_settings.lazy_rag_summary_model

    @pytest.mark.asyncio
    async def test_find_seed_entities_uses_graphiti(self, retriever, mock_graphiti_client):
        """Should use Graphiti for seed entity search."""
        entities = await retriever._find_seed_entities(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            num_results=5,
        )

        mock_graphiti_client.client.search.assert_called_once()
        call_kwargs = mock_graphiti_client.client.search.call_args[1]
        assert call_kwargs["group_ids"] == ["tenant-1"]
        assert len(entities) == 2

    @pytest.mark.asyncio
    async def test_find_seed_entities_returns_lazy_rag_entities(self, retriever, mock_graphiti_client):
        """Should convert Graphiti nodes to LazyRAGEntity objects."""
        entities = await retriever._find_seed_entities(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            num_results=5,
        )

        assert all(isinstance(e, LazyRAGEntity) for e in entities)
        assert entities[0].name == "FastAPI"
        assert entities[1].name == "Python"

    @pytest.mark.asyncio
    async def test_find_seed_entities_falls_back_when_graphiti_disconnected(self):
        """Should fall back to Neo4j search when Graphiti is disconnected."""
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = False

        mock_neo4j = MagicMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.run = AsyncMock()
        session.run.return_value.data = AsyncMock(return_value=[
            {"id": "e1", "name": "Fallback Entity", "type": "Entity", "description": None, "summary": None}
        ])
        mock_neo4j.driver.session = MagicMock(return_value=session)

        settings = _make_mock_settings()
        retriever = LazyRAGRetriever(
            graphiti_client=mock_graphiti,
            neo4j_client=mock_neo4j,
            settings=settings,
        )

        entities = await retriever._find_seed_entities(
            query="test",
            tenant_id="tenant-1",
        )

        # Should use fallback and return entities from Neo4j
        assert len(entities) == 1
        assert entities[0].name == "Fallback Entity"

    def test_merge_entities_deduplicates(self, retriever):
        """Should merge and deduplicate entities."""
        seed = [
            LazyRAGEntity(id="e1", name="Entity1"),
            LazyRAGEntity(id="e2", name="Entity2"),
        ]
        expanded = [
            LazyRAGEntity(id="e2", name="Entity2 Dup"),  # Duplicate
            LazyRAGEntity(id="e3", name="Entity3"),
        ]

        merged = retriever._merge_entities(seed, expanded)

        assert len(merged) == 3
        # Seed entities come first and take priority
        assert merged[0].id == "e1"
        assert merged[1].id == "e2"
        assert merged[1].name == "Entity2"  # Original seed name preserved
        assert merged[2].id == "e3"

    def test_estimate_confidence_no_entities(self, retriever):
        """Should return 0 confidence for empty entities."""
        confidence = retriever._estimate_confidence(
            query="test query",
            entities=[],
            relationships=[],
        )

        assert confidence == 0.0

    def test_estimate_confidence_with_entities(self, retriever):
        """Should estimate confidence based on entity coverage."""
        entities = [
            LazyRAGEntity(id="e1", name="FastAPI", description="Python web framework"),
            LazyRAGEntity(id="e2", name="Python", description="Programming language"),
        ]
        relationships = [
            LazyRAGRelationship(source_id="e1", target_id="e2", type="USES"),
        ]

        confidence = retriever._estimate_confidence(
            query="What is FastAPI built with?",
            entities=entities,
            relationships=relationships,
        )

        # Should be > 0 since we have entities matching query terms
        assert 0.0 < confidence <= 1.0

    def test_estimate_confidence_increases_with_more_entities(self, retriever):
        """Confidence should increase with more relevant entities."""
        few_entities = [
            LazyRAGEntity(id="e1", name="FastAPI", description="Framework"),
        ]
        many_entities = [
            LazyRAGEntity(id="e1", name="FastAPI", description="Framework"),
            LazyRAGEntity(id="e2", name="Python", description="Language"),
            LazyRAGEntity(id="e3", name="Starlette", description="ASGI toolkit"),
            LazyRAGEntity(id="e4", name="Pydantic", description="Data validation"),
        ]

        conf_few = retriever._estimate_confidence("FastAPI", few_entities, [])
        conf_many = retriever._estimate_confidence("FastAPI", many_entities, [])

        assert conf_many >= conf_few

    def test_format_entities(self, retriever):
        """Should format entities for prompt context."""
        entities = [
            LazyRAGEntity(id="e1", name="FastAPI", type="Framework", description="Python web framework"),
            LazyRAGEntity(id="e2", name="Python", type="Language"),  # No description
        ]

        formatted = retriever._format_entities(entities)

        assert "FastAPI" in formatted
        assert "Framework" in formatted
        assert "Python web framework" in formatted
        assert "Python" in formatted

    def test_format_relationships(self, retriever):
        """Should format relationships for prompt context."""
        relationships = [
            LazyRAGRelationship(source_id="e1", target_id="e2", type="USES", fact="FastAPI uses async"),
        ]

        formatted = retriever._format_relationships(relationships)

        assert "--USES-->" in formatted
        assert "e1" in formatted
        assert "e2" in formatted


class TestLazyRAGResult:
    """Tests for LazyRAGResult dataclass."""

    def test_result_to_response(self):
        """Should convert to API response model."""
        result = LazyRAGResult(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            entities=[LazyRAGEntity(id="e1", name="FastAPI")],
            relationships=[],
            communities=[],
            summary="FastAPI is a Python web framework.",
            confidence=0.85,
            seed_entity_count=1,
            expanded_entity_count=1,
            processing_time_ms=150,
            missing_info=None,
        )

        response = result.to_response()

        assert response.query == "What is FastAPI?"
        assert response.tenant_id == "tenant-1"
        assert response.summary == "FastAPI is a Python web framework."
        assert response.confidence == 0.85
        assert len(response.entities) == 1
        assert response.processing_time_ms == 150


class TestSummaryResult:
    """Tests for SummaryResult dataclass."""

    def test_summary_result_creation(self):
        """Should create summary result with all fields."""
        result = SummaryResult(
            text="This is the summary.",
            confidence=0.9,
            missing_info="Some context is missing.",
        )

        assert result.text == "This is the summary."
        assert result.confidence == 0.9
        assert result.missing_info == "Some context is missing."


class TestLazyRAGRetrieverIntegration:
    """Integration-style tests for LazyRAGRetriever.query()."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client for summary generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "FastAPI is a modern Python web framework."
        return mock_response

    @pytest.mark.asyncio
    async def test_query_without_summary(self):
        """Should execute query without generating summary."""
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = True
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("n1", "FastAPI", "Framework", ["Entity"]),
        ]
        mock_graphiti.client.search = AsyncMock(return_value=search_result)

        mock_neo4j = MagicMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)

        # Mock the Neo4j queries to return empty results for expansion
        run_result = MagicMock()
        run_result.data = AsyncMock(return_value=[])
        session.run = AsyncMock(return_value=run_result)
        mock_neo4j.driver.session = MagicMock(return_value=session)

        settings = _make_mock_settings()
        retriever = LazyRAGRetriever(
            graphiti_client=mock_graphiti,
            neo4j_client=mock_neo4j,
            settings=settings,
        )

        result = await retriever.query(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            include_summary=False,
        )

        assert isinstance(result, LazyRAGResult)
        assert result.query == "What is FastAPI?"
        assert result.tenant_id == "tenant-1"
        assert result.summary is None  # No summary requested
        assert result.seed_entity_count >= 0
        assert result.processing_time_ms >= 0  # Can be 0 for very fast execution

    @pytest.mark.asyncio
    async def test_query_with_parameter_overrides(self):
        """Should respect parameter overrides."""
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = True
        search_result = MagicMock()
        search_result.nodes = []
        mock_graphiti.client.search = AsyncMock(return_value=search_result)

        settings = _make_mock_settings(
            lazy_rag_max_entities=50,
            lazy_rag_max_hops=2,
        )
        retriever = LazyRAGRetriever(
            graphiti_client=mock_graphiti,
            neo4j_client=None,
            settings=settings,
        )

        result = await retriever.query(
            query="test",
            tenant_id="tenant-1",
            max_entities=10,  # Override
            max_hops=1,  # Override
            include_summary=False,
        )

        # Verify the search was called with overridden limits
        call_kwargs = mock_graphiti.client.search.call_args[1]
        # num_results should be min(10, max_entities=10)
        assert call_kwargs["num_results"] <= 10


class TestLazyRAGCommunityContext:
    """Tests for community context integration (20-B1)."""

    @pytest.mark.asyncio
    async def test_get_community_context_when_detector_available(self):
        """Should fetch community context when detector is available."""
        mock_neo4j = MagicMock()
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        session.run = AsyncMock()
        session.run.return_value.data = AsyncMock(return_value=[
            {
                "id": "comm-1",
                "name": "Web Frameworks",
                "summary": "Python web framework community",
                "keywords": ["web", "framework"],
                "level": 1,
            }
        ])
        mock_neo4j.driver.session = MagicMock(return_value=session)

        mock_detector = MagicMock()
        settings = _make_mock_settings(lazy_rag_use_communities=True)

        retriever = LazyRAGRetriever(
            graphiti_client=None,
            neo4j_client=mock_neo4j,
            settings=settings,
            community_detector=mock_detector,
        )

        communities = await retriever._get_community_context(
            entity_ids=["e1", "e2"],
            tenant_id="tenant-1",
        )

        assert len(communities) == 1
        assert communities[0].name == "Web Frameworks"
        assert communities[0].level == 1

    @pytest.mark.asyncio
    async def test_get_community_context_returns_empty_when_no_detector(self):
        """Should return empty list when community detector is not available."""
        settings = _make_mock_settings()
        retriever = LazyRAGRetriever(
            graphiti_client=None,
            neo4j_client=None,
            settings=settings,
            community_detector=None,
        )

        communities = await retriever._get_community_context(
            entity_ids=["e1", "e2"],
            tenant_id="tenant-1",
        )

        assert communities == []


class TestLazyRAGModels:
    """Tests for Pydantic request/response models."""

    def test_query_request_validation(self):
        """Should validate query request fields."""
        from agentic_rag_backend.retrieval.lazy_rag_models import LazyRAGQueryRequest
        from uuid import UUID

        request = LazyRAGQueryRequest(
            query="What is FastAPI?",
            tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
            max_entities=30,
            max_hops=3,
            use_communities=True,
            include_summary=True,
        )

        assert request.query == "What is FastAPI?"
        assert request.max_entities == 30
        assert request.max_hops == 3

    def test_query_request_defaults(self):
        """Should use default values for optional fields."""
        from agentic_rag_backend.retrieval.lazy_rag_models import LazyRAGQueryRequest
        from uuid import UUID

        request = LazyRAGQueryRequest(
            query="Test query",
            tenant_id=UUID("12345678-1234-5678-1234-567812345678"),
        )

        assert request.max_entities == 50
        assert request.max_hops == 2
        assert request.use_communities is True
        assert request.include_summary is True

    def test_query_response_structure(self):
        """Should have all required response fields."""
        from agentic_rag_backend.retrieval.lazy_rag_models import LazyRAGQueryResponse

        response = LazyRAGQueryResponse(
            query="What is FastAPI?",
            tenant_id="tenant-1",
            summary="FastAPI is a framework.",
            confidence=0.85,
            entities=[],
            relationships=[],
            communities=[],
            seed_entity_count=5,
            expanded_entity_count=10,
            processing_time_ms=250,
        )

        assert response.confidence == 0.85
        assert response.processing_time_ms == 250

    def test_status_response_structure(self):
        """Should have all required status fields."""
        from agentic_rag_backend.retrieval.lazy_rag_models import LazyRAGStatusResponse

        status = LazyRAGStatusResponse(
            enabled=True,
            max_entities=50,
            max_hops=2,
            use_communities=True,
            summary_model="gpt-4o-mini",
            community_detection_available=True,
        )

        assert status.enabled is True
        assert status.community_detection_available is True
