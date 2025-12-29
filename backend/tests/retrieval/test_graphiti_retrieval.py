"""Tests for Graphiti-based hybrid retrieval integration."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agentic_rag_backend.retrieval_router import RetrievalStrategy


def _make_mock_node(uuid: str, name: str, summary: str, labels: list):
    """Create a mock node with proper name attribute."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)  # Use configure_mock for name attribute
    node.summary = summary
    node.labels = labels
    return node


def _make_mock_edge(uuid: str, source: str, target: str, name: str, fact: str):
    """Create a mock edge with proper name attribute."""
    edge = MagicMock()
    edge.uuid = uuid
    edge.source_node_uuid = source
    edge.target_node_uuid = target
    edge.configure_mock(name=name)  # Use configure_mock for name attribute
    edge.fact = fact
    edge.fact_embedding = [0.1] * 768
    return edge


class TestGraphitiRetrieval:
    """Tests for Graphiti hybrid retrieval service."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient."""
        client = MagicMock()
        client.client = MagicMock()
        
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node(
                "node-1", "FastAPI", "Modern web framework for Python", ["TechnicalConcept"]
            ),
            _make_mock_node(
                "node-2", "async/await", "Asynchronous programming pattern", ["CodePattern"]
            ),
        ]
        search_result.edges = [
            _make_mock_edge(
                "edge-1", "node-1", "node-2", "USES",
                "FastAPI uses async/await for handling requests"
            ),
        ]
        
        client.client.search = AsyncMock(return_value=search_result)
        client.is_connected = True
        return client

    @pytest.mark.asyncio
    async def test_search_returns_results(self, mock_graphiti_client):
        """Should return search results from Graphiti."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import (
            graphiti_search,
            GraphitiSearchResult,
        )

        result = await graphiti_search(
            graphiti_client=mock_graphiti_client,
            query="How does FastAPI handle async?",
            tenant_id="test-tenant",
        )

        assert result is not None
        assert isinstance(result, GraphitiSearchResult)
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_uses_tenant_as_group_id(self, mock_graphiti_client):
        """Should use tenant_id as group_ids for multi-tenancy."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        tenant_id = "my-tenant-123"
        await graphiti_search(
            graphiti_client=mock_graphiti_client,
            query="test query",
            tenant_id=tenant_id,
        )

        call_kwargs = mock_graphiti_client.client.search.call_args[1]
        assert call_kwargs.get("group_ids") == [tenant_id]

    @pytest.mark.asyncio
    async def test_search_respects_num_results(self, mock_graphiti_client):
        """Should pass num_results to Graphiti."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        await graphiti_search(
            graphiti_client=mock_graphiti_client,
            query="test query",
            tenant_id="test-tenant",
            num_results=10,
        )

        call_kwargs = mock_graphiti_client.client.search.call_args[1]
        assert call_kwargs.get("num_results") == 10

    @pytest.mark.asyncio
    async def test_search_not_connected_raises(self):
        """Should raise error if Graphiti client not connected."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        disconnected_client = MagicMock()
        disconnected_client.is_connected = False

        with pytest.raises(RuntimeError, match="not connected"):
            await graphiti_search(
                graphiti_client=disconnected_client,
                query="test query",
                tenant_id="test-tenant",
            )

    @pytest.mark.asyncio
    async def test_search_result_contains_nodes_and_edges(self, mock_graphiti_client):
        """Should include nodes and edges in result."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        result = await graphiti_search(
            graphiti_client=mock_graphiti_client,
            query="test query",
            tenant_id="test-tenant",
        )

        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.nodes[0].name == "FastAPI"
        assert result.edges[0].name == "USES"


class TestRetrievalBackendRouting:
    """Tests for retrieval backend feature flag routing."""

    def test_retrieval_backend_default_is_graphiti(self):
        """Default retrieval backend should be graphiti."""
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test",
            "DATABASE_URL": "postgresql://test",
            "NEO4J_URI": "bolt://localhost",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost",
        }, clear=False):
            from agentic_rag_backend.config import load_settings
            load_settings.cache_clear() if hasattr(load_settings, 'cache_clear') else None
            settings = load_settings()
            assert settings.retrieval_backend == "graphiti"

    def test_retrieval_backend_can_be_legacy(self):
        """Retrieval backend should support legacy value."""
        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "test",
            "DATABASE_URL": "postgresql://test",
            "NEO4J_URI": "bolt://localhost",
            "NEO4J_USER": "neo4j",
            "NEO4J_PASSWORD": "password",
            "REDIS_URL": "redis://localhost",
            "RETRIEVAL_BACKEND": "legacy",
        }, clear=False):
            from agentic_rag_backend.config import load_settings
            load_settings.cache_clear() if hasattr(load_settings, 'cache_clear') else None
            settings = load_settings()
            assert settings.retrieval_backend == "legacy"


class TestGraphitiSearchResult:
    """Tests for search result model."""

    def test_search_result_structure(self):
        """GraphitiSearchResult should have required fields."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import (
            GraphitiSearchResult,
            SearchNode,
            SearchEdge,
        )

        node = SearchNode(
            uuid="node-1",
            name="Test Concept",
            summary="A test concept",
            labels=["TechnicalConcept"],
        )
        edge = SearchEdge(
            uuid="edge-1",
            source_node_uuid="node-1",
            target_node_uuid="node-2",
            name="RELATES_TO",
            fact="Test concept relates to other",
        )
        result = GraphitiSearchResult(
            query="test query",
            tenant_id="test-tenant",
            nodes=[node],
            edges=[edge],
            processing_time_ms=150,
        )

        assert result.query == "test query"
        assert result.tenant_id == "test-tenant"
        assert len(result.nodes) == 1
        assert len(result.edges) == 1
        assert result.processing_time_ms == 150


class TestHybridRetrieval:
    """Tests for hybrid retrieval with backend routing."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient."""
        client = MagicMock()
        client.client = MagicMock()
        
        search_result = MagicMock()
        search_result.nodes = []
        search_result.edges = []
        
        client.client.search = AsyncMock(return_value=search_result)
        client.is_connected = True
        return client

    @pytest.mark.asyncio
    async def test_search_with_backend_routing_uses_graphiti(self, mock_graphiti_client):
        """Should route to Graphiti when backend is graphiti."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import (
            search_with_backend_routing,
        )

        result = await search_with_backend_routing(
            query="test query",
            tenant_id="test-tenant",
            graphiti_client=mock_graphiti_client,
            legacy_retriever=None,
            retrieval_backend="graphiti",
        )

        assert result is not None
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_backend_routing_raises_on_invalid_backend(
        self, mock_graphiti_client
    ):
        """Should raise ValueError on invalid backend."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import (
            search_with_backend_routing,
        )

        with pytest.raises(ValueError, match="Invalid retrieval backend"):
            await search_with_backend_routing(
                query="test query",
                tenant_id="test-tenant",
                graphiti_client=mock_graphiti_client,
                legacy_retriever=None,
                retrieval_backend="invalid",
            )

    @pytest.mark.asyncio
    async def test_search_with_backend_routing_graphiti_not_available(self):
        """Should raise error when graphiti selected but not available."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import (
            search_with_backend_routing,
        )

        with pytest.raises(RuntimeError, match="Graphiti client not available"):
            await search_with_backend_routing(
                query="test query",
                tenant_id="test-tenant",
                graphiti_client=None,
                legacy_retriever=None,
                retrieval_backend="graphiti",
            )
