"""Tests for Graphiti-based hybrid retrieval integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag_backend.core.errors import Neo4jError



def _make_mock_node(uuid: str, name: str, summary: str, labels: list):
    """Create a mock node with proper name attribute."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)  # Use configure_mock for name attribute
    node.summary = summary
    node.labels = labels
    # Explicitly set group_id to None to skip tenant validation
    node.group_id = None
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
    # Explicitly set group_id to None to skip tenant validation
    edge.group_id = None
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
            tenant_id="99999999-9999-9999-9999-999999999999",
        )

        assert result is not None
        assert isinstance(result, GraphitiSearchResult)
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_uses_tenant_as_group_id(self, mock_graphiti_client):
        """Should use tenant_id as group_ids for multi-tenancy."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        tenant_id = "my-11111111-1111-1111-1111-11111111111123"
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
            tenant_id="99999999-9999-9999-9999-999999999999",
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

        with pytest.raises(Neo4jError, match="not connected"):
            await graphiti_search(
                graphiti_client=disconnected_client,
                query="test query",
                tenant_id="99999999-9999-9999-9999-999999999999",
            )

    @pytest.mark.asyncio
    async def test_search_result_contains_nodes_and_edges(self, mock_graphiti_client):
        """Should include nodes and edges in result."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        result = await graphiti_search(
            graphiti_client=mock_graphiti_client,
            query="test query",
            tenant_id="99999999-9999-9999-9999-999999999999",
        )

        assert len(result.nodes) == 2
        assert len(result.edges) == 1
        assert result.nodes[0].name == "FastAPI"
        assert result.edges[0].name == "USES"


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
            tenant_id="99999999-9999-9999-9999-999999999999",
            nodes=[node],
            edges=[edge],
            processing_time_ms=150,
        )

        assert result.query == "test query"
        assert result.tenant_id == "99999999-9999-9999-9999-999999999999"
        assert len(result.nodes) == 1
        assert len(result.edges) == 1
        assert result.processing_time_ms == 150


class TestGraphitiSearchExceptionHandling:
    """Tests for exception handling in graphiti_search."""

    @pytest.mark.asyncio
    async def test_search_exception_is_logged_and_reraised(self):
        """Should log error and re-raise when search fails."""
        from agentic_rag_backend.retrieval.graphiti_retrieval import graphiti_search

        failing_client = MagicMock()
        failing_client.client = MagicMock()
        failing_client.client.search = AsyncMock(side_effect=ValueError("Search failed"))
        failing_client.is_connected = True

        with pytest.raises(ValueError, match="Search failed"):
            await graphiti_search(
                graphiti_client=failing_client,
                query="test query",
                tenant_id="99999999-9999-9999-9999-999999999999",
            )
