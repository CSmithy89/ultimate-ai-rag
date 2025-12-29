"""Tests for temporal query capabilities using Graphiti."""
from agentic_rag_backend.core.errors import Neo4jError

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock


def _make_mock_node(uuid: str, name: str, summary: str, labels: list):
    """Create a mock node with proper name attribute."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)
    node.summary = summary
    node.labels = labels
    return node


def _make_mock_edge(uuid: str, source: str, target: str, name: str, fact: str, 
                   valid_at: datetime = None, invalid_at: datetime = None):
    """Create a mock edge with temporal data."""
    edge = MagicMock()
    edge.uuid = uuid
    edge.source_node_uuid = source
    edge.target_node_uuid = target
    edge.configure_mock(name=name)
    edge.fact = fact
    edge.valid_at = valid_at or datetime.now(timezone.utc)
    edge.invalid_at = invalid_at
    edge.created_at = datetime.now(timezone.utc)
    return edge


class TestTemporalSearch:
    """Tests for temporal search capabilities."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient with temporal search."""
        client = MagicMock()
        client.client = MagicMock()
        
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("node-1", "FastAPI", "Web framework", ["TechnicalConcept"]),
        ]
        search_result.edges = [
            _make_mock_edge(
                "edge-1", "node-1", "node-2", "USES",
                "FastAPI uses Starlette",
                valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]
        
        client.client.search = AsyncMock(return_value=search_result)
        client.is_connected = True
        return client

    @pytest.mark.asyncio
    async def test_temporal_search_with_as_of_date(self, mock_graphiti_client):
        """Should search with point-in-time date."""
        from agentic_rag_backend.retrieval.temporal_retrieval import (
            temporal_search,
            TemporalSearchResult,
        )

        as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
        result = await temporal_search(
            graphiti_client=mock_graphiti_client,
            query="How does FastAPI work?",
            tenant_id="test-tenant",
            as_of_date=as_of,
        )

        assert result is not None
        assert isinstance(result, TemporalSearchResult)
        assert result.as_of_date == as_of
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_temporal_search_uses_tenant_as_group_id(self, mock_graphiti_client):
        """Should use tenant_id as group_ids."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        tenant_id = "my-tenant-456"
        await temporal_search(
            graphiti_client=mock_graphiti_client,
            query="test query",
            tenant_id=tenant_id,
        )

        call_kwargs = mock_graphiti_client.client.search.call_args[1]
        assert call_kwargs.get("group_ids") == [tenant_id]

    @pytest.mark.asyncio
    async def test_temporal_search_not_connected_raises(self):
        """Should raise error if client not connected."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        disconnected_client = MagicMock()
        disconnected_client.is_connected = False

        with pytest.raises(Neo4jError, match="not connected"):
            await temporal_search(
                graphiti_client=disconnected_client,
                query="test query",
                tenant_id="test-tenant",
            )


class TestGetChanges:
    """Tests for getting knowledge changes over time."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient for changes."""
        client = MagicMock()
        client.client = MagicMock()
        
        # Mock get_episodes_by_group_ids
        client.client.get_episodes_by_group_ids = AsyncMock(return_value=[
            MagicMock(
                uuid="ep-1",
                name="Document 1",
                created_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
                entity_references=[],
                edge_references=[],
            ),
            MagicMock(
                uuid="ep-2",
                name="Document 2",
                created_at=datetime(2024, 6, 20, tzinfo=timezone.utc),
                entity_references=["node-1"],
                edge_references=["edge-1"],
            ),
        ])
        client.is_connected = True
        return client

    @pytest.mark.asyncio
    async def test_get_changes_returns_episodes(self, mock_graphiti_client):
        """Should return episode-based changes."""
        from agentic_rag_backend.retrieval.temporal_retrieval import (
            get_knowledge_changes,
            KnowledgeChangesResult,
        )

        start_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 6, 30, tzinfo=timezone.utc)

        result = await get_knowledge_changes(
            graphiti_client=mock_graphiti_client,
            tenant_id="test-tenant",
            start_date=start_date,
            end_date=end_date,
        )

        assert result is not None
        assert isinstance(result, KnowledgeChangesResult)
        assert result.start_date == start_date
        assert result.end_date == end_date

    @pytest.mark.asyncio
    async def test_get_changes_uses_tenant_as_group_id(self, mock_graphiti_client):
        """Should filter by tenant group ID."""
        from agentic_rag_backend.retrieval.temporal_retrieval import get_knowledge_changes

        tenant_id = "tenant-789"
        await get_knowledge_changes(
            graphiti_client=mock_graphiti_client,
            tenant_id=tenant_id,
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
        )

        call_args = mock_graphiti_client.client.get_episodes_by_group_ids.call_args
        assert tenant_id in call_args[1].get("group_ids", call_args[0][0] if call_args[0] else [])

    @pytest.mark.asyncio
    async def test_get_changes_not_connected_raises(self):
        """Should raise error if client not connected."""
        from agentic_rag_backend.retrieval.temporal_retrieval import get_knowledge_changes

        disconnected_client = MagicMock()
        disconnected_client.is_connected = False

        with pytest.raises(Neo4jError, match="not connected"):
            await get_knowledge_changes(
                graphiti_client=disconnected_client,
                tenant_id="test-tenant",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )


class TestTemporalSearchResult:
    """Tests for temporal search result model."""

    def test_temporal_result_structure(self):
        """TemporalSearchResult should have required fields."""
        from agentic_rag_backend.retrieval.temporal_retrieval import (
            TemporalSearchResult,
            TemporalNode,
            TemporalEdge,
        )

        node = TemporalNode(
            uuid="node-1",
            name="Test Concept",
            summary="A test concept",
            labels=["TechnicalConcept"],
        )
        edge = TemporalEdge(
            uuid="edge-1",
            source_node_uuid="node-1",
            target_node_uuid="node-2",
            name="RELATES_TO",
            fact="Test concept relates to other",
            valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            invalid_at=None,
        )
        as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
        result = TemporalSearchResult(
            query="test query",
            tenant_id="test-tenant",
            as_of_date=as_of,
            nodes=[node],
            edges=[edge],
            processing_time_ms=150,
        )

        assert result.query == "test query"
        assert result.tenant_id == "test-tenant"
        assert result.as_of_date == as_of
        assert len(result.nodes) == 1
        assert len(result.edges) == 1


class TestKnowledgeChangesResult:
    """Tests for knowledge changes result model."""

    def test_changes_result_structure(self):
        """KnowledgeChangesResult should have required fields."""
        from agentic_rag_backend.retrieval.temporal_retrieval import (
            KnowledgeChangesResult,
            EpisodeChange,
        )

        episode = EpisodeChange(
            uuid="ep-1",
            name="Document 1",
            created_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
            entities_added=5,
            edges_added=3,
        )
        result = KnowledgeChangesResult(
            tenant_id="test-tenant",
            start_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
            episodes=[episode],
            total_entities_added=5,
            total_edges_added=3,
            processing_time_ms=200,
        )

        assert result.tenant_id == "test-tenant"
        assert len(result.episodes) == 1
        assert result.total_entities_added == 5
        assert result.total_edges_added == 3


class TestTemporalEdgeFiltering:
    """Tests for temporal edge filtering based on valid_at/invalid_at."""

    @pytest.fixture
    def mock_graphiti_client_with_temporal_edges(self):
        """Create a mock GraphitiClient with temporal edge data."""
        client = MagicMock()
        client.client = MagicMock()
        
        # Create edges with different temporal validity
        # Temporal edges for testing
        
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("node-1", "FastAPI", "Web framework", ["TechnicalConcept"]),
        ]
        search_result.edges = [
            # Edge valid from Jan 2024, never invalidated
            _make_mock_edge(
                "edge-1", "node-1", "node-2", "USES", "Active edge",
                valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                invalid_at=None,
            ),
            # Edge that was invalidated before query date
            _make_mock_edge(
                "edge-2", "node-1", "node-3", "DEPRECATED", "Old edge",
                valid_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
                invalid_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
            ),
            # Edge valid_at is AFTER the query date (not yet valid)
            _make_mock_edge(
                "edge-3", "node-1", "node-4", "FUTURE", "Future edge",
                valid_at=datetime(2024, 12, 1, tzinfo=timezone.utc),
                invalid_at=None,
            ),
        ]
        
        client.client.search = AsyncMock(return_value=search_result)
        client.is_connected = True
        return client

    @pytest.mark.asyncio
    async def test_temporal_search_filters_out_invalidated_edges(
        self, mock_graphiti_client_with_temporal_edges
    ):
        """Should filter edges where invalid_at <= as_of_date."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        # Query at June 2024 - edge-2 should be filtered (invalidated in March)
        as_of = datetime(2024, 6, 15, tzinfo=timezone.utc)
        result = await temporal_search(
            graphiti_client=mock_graphiti_client_with_temporal_edges,
            query="test query",
            tenant_id="test-tenant",
            as_of_date=as_of,
        )

        # edge-1: valid (valid_at=Jan, no invalid_at)
        # edge-2: invalid (invalidated in March, before June)
        # edge-3: invalid (valid_at=Dec, after June query)
        assert len(result.edges) == 1
        assert result.edges[0].uuid == "edge-1"

    @pytest.mark.asyncio
    async def test_temporal_search_filters_future_edges(
        self, mock_graphiti_client_with_temporal_edges
    ):
        """Should filter edges where valid_at > as_of_date."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        # Query at Feb 2024 - both edge-2 and edge-3 have issues
        as_of = datetime(2024, 2, 15, tzinfo=timezone.utc)
        result = await temporal_search(
            graphiti_client=mock_graphiti_client_with_temporal_edges,
            query="test query",
            tenant_id="test-tenant",
            as_of_date=as_of,
        )

        # edge-1: valid (valid_at=Jan, before Feb, no invalid)
        # edge-2: valid at this point (valid_at=Jan, invalid_at=March is after Feb)
        # edge-3: invalid (valid_at=Dec, after Feb)
        assert len(result.edges) == 2
        edge_uuids = {e.uuid for e in result.edges}
        assert "edge-1" in edge_uuids
        assert "edge-2" in edge_uuids
        assert "edge-3" not in edge_uuids


class TestTemporalSearchExceptionHandling:
    """Tests for exception handling in temporal_search."""

    @pytest.mark.asyncio
    async def test_temporal_search_exception_is_logged_and_reraised(self):
        """Should log error and re-raise when search fails."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        failing_client = MagicMock()
        failing_client.client = MagicMock()
        failing_client.client.search = AsyncMock(side_effect=ValueError("Temporal search failed"))
        failing_client.is_connected = True

        with pytest.raises(ValueError, match="Temporal search failed"):
            await temporal_search(
                graphiti_client=failing_client,
                query="test query",
                tenant_id="test-tenant",
            )


class TestGetChangesExceptionHandling:
    """Tests for exception handling in get_knowledge_changes."""

    @pytest.mark.asyncio
    async def test_get_changes_exception_is_logged_and_reraised(self):
        """Should log error and re-raise when get_episodes fails."""
        from agentic_rag_backend.retrieval.temporal_retrieval import get_knowledge_changes

        failing_client = MagicMock()
        failing_client.client = MagicMock()
        failing_client.client.get_episodes_by_group_ids = AsyncMock(
            side_effect=ValueError("Episodes fetch failed")
        )
        failing_client.is_connected = True

        with pytest.raises(ValueError, match="Episodes fetch failed"):
            await get_knowledge_changes(
                graphiti_client=failing_client,
                tenant_id="test-tenant",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 12, 31, tzinfo=timezone.utc),
            )
