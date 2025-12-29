"""Tests for temporal query capabilities using Graphiti."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


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

        with pytest.raises(RuntimeError, match="not connected"):
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

        with pytest.raises(RuntimeError, match="not connected"):
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
