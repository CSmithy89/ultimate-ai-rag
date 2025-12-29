"""Tests for temporal knowledge API endpoints."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


def _make_mock_node(uuid: str, name: str, summary: str, labels: list):
    """Create a mock node with proper name attribute."""
    node = MagicMock()
    node.uuid = uuid
    node.configure_mock(name=name)
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
    edge.configure_mock(name=name)
    edge.fact = fact
    edge.valid_at = datetime.now(timezone.utc)
    edge.invalid_at = None
    # Explicitly set group_id to None to skip tenant validation
    edge.group_id = None
    return edge


class TestTemporalSearchUnit:
    """Unit tests for temporal search - using mocks directly."""

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create a mock GraphitiClient."""
        client = MagicMock()
        client.client = MagicMock()
        
        search_result = MagicMock()
        search_result.nodes = [
            _make_mock_node("node-1", "FastAPI", "Web framework", ["TechnicalConcept"]),
        ]
        search_result.edges = [
            _make_mock_edge("edge-1", "node-1", "node-2", "USES", "FastAPI uses Starlette"),
        ]
        
        client.client.search = AsyncMock(return_value=search_result)
        mock_episode = MagicMock(
            uuid="ep-1",
            name="Document 1",
            created_at=datetime(2024, 6, 15, tzinfo=timezone.utc),
            entity_references=["node-1"],
            edge_references=["edge-1"],
        )
        # Explicitly set group_id to None to skip tenant validation in tests
        # (MagicMock auto-creates attributes as MagicMock objects otherwise)
        mock_episode.group_id = None
        client.client.get_episodes_by_group_ids = AsyncMock(return_value=[mock_episode])
        client.is_connected = True
        return client

    @pytest.mark.asyncio
    async def test_temporal_query_function(self, mock_graphiti_client):
        """Should execute temporal query via function."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        tenant_id = str(uuid4())
        result = await temporal_search(
            graphiti_client=mock_graphiti_client,
            query="How does FastAPI work?",
            tenant_id=tenant_id,
        )

        assert result is not None
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "FastAPI"
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_temporal_query_with_as_of_date(self, mock_graphiti_client):
        """Should execute temporal query with as_of_date."""
        from agentic_rag_backend.retrieval.temporal_retrieval import temporal_search

        as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
        result = await temporal_search(
            graphiti_client=mock_graphiti_client,
            query="How does FastAPI work?",
            tenant_id=str(uuid4()),
            as_of_date=as_of,
        )

        assert result is not None
        assert result.as_of_date == as_of

    @pytest.mark.asyncio
    async def test_get_changes_function(self, mock_graphiti_client):
        """Should get knowledge changes via function."""
        from agentic_rag_backend.retrieval.temporal_retrieval import get_knowledge_changes

        tenant_id = str(uuid4())
        result = await get_knowledge_changes(
            graphiti_client=mock_graphiti_client,
            tenant_id=tenant_id,
            start_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 6, 30, tzinfo=timezone.utc),
        )

        assert result is not None
        assert len(result.episodes) == 1
        assert result.total_entities_added == 1
        assert result.total_edges_added == 1


class TestTemporalQueryRequestModel:
    """Tests for request/response models."""

    def test_temporal_query_request_model(self):
        """TemporalQueryRequest should validate correctly."""
        from agentic_rag_backend.api.routes.knowledge import TemporalQueryRequest

        tenant_id = uuid4()
        req = TemporalQueryRequest(
            query="test query",
            tenant_id=tenant_id,
        )

        assert req.query == "test query"
        assert req.tenant_id == tenant_id
        assert req.as_of_date is None
        assert req.num_results == 5

    def test_temporal_query_request_with_as_of_date(self):
        """TemporalQueryRequest should accept as_of_date."""
        from agentic_rag_backend.api.routes.knowledge import TemporalQueryRequest

        as_of = datetime(2024, 6, 1, tzinfo=timezone.utc)
        req = TemporalQueryRequest(
            query="test query",
            tenant_id=uuid4(),
            as_of_date=as_of,
            num_results=10,
        )

        assert req.as_of_date == as_of
        assert req.num_results == 10


class TestTemporalEndpointValidation:
    """Tests for API endpoint validation."""

    def test_temporal_query_missing_query_validation(self):
        """Should validate query is required."""
        from pydantic import ValidationError
        from agentic_rag_backend.api.routes.knowledge import TemporalQueryRequest

        with pytest.raises(ValidationError):
            TemporalQueryRequest(
                tenant_id=uuid4(),
            )

    def test_temporal_query_missing_tenant_validation(self):
        """Should validate tenant_id is required."""
        from pydantic import ValidationError
        from agentic_rag_backend.api.routes.knowledge import TemporalQueryRequest

        with pytest.raises(ValidationError):
            TemporalQueryRequest(
                query="test query",
            )
