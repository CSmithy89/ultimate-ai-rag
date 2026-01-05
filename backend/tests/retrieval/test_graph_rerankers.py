"""Tests for graph-based rerankers (Story 20-C1).

Comprehensive test coverage for:
- GraphRerankerType enum
- GraphContext and GraphRerankedResult dataclasses
- EpisodeMentionsReranker
- NodeDistanceReranker
- HybridGraphReranker
- Factory function and adapter
- Multi-tenancy enforcement
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag_backend.retrieval.graph_rerankers import (
    GraphRerankerType,
    GraphContext,
    GraphRerankedResult,
    GraphRerankerAdapter,
    EpisodeMentionsReranker,
    NodeDistanceReranker,
    HybridGraphReranker,
    get_graph_reranker_adapter,
    create_graph_reranker,
)


class TestGraphRerankerType:
    """Tests for GraphRerankerType enum."""

    def test_episode_type(self):
        """Should have episode type."""
        assert GraphRerankerType.EPISODE.value == "episode"

    def test_distance_type(self):
        """Should have distance type."""
        assert GraphRerankerType.DISTANCE.value == "distance"

    def test_hybrid_type(self):
        """Should have hybrid type."""
        assert GraphRerankerType.HYBRID.value == "hybrid"

    def test_from_string(self):
        """Should create enum from string value."""
        assert GraphRerankerType("episode") == GraphRerankerType.EPISODE
        assert GraphRerankerType("distance") == GraphRerankerType.DISTANCE
        assert GraphRerankerType("hybrid") == GraphRerankerType.HYBRID

    def test_invalid_type_raises(self):
        """Should raise ValueError for invalid type."""
        with pytest.raises(ValueError):
            GraphRerankerType("invalid")


class TestGraphContext:
    """Tests for GraphContext dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        context = GraphContext()
        assert context.episode_mentions == 0
        assert context.min_distance is None
        assert context.episode_score == 0.0
        assert context.distance_score == 0.0
        assert context.query_entities == []
        assert context.result_entities == []

    def test_to_dict(self):
        """Should convert to dictionary."""
        context = GraphContext(
            episode_mentions=5,
            min_distance=2,
            episode_score=0.5,
            distance_score=0.67,
            query_entities=["entity-1"],
            result_entities=["entity-2", "entity-3"],
        )
        result = context.to_dict()

        assert result["episode_mentions"] == 5
        assert result["min_distance"] == 2
        assert result["episode_score"] == 0.5
        assert result["distance_score"] == 0.67
        assert result["query_entities"] == ["entity-1"]
        assert result["result_entities"] == ["entity-2", "entity-3"]


class TestGraphRerankedResult:
    """Tests for GraphRerankedResult dataclass."""

    def test_to_dict_merges_original(self):
        """Should merge original result with graph metadata."""
        original = {
            "id": "chunk-1",
            "content": "Test content",
            "score": 0.8,
        }
        context = GraphContext(episode_mentions=3, episode_score=0.3)

        result = GraphRerankedResult(
            original_result=original,
            original_score=0.8,
            graph_score=0.3,
            combined_score=0.65,
            graph_context=context,
        )

        output = result.to_dict()

        assert output["id"] == "chunk-1"
        assert output["content"] == "Test content"
        assert output["score"] == 0.65  # Combined score
        assert output["original_score"] == 0.8
        assert output["graph_score"] == 0.3
        assert "graph_context" in output
        assert output["graph_context"]["episode_mentions"] == 3


class TestGraphRerankerAdapter:
    """Tests for GraphRerankerAdapter configuration."""

    def test_frozen_dataclass(self):
        """Should be immutable."""
        adapter = GraphRerankerAdapter(
            enabled=True,
            reranker_type=GraphRerankerType.HYBRID,
            episode_weight=0.3,
            distance_weight=0.3,
            original_weight=0.4,
            episode_window_days=30,
            max_distance=3,
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            adapter.enabled = False


class TestEpisodeMentionsReranker:
    """Tests for EpisodeMentionsReranker."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        client = MagicMock()
        client.driver = MagicMock()
        return client

    @pytest.fixture
    def reranker(self, mock_neo4j_client):
        """Create reranker instance."""
        return EpisodeMentionsReranker(
            neo4j_client=mock_neo4j_client,
            episode_window_days=30,
            original_weight=0.7,
        )

    def test_extract_entities_from_entity_ids(self, reranker):
        """Should extract from entity_ids field."""
        result = {"entity_ids": ["id-1", "id-2"]}
        entities = reranker._extract_entities(result)
        assert set(entities) == {"id-1", "id-2"}

    def test_extract_entities_from_entities_list(self, reranker):
        """Should extract from entities list with dicts."""
        result = {
            "entities": [
                {"id": "id-1", "name": "Entity 1"},
                {"uuid": "id-2", "name": "Entity 2"},
            ]
        }
        entities = reranker._extract_entities(result)
        assert set(entities) == {"id-1", "id-2"}

    def test_extract_entities_from_metadata(self, reranker):
        """Should extract from metadata.entity_refs."""
        result = {
            "metadata": {
                "entity_refs": ["ref-1", "ref-2"]
            }
        }
        entities = reranker._extract_entities(result)
        assert set(entities) == {"ref-1", "ref-2"}

    def test_extract_entities_deduplicates(self, reranker):
        """Should deduplicate entities from multiple sources."""
        result = {
            "entity_ids": ["id-1"],
            "entities": [{"id": "id-1"}],
            "metadata": {"entity_refs": ["id-1"]},
        }
        entities = reranker._extract_entities(result)
        assert len(entities) == 1
        assert entities[0] == "id-1"

    def test_normalize_episode_score_zero(self, reranker):
        """Should return 0 for zero mentions."""
        assert reranker._normalize_episode_score(0) == 0.0

    def test_normalize_episode_score_max(self, reranker):
        """Should return 1.0 for 10+ mentions."""
        assert reranker._normalize_episode_score(10) == 1.0
        assert reranker._normalize_episode_score(15) == 1.0

    def test_normalize_episode_score_partial(self, reranker):
        """Should return proportional score for partial mentions."""
        assert reranker._normalize_episode_score(5) == 0.5

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, reranker):
        """Should return empty list for empty input."""
        result = await reranker.rerank("test query", [], "tenant-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_preserves_results(self, mock_neo4j_client, reranker):
        """Should include all original results in output."""
        # Mock session with 0 episode mentions
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"mention_count": 0})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        results = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.8},
        ]

        reranked = await reranker.rerank("test query", results, "tenant-1")

        assert len(reranked) == 2
        assert all(isinstance(r, GraphRerankedResult) for r in reranked)

    @pytest.mark.asyncio
    async def test_rerank_with_tenant_id(self, mock_neo4j_client, reranker):
        """Should pass tenant_id to Neo4j queries."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"mention_count": 5})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        results = [{"id": "1", "score": 0.9, "entity_ids": ["entity-1"]}]
        await reranker.rerank("test query", results, "tenant-123")

        # Verify tenant_id was passed to the query
        call_kwargs = mock_session.run.call_args[1]
        assert call_kwargs.get("tenant_id") == "tenant-123"


class TestNodeDistanceReranker:
    """Tests for NodeDistanceReranker."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        client = MagicMock()
        client.driver = MagicMock()
        client.search_entities_by_terms = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create mock Graphiti client."""
        client = MagicMock()
        client.is_connected = True

        # Mock search result
        mock_search_result = MagicMock()
        mock_node = MagicMock()
        mock_node.uuid = "query-entity-1"
        mock_search_result.nodes = [mock_node]

        client.client = MagicMock()
        client.client.search = AsyncMock(return_value=mock_search_result)
        return client

    @pytest.fixture
    def reranker(self, mock_neo4j_client, mock_graphiti_client):
        """Create reranker instance."""
        return NodeDistanceReranker(
            neo4j_client=mock_neo4j_client,
            graphiti_client=mock_graphiti_client,
            max_distance=3,
            original_weight=0.7,
        )

    def test_distance_to_score_none(self, reranker):
        """Should return 0 for None distance."""
        assert reranker._distance_to_score(None) == 0.0

    def test_distance_to_score_zero(self, reranker):
        """Should return 1.0 for distance 0 (same node)."""
        assert reranker._distance_to_score(0) == 1.0

    def test_distance_to_score_max(self, reranker):
        """Should return 0 for distance at or beyond max."""
        assert reranker._distance_to_score(3) == 0.0
        assert reranker._distance_to_score(5) == 0.0

    def test_distance_to_score_partial(self, reranker):
        """Should return proportional score."""
        # distance=1 with max=3: 1 - 1/3 = 0.667
        assert reranker._distance_to_score(1) == pytest.approx(0.667, rel=0.01)
        # distance=2 with max=3: 1 - 2/3 = 0.333
        assert reranker._distance_to_score(2) == pytest.approx(0.333, rel=0.01)

    @pytest.mark.asyncio
    async def test_extract_query_entities_with_graphiti(
        self, mock_neo4j_client, mock_graphiti_client, reranker
    ):
        """Should use Graphiti client when available."""
        entities = await reranker._extract_query_entities("test query", "tenant-1")

        assert "query-entity-1" in entities
        mock_graphiti_client.client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_query_entities_fallback(self, mock_neo4j_client):
        """Should fall back to Neo4j when Graphiti unavailable."""
        mock_neo4j_client.search_entities_by_terms = AsyncMock(
            return_value=[{"id": "fallback-entity"}]
        )

        reranker = NodeDistanceReranker(
            neo4j_client=mock_neo4j_client,
            graphiti_client=None,
            max_distance=3,
            original_weight=0.7,
        )

        entities = await reranker._extract_query_entities("test query", "tenant-1")

        assert "fallback-entity" in entities
        mock_neo4j_client.search_entities_by_terms.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, reranker):
        """Should return empty list for empty input."""
        result = await reranker.rerank("test query", [], "tenant-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_preserves_order_when_no_entities(
        self, mock_neo4j_client, mock_graphiti_client
    ):
        """Should preserve order when no query entities found."""
        # Configure Graphiti to return no entities
        mock_search_result = MagicMock()
        mock_search_result.nodes = []
        mock_graphiti_client.client.search = AsyncMock(return_value=mock_search_result)

        reranker = NodeDistanceReranker(
            neo4j_client=mock_neo4j_client,
            graphiti_client=mock_graphiti_client,
        )

        results = [
            {"id": "1", "score": 0.9},
            {"id": "2", "score": 0.8},
        ]

        reranked = await reranker.rerank("test query", results, "tenant-1")

        # Original order preserved (no graph signal to change it)
        assert reranked[0].original_result["id"] == "1"
        assert reranked[1].original_result["id"] == "2"


class TestHybridGraphReranker:
    """Tests for HybridGraphReranker."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        client = MagicMock()
        client.driver = MagicMock()
        client.search_entities_by_terms = AsyncMock(return_value=[])
        return client

    @pytest.fixture
    def mock_graphiti_client(self):
        """Create mock Graphiti client."""
        client = MagicMock()
        client.is_connected = True

        mock_search_result = MagicMock()
        mock_search_result.nodes = []
        client.client = MagicMock()
        client.client.search = AsyncMock(return_value=mock_search_result)
        return client

    def test_weight_normalization(self, mock_neo4j_client):
        """Should normalize weights that don't sum to 1.0."""
        reranker = HybridGraphReranker(
            neo4j_client=mock_neo4j_client,
            episode_weight=0.5,
            distance_weight=0.5,
            original_weight=0.5,
        )

        total = (
            reranker._episode_weight
            + reranker._distance_weight
            + reranker._original_weight
        )

        assert abs(total - 1.0) < 0.01

    def test_valid_weights(self, mock_neo4j_client):
        """Should preserve valid weights."""
        reranker = HybridGraphReranker(
            neo4j_client=mock_neo4j_client,
            episode_weight=0.3,
            distance_weight=0.3,
            original_weight=0.4,
        )

        assert reranker._episode_weight == pytest.approx(0.3)
        assert reranker._distance_weight == pytest.approx(0.3)
        assert reranker._original_weight == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, mock_neo4j_client, mock_graphiti_client):
        """Should return empty list for empty input."""
        reranker = HybridGraphReranker(
            neo4j_client=mock_neo4j_client,
            graphiti_client=mock_graphiti_client,
        )

        result = await reranker.rerank("test query", [], "tenant-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_combines_scores(self, mock_neo4j_client, mock_graphiti_client):
        """Should combine episode and distance scores."""
        # Mock sessions for both sub-rerankers
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"mention_count": 0})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        reranker = HybridGraphReranker(
            neo4j_client=mock_neo4j_client,
            graphiti_client=mock_graphiti_client,
            episode_weight=0.3,
            distance_weight=0.3,
            original_weight=0.4,
        )

        results = [{"id": "1", "score": 0.9}]
        reranked = await reranker.rerank("test query", results, "tenant-1")

        assert len(reranked) == 1
        result = reranked[0]

        # Should have graph context from both sub-rerankers
        assert hasattr(result.graph_context, "episode_score")
        assert hasattr(result.graph_context, "distance_score")


class TestFactoryFunctionAndAdapter:
    """Tests for factory function and adapter creation."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = MagicMock()
        settings.graph_reranker_enabled = True
        settings.graph_reranker_type = "hybrid"
        settings.graph_reranker_episode_weight = 0.3
        settings.graph_reranker_distance_weight = 0.3
        settings.graph_reranker_original_weight = 0.4
        settings.graph_reranker_episode_window_days = 30
        settings.graph_reranker_max_distance = 3
        return settings

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        client = MagicMock()
        client.driver = MagicMock()
        return client

    def test_get_graph_reranker_adapter(self, mock_settings):
        """Should create adapter from settings."""
        adapter = get_graph_reranker_adapter(mock_settings)

        assert adapter.enabled is True
        assert adapter.reranker_type == GraphRerankerType.HYBRID
        assert adapter.episode_weight == 0.3
        assert adapter.distance_weight == 0.3
        assert adapter.original_weight == 0.4
        assert adapter.episode_window_days == 30
        assert adapter.max_distance == 3

    def test_get_graph_reranker_adapter_invalid_type(self, mock_settings):
        """Should fall back to hybrid for invalid type."""
        mock_settings.graph_reranker_type = "invalid"
        adapter = get_graph_reranker_adapter(mock_settings)

        assert adapter.reranker_type == GraphRerankerType.HYBRID

    def test_create_graph_reranker_episode(self, mock_neo4j_client):
        """Should create EpisodeMentionsReranker."""
        adapter = GraphRerankerAdapter(
            enabled=True,
            reranker_type=GraphRerankerType.EPISODE,
            episode_weight=0.3,
            distance_weight=0.3,
            original_weight=0.4,
            episode_window_days=30,
            max_distance=3,
        )

        reranker = create_graph_reranker(adapter, mock_neo4j_client)
        assert isinstance(reranker, EpisodeMentionsReranker)

    def test_create_graph_reranker_distance(self, mock_neo4j_client):
        """Should create NodeDistanceReranker."""
        adapter = GraphRerankerAdapter(
            enabled=True,
            reranker_type=GraphRerankerType.DISTANCE,
            episode_weight=0.3,
            distance_weight=0.3,
            original_weight=0.4,
            episode_window_days=30,
            max_distance=3,
        )

        reranker = create_graph_reranker(adapter, mock_neo4j_client)
        assert isinstance(reranker, NodeDistanceReranker)

    def test_create_graph_reranker_hybrid(self, mock_neo4j_client):
        """Should create HybridGraphReranker."""
        adapter = GraphRerankerAdapter(
            enabled=True,
            reranker_type=GraphRerankerType.HYBRID,
            episode_weight=0.3,
            distance_weight=0.3,
            original_weight=0.4,
            episode_window_days=30,
            max_distance=3,
        )

        reranker = create_graph_reranker(adapter, mock_neo4j_client)
        assert isinstance(reranker, HybridGraphReranker)


class TestMultiTenancy:
    """Tests for multi-tenancy enforcement."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client with query tracking."""
        client = MagicMock()
        client.driver = MagicMock()
        client.search_entities_by_terms = AsyncMock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_episode_reranker_tenant_filtering(self, mock_neo4j_client):
        """Should include tenant_id in episode count queries."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"mention_count": 5})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        reranker = EpisodeMentionsReranker(mock_neo4j_client)

        results = [{"id": "1", "score": 0.9, "entity_ids": ["entity-1"]}]
        await reranker.rerank("query", results, "tenant-abc")

        # Verify query contained tenant_id parameter
        call_args = mock_session.run.call_args
        query = call_args[0][0]
        kwargs = call_args[1]

        assert "tenant_id" in kwargs
        assert kwargs["tenant_id"] == "tenant-abc"
        assert "tenant_id" in query

    @pytest.mark.asyncio
    async def test_distance_reranker_tenant_filtering(self, mock_neo4j_client):
        """Should include tenant_id in distance queries."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        # Mock graphiti to return query entities
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = True
        mock_search_result = MagicMock()
        mock_node = MagicMock()
        mock_node.uuid = "query-entity-1"
        mock_search_result.nodes = [mock_node]
        mock_graphiti.client = MagicMock()
        mock_graphiti.client.search = AsyncMock(return_value=mock_search_result)

        reranker = NodeDistanceReranker(mock_neo4j_client, mock_graphiti)

        results = [{"id": "1", "score": 0.9, "entity_ids": ["result-entity-1"]}]
        await reranker.rerank("query", results, "tenant-xyz")

        # Verify graphiti search used tenant_id as group_ids
        graphiti_call = mock_graphiti.client.search.call_args[1]
        assert graphiti_call.get("group_ids") == ["tenant-xyz"]


class TestGracefulFallback:
    """Tests for graceful fallback when no graph context available."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        client = MagicMock()
        client.driver = MagicMock()
        client.search_entities_by_terms = AsyncMock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_episode_reranker_no_entities(self, mock_neo4j_client):
        """Should use 0 episode score when no entities in result."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"mention_count": 0})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        reranker = EpisodeMentionsReranker(mock_neo4j_client, original_weight=0.7)

        # Result with no entities
        results = [{"id": "1", "score": 0.9}]
        reranked = await reranker.rerank("query", results, "tenant-1")

        assert len(reranked) == 1
        result = reranked[0]

        # Graph score should be 0
        assert result.graph_score == 0.0
        # Combined score should just be weighted original
        assert result.combined_score == pytest.approx(0.9 * 0.7)

    @pytest.mark.asyncio
    async def test_distance_reranker_no_path(self, mock_neo4j_client):
        """Should use 0 distance score when no path exists."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)  # No path found
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        # Mock graphiti with query entities
        mock_graphiti = MagicMock()
        mock_graphiti.is_connected = True
        mock_search_result = MagicMock()
        mock_node = MagicMock()
        mock_node.uuid = "query-entity"
        mock_search_result.nodes = [mock_node]
        mock_graphiti.client = MagicMock()
        mock_graphiti.client.search = AsyncMock(return_value=mock_search_result)

        reranker = NodeDistanceReranker(
            mock_neo4j_client, mock_graphiti, original_weight=0.7
        )

        results = [{"id": "1", "score": 0.9, "entity_ids": ["result-entity"]}]
        reranked = await reranker.rerank("query", results, "tenant-1")

        assert len(reranked) == 1
        result = reranked[0]

        # Distance is None, so score is 0
        assert result.graph_context.min_distance is None
        assert result.graph_score == 0.0

    @pytest.mark.asyncio
    async def test_neo4j_error_handled_gracefully(self, mock_neo4j_client):
        """Should handle Neo4j errors gracefully."""
        mock_session = AsyncMock()
        mock_session.run = AsyncMock(side_effect=Exception("Neo4j connection error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        reranker = EpisodeMentionsReranker(mock_neo4j_client)

        results = [{"id": "1", "score": 0.9, "entity_ids": ["entity-1"]}]

        # Should not raise, just use 0 for graph score
        reranked = await reranker.rerank("query", results, "tenant-1")

        assert len(reranked) == 1
        assert reranked[0].graph_context.episode_mentions == 0


class TestScoreNormalization:
    """Tests for score normalization (0-1 range)."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client."""
        client = MagicMock()
        client.driver = MagicMock()
        return client

    def test_episode_score_normalized(self, mock_neo4j_client):
        """Episode scores should be in 0-1 range."""
        reranker = EpisodeMentionsReranker(mock_neo4j_client)

        # Test various mention counts
        assert 0 <= reranker._normalize_episode_score(0) <= 1
        assert 0 <= reranker._normalize_episode_score(5) <= 1
        assert 0 <= reranker._normalize_episode_score(15) <= 1
        assert 0 <= reranker._normalize_episode_score(100) <= 1

    def test_distance_score_normalized(self, mock_neo4j_client):
        """Distance scores should be in 0-1 range."""
        reranker = NodeDistanceReranker(mock_neo4j_client, max_distance=3)

        # Test various distances
        assert 0 <= reranker._distance_to_score(None) <= 1
        assert 0 <= reranker._distance_to_score(0) <= 1
        assert 0 <= reranker._distance_to_score(1) <= 1
        assert 0 <= reranker._distance_to_score(3) <= 1
        assert 0 <= reranker._distance_to_score(10) <= 1


class TestLatencyRequirements:
    """Tests for latency requirements (<200ms for reranking)."""

    @pytest.fixture
    def mock_neo4j_client(self):
        """Create mock Neo4j client with fast responses."""
        client = MagicMock()
        client.driver = MagicMock()
        client.search_entities_by_terms = AsyncMock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_reranking_completes_quickly(self, mock_neo4j_client):
        """Reranking should complete within time budget."""
        import time

        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value={"mention_count": 0})
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver.session = MagicMock(return_value=mock_session)

        reranker = EpisodeMentionsReranker(mock_neo4j_client)

        # 10 results
        results = [{"id": str(i), "score": 0.9 - i * 0.05} for i in range(10)]

        start = time.perf_counter()
        await reranker.rerank("query", results, "tenant-1")
        elapsed_ms = (time.perf_counter() - start) * 1000

        # With mocks, should be very fast
        # In production, target is <200ms
        assert elapsed_ms < 1000  # Very generous for mocked test
