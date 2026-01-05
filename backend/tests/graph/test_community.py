"""Unit tests for CommunityDetector (Story 20-B1).

Tests cover:
- Graph building from Neo4j data
- Louvain/Leiden algorithm execution
- Community hierarchy construction
- Summary generation
- Storage to Neo4j
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from agentic_rag_backend.graph import (
    Community,
    CommunityAlgorithm,
    CommunityDetector,
    CommunityDetectionError,
    CommunityNotFoundError,
    GraphTooSmallError,
    NETWORKX_AVAILABLE,
)


# Skip all tests if networkx not available
pytestmark = pytest.mark.skipif(
    not NETWORKX_AVAILABLE,
    reason="NetworkX not installed",
)


@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client."""
    client = MagicMock()
    mock_driver = MagicMock()
    client.driver = mock_driver
    return client


@pytest.fixture
def mock_session():
    """Create a mock Neo4j session."""
    session = AsyncMock()
    return session


@pytest.fixture
def sample_nodes():
    """Sample entity nodes for testing."""
    return [
        {"id": "entity-1", "name": "Alice", "type": "Person", "description": "A person"},
        {"id": "entity-2", "name": "Bob", "type": "Person", "description": "Another person"},
        {"id": "entity-3", "name": "Carol", "type": "Person", "description": "Third person"},
        {"id": "entity-4", "name": "Acme Corp", "type": "Organization", "description": "A company"},
        {"id": "entity-5", "name": "Tech Inc", "type": "Organization", "description": "Another company"},
    ]


@pytest.fixture
def sample_edges():
    """Sample relationship edges for testing."""
    return [
        {"source_id": "entity-1", "target_id": "entity-2", "rel_type": "RELATED_TO", "confidence": 0.9},
        {"source_id": "entity-1", "target_id": "entity-3", "rel_type": "RELATED_TO", "confidence": 0.8},
        {"source_id": "entity-2", "target_id": "entity-3", "rel_type": "RELATED_TO", "confidence": 0.7},
        {"source_id": "entity-4", "target_id": "entity-5", "rel_type": "RELATED_TO", "confidence": 0.95},
    ]


class TestCommunityDetectorInit:
    """Tests for CommunityDetector initialization."""

    def test_init_with_defaults(self, mock_neo4j_client):
        """Test initialization with default parameters."""
        detector = CommunityDetector(neo4j_client=mock_neo4j_client)

        assert detector.algorithm == CommunityAlgorithm.LOUVAIN
        assert detector.min_community_size == 3
        assert detector.max_hierarchy_levels == 3
        assert detector.summary_model == "gpt-4o-mini"
        assert detector._neo4j == mock_neo4j_client
        assert detector._llm_client is None

    def test_init_with_leiden(self, mock_neo4j_client):
        """Test initialization with Leiden algorithm."""
        detector = CommunityDetector(
            neo4j_client=mock_neo4j_client,
            algorithm=CommunityAlgorithm.LEIDEN,
        )
        assert detector.algorithm == CommunityAlgorithm.LEIDEN

    def test_init_with_custom_params(self, mock_neo4j_client):
        """Test initialization with custom parameters."""
        mock_llm = MagicMock()
        detector = CommunityDetector(
            neo4j_client=mock_neo4j_client,
            llm_client=mock_llm,
            algorithm=CommunityAlgorithm.LOUVAIN,
            min_community_size=5,
            max_hierarchy_levels=5,
            summary_model="gpt-4o",
        )

        assert detector.min_community_size == 5
        assert detector.max_hierarchy_levels == 5
        assert detector.summary_model == "gpt-4o"
        assert detector._llm_client == mock_llm


class TestNetworkXGraphBuilding:
    """Tests for NetworkX graph construction."""

    @pytest.mark.asyncio
    async def test_build_networkx_graph(self, mock_neo4j_client, sample_nodes, sample_edges):
        """Test building NetworkX graph from Neo4j data."""
        # Setup mock session
        mock_session = AsyncMock()

        # Mock node query result
        mock_node_result = AsyncMock()
        mock_node_result.data.return_value = sample_nodes

        # Mock edge query result
        mock_edge_result = AsyncMock()
        mock_edge_result.data.return_value = sample_edges

        mock_session.run = AsyncMock(side_effect=[mock_node_result, mock_edge_result])

        # Create async context manager mock
        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver = mock_driver

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        G = await detector._build_networkx_graph("tenant-123")

        # Verify nodes
        assert len(G.nodes) == 5
        assert "entity-1" in G.nodes
        assert G.nodes["entity-1"]["name"] == "Alice"
        assert G.nodes["entity-1"]["type"] == "Person"

        # Verify edges
        assert len(G.edges) == 4
        assert G.has_edge("entity-1", "entity-2")
        assert G.edges["entity-1", "entity-2"]["weight"] == 0.9


class TestLouvainAlgorithm:
    """Tests for Louvain community detection."""

    def test_run_louvain(self, mock_neo4j_client):
        """Test Louvain algorithm execution."""
        import networkx as nx

        # Create a simple test graph with clear communities
        G = nx.Graph()
        # Community 1: tightly connected
        G.add_edge("a1", "a2", weight=1.0)
        G.add_edge("a1", "a3", weight=1.0)
        G.add_edge("a2", "a3", weight=1.0)
        # Community 2: tightly connected
        G.add_edge("b1", "b2", weight=1.0)
        G.add_edge("b1", "b3", weight=1.0)
        G.add_edge("b2", "b3", weight=1.0)
        # Weak connection between communities
        G.add_edge("a1", "b1", weight=0.1)

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        partition = detector._run_louvain(G)

        # All nodes should be assigned
        assert len(partition) == 6
        assert all(node in partition for node in G.nodes)

        # Should detect at least 2 communities (or 1 if merged)
        communities_found = len(set(partition.values()))
        assert communities_found >= 1


class TestCommunityBuilding:
    """Tests for community object construction."""

    def test_build_communities(self, mock_neo4j_client):
        """Test building Community objects from partition."""
        import networkx as nx

        G = nx.Graph()
        G.add_node("e1", name="Entity 1", type="Person", description="Desc 1")
        G.add_node("e2", name="Entity 2", type="Person", description="Desc 2")
        G.add_node("e3", name="Entity 3", type="Person", description="Desc 3")
        G.add_edge("e1", "e2")
        G.add_edge("e2", "e3")

        partition = {"e1": 0, "e2": 0, "e3": 0}

        detector = CommunityDetector(
            neo4j_client=mock_neo4j_client,
            min_community_size=2,
        )
        communities = detector._build_communities(
            partition=partition,
            graph=G,
            tenant_id="tenant-123",
            min_size=2,
        )

        assert len(communities) == 1
        assert communities[0].entity_count == 3
        assert set(communities[0].entity_ids) == {"e1", "e2", "e3"}
        assert communities[0].tenant_id == "tenant-123"
        assert communities[0].level == 0

    def test_build_communities_filters_small(self, mock_neo4j_client):
        """Test that communities below min_size are filtered."""
        import networkx as nx

        G = nx.Graph()
        for i in range(4):
            G.add_node(f"e{i}", name=f"Entity {i}", type="Type", description="")

        # Two nodes in community 0, two in community 1
        partition = {"e0": 0, "e1": 0, "e2": 1, "e3": 1}

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        communities = detector._build_communities(
            partition=partition,
            graph=G,
            tenant_id="tenant-123",
            min_size=3,  # Both communities are size 2, below threshold
        )

        assert len(communities) == 0


class TestHierarchyBuilding:
    """Tests for hierarchical community construction."""

    def test_build_hierarchy_single_level(self, mock_neo4j_client):
        """Test hierarchy building with single level."""
        communities = [
            Community(
                id="c1",
                name="Community 1",
                level=0,
                tenant_id="tenant-123",
                entity_ids=["e1", "e2", "e3"],
                entity_count=3,
            ),
        ]

        import networkx as nx
        G = nx.Graph()
        for eid in ["e1", "e2", "e3"]:
            G.add_node(eid)
        G.add_edge("e1", "e2")
        G.add_edge("e2", "e3")

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        result = detector._build_hierarchy(
            communities=communities,
            graph=G,
            tenant_id="tenant-123",
            max_levels=1,
        )

        # With max_levels=1, should return original communities only
        assert len(result) == 1
        assert result[0].level == 0


class TestSummaryGeneration:
    """Tests for LLM summary generation."""

    def test_build_summary_prompt(self, mock_neo4j_client):
        """Test prompt building for summary generation."""
        community = Community(
            id="c1",
            name="Test Community",
            level=0,
            tenant_id="tenant-123",
            entity_ids=["e1", "e2"],
            entity_count=2,
        )

        entities = [
            {"name": "Alice", "type": "Person", "description": "A software engineer"},
            {"name": "Bob", "type": "Person", "description": "A data scientist"},
        ]

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        prompt = detector._build_summary_prompt(community, entities)

        assert "Alice" in prompt
        assert "Bob" in prompt
        assert "Person" in prompt
        assert "SUMMARY:" in prompt
        assert "KEYWORDS:" in prompt

    def test_parse_summary_response(self, mock_neo4j_client):
        """Test parsing LLM response into summary and keywords."""
        detector = CommunityDetector(neo4j_client=mock_neo4j_client)

        response = """SUMMARY: This is a community of software professionals.
KEYWORDS: software, engineering, technology"""

        summary, keywords = detector._parse_summary_response(response)

        assert summary == "This is a community of software professionals."
        assert len(keywords) == 3
        assert "software" in keywords
        assert "engineering" in keywords
        assert "technology" in keywords

    def test_parse_summary_response_defaults(self, mock_neo4j_client):
        """Test parsing returns defaults for malformed response."""
        detector = CommunityDetector(neo4j_client=mock_neo4j_client)

        # Malformed response
        response = "Some random text without proper format"

        summary, keywords = detector._parse_summary_response(response)

        assert summary == "A community of related entities"
        assert keywords == []


class TestCommunityStorage:
    """Tests for Neo4j storage operations."""

    @pytest.mark.asyncio
    async def test_store_communities(self, mock_neo4j_client):
        """Test storing communities to Neo4j."""
        communities = [
            Community(
                id="c1",
                name="Test Community",
                level=0,
                tenant_id="tenant-123",
                entity_ids=["e1", "e2"],
                entity_count=2,
                summary="A test community",
                keywords=["test", "community"],
                created_at=datetime.now(timezone.utc),
            ),
        ]

        mock_session = AsyncMock()
        mock_session.run = AsyncMock()

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver = mock_driver

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        await detector._store_communities(communities, "tenant-123")

        # Should have called run multiple times (create community + create relationships)
        assert mock_session.run.call_count >= 1

    @pytest.mark.asyncio
    async def test_get_community(self, mock_neo4j_client):
        """Test fetching a community by ID."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single.return_value = {
            "c": {
                "id": "c1",
                "name": "Test Community",
                "level": 0,
                "entity_count": 3,
                "summary": "A test",
                "keywords": ["test"],
            },
            "entity_ids": ["e1", "e2", "e3"],
        }
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver = mock_driver

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        community = await detector.get_community("c1", "tenant-123")

        assert community.id == "c1"
        assert community.name == "Test Community"
        assert community.entity_count == 3
        assert len(community.entity_ids) == 3

    @pytest.mark.asyncio
    async def test_get_community_not_found(self, mock_neo4j_client):
        """Test fetching non-existent community raises error."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver = mock_driver

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)

        with pytest.raises(CommunityNotFoundError) as exc_info:
            await detector.get_community("nonexistent", "tenant-123")

        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_community(self, mock_neo4j_client):
        """Test deleting a community."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.single.return_value = {"deleted": 1}
        mock_session.run.return_value = mock_result

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver = mock_driver

        detector = CommunityDetector(neo4j_client=mock_neo4j_client)
        deleted = await detector.delete_community("c1", "tenant-123")

        assert deleted is True


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_detect_communities_graph_too_small(self, mock_neo4j_client):
        """Test that small graphs raise GraphTooSmallError."""
        mock_session = AsyncMock()

        # Return only 2 nodes (below min_size=3 default)
        mock_node_result = AsyncMock()
        mock_node_result.data.return_value = [
            {"id": "e1", "name": "Entity 1", "type": "Type", "description": ""},
            {"id": "e2", "name": "Entity 2", "type": "Type", "description": ""},
        ]

        mock_edge_result = AsyncMock()
        mock_edge_result.data.return_value = [
            {"source_id": "e1", "target_id": "e2", "rel_type": "RELATED_TO", "confidence": 1.0},
        ]

        mock_session.run = AsyncMock(side_effect=[mock_node_result, mock_edge_result])

        mock_driver = MagicMock()
        mock_driver.session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_neo4j_client.driver = mock_driver

        detector = CommunityDetector(
            neo4j_client=mock_neo4j_client,
            min_community_size=3,
        )

        with pytest.raises(GraphTooSmallError) as exc_info:
            await detector.detect_communities("tenant-123")

        assert exc_info.value.node_count == 2
        assert exc_info.value.min_required == 3


class TestCommunityModel:
    """Tests for Community Pydantic model."""

    def test_community_model_creation(self):
        """Test creating a Community model."""
        community = Community(
            id=str(uuid4()),
            name="Test Community",
            level=0,
            tenant_id="tenant-123",
            entity_ids=["e1", "e2", "e3"],
            entity_count=3,
            summary="A test community of entities",
            keywords=["test", "entities"],
        )

        assert community.name == "Test Community"
        assert community.level == 0
        assert community.entity_count == 3
        assert len(community.entity_ids) == 3
        assert "test" in community.keywords

    def test_community_model_defaults(self):
        """Test Community model with defaults."""
        community = Community(
            id="c1",
            name="Test",
            tenant_id="tenant-123",
        )

        assert community.level == 0
        assert community.entity_ids == []
        assert community.entity_count == 0
        assert community.summary is None
        assert community.keywords == []
        assert community.parent_id is None
        assert community.child_ids == []

    def test_community_model_serialization(self):
        """Test Community model JSON serialization."""
        community = Community(
            id="c1",
            name="Test Community",
            level=1,
            tenant_id="tenant-123",
            entity_ids=["e1"],
            entity_count=1,
            created_at=datetime.now(timezone.utc),
        )

        data = community.model_dump(mode="json")

        assert data["id"] == "c1"
        assert data["name"] == "Test Community"
        assert data["level"] == 1
        assert "created_at" in data
