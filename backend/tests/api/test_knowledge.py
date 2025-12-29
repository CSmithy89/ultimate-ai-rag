"""Tests for Knowledge Graph API endpoints."""

import os

# Set environment variables BEFORE any imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def sample_graph_data():
    """Sample graph data for testing."""
    return {
        "nodes": [
            {
                "id": str(uuid4()),
                "label": "OpenAI",
                "type": "Organization",
                "properties": {"description": "AI research company"},
                "is_orphan": False,
            },
            {
                "id": str(uuid4()),
                "label": "GPT-4",
                "type": "Technology",
                "properties": {"description": "Large language model"},
                "is_orphan": False,
            },
            {
                "id": str(uuid4()),
                "label": "Python",
                "type": "Technology",
                "properties": {"description": "Programming language"},
                "is_orphan": True,
            },
        ],
        "edges": [
            {
                "id": "edge-1",
                "source": "node-1",
                "target": "node-2",
                "type": "AUTHORED_BY",
                "label": "AUTHORED_BY",
                "properties": {"confidence": 0.95},
            },
        ],
    }


@pytest.fixture
def sample_stats_data():
    """Sample stats data for testing."""
    return {
        "node_count": 1500,
        "edge_count": 3200,
        "orphan_count": 12,
        "entity_type_counts": {
            "Person": 200,
            "Organization": 300,
            "Technology": 500,
            "Concept": 400,
            "Location": 100,
        },
        "relationship_type_counts": {
            "USES": 800,
            "MENTIONS": 500,
            "AUTHORED_BY": 400,
            "PART_OF": 300,
            "RELATED_TO": 1200,
        },
    }


@pytest.fixture
def sample_orphan_nodes():
    """Sample orphan nodes for testing."""
    return [
        {
            "id": str(uuid4()),
            "label": "Orphan Entity 1",
            "type": "Technology",
            "properties": {"description": "No relationships"},
            "is_orphan": True,
        },
        {
            "id": str(uuid4()),
            "label": "Orphan Entity 2",
            "type": "Concept",
            "properties": {"description": "Also no relationships"},
            "is_orphan": True,
        },
    ]


@pytest.fixture
def mock_neo4j_client_with_data(sample_graph_data, sample_stats_data, sample_orphan_nodes):
    """Mock Neo4jClient with visualization data."""
    from agentic_rag_backend.db.neo4j import Neo4jClient

    client = MagicMock(spec=Neo4jClient)
    client.get_graph_data = AsyncMock(return_value=sample_graph_data)
    client.get_visualization_stats = AsyncMock(return_value=sample_stats_data)
    client.get_orphan_nodes = AsyncMock(return_value=sample_orphan_nodes)
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.create_indexes = AsyncMock()
    return client


@pytest.fixture
def knowledge_client(mock_neo4j_client_with_data, mock_redis_client, mock_postgres_client, monkeypatch):
    """Create FastAPI test client with mocked Neo4j for knowledge endpoints."""
    from agentic_rag_backend.main import app

    # Mock the dependency injection functions
    async def mock_get_neo4j():
        return mock_neo4j_client_with_data

    async def mock_get_redis():
        return mock_redis_client

    async def mock_get_postgres():
        return mock_postgres_client

    # Override dependencies
    from agentic_rag_backend.api.routes.knowledge import get_neo4j
    from agentic_rag_backend.api.routes.ingest import get_redis, get_postgres

    app.dependency_overrides[get_neo4j] = mock_get_neo4j
    app.dependency_overrides[get_redis] = mock_get_redis
    app.dependency_overrides[get_postgres] = mock_get_postgres

    # Mock environment variables for settings
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestGetGraph:
    """Tests for GET /api/v1/knowledge/graph endpoint."""

    def test_get_graph_success(self, knowledge_client, sample_tenant_id):
        """Test successful graph data retrieval."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert "nodes" in data["data"]
        assert "edges" in data["data"]
        assert "requestId" in data["meta"]
        assert "timestamp" in data["meta"]

    def test_get_graph_with_pagination(self, knowledge_client, sample_tenant_id):
        """Test graph retrieval with pagination parameters."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={
                "tenant_id": str(sample_tenant_id),
                "limit": 50,
                "offset": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_graph_with_entity_type_filter(self, knowledge_client, sample_tenant_id):
        """Test graph retrieval with entity type filter."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={
                "tenant_id": str(sample_tenant_id),
                "entity_type": "Technology",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_graph_with_relationship_type_filter(self, knowledge_client, sample_tenant_id):
        """Test graph retrieval with relationship type filter."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={
                "tenant_id": str(sample_tenant_id),
                "relationship_type": "USES",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_graph_missing_tenant_id(self, knowledge_client):
        """Test graph retrieval without required tenant_id."""
        response = knowledge_client.get("/api/v1/knowledge/graph")

        assert response.status_code == 422  # Validation error

    def test_get_graph_invalid_tenant_id(self, knowledge_client):
        """Test graph retrieval with invalid tenant_id format."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={"tenant_id": "not-a-uuid"},
        )

        assert response.status_code == 422  # Validation error

    def test_get_graph_limit_validation(self, knowledge_client, sample_tenant_id):
        """Test graph retrieval limit parameter validation."""
        # Test limit too high
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={
                "tenant_id": str(sample_tenant_id),
                "limit": 2000,  # Exceeds max 1000
            },
        )

        assert response.status_code == 422  # Validation error


class TestGetStats:
    """Tests for GET /api/v1/knowledge/stats endpoint."""

    def test_get_stats_success(self, knowledge_client, sample_tenant_id, sample_stats_data):
        """Test successful stats retrieval."""
        response = knowledge_client.get(
            "/api/v1/knowledge/stats",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert data["data"]["node_count"] == sample_stats_data["node_count"]
        assert data["data"]["edge_count"] == sample_stats_data["edge_count"]
        assert data["data"]["orphan_count"] == sample_stats_data["orphan_count"]
        assert "entity_type_counts" in data["data"]
        assert "relationship_type_counts" in data["data"]

    def test_get_stats_missing_tenant_id(self, knowledge_client):
        """Test stats retrieval without required tenant_id."""
        response = knowledge_client.get("/api/v1/knowledge/stats")

        assert response.status_code == 422  # Validation error


class TestGetOrphans:
    """Tests for GET /api/v1/knowledge/orphans endpoint."""

    def test_get_orphans_success(self, knowledge_client, sample_tenant_id, sample_orphan_nodes):
        """Test successful orphan nodes retrieval."""
        response = knowledge_client.get(
            "/api/v1/knowledge/orphans",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert "orphans" in data["data"]
        assert "count" in data["data"]
        assert data["data"]["count"] == len(sample_orphan_nodes)

        # Verify all returned nodes are orphans
        for orphan in data["data"]["orphans"]:
            assert orphan["is_orphan"] is True

    def test_get_orphans_with_limit(self, knowledge_client, sample_tenant_id):
        """Test orphan nodes retrieval with custom limit."""
        response = knowledge_client.get(
            "/api/v1/knowledge/orphans",
            params={
                "tenant_id": str(sample_tenant_id),
                "limit": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_get_orphans_missing_tenant_id(self, knowledge_client):
        """Test orphan nodes retrieval without required tenant_id."""
        response = knowledge_client.get("/api/v1/knowledge/orphans")

        assert response.status_code == 422  # Validation error


class TestTenantIsolation:
    """Tests for multi-tenancy isolation."""

    def test_graph_tenant_isolation(
        self, knowledge_client, sample_tenant_id, mock_neo4j_client_with_data
    ):
        """Test that graph data is fetched with correct tenant_id."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200

        # Verify Neo4j client was called with correct tenant_id
        mock_neo4j_client_with_data.get_graph_data.assert_called_once()
        call_args = mock_neo4j_client_with_data.get_graph_data.call_args
        assert call_args.kwargs["tenant_id"] == str(sample_tenant_id)

    def test_stats_tenant_isolation(
        self, knowledge_client, sample_tenant_id, mock_neo4j_client_with_data
    ):
        """Test that stats are fetched with correct tenant_id."""
        response = knowledge_client.get(
            "/api/v1/knowledge/stats",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200

        # Verify Neo4j client was called with correct tenant_id
        mock_neo4j_client_with_data.get_visualization_stats.assert_called_once()
        call_args = mock_neo4j_client_with_data.get_visualization_stats.call_args
        assert call_args.kwargs["tenant_id"] == str(sample_tenant_id)

    def test_orphans_tenant_isolation(
        self, knowledge_client, sample_tenant_id, mock_neo4j_client_with_data
    ):
        """Test that orphans are fetched with correct tenant_id."""
        response = knowledge_client.get(
            "/api/v1/knowledge/orphans",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200

        # Verify Neo4j client was called with correct tenant_id
        mock_neo4j_client_with_data.get_orphan_nodes.assert_called_once()
        call_args = mock_neo4j_client_with_data.get_orphan_nodes.call_args
        assert call_args.kwargs["tenant_id"] == str(sample_tenant_id)


class TestResponseFormat:
    """Tests for API response format compliance."""

    def test_graph_response_format(self, knowledge_client, sample_tenant_id):
        """Test graph response follows standard format."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        # Verify meta fields
        assert "meta" in data
        assert "requestId" in data["meta"]
        assert "timestamp" in data["meta"]

        # Verify UUID format for requestId
        import uuid
        uuid.UUID(data["meta"]["requestId"])  # Raises if invalid

        # Verify ISO8601 timestamp format
        assert data["meta"]["timestamp"].endswith("Z")

    def test_node_format(self, knowledge_client, sample_tenant_id):
        """Test node format in graph response."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        for node in data["data"]["nodes"]:
            assert "id" in node
            assert "label" in node
            assert "type" in node
            assert "is_orphan" in node
            assert isinstance(node["is_orphan"], bool)

    def test_edge_format(self, knowledge_client, sample_tenant_id):
        """Test edge format in graph response."""
        response = knowledge_client.get(
            "/api/v1/knowledge/graph",
            params={"tenant_id": str(sample_tenant_id)},
        )

        assert response.status_code == 200
        data = response.json()

        for edge in data["data"]["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge
            assert "type" in edge
            assert "label" in edge
