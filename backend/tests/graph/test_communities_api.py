"""API tests for communities endpoints (Story 20-B1).

Tests cover:
- POST /api/v1/communities/detect - Trigger community detection
- GET /api/v1/communities - List communities
- GET /api/v1/communities/{id} - Get single community
- DELETE /api/v1/communities/{id} - Delete community
- POST /api/v1/communities/search - Search communities
- DELETE /api/v1/communities - Delete all communities
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

# Set test environment variables BEFORE imports
os.environ.setdefault("COMMUNITY_DETECTION_ENABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag_backend.api.routes.communities import router
from agentic_rag_backend.graph import (
    Community,
    CommunityAlgorithm,
    CommunityDetector,
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
def mock_settings():
    """Create mock settings with community detection enabled."""
    settings = MagicMock()
    settings.community_detection_enabled = True
    settings.community_algorithm = "louvain"
    settings.community_min_size = 3
    settings.community_max_levels = 3
    settings.community_summary_model = "gpt-4o-mini"
    return settings


@pytest.fixture
def mock_neo4j_client():
    """Create a mock Neo4j client."""
    client = MagicMock()
    mock_driver = MagicMock()
    client.driver = mock_driver
    return client


@pytest.fixture
def mock_detector():
    """Create a mock CommunityDetector."""
    detector = AsyncMock(spec=CommunityDetector)
    return detector


@pytest.fixture
def sample_communities():
    """Sample communities for testing."""
    return [
        Community(
            id="community-1",
            name="Tech Community",
            level=0,
            tenant_id="test-tenant",
            entity_ids=["e1", "e2", "e3"],
            entity_count=3,
            summary="A community about technology",
            keywords=["tech", "software"],
            created_at=datetime.now(timezone.utc),
        ),
        Community(
            id="community-2",
            name="Business Community",
            level=0,
            tenant_id="test-tenant",
            entity_ids=["e4", "e5", "e6", "e7"],
            entity_count=4,
            summary="A community about business",
            keywords=["business", "finance"],
            created_at=datetime.now(timezone.utc),
        ),
    ]


@pytest.fixture
def app(mock_settings, mock_neo4j_client, mock_detector):
    """Create FastAPI test app with mocked dependencies."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Set up app.state
    app.state.settings = mock_settings
    app.state.neo4j = mock_neo4j_client
    app.state.community_detector = mock_detector

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestDetectCommunities:
    """Tests for POST /api/v1/communities/detect endpoint."""

    def test_detect_communities_success(self, client, mock_detector, sample_communities):
        """Test successful community detection."""
        mock_detector.delete_all_communities.return_value = 0
        mock_detector.detect_communities.return_value = sample_communities

        response = client.post(
            "/api/v1/communities/detect",
            json={
                "tenant_id": str(uuid4()),
                "algorithm": "louvain",
                "generate_summaries": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert data["data"]["communities_created"] == 2
        assert data["data"]["algorithm"] == "louvain"
        assert "meta" in data
        assert "requestId" in data["meta"]

    def test_detect_communities_leiden(self, client, mock_detector, sample_communities):
        """Test community detection with Leiden algorithm."""
        mock_detector.delete_all_communities.return_value = 0
        mock_detector.detect_communities.return_value = sample_communities

        response = client.post(
            "/api/v1/communities/detect",
            json={
                "tenant_id": str(uuid4()),
                "algorithm": "leiden",
                "generate_summaries": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["algorithm"] == "leiden"

    def test_detect_communities_graph_too_small(self, client, mock_detector):
        """Test detection with graph too small."""
        mock_detector.delete_all_communities.return_value = 0
        mock_detector.detect_communities.side_effect = GraphTooSmallError(
            node_count=2,
            min_required=3,
            tenant_id="test-tenant",
        )

        response = client.post(
            "/api/v1/communities/detect",
            json={
                "tenant_id": str(uuid4()),
            },
        )

        assert response.status_code == 422
        assert "too small" in response.json()["detail"].lower()

    def test_detect_communities_feature_disabled(self, app, client):
        """Test detection when feature is disabled."""
        app.state.settings.community_detection_enabled = False

        response = client.post(
            "/api/v1/communities/detect",
            json={"tenant_id": str(uuid4())},
        )

        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


class TestListCommunities:
    """Tests for GET /api/v1/communities endpoint."""

    def test_list_communities_success(self, client, mock_detector, sample_communities):
        """Test listing communities."""
        mock_detector.list_communities.return_value = (sample_communities, 2)

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities?tenant_id={tenant_id}",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 2
        assert len(data["data"]["communities"]) == 2

    def test_list_communities_with_level_filter(self, client, mock_detector, sample_communities):
        """Test listing communities filtered by level."""
        mock_detector.list_communities.return_value = (sample_communities, 2)

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities?tenant_id={tenant_id}&level=0",
        )

        assert response.status_code == 200
        # Verify level parameter was passed
        mock_detector.list_communities.assert_called_once()
        call_kwargs = mock_detector.list_communities.call_args.kwargs
        assert call_kwargs.get("level") == 0

    def test_list_communities_pagination(self, client, mock_detector, sample_communities):
        """Test listing communities with pagination."""
        mock_detector.list_communities.return_value = (sample_communities[:1], 2)

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities?tenant_id={tenant_id}&limit=1&offset=0",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["limit"] == 1
        assert data["data"]["offset"] == 0
        assert len(data["data"]["communities"]) == 1

    def test_list_communities_missing_tenant_id(self, client):
        """Test listing communities without tenant_id."""
        response = client.get("/api/v1/communities")
        assert response.status_code == 422  # Validation error


class TestGetCommunity:
    """Tests for GET /api/v1/communities/{id} endpoint."""

    def test_get_community_success(self, client, mock_detector, sample_communities):
        """Test getting a single community."""
        mock_detector.get_community.return_value = sample_communities[0]

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities/community-1?tenant_id={tenant_id}",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["id"] == "community-1"
        assert data["data"]["name"] == "Tech Community"

    def test_get_community_not_found(self, client, mock_detector):
        """Test getting non-existent community."""
        mock_detector.get_community.side_effect = CommunityNotFoundError(
            community_id="nonexistent",
            tenant_id="test-tenant",
        )

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities/nonexistent?tenant_id={tenant_id}",
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_community_with_entities(self, client, mock_detector, sample_communities):
        """Test getting community with entity details."""
        mock_detector.get_community.return_value = sample_communities[0]

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities/community-1?tenant_id={tenant_id}&include_entities=true",
        )

        assert response.status_code == 200
        # Verify include_entities was passed
        mock_detector.get_community.assert_called_once()
        call_kwargs = mock_detector.get_community.call_args.kwargs
        assert call_kwargs.get("include_entities") is True


class TestDeleteCommunity:
    """Tests for DELETE /api/v1/communities/{id} endpoint."""

    def test_delete_community_success(self, client, mock_detector):
        """Test deleting a community."""
        mock_detector.delete_community.return_value = True

        tenant_id = str(uuid4())
        response = client.delete(
            f"/api/v1/communities/community-1?tenant_id={tenant_id}",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["deleted"] is True
        assert data["data"]["community_id"] == "community-1"

    def test_delete_community_not_found(self, client, mock_detector):
        """Test deleting non-existent community."""
        mock_detector.delete_community.return_value = False

        tenant_id = str(uuid4())
        response = client.delete(
            f"/api/v1/communities/nonexistent?tenant_id={tenant_id}",
        )

        assert response.status_code == 404


class TestSearchCommunities:
    """Tests for POST /api/v1/communities/search endpoint."""

    def test_search_communities_success(self, client, mock_detector, sample_communities):
        """Test searching communities."""
        mock_detector.search_communities.return_value = [sample_communities[0]]

        response = client.post(
            "/api/v1/communities/search",
            json={
                "query": "technology",
                "tenant_id": str(uuid4()),
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total"] == 1
        assert data["data"]["query"] == "technology"
        assert len(data["data"]["communities"]) == 1

    def test_search_communities_with_level(self, client, mock_detector, sample_communities):
        """Test searching communities with level filter."""
        mock_detector.search_communities.return_value = [sample_communities[0]]

        response = client.post(
            "/api/v1/communities/search",
            json={
                "query": "tech",
                "tenant_id": str(uuid4()),
                "level": 0,
                "limit": 5,
            },
        )

        assert response.status_code == 200
        # Verify parameters were passed correctly
        mock_detector.search_communities.assert_called_once()
        call_kwargs = mock_detector.search_communities.call_args.kwargs
        assert call_kwargs.get("level") == 0
        assert call_kwargs.get("limit") == 5

    def test_search_communities_empty_query(self, client):
        """Test searching with empty query fails validation."""
        response = client.post(
            "/api/v1/communities/search",
            json={
                "query": "",
                "tenant_id": str(uuid4()),
            },
        )

        assert response.status_code == 422  # Validation error


class TestDeleteAllCommunities:
    """Tests for DELETE /api/v1/communities endpoint."""

    def test_delete_all_communities_success(self, client, mock_detector):
        """Test deleting all communities for a tenant."""
        mock_detector.delete_all_communities.return_value = 5

        tenant_id = str(uuid4())
        response = client.delete(
            f"/api/v1/communities?tenant_id={tenant_id}",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["deleted_count"] == 5
        assert data["data"]["tenant_id"] == str(tenant_id)

    def test_delete_all_communities_none_existing(self, client, mock_detector):
        """Test deleting when no communities exist."""
        mock_detector.delete_all_communities.return_value = 0

        tenant_id = str(uuid4())
        response = client.delete(
            f"/api/v1/communities?tenant_id={tenant_id}",
        )

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["deleted_count"] == 0


class TestResponseFormat:
    """Tests for response format consistency."""

    def test_success_response_format(self, client, mock_detector, sample_communities):
        """Test that all endpoints follow the success response format."""
        mock_detector.list_communities.return_value = (sample_communities, 2)

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities?tenant_id={tenant_id}",
        )

        assert response.status_code == 200
        data = response.json()

        # Check standard response wrapper
        assert "data" in data
        assert "meta" in data
        assert "requestId" in data["meta"]
        assert "timestamp" in data["meta"]

        # Verify timestamp format (ISO 8601)
        timestamp = data["meta"]["timestamp"]
        assert "T" in timestamp  # ISO format includes T separator

    def test_error_response_format(self, client, mock_detector):
        """Test error responses include detail field."""
        mock_detector.get_community.side_effect = CommunityNotFoundError(
            community_id="test",
            tenant_id="test",
        )

        tenant_id = str(uuid4())
        response = client.get(
            f"/api/v1/communities/test?tenant_id={tenant_id}",
        )

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data


class TestAlgorithmEnum:
    """Tests for CommunityAlgorithm enum handling."""

    def test_valid_algorithm_values(self, client, mock_detector, sample_communities):
        """Test that valid algorithm values are accepted."""
        mock_detector.delete_all_communities.return_value = 0
        mock_detector.detect_communities.return_value = sample_communities

        for algo in ["louvain", "leiden"]:
            response = client.post(
                "/api/v1/communities/detect",
                json={
                    "tenant_id": str(uuid4()),
                    "algorithm": algo,
                },
            )
            assert response.status_code == 200

    def test_invalid_algorithm_value(self, client):
        """Test that invalid algorithm values are rejected."""
        response = client.post(
            "/api/v1/communities/detect",
            json={
                "tenant_id": str(uuid4()),
                "algorithm": "invalid_algo",
            },
        )
        assert response.status_code == 422  # Validation error
