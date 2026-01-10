"""Tests for LazyRAG API endpoints (Story 20-B2).

This module tests the LazyRAG REST API endpoint handlers directly,
without spinning up the full application (which requires databases).

Tests cover:
- POST /api/v1/lazy-rag/query - Execute lazy RAG query
- POST /api/v1/lazy-rag/expand - Expand subgraph only
- GET /api/v1/lazy-rag/status - Feature status

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by LAZY_RAG_ENABLED configuration flag.
"""

import os

# Set environment variables BEFORE any imports
os.environ["SKIP_DB_POOL"] = "1"
os.environ["SKIP_GRAPHITI"] = "1"
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_mock_settings(
    lazy_rag_enabled: bool = True,
    lazy_rag_max_entities: int = 50,
    lazy_rag_max_hops: int = 2,
    lazy_rag_use_communities: bool = True,
    lazy_rag_summary_model: str = "gpt-4o-mini",
    community_detection_enabled: bool = False,
):
    """Create mock settings for LazyRAG tests."""
    settings = MagicMock()
    settings.lazy_rag_enabled = lazy_rag_enabled
    settings.lazy_rag_max_entities = lazy_rag_max_entities
    settings.lazy_rag_max_hops = lazy_rag_max_hops
    settings.lazy_rag_use_communities = lazy_rag_use_communities
    settings.lazy_rag_summary_model = lazy_rag_summary_model
    settings.community_detection_enabled = community_detection_enabled
    # Add other required settings for LLM
    settings.llm_provider = "openai"
    settings.llm_api_key = "test-key"
    settings.llm_base_url = None
    settings.embedding_provider = "openai"
    settings.embedding_api_key = "test-key"
    settings.embedding_base_url = None
    settings.embedding_model = "text-embedding-3-small"
    return settings


def _make_mock_lazy_rag_result():
    """Create a mock LazyRAGResult for testing."""
    from agentic_rag_backend.retrieval.lazy_rag_models import (
        LazyRAGEntity,
        LazyRAGRelationship,
        LazyRAGResult,
    )

    return LazyRAGResult(
        query="What is FastAPI?",
        tenant_id="test-tenant",
        entities=[
            LazyRAGEntity(id="e1", name="FastAPI", type="Framework"),
            LazyRAGEntity(id="e2", name="Python", type="Language"),
        ],
        relationships=[
            LazyRAGRelationship(source_id="e1", target_id="e2", type="BUILT_WITH"),
        ],
        communities=[],
        summary="FastAPI is a modern Python web framework.",
        confidence=0.85,
        seed_entity_count=2,
        expanded_entity_count=2,
        processing_time_ms=150,
        missing_info=None,
    )


def _make_mock_expansion_result():
    """Create a mock SubgraphExpansionResult for testing."""
    from agentic_rag_backend.retrieval.lazy_rag_models import (
        LazyRAGEntity,
        LazyRAGRelationship,
        SubgraphExpansionResult,
    )

    return SubgraphExpansionResult(
        entities=[
            LazyRAGEntity(id="e1", name="FastAPI", type="Framework"),
        ],
        relationships=[],
        seed_count=1,
        expanded_count=1,
    )


@pytest.fixture
def sample_tenant_id():
    """Provide a sample tenant ID."""
    return uuid4()


@pytest.fixture
def mock_lazy_rag_retriever():
    """Create a mock LazyRAGRetriever."""
    retriever = MagicMock()
    retriever.query = AsyncMock(return_value=_make_mock_lazy_rag_result())
    retriever.expand_only = AsyncMock(return_value=_make_mock_expansion_result())
    return retriever


@pytest.fixture
def mock_neo4j_for_lazy_rag():
    """Create a mock Neo4j client for LazyRAG tests."""
    client = MagicMock()
    client.driver = MagicMock()
    return client


@pytest.fixture
def lazy_rag_app(mock_lazy_rag_retriever, mock_neo4j_for_lazy_rag):
    """Create a minimal FastAPI app with only LazyRAG routes for testing."""
    from agentic_rag_backend.api.routes.lazy_rag import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Set up app state
    settings = _make_mock_settings(lazy_rag_enabled=True)
    app.state.settings = settings
    app.state.neo4j = mock_neo4j_for_lazy_rag
    app.state.lazy_rag_retriever = mock_lazy_rag_retriever
    app.state.community_detector = None

    return app


@pytest.fixture
def lazy_rag_disabled_app():
    """Create a minimal FastAPI app with LazyRAG disabled."""
    from agentic_rag_backend.api.routes.lazy_rag import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Set up app state with LazyRAG disabled
    settings = _make_mock_settings(lazy_rag_enabled=False)
    app.state.settings = settings
    app.state.neo4j = None
    app.state.community_detector = None

    return app


@pytest.fixture
def lazy_rag_client(lazy_rag_app):
    """Create test client for LazyRAG app."""
    return TestClient(lazy_rag_app)


@pytest.fixture
def lazy_rag_disabled_client(lazy_rag_disabled_app):
    """Create test client for disabled LazyRAG app."""
    return TestClient(lazy_rag_disabled_app)


class TestLazyRAGQueryEndpoint:
    """Tests for POST /api/v1/lazy-rag/query endpoint."""

    def test_query_success(self, lazy_rag_client, sample_tenant_id):
        """Test successful LazyRAG query."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert "query" in data["data"]
        assert "summary" in data["data"]
        assert "confidence" in data["data"]
        assert "entities" in data["data"]
        assert "relationships" in data["data"]
        assert "processing_time_ms" in data["data"]

    def test_query_with_custom_parameters(self, lazy_rag_client, sample_tenant_id):
        """Test query with custom max_entities and max_hops."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
                "max_entities": 30,
                "max_hops": 3,
                "use_communities": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_query_without_summary(self, lazy_rag_client, sample_tenant_id):
        """Test query with include_summary=false."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
                "include_summary": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_query_missing_query(self, lazy_rag_client, sample_tenant_id):
        """Test query without required query field."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 422  # Validation error

    def test_query_missing_tenant_id(self, lazy_rag_client):
        """Test query without required tenant_id."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_query_invalid_tenant_id(self, lazy_rag_client):
        """Test query with invalid tenant_id format."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": "not-a-uuid",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_query_max_entities_validation(self, lazy_rag_client, sample_tenant_id):
        """Test query with max_entities exceeding limit."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
                "max_entities": 500,  # Exceeds max 200
            },
        )

        assert response.status_code == 422  # Validation error

    def test_query_max_hops_validation(self, lazy_rag_client, sample_tenant_id):
        """Test query with max_hops exceeding limit."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
                "max_hops": 10,  # Exceeds max 5
            },
        )

        assert response.status_code == 422  # Validation error

    def test_query_disabled_feature(self, lazy_rag_disabled_client, sample_tenant_id):
        """Test query when LazyRAG is disabled."""
        response = lazy_rag_disabled_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


class TestLazyRAGExpandEndpoint:
    """Tests for POST /api/v1/lazy-rag/expand endpoint."""

    def test_expand_success(self, lazy_rag_client, sample_tenant_id):
        """Test successful subgraph expansion."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/expand",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert "query" in data["data"]
        assert "entities" in data["data"]
        assert "relationships" in data["data"]
        assert "seed_entity_count" in data["data"]
        assert "expanded_entity_count" in data["data"]
        assert "processing_time_ms" in data["data"]

    def test_expand_with_custom_parameters(self, lazy_rag_client, sample_tenant_id):
        """Test expand with custom parameters."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/expand",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
                "max_entities": 20,
                "max_hops": 1,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_expand_disabled_feature(self, lazy_rag_disabled_client, sample_tenant_id):
        """Test expand when LazyRAG is disabled."""
        response = lazy_rag_disabled_client.post(
            "/api/v1/lazy-rag/expand",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


class TestLazyRAGStatusEndpoint:
    """Tests for GET /api/v1/lazy-rag/status endpoint."""

    def test_status_enabled(self, lazy_rag_client):
        """Test status when LazyRAG is enabled."""
        response = lazy_rag_client.get("/api/v1/lazy-rag/status")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert data["data"]["enabled"] is True
        assert "max_entities" in data["data"]
        assert "max_hops" in data["data"]
        assert "use_communities" in data["data"]
        assert "summary_model" in data["data"]
        assert "community_detection_available" in data["data"]

    def test_status_disabled(self, lazy_rag_disabled_client):
        """Test status when LazyRAG is disabled."""
        response = lazy_rag_disabled_client.get("/api/v1/lazy-rag/status")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["enabled"] is False


class TestLazyRAGResponseFormat:
    """Tests for API response format compliance."""

    def test_query_response_format(self, lazy_rag_client, sample_tenant_id):
        """Test query response follows standard format."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
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

    def test_entity_format(self, lazy_rag_client, sample_tenant_id):
        """Test entity format in response."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        for entity in data["data"]["entities"]:
            assert "id" in entity
            assert "name" in entity
            assert "type" in entity

    def test_relationship_format(self, lazy_rag_client, sample_tenant_id):
        """Test relationship format in response."""
        response = lazy_rag_client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        for rel in data["data"]["relationships"]:
            assert "source_id" in rel
            assert "target_id" in rel
            assert "type" in rel


class TestLazyRAGTenantIsolation:
    """Tests for multi-tenancy isolation."""

    def test_query_uses_tenant_id(
        self, lazy_rag_app, mock_lazy_rag_retriever, sample_tenant_id
    ):
        """Test that query passes tenant_id correctly."""
        client = TestClient(lazy_rag_app)
        response = client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200

        # Verify retriever was called with correct tenant_id
        mock_lazy_rag_retriever.query.assert_called_once()
        call_kwargs = mock_lazy_rag_retriever.query.call_args[1]
        assert call_kwargs["tenant_id"] == str(sample_tenant_id)

    def test_expand_uses_tenant_id(
        self, lazy_rag_app, mock_lazy_rag_retriever, sample_tenant_id
    ):
        """Test that expand passes tenant_id correctly."""
        client = TestClient(lazy_rag_app)
        response = client.post(
            "/api/v1/lazy-rag/expand",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200

        # Verify retriever was called with correct tenant_id
        mock_lazy_rag_retriever.expand_only.assert_called_once()
        call_kwargs = mock_lazy_rag_retriever.expand_only.call_args[1]
        assert call_kwargs["tenant_id"] == str(sample_tenant_id)


class TestLazyRAGErrorHandling:
    """Tests for error handling in LazyRAG API."""

    def test_query_error_returns_500(
        self, lazy_rag_app, mock_lazy_rag_retriever, sample_tenant_id
    ):
        """Test that query errors return 500 status with generic message (no internal details leaked)."""
        mock_lazy_rag_retriever.query.side_effect = RuntimeError("Test error")

        client = TestClient(lazy_rag_app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/lazy-rag/query",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 500
        # Security: Generic error message returned (internal details not leaked)
        assert "internal server error" in response.json()["detail"]
        assert "Test error" not in response.json()["detail"]

    def test_expand_error_returns_500(
        self, lazy_rag_app, mock_lazy_rag_retriever, sample_tenant_id
    ):
        """Test that expand errors return 500 status with generic message (no internal details leaked)."""
        mock_lazy_rag_retriever.expand_only.side_effect = RuntimeError("Expand error")

        client = TestClient(lazy_rag_app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/lazy-rag/expand",
            json={
                "query": "What is FastAPI?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 500
        # Security: Generic error message returned (internal details not leaked)
        assert "internal server error" in response.json()["detail"]
        assert "Expand error" not in response.json()["detail"]
