"""Tests for Query Router API endpoints (Story 20-B3).

This module tests the Query Router REST API endpoint handlers directly,
without spinning up the full application (which requires databases).

Tests cover:
- POST /api/v1/query-router/route - Route a query
- GET /api/v1/query-router/patterns - List patterns (debug)
- GET /api/v1/query-router/status - Feature status

All endpoints respect multi-tenancy via tenant_id filtering.
Feature is gated by QUERY_ROUTING_ENABLED configuration flag.
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
    query_routing_enabled: bool = True,
    query_routing_use_llm: bool = False,
    query_routing_llm_model: str = "gpt-4o-mini",
    query_routing_confidence_threshold: float = 0.7,
    community_detection_enabled: bool = False,
    lazy_rag_enabled: bool = False,
):
    """Create mock settings for query router tests."""
    settings = MagicMock()
    settings.query_routing_enabled = query_routing_enabled
    settings.query_routing_use_llm = query_routing_use_llm
    settings.query_routing_llm_model = query_routing_llm_model
    settings.query_routing_confidence_threshold = query_routing_confidence_threshold
    settings.community_detection_enabled = community_detection_enabled
    settings.lazy_rag_enabled = lazy_rag_enabled
    # Add other required settings for LLM
    settings.llm_provider = "openai"
    settings.llm_api_key = "test-key"
    settings.llm_base_url = None
    settings.embedding_provider = "openai"
    settings.embedding_api_key = "test-key"
    settings.embedding_base_url = None
    settings.embedding_model = "text-embedding-3-small"
    return settings


def _make_mock_routing_decision():
    """Create a mock RoutingDecision for testing."""
    from agentic_rag_backend.retrieval.query_router_models import (
        QueryType,
        RoutingDecision,
    )

    return RoutingDecision(
        query_type=QueryType.GLOBAL,
        confidence=0.85,
        reasoning="High global pattern match",
        global_weight=1.0,
        local_weight=0.0,
        classification_method="rule_based",
        global_matches=2,
        local_matches=0,
        processing_time_ms=5,
    )


@pytest.fixture
def sample_tenant_id():
    """Provide a sample tenant ID."""
    return uuid4()


@pytest.fixture
def mock_query_router():
    """Create a mock QueryRouter."""
    from agentic_rag_backend.retrieval.query_router import QueryRouter

    router = MagicMock(spec=QueryRouter)
    router.route = AsyncMock(return_value=_make_mock_routing_decision())
    router.get_global_patterns = MagicMock(return_value=["pattern1", "pattern2"])
    router.get_local_patterns = MagicMock(return_value=["pattern3"])
    return router


@pytest.fixture
def query_router_app(mock_query_router):
    """Create a minimal FastAPI app with only query router routes for testing."""
    from agentic_rag_backend.api.routes.query_router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Set up app state
    settings = _make_mock_settings(query_routing_enabled=True)
    app.state.settings = settings
    app.state.query_router = mock_query_router

    return app


@pytest.fixture
def query_router_disabled_app():
    """Create a minimal FastAPI app with query router disabled."""
    from agentic_rag_backend.api.routes.query_router import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Set up app state with query routing disabled
    settings = _make_mock_settings(query_routing_enabled=False)
    app.state.settings = settings

    return app


@pytest.fixture
def query_router_client(query_router_app):
    """Create test client for query router app."""
    return TestClient(query_router_app)


@pytest.fixture
def query_router_disabled_client(query_router_disabled_app):
    """Create test client for disabled query router app."""
    return TestClient(query_router_disabled_app)


class TestQueryRouterRouteEndpoint:
    """Tests for POST /api/v1/query-router/route endpoint."""

    def test_route_success(self, query_router_client, sample_tenant_id):
        """Test successful query routing."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes in this document?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert "query_type" in data["data"]
        assert "confidence" in data["data"]
        assert "reasoning" in data["data"]
        assert "global_weight" in data["data"]
        assert "local_weight" in data["data"]
        assert "classification_method" in data["data"]
        assert "processing_time_ms" in data["data"]

    def test_route_with_use_llm_override(self, query_router_client, sample_tenant_id):
        """Test route with use_llm override."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": str(sample_tenant_id),
                "use_llm": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    def test_route_missing_query(self, query_router_client, sample_tenant_id):
        """Test route without required query field."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 422  # Validation error

    def test_route_missing_tenant_id(self, query_router_client):
        """Test route without required tenant_id."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_route_invalid_tenant_id(self, query_router_client):
        """Test route with invalid tenant_id format."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": "not-a-uuid",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_route_empty_query(self, query_router_client, sample_tenant_id):
        """Test route with empty query string."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 422  # Validation error

    def test_route_disabled_feature(self, query_router_disabled_client, sample_tenant_id):
        """Test route when feature is disabled."""
        response = query_router_disabled_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


class TestQueryRouterPatternsEndpoint:
    """Tests for GET /api/v1/query-router/patterns endpoint."""

    def test_patterns_success(self, query_router_client):
        """Test successful pattern list retrieval."""
        response = query_router_client.get("/api/v1/query-router/patterns")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert "global_patterns" in data["data"]
        assert "local_patterns" in data["data"]
        assert "global_pattern_count" in data["data"]
        assert "local_pattern_count" in data["data"]

        # Verify pattern counts match
        assert len(data["data"]["global_patterns"]) == data["data"]["global_pattern_count"]
        assert len(data["data"]["local_patterns"]) == data["data"]["local_pattern_count"]

    def test_patterns_disabled_feature(self, query_router_disabled_client):
        """Test patterns when feature is disabled."""
        response = query_router_disabled_client.get("/api/v1/query-router/patterns")

        assert response.status_code == 404
        assert "not enabled" in response.json()["detail"].lower()


class TestQueryRouterStatusEndpoint:
    """Tests for GET /api/v1/query-router/status endpoint."""

    def test_status_enabled(self, query_router_client):
        """Test status when feature is enabled."""
        response = query_router_client.get("/api/v1/query-router/status")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "data" in data
        assert "meta" in data
        assert data["data"]["enabled"] is True
        assert "use_llm" in data["data"]
        assert "llm_model" in data["data"]
        assert "confidence_threshold" in data["data"]
        assert "community_detection_available" in data["data"]
        assert "lazy_rag_available" in data["data"]

    def test_status_disabled(self, query_router_disabled_client):
        """Test status when feature is disabled."""
        response = query_router_disabled_client.get("/api/v1/query-router/status")

        assert response.status_code == 200
        data = response.json()
        assert data["data"]["enabled"] is False


class TestQueryRouterResponseFormat:
    """Tests for API response format compliance."""

    def test_route_response_format(self, query_router_client, sample_tenant_id):
        """Test route response follows standard format."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
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

    def test_route_response_query_type_values(self, query_router_client, sample_tenant_id):
        """Test route response has valid query_type values."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        query_type = data["data"]["query_type"]
        assert query_type in ["global", "local", "hybrid"]

    def test_route_response_weights_sum(self, query_router_client, sample_tenant_id):
        """Test route response weights are valid."""
        response = query_router_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        global_weight = data["data"]["global_weight"]
        local_weight = data["data"]["local_weight"]

        assert 0.0 <= global_weight <= 1.0
        assert 0.0 <= local_weight <= 1.0


class TestQueryRouterTenantIsolation:
    """Tests for multi-tenancy isolation."""

    def test_route_uses_tenant_id(
        self, query_router_app, mock_query_router, sample_tenant_id
    ):
        """Test that route passes tenant_id correctly."""
        client = TestClient(query_router_app)
        response = client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200

        # Verify router was called with correct tenant_id
        mock_query_router.route.assert_called_once()
        call_kwargs = mock_query_router.route.call_args[1]
        assert call_kwargs["tenant_id"] == str(sample_tenant_id)


class TestQueryRouterErrorHandling:
    """Tests for error handling in Query Router API."""

    def test_route_error_returns_500(
        self, query_router_app, mock_query_router, sample_tenant_id
    ):
        """Test that route errors return 500 status."""
        mock_query_router.route.side_effect = RuntimeError("Test error")
    
        client = TestClient(query_router_app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes?",
                "tenant_id": str(sample_tenant_id),
            },
        )
    
        assert response.status_code == 500
        # Sanitized error message
        assert "internal server error" in response.json()["detail"].lower()
    
class TestQueryRouterIntegration:
    """Integration tests using real QueryRouter (without mocks)."""

    @pytest.fixture
    def integration_app(self):
        """Create app with real QueryRouter."""
        from agentic_rag_backend.api.routes.query_router import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")

        # Set up app state with real-ish settings
        settings = _make_mock_settings(
            query_routing_enabled=True,
            query_routing_use_llm=False,
        )
        app.state.settings = settings
        # Don't set query_router - let endpoint create it

        return app

    @pytest.fixture
    def integration_client(self, integration_app):
        """Create test client for integration app."""
        return TestClient(integration_app)

    def test_route_global_query_integration(self, integration_client, sample_tenant_id):
        """Test routing a global query end-to-end."""
        response = integration_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What are the main themes in this codebase?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should classify as global
        assert data["data"]["query_type"] == "global"
        assert data["data"]["confidence"] > 0.5
        assert data["data"]["global_matches"] > 0

    def test_route_local_query_integration(self, integration_client, sample_tenant_id):
        """Test routing a local query end-to-end."""
        response = integration_client.post(
            "/api/v1/query-router/route",
            json={
                "query": "What is the QueryRouter class?",
                "tenant_id": str(sample_tenant_id),
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Should classify as local
        assert data["data"]["query_type"] == "local"
        assert data["data"]["confidence"] > 0.5
        assert data["data"]["local_matches"] > 0

    def test_patterns_integration(self, integration_client):
        """Test patterns endpoint end-to-end."""
        response = integration_client.get("/api/v1/query-router/patterns")

        assert response.status_code == 200
        data = response.json()

        # Should have multiple patterns
        assert data["data"]["global_pattern_count"] > 5
        assert data["data"]["local_pattern_count"] > 5
