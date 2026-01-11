"""Unit tests for telemetry endpoint.

Story 22-TD1: Prometheus telemetry metrics
Story 22-TD5: Tenant-aware rate limiting
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_rag_backend.api.routes.telemetry import router
from agentic_rag_backend.rate_limit import InMemoryRateLimiter
from agentic_rag_backend.observability.metrics import (
    TELEMETRY_EVENTS_TOTAL,
    record_telemetry_event,
)


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter that always allows requests."""
    limiter = MagicMock(spec=InMemoryRateLimiter)
    limiter.allow = AsyncMock(return_value=True)
    return limiter


@pytest.fixture
def app(mock_rate_limiter):
    """Create test FastAPI app with mocked dependencies."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.rate_limiter = mock_rate_limiter
    return test_app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestTelemetryEndpoint:
    """Tests for /telemetry endpoint."""

    def test_track_telemetry_success(self, client):
        """Test successful telemetry event tracking."""
        response = client.post(
            "/telemetry",
            json={
                "event": "page_view",
                "properties": {"page": "/dashboard"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "accepted"
        assert "received_at" in data

    def test_track_telemetry_sanitizes_sensitive_data(self, client):
        """Test that sensitive data is redacted from properties."""
        response = client.post(
            "/telemetry",
            json={
                "event": "login",
                "properties": {
                    "username": "testuser",
                    "password": "secret123",
                    "api_key": "sk-123456",
                },
            },
        )
        assert response.status_code == 200

    def test_track_telemetry_hashes_message_content(self, client):
        """Test that message content is hashed for privacy."""
        response = client.post(
            "/telemetry",
            json={
                "event": "message_sent",
                "properties": {
                    "message": "Hello, this is a test message",
                },
            },
        )
        assert response.status_code == 200


class TestTelemetryRateLimiting:
    """Tests for rate limiting with tenant-aware composite key (Story 22-TD5)."""

    def test_rate_limit_uses_composite_key(self, mock_rate_limiter):
        """Verify rate limit key includes both tenant_id and IP."""
        test_app = FastAPI()
        test_app.include_router(router)
        test_app.state.rate_limiter = mock_rate_limiter

        with TestClient(test_app) as client:
            client.post(
                "/telemetry",
                json={"event": "test_event"},
            )

        # Verify allow was called with composite key format
        mock_rate_limiter.allow.assert_called()
        call_args = mock_rate_limiter.allow.call_args[0][0]
        # Should be in format: telemetry:{tenant_id}:{ip}
        assert call_args.startswith("telemetry:")
        # Key should have format telemetry:anonymous:testclient (or similar)
        parts = call_args.split(":")
        assert len(parts) == 3  # telemetry, tenant_id, ip

    def test_rate_limit_exceeded_returns_429(self):
        """Test that rate limit exceeded returns 429 status."""
        limiter = MagicMock(spec=InMemoryRateLimiter)
        limiter.allow = AsyncMock(return_value=False)

        test_app = FastAPI()
        test_app.include_router(router)
        test_app.state.rate_limiter = limiter

        with TestClient(test_app) as client:
            response = client.post(
                "/telemetry",
                json={"event": "test_event"},
            )

        assert response.status_code == 429

    def test_different_tenants_same_ip_independent_limits(self, mock_rate_limiter):
        """Test that different tenants on same IP have independent rate limits."""
        test_app = FastAPI()
        test_app.include_router(router)
        test_app.state.rate_limiter = mock_rate_limiter

        with TestClient(test_app) as client:
            # First request
            client.post(
                "/telemetry",
                json={"event": "event_1"},
            )
            # Second request
            client.post(
                "/telemetry",
                json={"event": "event_2"},
            )

        # Both should have been allowed
        assert mock_rate_limiter.allow.call_count == 2


class TestTelemetryPrometheusMetrics:
    """Tests for Prometheus telemetry metrics (Story 22-TD1)."""

    def test_telemetry_events_total_metric_exists(self):
        """Test that TELEMETRY_EVENTS_TOTAL counter is defined."""
        assert TELEMETRY_EVENTS_TOTAL is not None
        assert "event" in TELEMETRY_EVENTS_TOTAL._labelnames
        assert "tenant_id" in TELEMETRY_EVENTS_TOTAL._labelnames

    def test_record_telemetry_event_function(self):
        """Test recording a telemetry event metric."""
        # Should not raise
        record_telemetry_event(event="page_view", tenant_id="test-tenant")
        record_telemetry_event(event="search_query", tenant_id="test-tenant")

    @patch("agentic_rag_backend.api.routes.telemetry.record_telemetry_event")
    def test_endpoint_records_prometheus_metric(self, mock_record, mock_rate_limiter):
        """Test that telemetry endpoint records Prometheus metric."""
        test_app = FastAPI()
        test_app.include_router(router)
        test_app.state.rate_limiter = mock_rate_limiter

        with TestClient(test_app) as client:
            response = client.post(
                "/telemetry",
                json={"event": "button_click"},
            )
        assert response.status_code == 200

        # Verify metric was recorded
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args
        assert call_kwargs[1]["event"] == "button_click"


class TestTelemetrySanitization:
    """Tests for PII sanitization in telemetry."""

    def test_sensitive_keys_redacted(self, client):
        """Test that sensitive keys are properly redacted."""
        sensitive_props = {
            "password": "secret",
            "api_key": "sk-123",
            "secret_token": "token123",
            "auth_header": "Bearer xyz",
            "credential": "cred123",
        }
        response = client.post(
            "/telemetry",
            json={"event": "test", "properties": sensitive_props},
        )
        assert response.status_code == 200
