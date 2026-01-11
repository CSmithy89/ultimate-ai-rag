"""Integration tests for A2A Resource Limits API endpoints.

Story 22-A2: Implement A2A Session Resource Limits

Tests cover:
- GET /a2a/metrics/{tenant_id} - Resource usage metrics endpoint
- Tenant isolation for metrics access
- Rate limiting on metrics endpoint
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agentic_rag_backend.api.routes.a2a import router
from agentic_rag_backend.core.errors import AppError, app_error_handler
from agentic_rag_backend.protocols.a2a_resource_limits import (
    A2AResourceLimits,
    InMemoryA2AResourceManager,
)
from agentic_rag_backend.rate_limit import InMemoryRateLimiter


pytestmark = pytest.mark.integration

# Valid UUID-format tenant IDs for testing
TENANT_UUID_1 = "12345678-1234-1234-1234-123456789abc"
TENANT_UUID_2 = "87654321-4321-4321-4321-cba987654321"


@pytest.fixture
def mock_rate_limiter() -> InMemoryRateLimiter:
    """Create an in-memory rate limiter for testing."""
    return InMemoryRateLimiter(max_requests=100, window_seconds=60)


@pytest.fixture
def resource_limits() -> A2AResourceLimits:
    """Create resource limits for testing."""
    return A2AResourceLimits(
        session_limit_per_tenant=10,
        message_limit_per_session=100,
        session_ttl_hours=1,
        message_rate_limit=30,
        cleanup_interval_minutes=1,
    )


@pytest.fixture
def resource_manager(resource_limits: A2AResourceLimits) -> InMemoryA2AResourceManager:
    """Create an in-memory resource manager for testing."""
    return InMemoryA2AResourceManager(limits=resource_limits)


@pytest.fixture
def app(
    resource_manager: InMemoryA2AResourceManager,
    mock_rate_limiter: InMemoryRateLimiter,
) -> FastAPI:
    """Create a test FastAPI app with the A2A router."""
    test_app = FastAPI()
    # Router already has prefix="/a2a", don't add another
    test_app.include_router(router)
    # Add exception handler for AppError (RFC 7807 responses)
    test_app.add_exception_handler(AppError, app_error_handler)

    # Set up app state
    test_app.state.a2a_resource_manager = resource_manager
    test_app.state.rate_limiter = mock_rate_limiter
    # Set None for unused dependencies
    test_app.state.a2a_middleware = None
    test_app.state.a2a_manager = None
    test_app.state.a2a_registry = None
    test_app.state.a2a_delegation_manager = None

    return test_app


@pytest.fixture
def client(app: FastAPI):
    """Create an HTTP client for testing."""
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


class TestGetResourceMetrics:
    """Tests for GET /a2a/metrics/{tenant_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_metrics_empty_tenant(
        self,
        client: AsyncClient,
    ) -> None:
        """Test getting metrics for tenant with no activity."""
        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == TENANT_UUID_1
            assert data["active_sessions"] == 0
            assert data["total_messages"] == 0
            assert data["session_limit"] == 10
            assert data["message_limit_per_session"] == 100
            assert data["message_rate_limit"] == 30
            assert "meta" in data

    @pytest.mark.asyncio
    async def test_get_metrics_with_activity(
        self,
        client: AsyncClient,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test getting metrics for tenant with sessions and messages."""
        # Register sessions and send messages
        await resource_manager.register_session("session-1", TENANT_UUID_1)
        await resource_manager.register_session("session-2", TENANT_UUID_1)
        await resource_manager.record_message("session-1")
        await resource_manager.record_message("session-1")
        await resource_manager.record_message("session-2")

        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == TENANT_UUID_1
            assert data["active_sessions"] == 2
            assert data["total_messages"] == 3

    @pytest.mark.asyncio
    async def test_get_metrics_tenant_isolation(
        self,
        client: AsyncClient,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that tenants can only view their own metrics."""
        # Create activity for TENANT_UUID_1
        await resource_manager.register_session("session-1", TENANT_UUID_1)
        await resource_manager.record_message("session-1")

        # Try to view TENANT_UUID_1 metrics as TENANT_UUID_2 (should fail)
        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_2},
            )

            assert response.status_code == 403
            data = response.json()
            assert "another tenant" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_metrics_own_tenant_success(
        self,
        client: AsyncClient,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that tenant can view their own metrics."""
        # Create activity for TENANT_UUID_1
        await resource_manager.register_session("session-1", TENANT_UUID_1)

        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == TENANT_UUID_1
            assert data["active_sessions"] == 1

    @pytest.mark.asyncio
    async def test_get_metrics_missing_tenant_header(
        self,
        client: AsyncClient,
    ) -> None:
        """Test that tenant header is required."""
        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_metrics_invalid_tenant_header(
        self,
        client: AsyncClient,
    ) -> None:
        """Test that tenant header must be valid UUID format."""
        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": "invalid-not-uuid"},
            )

            assert response.status_code == 400
            data = response.json()
            assert "uuid format" in data["detail"].lower()


class TestMetricsRateLimiting:
    """Tests for rate limiting on metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_rate_limited(
        self,
        app: FastAPI,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that metrics endpoint is rate limited."""
        # Create rate limiter that immediately rejects
        strict_limiter = InMemoryRateLimiter(max_requests=0, window_seconds=60)
        app.state.rate_limiter = strict_limiter

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )

            assert response.status_code == 429


class TestMetricsAfterSessionClose:
    """Tests for metrics after session lifecycle changes."""

    @pytest.mark.asyncio
    async def test_metrics_update_after_session_close(
        self,
        app: FastAPI,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that metrics update after session is closed."""
        # Register sessions
        await resource_manager.register_session("session-1", TENANT_UUID_1)
        await resource_manager.register_session("session-2", TENANT_UUID_1)

        # Check initial metrics
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )
            assert response.json()["active_sessions"] == 2

        # Close one session
        await resource_manager.close_session("session-1")

        # Check updated metrics (create new client)
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )
            data = response.json()
            assert data["active_sessions"] == 1

    @pytest.mark.asyncio
    async def test_metrics_message_count_after_close(
        self,
        client: AsyncClient,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that total message count persists after session close."""
        # Register session and send messages
        await resource_manager.register_session("session-1", TENANT_UUID_1)
        await resource_manager.record_message("session-1")
        await resource_manager.record_message("session-1")

        # Close session
        await resource_manager.close_session("session-1")

        # Message count should persist even after session close
        async with client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )
            data = response.json()
            assert data["active_sessions"] == 0
            assert data["total_messages"] == 2


class TestMultipleTenantMetrics:
    """Tests for multiple tenant scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_tenant_isolation(
        self,
        app: FastAPI,
        resource_manager: InMemoryA2AResourceManager,
    ) -> None:
        """Test that multiple tenants have isolated metrics."""
        # Create activity for both tenants
        await resource_manager.register_session("session-1", TENANT_UUID_1)
        await resource_manager.register_session("session-2", TENANT_UUID_1)
        await resource_manager.record_message("session-1")

        await resource_manager.register_session("session-3", TENANT_UUID_2)
        await resource_manager.record_message("session-3")
        await resource_manager.record_message("session-3")
        await resource_manager.record_message("session-3")

        # Check TENANT_UUID_1 metrics
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_1}",
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )
            data = response.json()
            assert data["active_sessions"] == 2
            assert data["total_messages"] == 1

        # Check TENANT_UUID_2 metrics
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/a2a/metrics/{TENANT_UUID_2}",
                headers={"X-Tenant-ID": TENANT_UUID_2},
            )
            data = response.json()
            assert data["active_sessions"] == 1
            assert data["total_messages"] == 3
