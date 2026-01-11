"""Integration tests for A2A Middleware API endpoints.

Tests cover:
- POST /a2a/middleware/agents/register - Agent registration with tenant validation
- GET /a2a/middleware/agents - List agents for tenant
- GET /a2a/middleware/capabilities - Discover capabilities
- POST /a2a/middleware/agents/{agent_id}/delegate - Task delegation

Security tests:
- Tenant prefix validation in agent_id
- Cross-tenant access prevention
- Rate limiting behavior
- Tenant ID format validation (UUID format)
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agentic_rag_backend.api.routes.a2a import router
from agentic_rag_backend.core.errors import AppError, app_error_handler
from agentic_rag_backend.protocols.a2a_middleware import (
    A2AAgentCapability,
    A2AAgentInfo,
    A2AMiddlewareAgent,
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
def middleware() -> A2AMiddlewareAgent:
    """Create an A2A middleware for testing."""
    return A2AMiddlewareAgent(
        agent_id="test:middleware",
        name="Test Middleware",
    )


@pytest.fixture
def app(
    middleware: A2AMiddlewareAgent,
    mock_rate_limiter: InMemoryRateLimiter,
) -> FastAPI:
    """Create a test FastAPI app with the A2A router."""
    test_app = FastAPI()
    # Router already has prefix="/a2a", don't add another
    test_app.include_router(router)
    # Add exception handler for AppError (RFC 7807 responses)
    test_app.add_exception_handler(AppError, app_error_handler)

    # Set up app state
    test_app.state.a2a_middleware = middleware
    test_app.state.rate_limiter = mock_rate_limiter
    test_app.state.a2a_manager = None
    test_app.state.a2a_registry = None
    test_app.state.a2a_delegation_manager = None

    return test_app


@pytest.fixture
def client(app: FastAPI):
    """Create an HTTP client for testing."""
    # Use a sync fixture that returns a context manager
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    )


class TestRegisterAgent:
    """Tests for POST /a2a/middleware/agents/register endpoint."""

    @pytest.mark.asyncio
    async def test_register_agent_success(
        self,
        client: AsyncClient,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test successful agent registration."""
        async with client:
            response = await client.post(
                "/a2a/middleware/agents/register",
                json={
                    "agent_id": f"{TENANT_UUID_1}:my-agent",
                    "name": "My Agent",
                    "description": "A test agent",
                    "capabilities": [
                        {
                            "name": "vector_search",
                            "description": "Search with vectors",
                            "input_schema": {},
                            "output_schema": {},
                        }
                    ],
                    "endpoint": "http://agent.example.com:8001/ag-ui",
                },
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "registered"
            assert data["agent_id"] == f"{TENANT_UUID_1}:my-agent"
            assert "meta" in data

            # Verify agent was registered in middleware
            agent = middleware.get_agent(f"{TENANT_UUID_1}:my-agent")
            assert agent is not None
            assert agent.name == "My Agent"

    @pytest.mark.asyncio
    async def test_register_agent_tenant_prefix_mismatch(
        self,
        client: AsyncClient,
    ) -> None:
        """Test registration fails when agent_id doesn't match tenant."""
        async with client:
            response = await client.post(
                "/a2a/middleware/agents/register",
                json={
                    "agent_id": f"{TENANT_UUID_2}:my-agent",
                    "name": "My Agent",
                    "description": "A test agent",
                    "capabilities": [],
                    "endpoint": "http://agent.example.com:8001/ag-ui",
                },
                headers={"X-Tenant-ID": TENANT_UUID_1},
            )

            assert response.status_code == 400
            data = response.json()
            assert "tenant_id" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_register_agent_missing_tenant_header(
        self,
        client: AsyncClient,
    ) -> None:
        """Test registration fails without tenant header."""
        async with client:
            response = await client.post(
                "/a2a/middleware/agents/register",
                json={
                    "agent_id": f"{TENANT_UUID_1}:my-agent",
                    "name": "My Agent",
                    "description": "A test agent",
                    "capabilities": [],
                    "endpoint": "http://agent.example.com:8001/ag-ui",
                },
            )

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_register_agent_validation_errors(
        self,
        client: AsyncClient,
    ) -> None:
        """Test registration fails with invalid data."""
        async with client:
            # Missing required fields
            response = await client.post(
                "/a2a/middleware/agents/register",
                json={
                    "agent_id": f"{TENANT_UUID_1}:my-agent",
                    # Missing name, description, endpoint
                },
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 422


class TestListAgents:
    """Tests for GET /a2a/middleware/agents endpoint."""

    @pytest.mark.asyncio
    async def test_list_agents_empty(self, client: AsyncClient) -> None:
        """Test listing agents when none are registered."""
        async with client:
            response = await client.get(
                "/a2a/middleware/agents",
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["agents"] == []
            assert "meta" in data

    @pytest.mark.asyncio
    async def test_list_agents_with_registered(
        self,
        client: AsyncClient,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test listing agents after registration."""
        # Register an agent
        agent = A2AAgentInfo(
            agent_id=f"{TENANT_UUID_1}:agent1",
            name="Agent 1",
            description="Test agent",
            capabilities=[
                A2AAgentCapability(
                    name="search",
                    description="Search capability",
                )
            ],
            endpoint="http://agent.example.com:8001",
        )
        middleware.register_agent(agent)

        async with client:
            response = await client.get(
                "/a2a/middleware/agents",
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["agents"]) == 1
            assert data["agents"][0]["agent_id"] == f"{TENANT_UUID_1}:agent1"
            assert data["agents"][0]["name"] == "Agent 1"

    @pytest.mark.asyncio
    async def test_list_agents_tenant_isolation(
        self,
        client: AsyncClient,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test that agents are filtered by tenant."""
        # Register agents for different tenants
        agent1 = A2AAgentInfo(
            agent_id=f"{TENANT_UUID_1}:agent1",
            name="Agent 1",
            description="Tenant 123 agent",
            capabilities=[],
            endpoint="http://agent.example.com:8001",
        )
        agent2 = A2AAgentInfo(
            agent_id=f"{TENANT_UUID_2}:agent2",
            name="Agent 2",
            description="Tenant 456 agent",
            capabilities=[],
            endpoint="http://agent.example.com:8002",
        )
        middleware.register_agent(agent1)
        middleware.register_agent(agent2)

        async with client:
            # Request as tenant123
            response = await client.get(
                "/a2a/middleware/agents",
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["agents"]) == 1
            assert data["agents"][0]["agent_id"] == f"{TENANT_UUID_1}:agent1"


class TestListCapabilities:
    """Tests for GET /a2a/middleware/capabilities endpoint."""

    @pytest.mark.asyncio
    async def test_list_capabilities_empty(self, client: AsyncClient) -> None:
        """Test listing capabilities when none are registered."""
        async with client:
            response = await client.get(
                "/a2a/middleware/capabilities",
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["capabilities"] == []

    @pytest.mark.asyncio
    async def test_list_capabilities_with_registered(
        self,
        client: AsyncClient,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test listing capabilities from registered agents."""
        agent = A2AAgentInfo(
            agent_id=f"{TENANT_UUID_1}:agent1",
            name="Agent 1",
            description="Test agent",
            capabilities=[
                A2AAgentCapability(
                    name="vector_search",
                    description="Vector search",
                ),
                A2AAgentCapability(
                    name="graph_traverse",
                    description="Graph traversal",
                ),
            ],
            endpoint="http://agent.example.com:8001",
        )
        middleware.register_agent(agent)

        async with client:
            response = await client.get(
                "/a2a/middleware/capabilities",
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["capabilities"]) == 2

    @pytest.mark.asyncio
    async def test_list_capabilities_with_filter(
        self,
        client: AsyncClient,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test filtering capabilities by name."""
        agent = A2AAgentInfo(
            agent_id=f"{TENANT_UUID_1}:agent1",
            name="Agent 1",
            description="Test agent",
            capabilities=[
                A2AAgentCapability(
                    name="vector_search",
                    description="Vector search",
                ),
                A2AAgentCapability(
                    name="graph_traverse",
                    description="Graph traversal",
                ),
            ],
            endpoint="http://agent.example.com:8001",
        )
        middleware.register_agent(agent)

        async with client:
            response = await client.get(
                "/a2a/middleware/capabilities?filter=vector",
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data["capabilities"]) == 1
            assert data["capabilities"][0]["capability"]["name"] == "vector_search"


class TestDelegateTask:
    """Tests for POST /a2a/middleware/agents/{agent_id}/delegate endpoint."""

    @pytest.mark.asyncio
    async def test_delegate_task_tenant_mismatch(
        self,
        client: AsyncClient,
    ) -> None:
        """Test delegation fails for cross-tenant agent."""
        async with client:
            response = await client.post(
                f"/a2a/middleware/agents/{TENANT_UUID_2}:agent1/delegate",
                json={
                    "capability_name": "vector_search",
                    "input_data": {"query": "test"},
                },
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 403
            data = response.json()
            assert "scope" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_delegate_task_agent_not_found(
        self,
        client: AsyncClient,
    ) -> None:
        """Test delegation fails when agent not registered."""
        async with client:
            response = await client.post(
                f"/a2a/middleware/agents/{TENANT_UUID_1}:nonexistent/delegate",
                json={
                    "capability_name": "vector_search",
                    "input_data": {"query": "test"},
                },
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delegate_task_capability_not_found(
        self,
        client: AsyncClient,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test delegation fails when capability not found."""
        agent = A2AAgentInfo(
            agent_id=f"{TENANT_UUID_1}:agent1",
            name="Agent 1",
            description="Test agent",
            capabilities=[
                A2AAgentCapability(
                    name="vector_search",
                    description="Vector search",
                )
            ],
            endpoint="http://agent.example.com:8001",
        )
        middleware.register_agent(agent)

        async with client:
            response = await client.post(
                f"/a2a/middleware/agents/{TENANT_UUID_1}:agent1/delegate",
                json={
                    "capability_name": "nonexistent_capability",
                    "input_data": {"query": "test"},
                },
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 404


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limiting_on_registration(
        self,
        app: FastAPI,
    ) -> None:
        """Test that registration is rate limited."""
        # Create rate limiter that immediately rejects
        strict_limiter = InMemoryRateLimiter(max_requests=0, window_seconds=60)
        app.state.rate_limiter = strict_limiter

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/a2a/middleware/agents/register",
                json={
                    "agent_id": f"{TENANT_UUID_1}:my-agent",
                    "name": "My Agent",
                    "description": "A test agent",
                    "capabilities": [],
                    "endpoint": "http://agent.example.com:8001/ag-ui",
                },
                headers={"X-Tenant-ID": f"{TENANT_UUID_1}"},
            )

            assert response.status_code == 429
