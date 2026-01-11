"""Unit tests for A2A Middleware Agent.

Tests cover:
- Agent registration and unregistration
- Agent listing with tenant filtering
- Capability discovery with optional filtering
- Task delegation via AG-UI SSE streaming
- HTTP client lifecycle management
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag_backend.core.errors import (
    A2AAgentNotFoundError,
    A2ACapabilityNotFoundError,
)
from agentic_rag_backend.protocols.a2a_middleware import (
    A2AAgentCapability,
    A2AAgentInfo,
    A2AMiddlewareAgent,
)


@pytest.fixture
def sample_capability() -> A2AAgentCapability:
    """Create a sample capability for testing."""
    return A2AAgentCapability(
        name="vector_search",
        description="Search using vector embeddings",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        output_schema={"type": "array"},
    )


@pytest.fixture
def sample_agent_info(sample_capability: A2AAgentCapability) -> A2AAgentInfo:
    """Create sample agent info for testing."""
    # Use a public-looking domain for tests (not localhost, which triggers SSRF protection)
    return A2AAgentInfo(
        agent_id="tenant123:search-agent",
        name="Search Agent",
        description="Agent for vector search",
        capabilities=[sample_capability],
        endpoint="http://agent.example.com:8001/ag-ui",
    )


@pytest.fixture
def middleware() -> A2AMiddlewareAgent:
    """Create a middleware instance for testing."""
    return A2AMiddlewareAgent(
        agent_id="system:middleware",
        name="Test Middleware",
    )


class TestA2AMiddlewareAgentInit:
    """Tests for A2AMiddlewareAgent initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        middleware = A2AMiddlewareAgent(
            agent_id="test:middleware",
            name="Test",
        )
        assert middleware.agent_id == "test:middleware"
        assert middleware.name == "Test"
        assert middleware.capabilities == []
        assert middleware._registered_agents == {}
        assert middleware._http_client is None

    def test_init_with_capabilities(self, sample_capability: A2AAgentCapability) -> None:
        """Test initialization with capabilities."""
        middleware = A2AMiddlewareAgent(
            agent_id="test:middleware",
            name="Test",
            capabilities=[sample_capability],
        )
        assert len(middleware.capabilities) == 1
        assert middleware.capabilities[0].name == "vector_search"


class TestAgentRegistration:
    """Tests for agent registration operations."""

    def test_register_agent(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test registering an agent."""
        middleware.register_agent(sample_agent_info)

        assert "tenant123:search-agent" in middleware._registered_agents
        registered = middleware._registered_agents["tenant123:search-agent"]
        assert registered.name == "Search Agent"

    def test_register_agent_overwrites(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test that re-registering updates existing entry."""
        middleware.register_agent(sample_agent_info)

        updated_info = A2AAgentInfo(
            agent_id="tenant123:search-agent",
            name="Updated Agent",
            description="Updated description",
            capabilities=[],
            endpoint="http://localhost:8002/ag-ui",
        )
        middleware.register_agent(updated_info)

        assert len(middleware._registered_agents) == 1
        assert middleware._registered_agents["tenant123:search-agent"].name == "Updated Agent"

    def test_unregister_agent_exists(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test unregistering an existing agent."""
        middleware.register_agent(sample_agent_info)
        result = middleware.unregister_agent("tenant123:search-agent")

        assert result is True
        assert "tenant123:search-agent" not in middleware._registered_agents

    def test_unregister_agent_not_found(self, middleware: A2AMiddlewareAgent) -> None:
        """Test unregistering a non-existent agent."""
        result = middleware.unregister_agent("nonexistent:agent")
        assert result is False

    def test_get_agent_exists(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test getting an existing agent."""
        middleware.register_agent(sample_agent_info)
        agent = middleware.get_agent("tenant123:search-agent")

        assert agent is not None
        assert agent.name == "Search Agent"

    def test_get_agent_not_found(self, middleware: A2AMiddlewareAgent) -> None:
        """Test getting a non-existent agent."""
        agent = middleware.get_agent("nonexistent:agent")
        assert agent is None


class TestAgentListing:
    """Tests for listing agents."""

    def test_list_agents_for_tenant_empty(self, middleware: A2AMiddlewareAgent) -> None:
        """Test listing agents when none are registered."""
        agents = middleware.list_agents_for_tenant("tenant123")
        assert agents == []

    def test_list_agents_for_tenant_single(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test listing agents with one registered for tenant."""
        middleware.register_agent(sample_agent_info)
        agents = middleware.list_agents_for_tenant("tenant123")

        assert len(agents) == 1
        assert agents[0].agent_id == "tenant123:search-agent"

    def test_list_agents_for_tenant_filters_by_tenant(
        self,
        middleware: A2AMiddlewareAgent,
        sample_capability: A2AAgentCapability,
    ) -> None:
        """Test that agents are filtered by tenant prefix."""
        # Register agents for different tenants
        agent1 = A2AAgentInfo(
            agent_id="tenant123:agent1",
            name="Agent 1",
            description="First agent",
            capabilities=[sample_capability],
            endpoint="http://localhost:8001",
        )
        agent2 = A2AAgentInfo(
            agent_id="tenant456:agent2",
            name="Agent 2",
            description="Second agent",
            capabilities=[sample_capability],
            endpoint="http://localhost:8002",
        )

        middleware.register_agent(agent1)
        middleware.register_agent(agent2)

        # List should only include tenant123 agents
        agents = middleware.list_agents_for_tenant("tenant123")
        assert len(agents) == 1
        assert agents[0].agent_id == "tenant123:agent1"

        # List for tenant456
        agents = middleware.list_agents_for_tenant("tenant456")
        assert len(agents) == 1
        assert agents[0].agent_id == "tenant456:agent2"


class TestCapabilityDiscovery:
    """Tests for capability discovery."""

    def test_discover_capabilities_empty(self, middleware: A2AMiddlewareAgent) -> None:
        """Test discovering capabilities when none are registered."""
        capabilities = middleware.discover_capabilities("tenant123")
        assert capabilities == []

    def test_discover_capabilities_single_agent(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test discovering capabilities from one agent."""
        middleware.register_agent(sample_agent_info)
        capabilities = middleware.discover_capabilities("tenant123")

        assert len(capabilities) == 1
        assert capabilities[0]["agent_id"] == "tenant123:search-agent"
        assert capabilities[0]["capability"]["name"] == "vector_search"

    def test_discover_capabilities_with_filter(
        self,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test discovering capabilities with a filter."""
        cap1 = A2AAgentCapability(
            name="vector_search",
            description="Vector search",
        )
        cap2 = A2AAgentCapability(
            name="graph_traverse",
            description="Graph traversal",
        )

        agent = A2AAgentInfo(
            agent_id="tenant123:multi-agent",
            name="Multi Agent",
            description="Agent with multiple capabilities",
            capabilities=[cap1, cap2],
            endpoint="http://localhost:8001",
        )
        middleware.register_agent(agent)

        # Filter by "vector"
        capabilities = middleware.discover_capabilities("tenant123", "vector")
        assert len(capabilities) == 1
        assert capabilities[0]["capability"]["name"] == "vector_search"

        # Filter by "graph"
        capabilities = middleware.discover_capabilities("tenant123", "graph")
        assert len(capabilities) == 1
        assert capabilities[0]["capability"]["name"] == "graph_traverse"

    def test_discover_capabilities_filters_by_tenant(
        self,
        middleware: A2AMiddlewareAgent,
        sample_capability: A2AAgentCapability,
    ) -> None:
        """Test that capability discovery filters by tenant."""
        agent1 = A2AAgentInfo(
            agent_id="tenant123:agent1",
            name="Agent 1",
            description="First agent",
            capabilities=[sample_capability],
            endpoint="http://localhost:8001",
        )
        agent2 = A2AAgentInfo(
            agent_id="tenant456:agent2",
            name="Agent 2",
            description="Second agent",
            capabilities=[sample_capability],
            endpoint="http://localhost:8002",
        )

        middleware.register_agent(agent1)
        middleware.register_agent(agent2)

        # Only tenant123 capabilities
        capabilities = middleware.discover_capabilities("tenant123")
        assert len(capabilities) == 1
        assert capabilities[0]["agent_id"] == "tenant123:agent1"


class TestTaskDelegation:
    """Tests for task delegation."""

    @pytest.mark.asyncio
    async def test_delegate_task_agent_not_found(
        self,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test delegation fails when agent not found."""
        with pytest.raises(A2AAgentNotFoundError) as exc_info:
            async for _ in middleware.delegate_task(
                target_agent_id="nonexistent:agent",
                capability_name="vector_search",
                input_data={"query": "test"},
            ):
                pass

        assert "nonexistent:agent" in str(exc_info.value.details)

    @pytest.mark.asyncio
    async def test_delegate_task_capability_not_found(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test delegation fails when capability not found."""
        middleware.register_agent(sample_agent_info)

        with pytest.raises(A2ACapabilityNotFoundError) as exc_info:
            async for _ in middleware.delegate_task(
                target_agent_id="tenant123:search-agent",
                capability_name="nonexistent_capability",
                input_data={"query": "test"},
            ):
                pass

        assert "nonexistent_capability" in str(exc_info.value.details)

    @pytest.mark.asyncio
    async def test_delegate_task_success(
        self,
        middleware: A2AMiddlewareAgent,
        sample_agent_info: A2AAgentInfo,
    ) -> None:
        """Test successful task delegation with mocked HTTP response."""
        middleware.register_agent(sample_agent_info)

        # Mock the HTTP client response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_iter_lines():
            yield 'data: {"type": "result", "content": "test output"}'
            yield 'data: {"type": "done"}'

        mock_response.aiter_lines = mock_iter_lines

        with patch.object(middleware, "_get_http_client") as mock_client:
            mock_client.return_value.stream = MagicMock(
                return_value=AsyncContextManagerMock(mock_response)
            )

            events = []
            async for event in middleware.delegate_task(
                target_agent_id="tenant123:search-agent",
                capability_name="vector_search",
                input_data={"query": "test"},
            ):
                events.append(event)

            assert len(events) == 2
            assert events[0]["type"] == "result"
            assert events[1]["type"] == "done"


class TestHttpClientLifecycle:
    """Tests for HTTP client lifecycle management."""

    @pytest.mark.asyncio
    async def test_get_http_client_creates_client(
        self,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test that _get_http_client creates a client on first call."""
        assert middleware._http_client is None

        client = await middleware._get_http_client()

        assert client is not None
        assert middleware._http_client is client

    @pytest.mark.asyncio
    async def test_get_http_client_reuses_client(
        self,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test that _get_http_client reuses existing client."""
        client1 = await middleware._get_http_client()
        client2 = await middleware._get_http_client()

        assert client1 is client2

    @pytest.mark.asyncio
    async def test_close_with_no_client(
        self,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test close when no client was created."""
        # Should not raise
        await middleware.close()
        assert middleware._http_client is None

    @pytest.mark.asyncio
    async def test_close_with_client(
        self,
        middleware: A2AMiddlewareAgent,
    ) -> None:
        """Test close properly closes the HTTP client."""
        # Create a client
        await middleware._get_http_client()
        assert middleware._http_client is not None

        # Close it
        await middleware.close()
        assert middleware._http_client is None


class AsyncContextManagerMock:
    """Helper class to mock async context managers."""

    def __init__(self, mock_response: Any) -> None:
        self.mock_response = mock_response

    async def __aenter__(self) -> Any:
        return self.mock_response

    async def __aexit__(self, *args: Any) -> None:
        pass
