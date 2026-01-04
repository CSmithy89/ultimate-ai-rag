"""Tests for A2A Agent Registry with health monitoring."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agentic_rag_backend.protocols.a2a_messages import AgentCapability
from agentic_rag_backend.protocols.a2a_registry import (
    A2AAgentRegistry,
    RegistryConfig,
)


@pytest.fixture
def registry() -> A2AAgentRegistry:
    """Create a registry instance for testing."""
    config = RegistryConfig(
        heartbeat_interval_seconds=30,
        heartbeat_timeout_seconds=60,
        cleanup_interval_seconds=60,
    )
    return A2AAgentRegistry(config=config)


@pytest.fixture
def sample_capabilities() -> list[AgentCapability]:
    """Create sample capabilities for testing."""
    return [
        AgentCapability(
            name="hybrid_retrieve",
            description="Combined vector + graph retrieval",
        ),
        AgentCapability(
            name="vector_search",
            description="Semantic search",
        ),
    ]


@pytest.mark.asyncio
async def test_register_agent(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test agent registration."""
    registration = await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    assert registration.agent_id == "agent-001"
    assert registration.agent_type == "rag_engine"
    assert registration.endpoint_url == "http://localhost:8000"
    assert len(registration.capabilities) == 2
    assert registration.tenant_id == "tenant-123"
    assert registration.health_status == "healthy"


@pytest.mark.asyncio
async def test_register_agent_validation(registry: A2AAgentRegistry) -> None:
    """Test registration validation."""
    with pytest.raises(ValueError):
        await registry.register_agent(
            agent_id="",
            agent_type="rag_engine",
            endpoint_url="http://localhost:8000",
            capabilities=[],
            tenant_id="tenant-123",
        )


@pytest.mark.asyncio
async def test_register_agent_update_existing(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test updating an existing registration."""
    # Initial registration
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    # Update registration
    updated = await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:9000",  # New endpoint
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    assert updated.endpoint_url == "http://localhost:9000"


@pytest.mark.asyncio
async def test_register_agent_tenant_mismatch(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test that tenant mismatch raises PermissionError."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    with pytest.raises(PermissionError):
        await registry.register_agent(
            agent_id="agent-001",
            agent_type="rag_engine",
            endpoint_url="http://localhost:9000",
            capabilities=sample_capabilities,
            tenant_id="tenant-456",  # Different tenant
        )


@pytest.mark.asyncio
async def test_unregister_agent(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test agent unregistration."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    success = await registry.unregister_agent("agent-001", "tenant-123")
    assert success is True

    # Verify agent is gone
    agent = await registry.get_agent("agent-001", "tenant-123")
    assert agent is None


@pytest.mark.asyncio
async def test_unregister_agent_not_found(registry: A2AAgentRegistry) -> None:
    """Test unregistering non-existent agent."""
    success = await registry.unregister_agent("nonexistent", "tenant-123")
    assert success is False


@pytest.mark.asyncio
async def test_unregister_agent_tenant_mismatch(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test unregistering with wrong tenant."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    with pytest.raises(PermissionError):
        await registry.unregister_agent("agent-001", "tenant-456")


@pytest.mark.asyncio
async def test_heartbeat(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test heartbeat recording."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    success = await registry.heartbeat("agent-001", "tenant-123")
    assert success is True


@pytest.mark.asyncio
async def test_heartbeat_not_found(registry: A2AAgentRegistry) -> None:
    """Test heartbeat for non-existent agent."""
    success = await registry.heartbeat("nonexistent", "tenant-123")
    assert success is False


@pytest.mark.asyncio
async def test_heartbeat_tenant_mismatch(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test heartbeat with wrong tenant."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    with pytest.raises(PermissionError):
        await registry.heartbeat("agent-001", "tenant-456")


@pytest.mark.asyncio
async def test_get_agent(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test getting a specific agent."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    agent = await registry.get_agent("agent-001", "tenant-123")
    assert agent is not None
    assert agent.agent_id == "agent-001"


@pytest.mark.asyncio
async def test_get_agent_wrong_tenant(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test getting agent with wrong tenant returns None."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    agent = await registry.get_agent("agent-001", "tenant-456")
    assert agent is None


@pytest.mark.asyncio
async def test_list_agents(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test listing agents for a tenant."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )
    await registry.register_agent(
        agent_id="agent-002",
        agent_type="search_agent",
        endpoint_url="http://localhost:9000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )
    await registry.register_agent(
        agent_id="agent-003",
        agent_type="rag_engine",
        endpoint_url="http://localhost:7000",
        capabilities=sample_capabilities,
        tenant_id="tenant-456",  # Different tenant
    )

    agents = await registry.list_agents("tenant-123")
    assert len(agents) == 2
    agent_ids = {a.agent_id for a in agents}
    assert "agent-001" in agent_ids
    assert "agent-002" in agent_ids
    assert "agent-003" not in agent_ids


@pytest.mark.asyncio
async def test_list_agents_healthy_only(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test listing only healthy agents."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )
    await registry.register_agent(
        agent_id="agent-002",
        agent_type="rag_engine",
        endpoint_url="http://localhost:9000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    # Make one agent unhealthy by setting old heartbeat
    registry._agents["agent-002"].last_heartbeat = datetime.now(
        timezone.utc
    ) - timedelta(seconds=120)

    healthy_agents = await registry.list_agents("tenant-123", healthy_only=True)
    assert len(healthy_agents) == 1
    assert healthy_agents[0].agent_id == "agent-001"


@pytest.mark.asyncio
async def test_find_agents_by_capability(
    registry: A2AAgentRegistry,
) -> None:
    """Test finding agents by capability."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[
            AgentCapability(name="hybrid_retrieve", description="Hybrid retrieval"),
        ],
        tenant_id="tenant-123",
    )
    await registry.register_agent(
        agent_id="agent-002",
        agent_type="search_agent",
        endpoint_url="http://localhost:9000",
        capabilities=[
            AgentCapability(name="vector_search", description="Vector search"),
        ],
        tenant_id="tenant-123",
    )

    hybrid_agents = await registry.find_agents_by_capability(
        "hybrid_retrieve", "tenant-123"
    )
    assert len(hybrid_agents) == 1
    assert hybrid_agents[0].agent_id == "agent-001"

    vector_agents = await registry.find_agents_by_capability(
        "vector_search", "tenant-123"
    )
    assert len(vector_agents) == 1
    assert vector_agents[0].agent_id == "agent-002"


@pytest.mark.asyncio
async def test_find_agents_by_type(
    registry: A2AAgentRegistry,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test finding agents by type."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )
    await registry.register_agent(
        agent_id="agent-002",
        agent_type="search_agent",
        endpoint_url="http://localhost:9000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    rag_agents = await registry.find_agents_by_type("rag_engine", "tenant-123")
    assert len(rag_agents) == 1
    assert rag_agents[0].agent_id == "agent-001"


@pytest.mark.asyncio
async def test_register_self(registry: A2AAgentRegistry) -> None:
    """Test self-registration with predefined capabilities."""
    registration = await registry.register_self(
        agent_id="rag-system-001",
        endpoint_url="http://localhost:8000",
        tenant_id="tenant-123",
    )

    assert registration.agent_id == "rag-system-001"
    assert registration.agent_type == "rag_engine"
    assert len(registration.capabilities) > 0
    assert registration.has_capability("hybrid_retrieve")
    assert registration.metadata.get("protocol") == "a2a-v1"


@pytest.mark.asyncio
async def test_health_status_updates() -> None:
    """Test that health status is updated based on heartbeat timing."""
    config = RegistryConfig(
        heartbeat_timeout_seconds=1,
        cleanup_interval_seconds=1,
    )
    registry = A2AAgentRegistry(config=config)

    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[],
        tenant_id="tenant-123",
    )

    # Agent should initially be healthy
    agent = await registry.get_agent("agent-001", "tenant-123")
    assert agent is not None
    assert agent.health_status == "healthy"

    # Simulate time passing without heartbeat
    registry._agents["agent-001"].last_heartbeat = datetime.now(
        timezone.utc
    ) - timedelta(seconds=2)

    # Getting agent should update health status
    agent = await registry.get_agent("agent-001", "tenant-123")
    assert agent is not None
    assert agent.health_status == "unhealthy"


@pytest.mark.asyncio
async def test_cleanup_task() -> None:
    """Test that cleanup task removes stale agents."""
    config = RegistryConfig(
        heartbeat_timeout_seconds=1,
        cleanup_interval_seconds=1,
    )
    registry = A2AAgentRegistry(config=config)

    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[],
        tenant_id="tenant-123",
    )

    # Make agent very stale (beyond 3x timeout)
    registry._agents["agent-001"].last_heartbeat = datetime.now(
        timezone.utc
    ) - timedelta(seconds=5)

    # Run cleanup manually
    await registry._cleanup_unhealthy_agents()

    # Agent should be removed
    assert "agent-001" not in registry._agents


@pytest.mark.asyncio
async def test_cleanup_task_start_stop() -> None:
    """Test cleanup task can be started and stopped."""
    config = RegistryConfig(cleanup_interval_seconds=60)
    registry = A2AAgentRegistry(config=config)

    await registry.start_cleanup_task()
    assert registry._cleanup_task is not None
    assert not registry._cleanup_task.done()

    await registry.stop_cleanup_task()
    assert registry._cleanup_task is None


@pytest.mark.asyncio
async def test_concurrent_registration() -> None:
    """Test concurrent agent registration is thread-safe."""
    registry = A2AAgentRegistry()

    async def register_agent(idx: int) -> None:
        await registry.register_agent(
            agent_id=f"agent-{idx}",
            agent_type="rag_engine",
            endpoint_url=f"http://localhost:{8000 + idx}",
            capabilities=[],
            tenant_id="tenant-123",
        )

    await asyncio.gather(*[register_agent(i) for i in range(10)])

    agents = await registry.list_agents("tenant-123")
    assert len(agents) == 10


@pytest.mark.asyncio
async def test_redis_persistence() -> None:
    """Test that agents are persisted to Redis."""

    class FakeRedis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}
            self.sets: dict[str, set[str]] = {}

        async def set(self, key: str, value: str, ex: int | None = None) -> None:
            self.store[key] = value

        async def get(self, key: str) -> str | None:
            return self.store.get(key)

        async def delete(self, key: str) -> None:
            self.store.pop(key, None)

        async def sadd(self, key: str, value: str) -> None:
            if key not in self.sets:
                self.sets[key] = set()
            self.sets[key].add(value)

        async def srem(self, key: str, value: str) -> None:
            if key in self.sets:
                self.sets[key].discard(value)

        async def smembers(self, key: str):
            return self.sets.get(key, set())

        async def expire(self, key: str, ttl: int) -> None:
            pass

    redis_mock = MagicMock()
    redis_mock.client = FakeRedis()

    registry = A2AAgentRegistry(redis_client=redis_mock)

    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[],
        tenant_id="tenant-123",
    )

    # Verify data was persisted
    assert len(redis_mock.client.store) > 0

    # Create new registry and load from Redis
    registry2 = A2AAgentRegistry(redis_client=redis_mock)
    agent = await registry2.get_agent("agent-001", "tenant-123")

    assert agent is not None
    assert agent.agent_id == "agent-001"
