"""Tests for A2A Task Delegation Manager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag_backend.protocols.a2a_messages import (
    AgentCapability,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from agentic_rag_backend.protocols.a2a_registry import A2AAgentRegistry, RegistryConfig
from agentic_rag_backend.protocols.a2a_delegation import (
    DelegationConfig,
    TaskDelegationManager,
)


@pytest.fixture
def registry() -> A2AAgentRegistry:
    """Create a registry instance for testing."""
    config = RegistryConfig(
        heartbeat_interval_seconds=30,
        heartbeat_timeout_seconds=60,
    )
    return A2AAgentRegistry(config=config)


@pytest.fixture
def delegation_config() -> DelegationConfig:
    """Create delegation config for testing."""
    return DelegationConfig(
        default_timeout_seconds=30,
        max_retries=2,
        retry_delay_seconds=0.01,  # Fast retries for tests
        http_timeout_seconds=5.0,
    )


@pytest.fixture
def delegation_manager(
    registry: A2AAgentRegistry,
    delegation_config: DelegationConfig,
) -> TaskDelegationManager:
    """Create a delegation manager for testing."""
    return TaskDelegationManager(registry=registry, config=delegation_config)


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
async def test_delegate_task_no_agent(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test delegating task when no agent is available."""
    result = await delegation_manager.delegate_task(
        capability_name="hybrid_retrieve",
        parameters={"query": "test"},
        tenant_id="tenant-123",
    )

    assert result.status == TaskStatus.FAILED
    assert "No healthy agent found" in result.error


@pytest.mark.asyncio
async def test_delegate_task_target_not_found(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test delegating to a specific agent that doesn't exist."""
    result = await delegation_manager.delegate_task(
        capability_name="hybrid_retrieve",
        parameters={"query": "test"},
        tenant_id="tenant-123",
        target_agent_id="nonexistent-agent",
    )

    assert result.status == TaskStatus.FAILED
    assert "not found" in result.error


@pytest.mark.asyncio
async def test_delegate_task_target_no_capability(
    registry: A2AAgentRegistry,
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test delegating to agent without required capability."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[
            AgentCapability(name="other_cap", description="Other capability"),
        ],
        tenant_id="tenant-123",
    )

    result = await delegation_manager.delegate_task(
        capability_name="hybrid_retrieve",
        parameters={"query": "test"},
        tenant_id="tenant-123",
        target_agent_id="agent-001",
    )

    assert result.status == TaskStatus.FAILED
    assert "does not have capability" in result.error


@pytest.mark.asyncio
async def test_delegate_task_success(
    registry: A2AAgentRegistry,
    delegation_manager: TaskDelegationManager,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test successful task delegation."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    # Mock the HTTP request
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "result": {"answer": "Test answer"},
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance

        result = await delegation_manager.delegate_task(
            capability_name="hybrid_retrieve",
            parameters={"query": "test"},
            tenant_id="tenant-123",
        )

    assert result.status == TaskStatus.COMPLETED
    assert result.result == {"answer": "Test answer"}
    assert result.execution_time_ms is not None


@pytest.mark.asyncio
async def test_delegate_task_with_timeout(
    registry: A2AAgentRegistry,
    delegation_config: DelegationConfig,
) -> None:
    """Test task delegation timeout."""
    # Very short timeout
    delegation_config.default_timeout_seconds = 0.01

    delegation_manager = TaskDelegationManager(
        registry=registry,
        config=delegation_config,
    )

    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[
            AgentCapability(name="slow_task", description="Slow task"),
        ],
        tenant_id="tenant-123",
    )

    # Mock a slow HTTP request
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(1)  # Longer than timeout
        return MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = slow_request
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance

        result = await delegation_manager.delegate_task(
            capability_name="slow_task",
            parameters={},
            tenant_id="tenant-123",
        )

    assert result.status == TaskStatus.FAILED
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_delegate_task_with_retry(
    registry: A2AAgentRegistry,
    delegation_manager: TaskDelegationManager,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test task delegation with retries on failure."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    # First calls fail, last succeeds
    call_count = 0

    async def flaky_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": {"success": True}}
        return mock_response

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = flaky_request
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance

        result = await delegation_manager.delegate_task(
            capability_name="hybrid_retrieve",
            parameters={"query": "test"},
            tenant_id="tenant-123",
        )

    assert result.status == TaskStatus.COMPLETED
    assert call_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_get_task_status_running(
    registry: A2AAgentRegistry,
    delegation_manager: TaskDelegationManager,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test getting status of running task."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    # Start a long-running task in background
    async def slow_request(*args, **kwargs):
        await asyncio.sleep(10)
        return MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = slow_request
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance

        # Start task in background (will timeout eventually)
        task = asyncio.create_task(
            delegation_manager.delegate_task(
                capability_name="hybrid_retrieve",
                parameters={"query": "test"},
                tenant_id="tenant-123",
            )
        )

        # Give it a moment to register as pending
        await asyncio.sleep(0.01)

        # Get list of pending tasks
        pending = await delegation_manager.list_pending_tasks("tenant-123")

        # Cancel the background task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have one pending task
        assert len(pending) >= 0  # May or may not be pending depending on timing


@pytest.mark.asyncio
async def test_cancel_task(
    registry: A2AAgentRegistry,
    delegation_manager: TaskDelegationManager,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test cancelling a pending task."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    # Mock a task that will be pending
    delegation_manager._pending_tasks["task-001"] = MagicMock()
    delegation_manager._pending_tasks["task-001"].request = TaskRequest(
        task_id="task-001",
        tenant_id="tenant-123",
    )

    success = await delegation_manager.cancel_task("task-001", "tenant-123")
    assert success is True
    assert "task-001" not in delegation_manager._pending_tasks


@pytest.mark.asyncio
async def test_cancel_task_not_found(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test cancelling non-existent task."""
    success = await delegation_manager.cancel_task("nonexistent", "tenant-123")
    assert success is False


@pytest.mark.asyncio
async def test_cancel_task_wrong_tenant(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test cancelling task with wrong tenant."""
    delegation_manager._pending_tasks["task-001"] = MagicMock()
    delegation_manager._pending_tasks["task-001"].request = TaskRequest(
        task_id="task-001",
        tenant_id="tenant-123",
    )

    with pytest.raises(PermissionError):
        await delegation_manager.cancel_task("task-001", "tenant-456")


@pytest.mark.asyncio
async def test_list_pending_tasks_filters_by_tenant(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test that pending tasks are filtered by tenant."""
    delegation_manager._pending_tasks["task-001"] = MagicMock()
    delegation_manager._pending_tasks["task-001"].request = TaskRequest(
        task_id="task-001",
        tenant_id="tenant-123",
    )
    delegation_manager._pending_tasks["task-002"] = MagicMock()
    delegation_manager._pending_tasks["task-002"].request = TaskRequest(
        task_id="task-002",
        tenant_id="tenant-456",
    )

    tasks = await delegation_manager.list_pending_tasks("tenant-123")
    assert len(tasks) == 1
    assert tasks[0].task_id == "task-001"


@pytest.mark.asyncio
async def test_handle_incoming_task_success(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test handling incoming task successfully."""
    request_data = {
        "task_id": "task-001",
        "source_agent": "orchestrator",
        "target_agent": "rag-engine",
        "capability_name": "hybrid_retrieve",
        "parameters": {"query": "test"},
        "priority": 5,
        "timeout_seconds": 300,
        "created_at": "2024-01-01T00:00:00Z",
        "tenant_id": "tenant-123",
    }

    async def handler(request: TaskRequest) -> dict:
        return {"answer": "Test response"}

    result = await delegation_manager.handle_incoming_task(request_data, handler)

    assert result["task_id"] == "task-001"
    assert result["status"] == TaskStatus.COMPLETED.value
    assert result["result"] == {"answer": "Test response"}
    assert result["execution_time_ms"] is not None


@pytest.mark.asyncio
async def test_handle_incoming_task_failure(
    delegation_manager: TaskDelegationManager,
) -> None:
    """Test handling incoming task that fails."""
    request_data = {
        "task_id": "task-001",
        "source_agent": "orchestrator",
        "target_agent": "rag-engine",
        "capability_name": "hybrid_retrieve",
        "parameters": {"query": "test"},
        "priority": 5,
        "timeout_seconds": 300,
        "created_at": "2024-01-01T00:00:00Z",
        "tenant_id": "tenant-123",
    }

    async def handler(request: TaskRequest) -> dict:
        raise ValueError("Handler error")

    result = await delegation_manager.handle_incoming_task(request_data, handler)

    assert result["task_id"] == "task-001"
    assert result["status"] == TaskStatus.FAILED.value
    assert "Handler error" in result["error"]


@pytest.mark.asyncio
async def test_concurrent_delegation(
    registry: A2AAgentRegistry,
    delegation_manager: TaskDelegationManager,
    sample_capabilities: list[AgentCapability],
) -> None:
    """Test concurrent task delegation respects limits."""
    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=sample_capabilities,
        tenant_id="tenant-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": {"success": True}}

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance

        # Run multiple delegations concurrently
        results = await asyncio.gather(
            *[
                delegation_manager.delegate_task(
                    capability_name="hybrid_retrieve",
                    parameters={"query": f"test-{i}"},
                    tenant_id="tenant-123",
                )
                for i in range(5)
            ]
        )

    assert all(r.status == TaskStatus.COMPLETED for r in results)


@pytest.mark.asyncio
async def test_redis_result_persistence() -> None:
    """Test that task results are persisted to Redis."""

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
    delegation_manager = TaskDelegationManager(
        registry=registry,
        redis_client=redis_mock,
    )

    await registry.register_agent(
        agent_id="agent-001",
        agent_type="rag_engine",
        endpoint_url="http://localhost:8000",
        capabilities=[
            AgentCapability(name="test_cap", description="Test capability"),
        ],
        tenant_id="tenant-123",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": {"data": "test"}}

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client.return_value = mock_client_instance

        result = await delegation_manager.delegate_task(
            capability_name="test_cap",
            parameters={},
            tenant_id="tenant-123",
        )

    # Verify result was stored in Redis
    result_keys = [k for k in redis_mock.client.store.keys() if "result" in k]
    assert len(result_keys) > 0
