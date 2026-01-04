"""Tests for A2A message types and dataclasses."""

from datetime import datetime, timezone

import pytest

from agentic_rag_backend.protocols.a2a_messages import (
    A2AMessageType,
    AgentCapability,
    AgentRegistration,
    TaskRequest,
    TaskResult,
    TaskStatus,
    get_rag_capabilities,
)


class TestA2AMessageType:
    """Tests for A2AMessageType enum."""

    def test_message_types_exist(self) -> None:
        assert A2AMessageType.CAPABILITY_QUERY.value == "capability_query"
        assert A2AMessageType.CAPABILITY_RESPONSE.value == "capability_response"
        assert A2AMessageType.TASK_REQUEST.value == "task_request"
        assert A2AMessageType.TASK_PROGRESS.value == "task_progress"
        assert A2AMessageType.TASK_RESULT.value == "task_result"
        assert A2AMessageType.HEARTBEAT.value == "heartbeat"
        assert A2AMessageType.ERROR.value == "error"


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_statuses_exist(self) -> None:
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.ACCEPTED.value == "accepted"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_capability_creation(self) -> None:
        capability = AgentCapability(
            name="hybrid_retrieve",
            description="Combined vector + graph retrieval",
            parameters_schema={"type": "object"},
            returns_schema={"type": "object"},
            estimated_duration_ms=2000,
        )
        assert capability.name == "hybrid_retrieve"
        assert capability.description == "Combined vector + graph retrieval"
        assert capability.parameters_schema == {"type": "object"}
        assert capability.returns_schema == {"type": "object"}
        assert capability.estimated_duration_ms == 2000

    def test_capability_to_dict(self) -> None:
        capability = AgentCapability(
            name="test_cap",
            description="Test capability",
            estimated_duration_ms=1000,
        )
        result = capability.to_dict()
        assert result["name"] == "test_cap"
        assert result["description"] == "Test capability"
        assert result["parameters_schema"] == {}
        assert result["returns_schema"] == {}
        assert result["estimated_duration_ms"] == 1000

    def test_capability_to_dict_without_duration(self) -> None:
        capability = AgentCapability(
            name="test_cap",
            description="Test capability",
        )
        result = capability.to_dict()
        assert "estimated_duration_ms" not in result

    def test_capability_from_dict(self) -> None:
        data = {
            "name": "test_cap",
            "description": "Test capability",
            "parameters_schema": {"type": "string"},
            "returns_schema": {"type": "array"},
            "estimated_duration_ms": 500,
        }
        capability = AgentCapability.from_dict(data)
        assert capability.name == "test_cap"
        assert capability.description == "Test capability"
        assert capability.parameters_schema == {"type": "string"}
        assert capability.returns_schema == {"type": "array"}
        assert capability.estimated_duration_ms == 500

    def test_capability_from_dict_defaults(self) -> None:
        data = {}
        capability = AgentCapability.from_dict(data)
        assert capability.name == ""
        assert capability.description == ""
        assert capability.parameters_schema == {}
        assert capability.returns_schema == {}
        assert capability.estimated_duration_ms is None


class TestAgentRegistration:
    """Tests for AgentRegistration dataclass."""

    def test_registration_creation(self) -> None:
        now = datetime.now(timezone.utc)
        capabilities = [
            AgentCapability(name="cap1", description="Capability 1"),
        ]
        registration = AgentRegistration(
            agent_id="agent-001",
            agent_type="rag_engine",
            endpoint_url="http://localhost:8000",
            capabilities=capabilities,
            tenant_id="tenant-123",
            registered_at=now,
            last_heartbeat=now,
            health_status="healthy",
            metadata={"version": "1.0"},
        )
        assert registration.agent_id == "agent-001"
        assert registration.agent_type == "rag_engine"
        assert registration.endpoint_url == "http://localhost:8000"
        assert len(registration.capabilities) == 1
        assert registration.tenant_id == "tenant-123"
        assert registration.health_status == "healthy"
        assert registration.metadata == {"version": "1.0"}

    def test_registration_to_dict(self) -> None:
        now = datetime.now(timezone.utc)
        registration = AgentRegistration(
            agent_id="agent-001",
            agent_type="rag_engine",
            endpoint_url="http://localhost:8000",
            capabilities=[],
            tenant_id="tenant-123",
            registered_at=now,
            last_heartbeat=now,
        )
        result = registration.to_dict()
        assert result["agent_id"] == "agent-001"
        assert result["agent_type"] == "rag_engine"
        assert result["endpoint_url"] == "http://localhost:8000"
        assert result["tenant_id"] == "tenant-123"
        assert result["health_status"] == "healthy"
        assert result["registered_at"].endswith("Z")
        assert result["last_heartbeat"].endswith("Z")

    def test_registration_from_dict(self) -> None:
        data = {
            "agent_id": "agent-001",
            "agent_type": "rag_engine",
            "endpoint_url": "http://localhost:8000",
            "capabilities": [
                {"name": "cap1", "description": "Cap 1"},
            ],
            "tenant_id": "tenant-123",
            "registered_at": "2024-01-01T00:00:00Z",
            "last_heartbeat": "2024-01-01T01:00:00Z",
            "health_status": "healthy",
            "metadata": {},
        }
        registration = AgentRegistration.from_dict(data)
        assert registration.agent_id == "agent-001"
        assert registration.agent_type == "rag_engine"
        assert len(registration.capabilities) == 1
        assert registration.capabilities[0].name == "cap1"

    def test_has_capability(self) -> None:
        registration = AgentRegistration(
            agent_id="agent-001",
            agent_type="rag_engine",
            endpoint_url="http://localhost:8000",
            capabilities=[
                AgentCapability(name="hybrid_retrieve", description="Hybrid retrieval"),
                AgentCapability(name="vector_search", description="Vector search"),
            ],
            tenant_id="tenant-123",
            registered_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
        )
        assert registration.has_capability("hybrid_retrieve") is True
        assert registration.has_capability("vector_search") is True
        assert registration.has_capability("unknown_cap") is False


class TestTaskRequest:
    """Tests for TaskRequest dataclass."""

    def test_task_request_creation(self) -> None:
        request = TaskRequest(
            task_id="task-001",
            source_agent="orchestrator",
            target_agent="rag-engine",
            capability_name="hybrid_retrieve",
            parameters={"query": "test"},
            priority=7,
            timeout_seconds=120,
            tenant_id="tenant-123",
        )
        assert request.task_id == "task-001"
        assert request.source_agent == "orchestrator"
        assert request.target_agent == "rag-engine"
        assert request.capability_name == "hybrid_retrieve"
        assert request.parameters == {"query": "test"}
        assert request.priority == 7
        assert request.timeout_seconds == 120
        assert request.tenant_id == "tenant-123"

    def test_task_request_priority_clamping_low(self) -> None:
        request = TaskRequest(priority=-5, tenant_id="t1")
        assert request.priority == 1

    def test_task_request_priority_clamping_high(self) -> None:
        request = TaskRequest(priority=20, tenant_id="t1")
        assert request.priority == 10

    def test_task_request_auto_generated_id(self) -> None:
        request = TaskRequest(tenant_id="t1")
        assert request.task_id != ""
        assert len(request.task_id) == 36  # UUID format

    def test_task_request_to_dict(self) -> None:
        request = TaskRequest(
            task_id="task-001",
            source_agent="orchestrator",
            capability_name="test",
            tenant_id="t1",
        )
        result = request.to_dict()
        assert result["task_id"] == "task-001"
        assert result["source_agent"] == "orchestrator"
        assert result["capability_name"] == "test"
        assert result["created_at"].endswith("Z")

    def test_task_request_from_dict(self) -> None:
        data = {
            "task_id": "task-001",
            "source_agent": "orchestrator",
            "target_agent": "rag-engine",
            "capability_name": "hybrid_retrieve",
            "parameters": {"query": "test"},
            "priority": 5,
            "timeout_seconds": 300,
            "correlation_id": "corr-123",
            "created_at": "2024-01-01T00:00:00Z",
            "tenant_id": "t1",
        }
        request = TaskRequest.from_dict(data)
        assert request.task_id == "task-001"
        assert request.source_agent == "orchestrator"
        assert request.correlation_id == "corr-123"
        assert request.tenant_id == "t1"

    def test_task_request_from_dict_defaults(self) -> None:
        request = TaskRequest.from_dict({})
        assert request.source_agent == ""
        assert request.target_agent == ""
        assert request.capability_name == ""
        assert request.priority == 5
        assert request.timeout_seconds == 300


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_completed(self) -> None:
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.COMPLETED,
            result={"answer": "test"},
            execution_time_ms=500,
        )
        assert result.task_id == "task-001"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"answer": "test"}
        assert result.error is None
        assert result.execution_time_ms == 500
        assert result.is_success is True
        assert result.is_failure is False

    def test_task_result_failed(self) -> None:
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.FAILED,
            error="Something went wrong",
            execution_time_ms=100,
        )
        assert result.status == TaskStatus.FAILED
        assert result.error == "Something went wrong"
        assert result.result is None
        assert result.is_success is False
        assert result.is_failure is True

    def test_task_result_cancelled_is_failure(self) -> None:
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.CANCELLED,
        )
        assert result.is_success is False
        assert result.is_failure is True

    def test_task_result_to_dict(self) -> None:
        result = TaskResult(
            task_id="task-001",
            status=TaskStatus.COMPLETED,
            result={"data": "test"},
            execution_time_ms=250,
        )
        data = result.to_dict()
        assert data["task_id"] == "task-001"
        assert data["status"] == "completed"
        assert data["result"] == {"data": "test"}
        assert data["execution_time_ms"] == 250

    def test_task_result_from_dict(self) -> None:
        data = {
            "task_id": "task-001",
            "status": "completed",
            "result": {"answer": "test"},
            "error": None,
            "execution_time_ms": 500,
            "completed_at": "2024-01-01T00:00:00Z",
        }
        result = TaskResult.from_dict(data)
        assert result.task_id == "task-001"
        assert result.status == TaskStatus.COMPLETED
        assert result.result == {"answer": "test"}
        assert result.execution_time_ms == 500


class TestGetRagCapabilities:
    """Tests for get_rag_capabilities function."""

    def test_returns_list_of_capabilities(self) -> None:
        capabilities = get_rag_capabilities()
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert all(isinstance(c, AgentCapability) for c in capabilities)

    def test_contains_expected_capabilities(self) -> None:
        capabilities = get_rag_capabilities()
        capability_names = {c.name for c in capabilities}
        assert "hybrid_retrieve" in capability_names
        assert "ingest_url" in capability_names
        assert "ingest_pdf" in capability_names
        assert "ingest_youtube" in capability_names
        assert "vector_search" in capability_names
        assert "query_with_reranking" in capability_names

    def test_returns_copy(self) -> None:
        capabilities1 = get_rag_capabilities()
        capabilities2 = get_rag_capabilities()
        assert capabilities1 is not capabilities2
        # Modifying one should not affect the other
        capabilities1.pop()
        assert len(capabilities2) > len(capabilities1)
