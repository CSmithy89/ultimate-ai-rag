"""A2A message types and data structures for agent-to-agent communication.

This module defines the standardized message types, task statuses, and data
structures used in the A2A (Agent-to-Agent) protocol for multi-agent orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


class A2AMessageType(str, Enum):
    """Standardized A2A message types for protocol communication."""

    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    TASK_REQUEST = "task_request"
    TASK_PROGRESS = "task_progress"
    TASK_RESULT = "task_result"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Task execution status enumeration."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentCapability:
    """A capability offered by an agent in the A2A network.

    Attributes:
        name: Unique identifier for the capability (e.g., 'hybrid_retrieve')
        description: Human-readable description of what the capability does
        parameters_schema: JSON Schema describing expected parameters
        returns_schema: JSON Schema describing the return format
        estimated_duration_ms: Optional estimated execution time in milliseconds
    """

    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    returns_schema: dict[str, Any] = field(default_factory=dict)
    estimated_duration_ms: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize capability to dictionary format."""
        result: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.parameters_schema,
            "returns_schema": self.returns_schema,
        }
        if self.estimated_duration_ms is not None:
            result["estimated_duration_ms"] = self.estimated_duration_ms
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCapability":
        """Deserialize capability from dictionary format."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            parameters_schema=data.get("parameters_schema", {}),
            returns_schema=data.get("returns_schema", {}),
            estimated_duration_ms=data.get("estimated_duration_ms"),
        )


@dataclass
class AgentRegistration:
    """Registration record for an agent in the A2A network.

    Attributes:
        agent_id: Unique identifier for this agent
        agent_type: Type classification (e.g., 'rag_engine', 'search_agent')
        endpoint_url: HTTP endpoint for receiving task delegations
        capabilities: List of capabilities this agent offers
        tenant_id: Tenant scope for multi-tenancy isolation
        registered_at: Timestamp when agent was registered
        last_heartbeat: Timestamp of last heartbeat received
        health_status: Current health status ('healthy' or 'unhealthy')
        metadata: Additional agent metadata
    """

    agent_id: str
    agent_type: str
    endpoint_url: str
    capabilities: list[AgentCapability]
    tenant_id: str
    registered_at: datetime
    last_heartbeat: datetime
    health_status: str = "healthy"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize registration to dictionary format for JSON/Redis storage."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "endpoint_url": self.endpoint_url,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "tenant_id": self.tenant_id,
            "registered_at": self.registered_at.isoformat().replace("+00:00", "Z"),
            "last_heartbeat": self.last_heartbeat.isoformat().replace("+00:00", "Z"),
            "health_status": self.health_status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRegistration":
        """Deserialize registration from dictionary format."""
        registered_at = data.get("registered_at", "")
        if isinstance(registered_at, str):
            if registered_at.endswith("Z"):
                registered_at = registered_at.replace("Z", "+00:00")
            registered_at = datetime.fromisoformat(registered_at)

        last_heartbeat = data.get("last_heartbeat", "")
        if isinstance(last_heartbeat, str):
            if last_heartbeat.endswith("Z"):
                last_heartbeat = last_heartbeat.replace("Z", "+00:00")
            last_heartbeat = datetime.fromisoformat(last_heartbeat)

        capabilities = [
            AgentCapability.from_dict(cap) if isinstance(cap, dict) else cap
            for cap in data.get("capabilities", [])
        ]

        return cls(
            agent_id=data.get("agent_id", ""),
            agent_type=data.get("agent_type", ""),
            endpoint_url=data.get("endpoint_url", ""),
            capabilities=capabilities,
            tenant_id=data.get("tenant_id", ""),
            registered_at=registered_at,
            last_heartbeat=last_heartbeat,
            health_status=data.get("health_status", "healthy"),
            metadata=data.get("metadata", {}),
        )

    def has_capability(self, capability_name: str) -> bool:
        """Check if this agent has a specific capability."""
        return any(cap.name == capability_name for cap in self.capabilities)


@dataclass
class TaskRequest:
    """A task delegated from one agent to another.

    Attributes:
        task_id: Unique identifier for this task
        source_agent: Agent ID of the delegating agent
        target_agent: Agent ID of the receiving agent
        capability_name: The capability to invoke on the target agent
        parameters: Parameters to pass to the capability
        priority: Task priority (1-10, higher = more urgent)
        timeout_seconds: Maximum time to wait for result
        correlation_id: ID for request/response correlation
        created_at: Timestamp when task was created
        tenant_id: Tenant scope for multi-tenancy isolation
    """

    task_id: str = field(default_factory=lambda: str(uuid4()))
    source_agent: str = ""
    target_agent: str = ""
    capability_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, higher = more urgent
    timeout_seconds: int = 300
    correlation_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = ""

    def __post_init__(self) -> None:
        """Validate priority range after initialization."""
        if self.priority < 1:
            self.priority = 1
        elif self.priority > 10:
            self.priority = 10

    def to_dict(self) -> dict[str, Any]:
        """Serialize task request to dictionary format."""
        return {
            "task_id": self.task_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "capability_name": self.capability_name,
            "parameters": self.parameters,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRequest":
        """Deserialize task request from dictionary format."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            if created_at.endswith("Z"):
                created_at = created_at.replace("Z", "+00:00")
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            task_id=data.get("task_id", str(uuid4())),
            source_agent=data.get("source_agent", ""),
            target_agent=data.get("target_agent", ""),
            capability_name=data.get("capability_name", ""),
            parameters=data.get("parameters", {}),
            priority=data.get("priority", 5),
            timeout_seconds=data.get("timeout_seconds", 300),
            correlation_id=data.get("correlation_id"),
            created_at=created_at,
            tenant_id=data.get("tenant_id", ""),
        )


@dataclass
class TaskResult:
    """Result of a completed or failed task.

    Attributes:
        task_id: ID of the task this result corresponds to
        status: Final status of the task
        result: Result data (if completed successfully)
        error: Error message (if failed)
        execution_time_ms: Time taken to execute in milliseconds
        completed_at: Timestamp when task completed
        tenant_id: Tenant ID for multi-tenancy isolation
    """

    task_id: str
    status: TaskStatus
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tenant_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize task result to dictionary format."""
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "completed_at": self.completed_at.isoformat().replace("+00:00", "Z"),
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        """Deserialize task result from dictionary format."""
        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            if completed_at.endswith("Z"):
                completed_at = completed_at.replace("Z", "+00:00")
            completed_at = datetime.fromisoformat(completed_at)
        elif completed_at is None:
            completed_at = datetime.now(timezone.utc)

        status = data.get("status", "pending")
        if isinstance(status, str):
            status = TaskStatus(status)

        return cls(
            task_id=data.get("task_id", ""),
            status=status,
            result=data.get("result"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms"),
            completed_at=completed_at,
            tenant_id=data.get("tenant_id", ""),
        )

    @property
    def is_success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        """Check if task failed."""
        return self.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)


# Pre-defined RAG capabilities that this system offers
RAG_CAPABILITIES = [
    AgentCapability(
        name="hybrid_retrieve",
        description="Combined vector + graph retrieval with optional reranking",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 10, "description": "Number of results"},
                "use_reranking": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
        returns_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "sources": {"type": "array"},
            },
        },
        estimated_duration_ms=2000,
    ),
    AgentCapability(
        name="ingest_url",
        description="Crawl and ingest a URL into the knowledge base",
        parameters_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to crawl"},
                "depth": {"type": "integer", "default": 1, "description": "Crawl depth"},
            },
            "required": ["url"],
        },
        returns_schema={
            "type": "object",
            "properties": {
                "pages_ingested": {"type": "integer"},
                "chunks_created": {"type": "integer"},
            },
        },
        estimated_duration_ms=30000,
    ),
    AgentCapability(
        name="ingest_pdf",
        description="Parse and ingest a PDF document",
        parameters_schema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to PDF file"},
            },
            "required": ["file_path"],
        },
        returns_schema={
            "type": "object",
            "properties": {
                "pages_parsed": {"type": "integer"},
                "chunks_created": {"type": "integer"},
            },
        },
        estimated_duration_ms=15000,
    ),
    AgentCapability(
        name="ingest_youtube",
        description="Extract transcript and ingest a YouTube video",
        parameters_schema={
            "type": "object",
            "properties": {
                "video_url": {"type": "string", "description": "YouTube video URL"},
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Preferred transcript languages",
                },
            },
            "required": ["video_url"],
        },
        returns_schema={
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "duration_seconds": {"type": "number"},
                "chunks_created": {"type": "integer"},
            },
        },
        estimated_duration_ms=10000,
    ),
    AgentCapability(
        name="vector_search",
        description="Semantic search via pgvector",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        returns_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array"},
            },
        },
        estimated_duration_ms=500,
    ),
    AgentCapability(
        name="query_with_reranking",
        description="Query with explicit reranking control",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "reranker": {
                    "type": "string",
                    "default": "flashrank",
                    "enum": ["flashrank", "cohere"],
                },
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
        returns_schema={
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "reranker_used": {"type": "string"},
            },
        },
        estimated_duration_ms=1500,
    ),
]


def get_rag_capabilities() -> list[AgentCapability]:
    """Get the list of RAG capabilities this system offers."""
    return RAG_CAPABILITIES.copy()


# Capabilities that are actually implemented and can be executed
IMPLEMENTED_CAPABILITY_NAMES = {"hybrid_retrieve", "vector_search"}


def get_implemented_rag_capabilities() -> list[AgentCapability]:
    """Get only the RAG capabilities that are actually implemented.

    This returns a filtered list containing only capabilities that have
    working implementations in the execute endpoint. Currently implemented:
    - hybrid_retrieve: Combined vector + graph retrieval
    - vector_search: Semantic search via pgvector
    """
    return [cap for cap in RAG_CAPABILITIES if cap.name in IMPLEMENTED_CAPABILITY_NAMES]
