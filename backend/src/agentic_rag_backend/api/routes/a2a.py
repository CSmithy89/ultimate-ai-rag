"""A2A (Agent-to-Agent) collaboration endpoints.

This module provides the HTTP API for the A2A protocol including:
- Session management (existing)
- Agent registration and discovery (new)
- Task delegation (new)
- Health monitoring (new)
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
import jsonschema
import structlog

from ...api.utils import build_meta, rate_limit_exceeded
from ...core.errors import (
    A2AAgentNotFoundError,
    A2ACapabilityNotFoundError,
    A2ADelegationError,
    A2APermissionError,
    A2ARegistrationError,
    A2AServiceUnavailableError,
    A2ATaskNotFoundError,
)
from ...protocols.a2a import A2ASessionManager
from ...protocols.a2a_messages import (
    AgentCapability,
    TaskRequest,
    get_implemented_rag_capabilities,
)
from ...protocols.a2a_registry import A2AAgentRegistry
from ...protocols.a2a_delegation import TaskDelegationManager
from ...protocols.a2a_middleware import (
    A2AMiddlewareAgent,
    A2AAgentInfo as MiddlewareAgentInfo,
    A2AAgentCapability as MiddlewareAgentCapability,
)
from ...rate_limit import RateLimiter
from ...validation import TENANT_ID_PATTERN, TENANT_ID_REGEX

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/a2a", tags=["a2a"])


# ==================== Pydantic Models ====================

# Session Models (existing)


class CreateSessionRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)


class CreateSessionResponse(BaseModel):
    session: dict[str, Any]
    meta: dict[str, Any]


class MessageRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)
    sender: str = Field(..., min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=10000)
    metadata: dict[str, Any] | None = None


class SessionResponse(BaseModel):
    session: dict[str, Any]
    meta: dict[str, Any]


# Registration Models (new)


class CapabilityModel(BaseModel):
    """Pydantic model for agent capability."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    returns_schema: dict[str, Any] = Field(default_factory=dict)
    estimated_duration_ms: Optional[int] = Field(None, ge=0)

    @field_validator("parameters_schema", "returns_schema")
    @classmethod
    def validate_json_schema(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate that the schema is a valid JSON Schema."""
        if not v:
            return v
        try:
            jsonschema.Draft7Validator.check_schema(v)
        except jsonschema.SchemaError as exc:
            raise ValueError(f"Invalid JSON Schema: {exc.message}") from exc
        return v


class RegisterAgentRequest(BaseModel):
    """Request to register an agent."""

    agent_id: str = Field(..., min_length=1, max_length=100)
    agent_type: str = Field(..., min_length=1, max_length=50)
    endpoint_url: str = Field(..., min_length=1, max_length=500)
    capabilities: list[CapabilityModel]
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    """Response containing agent registration data."""

    agent: dict[str, Any]
    meta: dict[str, Any]


class AgentListResponse(BaseModel):
    """Response containing list of agents."""

    agents: list[dict[str, Any]]
    meta: dict[str, Any]


class HeartbeatRequest(BaseModel):
    """Request to record agent heartbeat."""

    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)


class HeartbeatResponse(BaseModel):
    """Response from heartbeat endpoint."""

    acknowledged: bool
    meta: dict[str, Any]


# Task Delegation Models (new)


class DelegateTaskRequest(BaseModel):
    """Request to delegate a task to another agent."""

    capability_name: str = Field(..., min_length=1, max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)
    priority: int = Field(5, ge=1, le=10)
    timeout_seconds: Optional[int] = Field(None, ge=1, le=3600)
    target_agent_id: Optional[str] = Field(None, max_length=100)


class TaskResponse(BaseModel):
    """Response containing task result."""

    task: dict[str, Any]
    meta: dict[str, Any]


class TaskListResponse(BaseModel):
    """Response containing list of pending tasks."""

    tasks: list[dict[str, Any]]
    meta: dict[str, Any]


class ExecuteTaskRequest(BaseModel):
    """Incoming task execution request from another agent."""

    task_id: str = Field(..., min_length=1, max_length=255)
    source_agent: str = Field(..., min_length=1, max_length=255)
    target_agent: str = Field(..., min_length=1, max_length=255)
    capability_name: str = Field(..., min_length=1, max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    correlation_id: Optional[str] = Field(None, max_length=255)
    created_at: str
    # Validate tenant_id format to prevent injection attacks
    tenant_id: str = Field(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN)


class ExecuteTaskResponse(BaseModel):
    """Response from task execution."""

    task_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class CapabilityListResponse(BaseModel):
    """Response containing list of capabilities."""

    capabilities: list[dict[str, Any]]
    meta: dict[str, Any]


# ==================== Middleware Models (Story 22-A1) ====================


class MiddlewareCapabilityModel(BaseModel):
    """Pydantic model for middleware agent capability."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)


class MiddlewareRegisterRequest(BaseModel):
    """Request to register an agent with the middleware.

    The agent_id MUST be prefixed with tenant_id: (e.g., tenant123:my-agent).
    """

    agent_id: str = Field(..., min_length=1, max_length=128)
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=500)
    capabilities: list[MiddlewareCapabilityModel]
    endpoint: str = Field(..., min_length=1, max_length=500)


class MiddlewareRegisterResponse(BaseModel):
    """Response from middleware agent registration."""

    status: str
    agent_id: str
    meta: dict[str, Any]


class MiddlewareAgentListResponse(BaseModel):
    """Response containing list of middleware agents."""

    agents: list[dict[str, Any]]
    meta: dict[str, Any]


class MiddlewareCapabilityListResponse(BaseModel):
    """Response containing list of middleware capabilities."""

    capabilities: list[dict[str, Any]]
    meta: dict[str, Any]


class MiddlewareDelegateRequest(BaseModel):
    """Request to delegate a task to an agent via middleware."""

    capability_name: str = Field(..., min_length=1, max_length=100)
    input_data: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] | None = None


class MiddlewareDelegateResponse(BaseModel):
    """Response from task delegation."""

    status: str
    events: list[dict[str, Any]]
    meta: dict[str, Any]


# ==================== Dependency Injection ====================


def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


def get_a2a_manager(request: Request) -> A2ASessionManager:
    """Get the A2A session manager from app state."""
    manager = getattr(request.app.state, "a2a_manager", None)
    if manager is None:
        raise RuntimeError("A2A session manager not initialized")
    return manager


def get_a2a_registry(request: Request) -> A2AAgentRegistry:
    """Get the A2A agent registry from app state."""
    registry = getattr(request.app.state, "a2a_registry", None)
    if registry is None:
        raise RuntimeError("A2A agent registry not initialized")
    return registry


def get_delegation_manager(request: Request) -> TaskDelegationManager:
    """Get the task delegation manager from app state."""
    manager = getattr(request.app.state, "a2a_delegation_manager", None)
    if manager is None:
        raise RuntimeError("A2A delegation manager not initialized")
    return manager


def get_a2a_middleware(request: Request) -> A2AMiddlewareAgent:
    """Get the A2A middleware agent from app state."""
    middleware = getattr(request.app.state, "a2a_middleware", None)
    if middleware is None:
        raise RuntimeError("A2A middleware not initialized")
    return middleware


def validate_tenant_id_header(
    x_tenant_id: str = Header(..., alias="X-Tenant-ID"),
) -> str:
    """Validate the X-Tenant-ID header against the TENANT_ID_PATTERN.

    Args:
        x_tenant_id: The tenant ID from the X-Tenant-ID header

    Returns:
        The validated tenant ID

    Raises:
        HTTPException: If the tenant ID doesn't match the required UUID pattern
    """
    if not TENANT_ID_REGEX.fullmatch(x_tenant_id):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid X-Tenant-ID header: must match UUID format ({TENANT_ID_PATTERN})",
        )
    return x_tenant_id


# A2A Middleware API Key configuration
# Set A2A_API_KEY environment variable to enable API key authentication
_A2A_API_KEY: Optional[str] = os.getenv("A2A_API_KEY")


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Optional[str]:
    """Verify API key for A2A middleware endpoints.

    Checks X-API-Key header or Authorization: Bearer <key> header.
    If A2A_API_KEY environment variable is not set, authentication is disabled.

    Args:
        x_api_key: API key from X-API-Key header
        authorization: Authorization header (Bearer token)

    Returns:
        The validated API key, or None if auth is disabled

    Raises:
        HTTPException: If API key is required but missing/invalid
    """
    if _A2A_API_KEY is None:
        # API key authentication disabled
        return None

    # Extract key from headers
    api_key: Optional[str] = None
    if x_api_key:
        api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header or Authorization: Bearer <key>",
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, _A2A_API_KEY):
        logger.warning("a2a_middleware_invalid_api_key")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key",
        )

    return api_key


# ==================== Session Endpoints (existing) ====================


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    request_body: CreateSessionRequest,
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> CreateSessionResponse:
    """Create a new A2A collaboration session."""
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    try:
        session = await manager.create_session(request_body.tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info("a2a_session_created", session_id=session["session_id"])

    return CreateSessionResponse(session=session, meta=build_meta())


@router.post("/sessions/{session_id}/messages", response_model=SessionResponse)
async def add_message(
    session_id: str,
    request_body: MessageRequest,
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> SessionResponse:
    """Add a message to an existing A2A session."""
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    try:
        session = await manager.add_message(
            session_id=session_id,
            tenant_id=request_body.tenant_id,
            sender=request_body.sender,
            content=request_body.content,
            metadata=request_body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except PermissionError as exc:
        raise A2APermissionError("Tenant not authorized for this session") from exc

    return SessionResponse(session=session, meta=build_meta())


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    manager: A2ASessionManager = Depends(get_a2a_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> SessionResponse:
    """Fetch session transcript for a tenant."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    session = await manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["tenant_id"] != tenant_id:
        raise HTTPException(status_code=403, detail="Tenant not authorized")

    return SessionResponse(session=session, meta=build_meta())


# ==================== Registration Endpoints (new) ====================


@router.post("/agents/register", response_model=AgentResponse)
async def register_agent(
    request_body: RegisterAgentRequest,
    registry: A2AAgentRegistry = Depends(get_a2a_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> AgentResponse:
    """Register an agent in the A2A network.

    Agents must register to be discoverable by other agents.
    Registration includes the agent's capabilities and endpoint URL.
    """
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    try:
        # Convert Pydantic models to dataclasses
        capabilities = [
            AgentCapability(
                name=cap.name,
                description=cap.description,
                parameters_schema=cap.parameters_schema,
                returns_schema=cap.returns_schema,
                estimated_duration_ms=cap.estimated_duration_ms,
            )
            for cap in request_body.capabilities
        ]

        registration = await registry.register_agent(
            agent_id=request_body.agent_id,
            agent_type=request_body.agent_type,
            endpoint_url=request_body.endpoint_url,
            capabilities=capabilities,
            tenant_id=request_body.tenant_id,
            metadata=request_body.metadata,
        )
    except ValueError as exc:
        raise A2ARegistrationError(request_body.agent_id, str(exc)) from exc
    except PermissionError as exc:
        raise A2APermissionError(str(exc), request_body.agent_id) from exc

    logger.info(
        "a2a_agent_registered_api",
        agent_id=registration.agent_id,
        tenant_id=registration.tenant_id,
    )

    return AgentResponse(agent=registration.to_dict(), meta=build_meta())


@router.delete("/agents/{agent_id}", response_model=dict[str, Any])
async def unregister_agent(
    agent_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    registry: A2AAgentRegistry = Depends(get_a2a_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> dict[str, Any]:
    """Unregister an agent from the A2A network."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    try:
        success = await registry.unregister_agent(agent_id, tenant_id)
    except PermissionError as exc:
        raise A2APermissionError(str(exc), agent_id) from exc

    if not success:
        raise A2AAgentNotFoundError(agent_id)

    return {"success": True, "meta": build_meta()}


@router.post("/agents/{agent_id}/heartbeat", response_model=HeartbeatResponse)
async def agent_heartbeat(
    agent_id: str,
    request_body: HeartbeatRequest,
    registry: A2AAgentRegistry = Depends(get_a2a_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> HeartbeatResponse:
    """Record heartbeat for an agent.

    Agents should send heartbeats periodically to maintain healthy status.
    """
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    try:
        success = await registry.heartbeat(agent_id, request_body.tenant_id)
    except PermissionError as exc:
        raise A2APermissionError(str(exc), agent_id) from exc

    if not success:
        raise A2AAgentNotFoundError(agent_id)

    return HeartbeatResponse(acknowledged=True, meta=build_meta())


# ==================== Discovery Endpoints (new) ====================


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    healthy_only: bool = Query(False),
    registry: A2AAgentRegistry = Depends(get_a2a_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> AgentListResponse:
    """List all registered agents for a tenant."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    agents = await registry.list_agents(tenant_id, healthy_only=healthy_only)

    return AgentListResponse(
        agents=[a.to_dict() for a in agents],
        meta=build_meta(),
    )


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    registry: A2AAgentRegistry = Depends(get_a2a_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> AgentResponse:
    """Get a specific agent by ID."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    agent = await registry.get_agent(agent_id, tenant_id)
    if not agent:
        raise A2AAgentNotFoundError(agent_id)

    return AgentResponse(agent=agent.to_dict(), meta=build_meta())


@router.get("/agents/by-capability/{capability_name}", response_model=AgentListResponse)
async def find_agents_by_capability(
    capability_name: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    healthy_only: bool = Query(True),
    registry: A2AAgentRegistry = Depends(get_a2a_registry),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> AgentListResponse:
    """Find agents that offer a specific capability."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    agents = await registry.find_agents_by_capability(
        capability_name,
        tenant_id,
        healthy_only=healthy_only,
    )

    return AgentListResponse(
        agents=[a.to_dict() for a in agents],
        meta=build_meta(),
    )


@router.get("/capabilities", response_model=CapabilityListResponse)
async def list_capabilities(
    limiter: RateLimiter = Depends(get_rate_limiter),
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
) -> CapabilityListResponse:
    """List all capabilities offered by this RAG system.

    Only returns capabilities that are actually implemented and can be executed.
    """
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    capabilities = get_implemented_rag_capabilities()

    return CapabilityListResponse(
        capabilities=[c.to_dict() for c in capabilities],
        meta=build_meta(),
    )


# ==================== Task Delegation Endpoints (new) ====================


@router.post("/tasks/delegate", response_model=TaskResponse)
async def delegate_task(
    request_body: DelegateTaskRequest,
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TaskResponse:
    """Delegate a task to an agent with the required capability.

    The system will automatically find a suitable healthy agent that
    offers the requested capability.
    """
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    result = await delegation_manager.delegate_task(
        capability_name=request_body.capability_name,
        parameters=request_body.parameters,
        tenant_id=request_body.tenant_id,
        priority=request_body.priority,
        timeout_seconds=request_body.timeout_seconds,
        target_agent_id=request_body.target_agent_id,
    )

    return TaskResponse(task=result.to_dict(), meta=build_meta())


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(
    task_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TaskResponse:
    """Get the status or result of a delegated task."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    result = await delegation_manager.get_task_status(task_id, tenant_id)
    if not result:
        raise A2ATaskNotFoundError(task_id)

    return TaskResponse(task=result.to_dict(), meta=build_meta())


@router.delete("/tasks/{task_id}", response_model=dict[str, Any])
async def cancel_task(
    task_id: str,
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> dict[str, Any]:
    """Cancel a pending delegated task."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    try:
        success = await delegation_manager.cancel_task(task_id, tenant_id)
    except PermissionError as exc:
        raise A2APermissionError(str(exc), task_id) from exc

    if not success:
        raise A2ATaskNotFoundError(task_id)

    return {"success": True, "meta": build_meta()}


@router.get("/tasks", response_model=TaskListResponse)
async def list_pending_tasks(
    tenant_id: str = Query(..., min_length=1, max_length=255, pattern=TENANT_ID_PATTERN),
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> TaskListResponse:
    """List all pending tasks for a tenant."""
    if not await limiter.allow(tenant_id):
        raise rate_limit_exceeded()

    tasks = await delegation_manager.list_pending_tasks(tenant_id)

    return TaskListResponse(
        tasks=[t.to_dict() for t in tasks],
        meta=build_meta(),
    )


@router.post("/execute", response_model=ExecuteTaskResponse)
async def execute_incoming_task(
    request_body: ExecuteTaskRequest,
    request: Request,
    delegation_manager: TaskDelegationManager = Depends(get_delegation_manager),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> ExecuteTaskResponse:
    """Execute an incoming task delegated from another agent.

    This endpoint is called by other agents to execute tasks on this system.
    """
    if not await limiter.allow(request_body.tenant_id):
        raise rate_limit_exceeded()

    # Get the orchestrator to execute capabilities
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise A2AServiceUnavailableError(
            service="Orchestrator",
            reason="Orchestrator not initialized - cannot execute task",
        )

    # Set of capabilities that are actually implemented in this endpoint
    EXECUTABLE_CAPABILITIES = {"hybrid_retrieve", "vector_search"}

    async def task_handler(task_request: TaskRequest) -> dict[str, Any]:
        """Execute the capability requested by the task."""
        capability = task_request.capability_name

        # Validate capability is implemented before attempting execution
        if capability not in EXECUTABLE_CAPABILITIES:
            raise ValueError(
                f"Capability '{capability}' not implemented. "
                f"Available: {sorted(EXECUTABLE_CAPABILITIES)}"
            )

        # Route to appropriate handler based on capability
        if capability == "hybrid_retrieve":
            result = await orchestrator.run(
                task_request.parameters.get("query", ""),
                task_request.tenant_id,
            )
            return {
                "answer": result.answer,
                "evidence": result.evidence,
            }
        elif capability == "vector_search":
            # Direct vector search using public property (not private _postgres)
            query = task_request.parameters.get("query", "")
            vector_service = orchestrator.vector_search_service
            if vector_service:
                hits = await vector_service.search(query, task_request.tenant_id)
                # Convert VectorHit objects to dicts for JSON serialization
                return {
                    "results": [
                        {
                            "chunk_id": hit.chunk_id,
                            "document_id": hit.document_id,
                            "content": hit.content,
                            "similarity": hit.similarity,
                            "metadata": hit.metadata,
                        }
                        for hit in hits
                    ]
                }
            raise A2AServiceUnavailableError(
                service="vector_search",
                reason="Vector search service not initialized",
            )
        # Note: 'else' clause removed - capability validation at start covers this

    response = await delegation_manager.handle_incoming_task(
        request_body.model_dump(),
        task_handler,
    )

    return ExecuteTaskResponse(
        task_id=response["task_id"],
        status=response["status"],
        result=response.get("result"),
        error=response.get("error"),
        execution_time_ms=response.get("execution_time_ms"),
    )


# ==================== Middleware Endpoints (Story 22-A1) ====================
# These use the /middleware prefix to avoid collision with registry endpoints


@router.post("/middleware/agents/register", response_model=MiddlewareRegisterResponse)
async def register_middleware_agent(
    body: MiddlewareRegisterRequest,
    x_tenant_id: str = Depends(validate_tenant_id_header),
    _api_key: Optional[str] = Depends(verify_api_key),
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> MiddlewareRegisterResponse:
    """Register an agent with the A2A middleware.

    The agent_id MUST be prefixed with the tenant_id.
    Rate limit: 10 registrations per minute per tenant.
    Requires API key authentication when A2A_API_KEY is configured.
    """
    if not await limiter.allow(f"a2a_register:{x_tenant_id}"):
        raise rate_limit_exceeded()

    # Validate tenant prefix in agent_id
    expected_prefix = f"{x_tenant_id}:"
    if not body.agent_id.startswith(expected_prefix):
        raise A2ARegistrationError(
            agent_id=body.agent_id,
            reason=f"agent_id must be prefixed with tenant_id '{x_tenant_id}:'",
        )

    # Convert to middleware model
    capabilities = [
        MiddlewareAgentCapability(
            name=c.name,
            description=c.description,
            input_schema=c.input_schema,
            output_schema=c.output_schema,
        )
        for c in body.capabilities
    ]

    agent_info = MiddlewareAgentInfo(
        agent_id=body.agent_id,
        name=body.name,
        description=body.description,
        capabilities=capabilities,
        endpoint=body.endpoint,
    )

    middleware.register_agent(agent_info)

    logger.info(
        "a2a_middleware_agent_registered",
        agent_id=body.agent_id,
        tenant_id=x_tenant_id,
        capabilities=[c.name for c in capabilities],
    )

    return MiddlewareRegisterResponse(
        status="registered",
        agent_id=body.agent_id,
        meta=build_meta(),
    )


@router.get("/middleware/agents", response_model=MiddlewareAgentListResponse)
async def list_middleware_agents(
    x_tenant_id: str = Depends(validate_tenant_id_header),
    _api_key: Optional[str] = Depends(verify_api_key),
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> MiddlewareAgentListResponse:
    """List all registered agents for the calling tenant.

    Requires API key authentication when A2A_API_KEY is configured.
    """
    if not await limiter.allow(x_tenant_id):
        raise rate_limit_exceeded()

    agents = middleware.list_agents_for_tenant(x_tenant_id)

    return MiddlewareAgentListResponse(
        agents=[
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "description": a.description,
                "capabilities": [c.model_dump() for c in a.capabilities],
                "endpoint": a.endpoint,
            }
            for a in agents
        ],
        meta=build_meta(),
    )


@router.get("/middleware/capabilities", response_model=MiddlewareCapabilityListResponse)
async def list_middleware_capabilities(
    x_tenant_id: str = Depends(validate_tenant_id_header),
    filter: str | None = Query(None, description="Substring filter for capability name"),
    _api_key: Optional[str] = Depends(verify_api_key),
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> MiddlewareCapabilityListResponse:
    """Discover capabilities across all registered agents for the tenant.

    Requires API key authentication when A2A_API_KEY is configured.
    """
    if not await limiter.allow(x_tenant_id):
        raise rate_limit_exceeded()

    capabilities = middleware.discover_capabilities(x_tenant_id, filter)

    return MiddlewareCapabilityListResponse(
        capabilities=capabilities,
        meta=build_meta(),
    )


@router.post("/middleware/agents/{agent_id}/delegate", response_model=MiddlewareDelegateResponse)
async def delegate_to_agent(
    agent_id: str,
    body: MiddlewareDelegateRequest,
    x_tenant_id: str = Depends(validate_tenant_id_header),
    _api_key: Optional[str] = Depends(verify_api_key),
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> MiddlewareDelegateResponse:
    """Delegate a task to a specific agent via AG-UI streaming.

    The target agent_id must belong to the calling tenant.
    Requires API key authentication when A2A_API_KEY is configured.
    """
    if not await limiter.allow(x_tenant_id):
        raise rate_limit_exceeded()

    # Validate tenant prefix in target agent_id
    expected_prefix = f"{x_tenant_id}:"
    if not agent_id.startswith(expected_prefix):
        raise A2APermissionError(
            reason="Cannot delegate to agent outside tenant scope",
            resource_id=agent_id,
        )

    # Collect events from the SSE stream with a max limit to prevent memory exhaustion
    # NOTE: For true streaming, consider using StreamingResponse with SSE format.
    # Current implementation limits to 1000 events as a safeguard.
    MAX_EVENTS = 1000
    events: list[dict[str, Any]] = []
    truncated = False

    try:
        async for event in middleware.delegate_task(
            target_agent_id=agent_id,
            capability_name=body.capability_name,
            input_data=body.input_data,
            context=body.context,
        ):
            if len(events) >= MAX_EVENTS:
                truncated = True
                logger.warning(
                    "a2a_delegation_events_truncated",
                    agent_id=agent_id,
                    capability=body.capability_name,
                    max_events=MAX_EVENTS,
                )
                break
            events.append(event)

        status = "completed_truncated" if truncated else "completed"
        return MiddlewareDelegateResponse(
            status=status,
            events=events,
            meta=build_meta(),
        )
    except A2AAgentNotFoundError:
        raise
    except A2ACapabilityNotFoundError:
        raise
    except Exception as e:
        logger.error(
            "a2a_delegation_failed",
            agent_id=agent_id,
            capability=body.capability_name,
            error=str(e),
        )
        raise A2ADelegationError(
            reason=str(e),
            task_id=None,
        ) from e
