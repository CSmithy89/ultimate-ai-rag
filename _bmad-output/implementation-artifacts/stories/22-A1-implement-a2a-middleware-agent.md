# Story 22-A1: Implement A2A Middleware Agent

Status: drafted

Epic: 22 - Advanced Protocol Integration
Priority: P0 - HIGH
Story Points: 8
Owner: Backend

## Story

As a **backend developer**,
I want **to implement an A2AMiddlewareAgent for agent delegation and collaboration**,
So that **agents can discover, register, and delegate tasks to each other through a standardized A2A protocol with tenant-scoped security**.

## Background

Epic 22 builds on Epic 21's CopilotKit Full Integration to deliver enterprise-grade protocol capabilities. The A2AMiddlewareAgent is the foundational component that enables:

1. **Agent Registration** - Agents can register themselves with capabilities for discovery
2. **Capability Discovery** - Find agents that can handle specific tasks across the system
3. **Task Delegation** - Route tasks to specialized agents while preserving context
4. **Tenant Isolation** - All operations are scoped to the requesting tenant

### A2A Protocol Context

The A2A (Agent-to-Agent) protocol enables multi-agent collaboration where:
- An orchestrator agent can delegate specialized tasks to domain agents
- Agents advertise capabilities via registration endpoints
- Delegation uses AG-UI streaming for real-time response handling
- All operations are tenant-scoped for multi-tenant safety

### Related Prior Work

| Epic/Story | Relationship |
|------------|-------------|
| Epic 7-2: A2A Agent Collaboration | Original A2A foundation (session/message lifecycle) |
| Epic 14-2: Implement Robust A2A Protocol | Enhanced A2A with RFC 7807 errors, retries |
| Epic 21: CopilotKit Full Integration | AG-UI transport and MCP client (prerequisite) |

## Acceptance Criteria

1. **Given** the A2AMiddlewareAgent class exists in `backend/src/agentic_rag_backend/protocols/a2a_middleware.py`, **when** it is instantiated, **then** it initializes with agent_id, name, capabilities, and an empty registered agents registry.

2. **Given** an agent registration request is received via `POST /a2a/agents/register`, **when** the tenant_id from `X-Tenant-ID` header matches the agent_id prefix (e.g., `tenant123:agent-name`), **then** the agent is registered successfully with a 200 response.

3. **Given** an agent registration request with an agent_id that does NOT match the tenant_id prefix, **when** the request is processed, **then** a 403 Forbidden response is returned with RFC 7807 error format.

4. **Given** agents are registered, **when** `GET /a2a/agents` is called, **then** all registered agents for the calling tenant are returned (tenant-scoped list).

5. **Given** agents are registered with capabilities, **when** `GET /a2a/capabilities?filter=search` is called, **then** capabilities matching the filter across all tenant agents are returned with their agent_id.

6. **Given** a delegation request to a registered agent, **when** `delegate_task(target_agent_id, capability_name, input_data, context)` is called, **then** the middleware invokes the target agent's AG-UI endpoint and streams responses back via AsyncIterator.

7. **Given** a delegation request to a non-existent agent, **when** `delegate_task()` is called, **then** an `A2AAgentNotFoundError` is raised.

8. **Given** a delegation request for a non-existent capability, **when** `delegate_task()` is called, **then** an `A2ACapabilityNotFoundError` is raised.

9. **Given** all A2A operations, **when** executed, **then** they are logged via structlog with `a2a_agent_registered`, `a2a_task_delegated`, etc.

10. **Given** the middleware uses HTTP client for delegation, **when** the middleware is closed, **then** the pooled HTTP client is properly cleaned up via `close()` method.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: **Addressed** - Agent IDs must be prefixed with tenant_id, all queries filtered by tenant
- [x] Rate limiting / abuse protection: **Addressed** - Registration endpoint rate-limited to 10/min per tenant (via existing rate limiter)
- [x] Input validation / schema enforcement: **Addressed** - Pydantic models for all request/response payloads
- [x] Tests (unit/integration): **Addressed** - Unit tests for middleware, integration tests for API endpoints
- [x] Error handling + logging: **Addressed** - RFC 7807 errors, structlog for all operations
- [x] Documentation updates: **Addressed** - Docstrings on all public methods

## Security Checklist

- [ ] **Cross-tenant isolation verified**: Agent IDs must be prefixed with `{tenant_id}:`
- [ ] **Authorization checked**: API key validation via `verify_api_key` dependency
- [ ] **No information leakage**: Agent listings scoped to requesting tenant only
- [ ] **Redis keys include tenant scope**: N/A - No Redis in this story (see 22-A2)
- [ ] **Integration tests for access control**: Cross-tenant registration rejection tested
- [ ] **RFC 7807 error responses**: All errors use AppError pattern
- [ ] **File-path inputs scoped**: N/A - No file path handling

## Tasks / Subtasks

- [ ] **Task 1: Create A2A Middleware Core Classes** (AC: 1, 9, 10)
  - [ ] Create `backend/src/agentic_rag_backend/protocols/a2a_middleware.py`
  - [ ] Define `A2AAgentCapability` Pydantic model (name, description, input_schema, output_schema)
  - [ ] Define `A2AAgentInfo` Pydantic model (agent_id, name, description, capabilities, endpoint)
  - [ ] Implement `A2AMiddlewareAgent` class with `__init__`, `register_agent`, `discover_capabilities`
  - [ ] Add structlog logging for all operations
  - [ ] Add `_get_http_client()` with connection pooling (httpx.AsyncClient)
  - [ ] Add `close()` method to clean up HTTP client

- [ ] **Task 2: Implement Task Delegation** (AC: 6, 7, 8)
  - [ ] Implement `delegate_task()` async method that returns `AsyncIterator[dict[str, Any]]`
  - [ ] Implement `_invoke_agent()` internal method for AG-UI SSE streaming
  - [ ] Define `A2AAgentNotFoundError` exception
  - [ ] Define `A2ACapabilityNotFoundError` exception
  - [ ] Add timeout configuration (30s default, 5s connect)
  - [ ] Log delegation events with from_agent, to_agent, capability

- [ ] **Task 3: Implement Agent Registration API Endpoint** (AC: 2, 3)
  - [ ] Extend `backend/src/agentic_rag_backend/api/routes/a2a.py`
  - [ ] Add `POST /a2a/agents/register` endpoint
  - [ ] Add `get_tenant_id` dependency from `X-Tenant-ID` header
  - [ ] Add `verify_api_key` dependency for authorization
  - [ ] Validate agent_id prefix matches tenant_id
  - [ ] Return 403 with RFC 7807 format if prefix mismatch
  - [ ] Return 200 with `{"status": "registered", "agent_id": "..."}` on success

- [ ] **Task 4: Implement Agent Discovery API Endpoints** (AC: 4, 5)
  - [ ] Add `GET /a2a/agents` endpoint (list registered agents for tenant)
  - [ ] Add `GET /a2a/capabilities` endpoint with optional `filter` query param
  - [ ] Scope agent list to calling tenant only
  - [ ] Return capability tuples as `[{"agent_id": "...", "capability": {...}}]`

- [ ] **Task 5: Wire Middleware to Application Lifecycle**
  - [ ] Add `get_a2a_middleware` dependency in `api/dependencies.py`
  - [ ] Wire middleware to `app.state` in `main.py` startup event
  - [ ] Wire middleware cleanup to `main.py` shutdown event
  - [ ] Export new classes in `protocols/__init__.py`

- [ ] **Task 6: Add Unit Tests** (AC: 1, 6, 7, 8, 9, 10)
  - [ ] Create `tests/protocols/test_a2a_middleware.py`
  - [ ] Test `A2AMiddlewareAgent` initialization
  - [ ] Test `register_agent()` with valid agent info
  - [ ] Test `discover_capabilities()` with and without filter
  - [ ] Test `delegate_task()` with mocked httpx client
  - [ ] Test `A2AAgentNotFoundError` raised for unknown agent
  - [ ] Test `A2ACapabilityNotFoundError` raised for unknown capability
  - [ ] Test `close()` cleans up HTTP client

- [ ] **Task 7: Add Integration Tests** (AC: 2, 3, 4, 5)
  - [ ] Create `tests/integration/test_a2a_endpoints.py`
  - [ ] Test `POST /a2a/agents/register` with valid tenant-scoped agent_id
  - [ ] Test `POST /a2a/agents/register` returns 403 for mismatched tenant prefix
  - [ ] Test `GET /a2a/agents` returns only tenant-scoped agents
  - [ ] Test `GET /a2a/capabilities` with filter
  - [ ] Test registration requires valid API key

## Technical Notes

### A2AMiddlewareAgent Class Structure

```python
# backend/src/agentic_rag_backend/protocols/a2a_middleware.py
from typing import Any, AsyncIterator
from pydantic import BaseModel
import httpx
import structlog

logger = structlog.get_logger(__name__)

class A2AAgentCapability(BaseModel):
    """Advertised capability of an A2A agent."""
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]

class A2AAgentInfo(BaseModel):
    """Registered A2A agent information."""
    agent_id: str
    name: str
    description: str
    capabilities: list[A2AAgentCapability]
    endpoint: str  # AG-UI endpoint for this agent

class A2AMiddlewareAgent:
    """Middleware agent for A2A protocol collaboration."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: list[A2AAgentCapability],
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self._registered_agents: dict[str, A2AAgentInfo] = {}
        self._http_client: httpx.AsyncClient | None = None

    def register_agent(self, agent_info: A2AAgentInfo) -> None:
        """Register an agent for collaboration."""
        # Implementation per tech spec

    def discover_capabilities(
        self,
        capability_filter: str | None = None,
    ) -> list[tuple[str, A2AAgentCapability]]:
        """Discover capabilities across all registered agents."""
        # Implementation per tech spec

    async def delegate_task(
        self,
        target_agent_id: str,
        capability_name: str,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate a task to another agent."""
        # Implementation per tech spec

    async def close(self) -> None:
        """Close the HTTP client pool."""
        # Implementation per tech spec
```

### API Registration Endpoint Pattern

```python
# backend/src/agentic_rag_backend/api/routes/a2a.py (extend)
@router.post("/a2a/agents/register")
async def register_agent(
    agent_info: A2AAgentInfo,
    tenant_id: str = Depends(get_tenant_id),  # From X-Tenant-ID header
    api_key: str = Depends(verify_api_key),    # Validates Bearer token
    middleware: A2AMiddlewareAgent = Depends(get_a2a_middleware),
) -> dict[str, str]:
    """Register an agent for A2A collaboration (tenant-scoped)."""
    # Validate agent belongs to this tenant
    if not agent_info.agent_id.startswith(f"{tenant_id}:"):
        raise HTTPException(403, "Agent ID must be prefixed with tenant ID")
    middleware.register_agent(agent_info)
    return {"status": "registered", "agent_id": agent_info.agent_id}
```

### HTTP Client Configuration

```python
def _get_http_client(self) -> httpx.AsyncClient:
    """Get or create pooled HTTP client."""
    if not hasattr(self, "_http_client") or self._http_client is None:
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            timeout=httpx.Timeout(30.0, connect=5.0),
        )
    return self._http_client
```

### SSE Streaming Pattern

```python
async def _invoke_agent(
    self,
    endpoint: str,
    capability: str,
    input_data: dict[str, Any],
    context: dict[str, Any] | None,
) -> AsyncIterator[dict[str, Any]]:
    """Invoke an agent's capability via AG-UI protocol."""
    async with self._get_http_client().stream(
        "POST",
        endpoint,
        json={
            "capability": capability,
            "input": input_data,
            "context": context or {},
        },
        headers={"Accept": "text/event-stream"},
        timeout=httpx.Timeout(30.0, connect=5.0),
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                import json
                yield json.loads(line[6:])
```

### Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/protocols/a2a_middleware.py` | Create | Middleware agent core classes |
| `backend/src/agentic_rag_backend/api/routes/a2a.py` | Modify | Add registration/discovery endpoints |
| `backend/src/agentic_rag_backend/api/dependencies.py` | Modify | Add `get_a2a_middleware` dependency |
| `backend/src/agentic_rag_backend/main.py` | Modify | Wire middleware to app lifecycle |
| `backend/src/agentic_rag_backend/protocols/__init__.py` | Modify | Export new classes |
| `tests/protocols/test_a2a_middleware.py` | Create | Unit tests |
| `tests/integration/test_a2a_endpoints.py` | Create | Integration tests |

### Dependencies

- `httpx ^0.27.x` - Already installed, for pooled HTTP client
- `structlog` - Already installed, for structured logging
- `pydantic` - Already installed, for models

## Dependencies

- **Epic 21 completed** - AG-UI transport must be available for delegation streaming
- **Epic 7-2** - Basic A2A session/message lifecycle (completed)
- **Epic 14-2** - Robust A2A protocol foundation (completed)

## Definition of Done

- [ ] `A2AMiddlewareAgent` class implemented with registration, discovery, delegation
- [ ] `A2AAgentCapability` and `A2AAgentInfo` Pydantic models created
- [ ] `A2AAgentNotFoundError` and `A2ACapabilityNotFoundError` exceptions defined
- [ ] `POST /a2a/agents/register` endpoint functional with tenant validation
- [ ] `GET /a2a/agents` endpoint returns tenant-scoped agent list
- [ ] `GET /a2a/capabilities` endpoint with filter support
- [ ] Structlog logging for all A2A operations
- [ ] HTTP client pooling with proper lifecycle management
- [ ] Unit tests for middleware (>90% coverage)
- [ ] Integration tests for API endpoints
- [ ] RFC 7807 error format for all error responses
- [ ] Code review approved
- [ ] Story file updated with Dev Notes

## Dev Notes

*To be completed after implementation*

## Dev Agent Record

*To be completed after implementation*

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

## Test Outcomes

*To be completed after implementation*

## Challenges Encountered

*To be completed after implementation*

## Senior Developer Review

*To be completed after code review*
