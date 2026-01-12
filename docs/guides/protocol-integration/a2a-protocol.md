# A2A Protocol Guide

**Version:** 1.0
**Last Updated:** 2026-01-11
**Related Stories:** 22-A1, 22-A2

---

## Overview

A2A (Agent-to-Agent) is the internal protocol for agent delegation and collaboration within the Agentic RAG platform. It enables agents to discover capabilities, delegate tasks, and share context.

## Architecture

```mermaid
graph TB
    subgraph A2A["A2A Manager"]
        REG[Agent Registry]
        MW[Middleware Agent]
        LIM[Resource Limits]
        MSG[Message Router]
    end

    subgraph Agents["Registered Agents"]
        ORC[Orchestrator]
        RET[Retrieval Agent]
        IDX[Indexing Agent]
        CUSTOM[Custom Agents]
    end

    subgraph Storage["Storage Layer"]
        REDIS[(Redis)]
    end

    ORC -->|Register| REG
    RET -->|Register| REG
    IDX -->|Register| REG
    CUSTOM -->|Register| REG

    ORC -->|Delegate| MW
    MW -->|Route| MSG
    MSG --> RET
    MSG --> IDX

    LIM -->|Check Limits| REDIS
    REG -->|Store| REDIS
    MSG -->|Track Usage| LIM
```

## Core Components

### Agent Registry

Manages agent registration and capability discovery.

```python
class A2AAgentRegistration(BaseModel):
    agent_id: str           # Unique identifier
    tenant_id: str          # Tenant scope
    capabilities: list[str] # Supported operations
    endpoint: str           # Callback URL
    metadata: dict          # Additional info
```

### Middleware Agent

Routes messages between agents with context preservation.

```python
class A2AMiddlewareAgent:
    async def delegate(
        self,
        capability: str,
        payload: dict,
        tenant_id: str,
        context: dict | None = None,
    ) -> AsyncIterator[A2AMessage]:
        """Delegate a task to an agent with the requested capability."""
```

### Resource Limits

Enforces per-tenant and per-session limits.

```python
class A2AResourceLimits(BaseModel):
    session_limit_per_tenant: int = 100    # Max concurrent sessions
    message_limit_per_session: int = 1000  # Max messages per session
    session_ttl_hours: int = 24            # Session expiration
    message_rate_limit: int = 60           # Messages per minute
```

## Capabilities

### Standard Capabilities

| Capability | Description | Agent |
|------------|-------------|-------|
| `QUERY` | Information retrieval | Retrieval Agent |
| `INGEST` | Document ingestion | Indexing Agent |
| `ANALYZE` | Data analysis | Analysis Agent |
| `SUMMARIZE` | Content summarization | Summarization Agent |

### Capability Discovery

```python
# Find agents with specific capability
agents = await registry.find_by_capability("QUERY", tenant_id)

# Get all capabilities for tenant
capabilities = await registry.list_capabilities(tenant_id)
```

## Message Format

### A2A Message Structure

```python
class A2AMessage(BaseModel):
    message_id: str
    sender_id: str
    recipient_id: str
    capability: str
    payload: dict
    context: dict | None
    timestamp: datetime
    tenant_id: str
```

### Delegation Request

```json
{
  "capability": "QUERY",
  "payload": {
    "query": "What is RAG?",
    "filters": {"source": "documentation"}
  },
  "context": {
    "session_id": "sess_abc123",
    "parent_task_id": "task_xyz"
  }
}
```

### Delegation Response

```json
{
  "status": "success",
  "result": {
    "answer": "RAG stands for...",
    "sources": [{"title": "...", "url": "..."}]
  },
  "agent_id": "retrieval-agent-1",
  "execution_time_ms": 450
}
```

## Resource Limits

### Limit Types

| Limit | Default | Description |
|-------|---------|-------------|
| `session_limit_per_tenant` | 100 | Max concurrent sessions |
| `message_limit_per_session` | 1000 | Max messages per session |
| `session_ttl_hours` | 24 | Session expiration |
| `message_rate_limit` | 60/min | Rate limit per session |

### Enforcement

```python
# Create resource manager
manager = A2AResourceManagerFactory.create(
    implementation="redis",
    redis_client=redis,
    limits=A2AResourceLimits(
        session_limit_per_tenant=50,
        message_rate_limit=30,
    ),
)

# Check before processing
if not await manager.check_and_increment(tenant_id, session_id):
    raise RateLimitError("Session limit exceeded")
```

### Redis Key Structure

```
a2a:sessions:{session_id}                 # Session metadata (A2ASessionManager)
a2a:tenant:{tenant_id}                    # Resource limits (tenant usage)
a2a:session:{session_id}:info             # Resource limits (session usage)
a2a:session:{session_id}:rate             # Resource limits (rate window)
```

## Configuration

### Environment Variables

```bash
# Enable A2A
A2A_ENABLED=true

# Delegation behavior
A2A_TASK_DEFAULT_TIMEOUT_SECONDS=300
A2A_TASK_MAX_RETRIES=3
A2A_SIGNING_SECRET=  # Optional shared secret for agent-to-agent signing
A2A_SIGNING_TTL_SECONDS=300

# Resource limits
A2A_SESSION_LIMIT_PER_TENANT=100
A2A_MESSAGE_LIMIT_PER_SESSION=1000
A2A_SESSION_TTL_HOURS=24
A2A_MESSAGE_RATE_LIMIT=60

# Limits backend
A2A_LIMITS_BACKEND=memory  # memory | redis
# Redis configuration (used when A2A_LIMITS_BACKEND=redis)
REDIS_URL=redis://localhost:6379/1
```

### Python Configuration

```python
# settings.py
class A2ASettings(BaseModel):
    enabled: bool = True
    task_default_timeout_seconds: int = 300
    task_max_retries: int = 3
    limits: A2AResourceLimits = A2AResourceLimits()
```

## Code Examples

### Registering an Agent

```python
from agentic_rag_backend.protocols.a2a_registry import A2ARegistry

registry = A2ARegistry(redis_client)

await registry.register(
    A2AAgentRegistration(
        agent_id="retrieval-agent-1",
        tenant_id="tenant_abc",
        capabilities=["QUERY", "SUMMARIZE"],
        endpoint="http://internal:8001/invoke",
        metadata={"version": "1.0"},
    )
)
```

### Delegating a Task

```python
from agentic_rag_backend.protocols.a2a_middleware import A2AMiddlewareAgent

middleware = A2AMiddlewareAgent(registry, resource_manager)

async for message in middleware.delegate(
    capability="QUERY",
    payload={"query": "What is GraphRAG?"},
    tenant_id="tenant_abc",
    context={"session_id": "sess_123"},
):
    process_message(message)
```

### Handling Rate Limits

```python
from agentic_rag_backend.protocols.a2a_resource_limits import (
    A2AResourceManagerFactory,
)

manager = A2AResourceManagerFactory.create("redis", redis_client)

try:
    await manager.acquire_session(tenant_id, session_id)
    # Process messages
    for msg in messages:
        if not await manager.check_message_limit(tenant_id, session_id):
            raise RateLimitError("Message limit exceeded")
        await process(msg)
finally:
    await manager.release_session(tenant_id, session_id)
```

## Security Considerations

### SSRF Protection

The middleware validates all agent endpoints to prevent SSRF attacks:

```python
def is_safe_endpoint_url(url: str) -> bool:
    """Rejects:
    - Non-HTTP(S) schemes
    - Localhost variants
    - Private IP ranges (10.x, 192.168.x, 172.16-31.x)
    - Link-local and reserved addresses
    """
```

### Request Signing

If `A2A_SIGNING_SECRET` is configured, incoming `POST /a2a/execute` requests must
include signed headers:

- `X-A2A-Timestamp`: Unix timestamp (seconds)
- `X-A2A-Signature`: HMAC-SHA256 of `{timestamp}.{canonical_json_body}`

Requests are rejected when the timestamp skew exceeds `A2A_SIGNING_TTL_SECONDS`.

### Tenant Isolation

- Agent IDs are prefixed with tenant_id
- All queries include tenant_id filter
- Sessions are scoped to tenant

### Rate Limiting

- Per-session message limits prevent abuse
- Rate limiting prevents burst attacks
- Session TTL ensures cleanup

## Troubleshooting

### Agent Not Found

**Symptoms**: `A2AAgentNotFoundError` when delegating

**Causes**:
- Agent not registered
- Wrong tenant_id
- Agent registration expired

**Solutions**:
```python
# Verify registration
agents = await registry.list_agents(tenant_id)
print(f"Registered agents: {agents}")

# Re-register if needed
await registry.register(agent_config)
```

### Session Limit Exceeded

**Symptoms**: `SessionLimitExceeded` error

**Causes**:
- Too many concurrent sessions
- Sessions not being released

**Solutions**:
```python
# Check current usage
usage = await manager.get_tenant_usage(tenant_id)
print(f"Active sessions: {usage.active_sessions}")

# Force cleanup stale sessions
await manager.cleanup_expired_sessions(tenant_id)
```

### Rate Limit Exceeded

**Symptoms**: `RateLimitExceeded` error

**Causes**:
- Messages sent too quickly
- Rate limit too low for use case

**Solutions**:
```bash
# Increase rate limit
export A2A_MESSAGE_RATE_LIMIT=120

# Or implement backoff in client
```

### Circular Delegation

**Symptoms**: Stack overflow or timeout

**Causes**:
- Agent A delegates to B, B delegates back to A

**Solutions**:
```python
# Middleware tracks delegation depth
# MAX_DELEGATION_DEPTH prevents infinite loops

# Check logs for delegation chain
logger.info("delegation_chain", chain=context.get("delegation_path"))
```

## Prometheus Metrics

| Metric | Labels | Description |
|--------|--------|-------------|
| `a2a_delegations_total` | `tenant_id`, `capability` | Total delegations |
| `a2a_delegation_duration_seconds` | `tenant_id` | Delegation latency |
| `a2a_active_sessions` | `tenant_id` | Active sessions |
| `a2a_rate_limit_rejections_total` | `tenant_id` | Rate limit hits |

## Framework Integration (Headless Agent Pattern)

The A2A protocol enables framework-agnostic integration, allowing external agent frameworks to leverage the Agentic RAG platform as a knowledge service. This "headless" pattern means any framework with A2A or MCP support can connect without custom adapters.

### Supported Frameworks

Modern agent frameworks have native protocol support:

| Framework | A2A Support | MCP Support | Integration Method |
|-----------|-------------|-------------|-------------------|
| **PydanticAI** | Native | Native | Direct MCP client or A2A delegation |
| **CrewAI** | Native | Native | Tool wrapping via MCP |
| **LangGraph** | Native | Native | State-aware A2A sessions |
| **Anthropic SDK** | Via Agent Skills | Native | Claude Desktop/Code integration |
| **AutoGen** | Native | Native | Multi-agent orchestration |

### Integration Patterns

#### Pattern 1: MCP Tool Access (Recommended)

External frameworks call RAG tools directly via MCP protocol:

```python
# PydanticAI example - direct MCP tool access
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

agent = Agent(
    "openai:gpt-4o",
    mcp_servers=[
        MCPServerHTTP(url="http://localhost:3001")  # Agentic RAG MCP server
    ],
)

# Agent can now use vector_search, hybrid_retrieve, ingest_url, etc.
result = await agent.run("What do the docs say about GraphRAG?")
```

#### Pattern 2: A2A Delegation (Multi-Agent)

For multi-agent scenarios, register your agent and delegate tasks:

```python
# Register external agent
await registry.register(
    A2AAgentRegistration(
        agent_id="external-pydantic-agent",
        tenant_id="tenant_abc",
        capabilities=["CUSTOM_ANALYSIS"],
        endpoint="http://my-agent:8080/invoke",
        metadata={"framework": "pydantic_ai", "version": "0.1.0"},
    )
)

# Delegate from orchestrator to your agent
async for message in middleware.delegate(
    capability="CUSTOM_ANALYSIS",
    payload={"data": retrieved_context},
    tenant_id="tenant_abc",
):
    process_message(message)
```

#### Pattern 3: Hybrid (A2A + MCP)

Combine A2A for orchestration with MCP for tool access:

```python
# CrewAI example - hybrid integration
from crewai import Agent, Task, Crew
from crewai.tools import MCPTool

# Create tools from MCP server
rag_tools = MCPTool.from_server("http://localhost:3001")

# Define agent with RAG capabilities
researcher = Agent(
    role="Knowledge Researcher",
    goal="Find accurate information from the knowledge graph",
    tools=[rag_tools.vector_search, rag_tools.hybrid_retrieve],
)

# Register with A2A for delegation
# (CrewAI agent can now receive delegated tasks)
```

### Agent Interface Contract

When implementing agents that participate in A2A delegation, follow this interface:

```python
from typing import AsyncIterator
from pydantic import BaseModel

class AgentInput(BaseModel):
    """Standard input for A2A agent invocation."""
    query: str
    history: list[dict] = []
    context: dict = {}
    tenant_id: str

class AgentResponse(BaseModel):
    """Standard response from A2A agent."""
    content: str
    sources: list[dict] = []
    trajectory: list[dict] = []  # For debugging/observability
    metadata: dict = {}

# Your agent endpoint should accept AgentInput and return AgentResponse
# Example FastAPI endpoint:
@app.post("/invoke")
async def invoke_agent(input: AgentInput) -> AgentResponse:
    result = await my_framework_agent.run(input.query, context=input.context)
    return AgentResponse(
        content=result.text,
        sources=result.citations,
        trajectory=result.steps,
    )
```

### Framework-Specific Examples

See the starter templates for complete integration examples:

- `templates/pydantic_ai/` - Type-safe outputs with MCP
- `templates/crew_ai/` - Multi-agent with A2A delegation
- `templates/langgraph/` - Stateful workflows with session persistence
- `templates/anthropic/` - Claude Desktop Agent Skills

### Why No Custom Adapters?

Earlier designs considered framework-specific adapters (PydanticAIAdapter, CrewAIAdapter, etc.). This approach was deprecated because:

1. **Native Protocol Support**: Modern frameworks already implement A2A and MCP natively
2. **Maintenance Burden**: Custom adapters require updates when frameworks change
3. **Reduced Flexibility**: Protocol-based integration allows any compliant agent to participate
4. **Industry Alignment**: A2A and MCP are becoming standard agent protocols

The current architecture treats the Agentic RAG platform as a **knowledge service** that any framework can consume via standard protocols.

## Related Documentation

- [Overview](./overview.md)
- [AG-UI Protocol](./ag-ui-protocol.md)
- [MCP Integration](./mcp-integration.md)
