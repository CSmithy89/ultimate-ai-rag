# ADR-002: Protocol Selection (MCP, A2A, AG-UI)

**Status:** Accepted
**Date:** 2025-12-28
**Deciders:** Architecture Team
**Technical Story:** FR6-FR10 (Agentic Orchestration), FR19-FR22 (Copilot Interface)

## Context

The Agentic RAG platform requires standardized protocols for three critical communication patterns:

1. **Agent-to-Tool Communication** - How agents invoke external tools and services
2. **Agent-to-Agent Communication** - How agents delegate tasks to other agents
3. **Agent-to-Frontend Communication** - How agents stream state and UI updates to the frontend

The AI ecosystem in 2025 has converged on emerging protocol standards that enable interoperability between frameworks. The PRD mandates 100% protocol compliance (NFR7) for ecosystem compatibility.

### Protocol Landscape (2025)

| Protocol | Purpose | Adoption |
|----------|---------|----------|
| MCP (Model Context Protocol) | Tool/resource access | Anthropic, OpenAI, VS Code, Cursor |
| A2A (Agent-to-Agent) | Multi-agent orchestration | Google, CopilotKit, emerging standard |
| AG-UI (Agent-User Interface) | Real-time frontend sync | CopilotKit native |

## Decision

We adopt **MCP, A2A, and AG-UI** as the three core protocols for the platform:

### 1. MCP (Model Context Protocol) for Tool Execution

**Specification:** JSON-RPC 2.0 over HTTP/SSE
**Transport:** HTTP for request/response, SSE for streaming

MCP provides standardized tool definitions that can be consumed by:
- Claude Desktop / Claude Code
- Cursor IDE
- VS Code Copilot
- Any MCP-compatible client

### 2. A2A (Agent-to-Agent) for Delegation

**Specification:** Google/CopilotKit A2A Draft
**Transport:** HTTPS with session management

A2A enables:
- Task delegation between agents
- Context preservation across agent boundaries
- Capability-based agent discovery

### 3. AG-UI (Agent-User Interface) for Frontend Sync

**Specification:** CopilotKit AG-UI Events
**Transport:** Server-Sent Events (SSE)

AG-UI streams:
- Agent state changes (STATE_SNAPSHOT, STATE_DELTA)
- Message history (MESSAGES_SNAPSHOT)
- Generative UI components (A2UI widgets)
- Tool call visualization

### Protocol Stack Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Protocol Layer                            │
├─────────────┬─────────────┬─────────────┬─────────────┬─────┤
│    MCP      │    A2A      │   AG-UI     │   MCP-UI    │O-J-UI│
│  (Tools)    │ (Delegate)  │  (Stream)   │  (Embed)    │(Decl)│
├─────────────┴─────────────┴─────────────┴─────────────┴─────┤
│                   Observability Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Prometheus  │  │  Telemetry  │  │  Benchmarks │          │
│  │   Metrics   │  │   Endpoint  │  │   Suite     │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                    Security Layer                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Tenant    │  │   Origin    │  │    Rate     │          │
│  │  Isolation  │  │ Validation  │  │   Limits    │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Consequences

### Positive

- **Ecosystem Compatibility**: Tools work with Claude Desktop, Cursor, and any MCP client
- **Framework Agnostic**: PydanticAI, CrewAI, LangGraph all have native protocol support
- **Future-Proof**: Protocols are emerging industry standards with growing adoption
- **Interoperability**: External agents can participate via A2A
- **Rich UX**: AG-UI enables real-time streaming and generative UI

### Negative

- **Protocol Complexity**: Three protocols to implement and maintain
- **Versioning Risk**: Protocols are still evolving (not finalized standards)
- **Learning Curve**: Developers need to understand all three protocols
- **Testing Overhead**: Protocol compliance tests add to test suite

### Neutral

- **Transport Standardization**: All use HTTP-based transports (no custom protocols)
- **JSON-Based**: All protocols use JSON payloads (easy debugging)

## Alternatives Considered

### 1. Custom REST APIs Only

**Rejected because:**
- No ecosystem interoperability
- Would require custom integrations for every client
- Loses real-time streaming capabilities
- Not compatible with agent marketplaces

### 2. LangChain Tool Interface

**Rejected because:**
- LangChain-specific, not portable
- Conflicts with Agno framework choice
- No standardized discovery mechanism
- Limited adoption outside LangChain ecosystem

### 3. OpenAI Function Calling Only

**Rejected because:**
- Vendor lock-in to OpenAI
- No agent-to-agent delegation
- No real-time state streaming
- Limited tool metadata support

### 4. gRPC for All Protocols

**Rejected because:**
- Higher complexity for browser clients
- Protocol buffer maintenance overhead
- Overkill for our use case
- Less developer-friendly for debugging

### 5. WebSockets Instead of SSE

**Rejected because:**
- SSE is simpler for unidirectional streaming
- AG-UI specification uses SSE
- WebSocket complexity not needed for our use case
- SSE has better proxy/CDN compatibility

## Implementation Notes

### MCP Server Implementation

```python
# backend/src/agentic_rag_backend/protocols/mcp_server.py
@mcp.tool()
async def vector_search(query: str, top_k: int = 10) -> list[SearchResult]:
    """Search knowledge base using semantic similarity."""
    ...

@mcp.tool()
async def query_with_reranking(query: str) -> RerankedResults:
    """Execute hybrid retrieval with cross-encoder reranking."""
    ...
```

### A2A Middleware

```python
# backend/src/agentic_rag_backend/protocols/a2a_middleware.py
class A2AMiddlewareAgent:
    """Agent delegation with resource limits and context preservation."""

    async def delegate(
        self,
        target_agent: str,
        task: str,
        context: dict
    ) -> A2AResponse:
        ...
```

### AG-UI Event Flow

```
RUN_STARTED (run_id) --> STATE_SNAPSHOT --> STATE_DELTA* --> RUN_FINISHED
                              |
                              v
                       MESSAGES_SNAPSHOT
                              |
                              v
                       A2UI_WIDGETS (optional)
```

### Configuration Options

| Config | Purpose | Default |
|--------|---------|---------|
| `MCP_ENABLED` | Enable MCP tool server | `true` |
| `MCP_STDIO_ENABLED` | Enable stdio transport | `false` |
| `A2A_ENABLED` | Enable A2A middleware | `true` |
| `A2A_LIMITS_BACKEND` | Resource limits storage | `memory` |
| `A2A_SESSION_LIMIT_PER_TENANT` | Max sessions per tenant | `100` |
| `AG_UI_ENABLED` | Enable AG-UI streaming | `true` |
| `MCP_UI_ENABLED` | Enable iframe embedding | `false` |
| `OPEN_JSON_UI_ENABLED` | Enable declarative UI | `true` |

### Error Taxonomy (RFC 7807)

| Error Code | HTTP Status | Description |
|------------|-------------|-------------|
| `AGENT_EXECUTION_ERROR` | 500 | Internal agent failure |
| `TENANT_REQUIRED` | 401 | Missing X-Tenant-ID |
| `RATE_LIMITED` | 429 | Request throttled |
| `TIMEOUT` | 504 | Operation timeout |
| `CAPABILITY_NOT_FOUND` | 404 | Unknown capability |

## References

- [Architecture Document](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/architecture.md) - Protocol sections
- [Epic 7: Protocol Integration](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/project-planning-artifacts/epics.md)
- [Epic 14: MCP Wrapper Architecture](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/epics/epic-14-tech-spec.md)
- [Epic 22: Advanced Protocol Integration](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/_bmad-output/epics/epic-22-tech-spec.md)
- [Protocol Integration Guide](/home/chris/projects/work/Agentic Rag and Graphrag with copilot/docs/guides/protocol-integration/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [CopilotKit AG-UI](https://docs.copilotkit.ai/coagents/concepts/agent-ui-protocol)
