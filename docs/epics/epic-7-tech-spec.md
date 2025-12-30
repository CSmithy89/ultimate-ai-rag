# Epic 7 Tech Spec: Protocol Integration & Extensibility

**Version:** 1.0
**Created:** 2025-12-30
**Status:** Ready for Implementation

---

## Overview

Epic 7 delivers a protocol layer that makes the platform interoperable with external agents and tools. We add an MCP-style tool server, a lightweight A2A collaboration API, a Python extension SDK for third-party integrations, and a universal AG-UI endpoint for non-Copilot clients.

### Business Value

- External agents can discover and invoke platform tools via a standardized interface.
- Teams can federate multiple agents through a collaboration workflow.
- Python developers get a supported SDK for integrating with the platform.
- AG-UI access is opened to non-Copilot clients without duplicating logic.

### Functional Requirements Covered

| FR | Description | Story |
|----|-------------|-------|
| FR23 | MCP-compatible tool discovery and invocation | 7-1 |
| FR24 | Agent-to-agent collaboration endpoints | 7-2 |
| FR25 | Python extension SDK for integrations | 7-3 |
| FR26 | Universal AG-UI protocol access | 7-4 |

### NFRs Addressed

| NFR | Requirement | Implementation |
|-----|-------------|----------------|
| NFR8 | Protocol responses < 2s | Cached tool registry + fast routing |
| NFR9 | Secure multi-tenant isolation | Tenant-aware routing + validation |
| NFR10 | Extensible integrations | SDK + pluggable tool registry |

---

## Architecture Decisions

### 1. MCP Tool Registry

**Decision:** Implement an internal tool registry exposed via FastAPI endpoints.

**Rationale:** We can support MCP clients without pulling in extra dependencies by mapping MCP-style tool definitions to existing backend capabilities.

**Endpoints:**
- `GET /api/v1/mcp/tools` - list tools and JSON schemas
- `POST /api/v1/mcp/call` - invoke a tool by name

### 2. A2A Collaboration API

**Decision:** Provide REST endpoints for session creation and message exchange, backed by an in-memory store with optional future persistence.

**Rationale:** Enables agent-to-agent workflows without introducing new infrastructure; can be upgraded to Redis later.

### 3. Python Extension SDK

**Decision:** Create a lightweight SDK in `backend/src/agentic_rag_backend/sdk` using `httpx` for HTTP calls.

**Rationale:** Keeps distribution simple while providing typed helpers for MCP and AG-UI interactions.

### 4. Universal AG-UI Endpoint

**Decision:** Add a generic AG-UI endpoint that accepts standard AG-UI payloads and streams events, reusing the existing bridge.

**Rationale:** Avoid duplicating logic from CopilotKit while enabling other clients to integrate.

---

## Component Changes

### New Modules

| Module | Purpose |
|--------|---------|
| `backend/src/agentic_rag_backend/protocols/mcp.py` | Tool registry + MCP handlers |
| `backend/src/agentic_rag_backend/protocols/a2a.py` | A2A session manager |
| `backend/src/agentic_rag_backend/api/routes/mcp.py` | MCP endpoints |
| `backend/src/agentic_rag_backend/api/routes/a2a.py` | A2A endpoints |
| `backend/src/agentic_rag_backend/api/routes/ag_ui.py` | Universal AG-UI endpoint |
| `backend/src/agentic_rag_backend/sdk/client.py` | Python SDK client |
| `backend/src/agentic_rag_backend/sdk/models.py` | SDK data models |

### Modified Modules

| Module | Change |
|--------|--------|
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export new routers |
| `backend/src/agentic_rag_backend/main.py` | Register new routers |
| `backend/src/agentic_rag_backend/config.py` | Add protocol settings |

---

## API Contracts

### MCP Tool Registry

```json
GET /api/v1/mcp/tools
{
  "tools": [
    {
      "name": "knowledge.search",
      "description": "Run a RAG query",
      "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
    }
  ]
}
```

```json
POST /api/v1/mcp/call
{
  "tool": "knowledge.search",
  "arguments": {"query": "What is Graphiti?", "tenant_id": "..."}
}
```

### A2A Collaboration

- `POST /api/v1/a2a/sessions` - create session
- `POST /api/v1/a2a/sessions/{id}/messages` - add message
- `GET /api/v1/a2a/sessions/{id}` - fetch session transcript

### Universal AG-UI

- `POST /api/v1/ag-ui` - stream AG-UI events (SSE)
- Request body aligns with `CopilotRequest` while remaining protocol-agnostic

---

## Story Breakdown

1. **7-1 MCP Tool Server Implementation**
   - Build tool registry + endpoints
   - Add tests for list/call behavior

2. **7-2 A2A Agent Collaboration**
   - Implement session storage + APIs
   - Add validation + rate limiting

3. **7-3 Python Extension SDK**
   - Provide HTTP client + typed models
   - Document usage and integration

4. **7-4 Universal AG-UI Protocol Access**
   - Add generic AG-UI endpoint
   - Reuse AG-UI bridge for events

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Protocol mismatch with MCP clients | Medium | Strict schema validation + tests |
| A2A abuse or spam | Medium | Rate limiting + tenant validation |
| SDK drift from API | Medium | Versioned client with typed models |
| AG-UI integration regressions | Low | Shared bridge + regression tests |

---

## Testing Strategy

- Unit tests for MCP tool registry + A2A session manager
- API tests for MCP list/call + A2A session flow + AG-UI SSE
- Contract tests for SDK serialization

---

## Out of Scope

- Full MCP JSON-RPC over stdio support
- Persistent A2A storage (future epic)
- Public package publishing (SDK stays internal for now)
