# Story 7-1: MCP Tool Server Implementation

Status: done
Epic: 7 - Protocol Integration & Extensibility
Priority: High

## User Story

As a **platform integrator**,
I want **a tool server that exposes MCP-style tool discovery and invocation**,
So that **external agents can safely call platform capabilities via a standard interface**.

## Acceptance Criteria

- Given the backend is running
- When a client calls `GET /api/v1/mcp/tools`
- Then it receives a list of tools with `name`, `description`, and `input_schema`
- When a client calls `POST /api/v1/mcp/call` with a valid tool name and arguments
- Then the tool executes and returns a structured result payload
- And invalid tool names return a clear 404-style error response
- And tool calls require a tenant_id for multi-tenant isolation
- And requests are rate-limited per tenant

## Technical Approach

### 1. MCP Tool Registry

Create a lightweight registry that defines available tools and their input schemas. The registry supports:
- Listing tool metadata (name, description, schema)
- Invoking a tool by name with validated arguments

### 2. MCP API Routes

Add a new FastAPI router with endpoints:
- `GET /api/v1/mcp/tools`
- `POST /api/v1/mcp/call`

Tool invocation will use existing backend services:
- `knowledge.query` -> OrchestratorAgent.run
- `knowledge.graph_stats` -> Neo4j stats

### 3. Validation + Rate Limiting

Reuse the existing rate limiter and enforce a required tenant_id in tool arguments.

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/protocols/mcp.py` | Tool registry and invocation logic |
| `backend/src/agentic_rag_backend/api/routes/mcp.py` | MCP endpoints |
| `backend/tests/api/routes/test_mcp.py` | API tests for MCP endpoints |

### Modified Files

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export MCP router |
| `backend/src/agentic_rag_backend/main.py` | Register MCP routes |

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| Tool registry list + call | `backend/tests/api/routes/test_mcp.py` |

### Manual Verification Steps

1. Start the backend server
2. `curl http://localhost:8000/api/v1/mcp/tools`
3. `curl -X POST http://localhost:8000/api/v1/mcp/call -d '{"tool": "knowledge.query", "arguments": {"tenant_id": "demo", "query": "test"}}'`

## Dependencies

- No new dependencies required (reuse FastAPI + Pydantic)

## Development Log

### Implementation Summary

- Added MCP tool registry with built-in `knowledge.query` and `knowledge.graph_stats` tools.
- Exposed MCP discovery + call endpoints with tenant-aware rate limiting.
- Added API tests covering tool listing, invocation, and error paths.

### Files Changed

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/protocols/mcp.py` | New MCP tool registry + handlers |
| `backend/src/agentic_rag_backend/api/routes/mcp.py` | New MCP endpoints |
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export MCP router |
| `backend/src/agentic_rag_backend/main.py` | Register MCP routes |
| `backend/src/agentic_rag_backend/protocols/__init__.py` | Export MCP registry |
| `backend/tests/api/routes/test_mcp.py` | Added MCP endpoint tests |

### Tests

- `backend/tests/api/routes/test_mcp.py`

## Senior Developer Review

**Reviewer:** Code Review Agent
**Review Round:** 1

### Review Outcome: APPROVE

### Notes

- MCP endpoints enforce tenant_id and rate limiting consistently with existing patterns.
- Tool registry uses existing orchestrator and Neo4j services without new dependencies.
- Test coverage includes success + error paths for discovery and invocation.
