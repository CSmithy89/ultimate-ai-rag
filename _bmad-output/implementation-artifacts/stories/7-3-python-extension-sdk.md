# Story 7-3: Python Extension SDK

Status: done
Epic: 7 - Protocol Integration & Extensibility
Priority: Medium

## User Story

As a **Python integrator**,
I want **a supported SDK for MCP and A2A APIs**,
So that **I can build integrations without hand-crafting HTTP requests**.

## Acceptance Criteria

- Given the backend API base URL
- When a developer instantiates the SDK client
- Then they can list MCP tools and call a tool by name
- And they can create A2A sessions and post messages
- And SDK responses are typed and validated
- And errors surface as HTTP exceptions when requests fail

## Technical Approach

### 1. SDK Module

Create a lightweight SDK package under `backend/src/agentic_rag_backend/sdk` with:
- `AgenticRagClient` async client
- Pydantic models for MCP + A2A responses

### 2. HTTP Layer

Use `httpx.AsyncClient` with configurable base URL, timeout, and headers.

### 3. Typed Responses

Parse JSON into SDK models to provide consistent shapes for integrations.

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/sdk/client.py` | SDK async client |
| `backend/src/agentic_rag_backend/sdk/models.py` | SDK response models |
| `backend/src/agentic_rag_backend/sdk/__init__.py` | SDK exports |
| `backend/tests/sdk/test_client.py` | SDK client tests |

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| SDK list tools + call tool | `backend/tests/sdk/test_client.py` |
| SDK A2A session flow | `backend/tests/sdk/test_client.py` |

## Dependencies

- Uses existing `httpx` dependency

## Development Log

### Implementation Summary

- Added Python SDK client with MCP and A2A helpers using httpx.
- Added typed SDK models for MCP tool responses and A2A sessions.
- Added SDK tests with mock transports.

### Files Changed

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/sdk/client.py` | SDK async client |
| `backend/src/agentic_rag_backend/sdk/models.py` | SDK response models |
| `backend/src/agentic_rag_backend/sdk/__init__.py` | SDK exports |
| `backend/tests/sdk/test_client.py` | SDK tests |

### Tests

- `backend/tests/sdk/test_client.py`

## Senior Developer Review

**Reviewer:** Code Review Agent
**Review Round:** 1

### Review Outcome: APPROVE

### Notes

- SDK wraps MCP and A2A endpoints with typed response models.
- Client supports injected httpx client for tests and custom transports.
- Tests validate MCP and A2A flows with mock responses.
