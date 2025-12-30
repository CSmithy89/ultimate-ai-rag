# Story 7-4: Universal AG-UI Protocol Access

Status: done
Epic: 7 - Protocol Integration & Extensibility
Priority: High

## User Story

As an **AG-UI client developer**,
I want **a universal AG-UI endpoint that is not CopilotKit-specific**,
So that **any compatible client can stream AG-UI events without extra adapters**.

## Acceptance Criteria

- Given a standard AG-UI request payload with messages and tenant_id
- When a client posts to `POST /api/v1/ag-ui`
- Then the backend streams AG-UI events as SSE
- And the request is rate-limited per tenant
- And missing tenant_id returns a clear error response
- And the endpoint reuses the existing AG-UI bridge

## Technical Approach

### 1. Universal AG-UI Request Model

Create a simplified request model that accepts:
- messages (role + content)
- tenant_id
- optional session_id
- optional actions

### 2. AG-UI Endpoint

Add a new FastAPI router that converts the request into a `CopilotRequest` and
streams events using `AGUIBridge`.

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/api/routes/ag_ui.py` | Universal AG-UI endpoint |
| `backend/tests/api/routes/test_ag_ui.py` | AG-UI endpoint tests |

### Modified Files

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export AG-UI router |
| `backend/src/agentic_rag_backend/main.py` | Register AG-UI routes |

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| AG-UI endpoint streams events | `backend/tests/api/routes/test_ag_ui.py` |

## Dependencies

- No new dependencies required (reuse FastAPI + AG-UI bridge)

## Development Log

### Implementation Summary

- Added universal AG-UI endpoint that accepts tenant-scoped requests.
- Reused AGUIBridge to stream SSE events for non-Copilot clients.
- Added AG-UI endpoint tests for event streaming and rate limiting.

### Files Changed

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/api/routes/ag_ui.py` | New AG-UI endpoint |
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export AG-UI router |
| `backend/src/agentic_rag_backend/main.py` | Register AG-UI routes |
| `backend/tests/api/routes/test_ag_ui.py` | Added AG-UI endpoint tests |

### Tests

- `backend/tests/api/routes/test_ag_ui.py`

## Senior Developer Review

**Reviewer:** Code Review Agent
**Review Round:** 1

### Review Outcome: APPROVE

### Notes

- Universal endpoint reuses AGUIBridge for consistent event sequencing.
- Tenant-based rate limiting mirrors existing Copilot endpoint behavior.
- Tests validate SSE event stream and rate limit handling.
