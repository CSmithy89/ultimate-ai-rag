# Story 7-2: A2A Agent Collaboration

Status: done
Epic: 7 - Protocol Integration & Extensibility
Priority: High

## User Story

As a **solution architect**,
I want **agent-to-agent collaboration endpoints**,
So that **multiple agents can share context and coordinate tasks through a single backend**.

## Acceptance Criteria

- Given a tenant ID
- When a client creates an A2A session
- Then the backend returns a session ID and creation timestamp
- When a client posts a message to the session
- Then the message is stored and returned in the session transcript
- And session access is scoped to the tenant_id
- And requests are rate-limited per tenant
- And invalid session IDs return a 404 response

## Technical Approach

### 1. A2A Session Manager

Implement an in-memory A2A session manager that stores:
- session_id
- tenant_id
- created_at
- messages (sender, content, timestamp, metadata)

### 2. A2A API Routes

Add FastAPI endpoints:
- `POST /api/v1/a2a/sessions`
- `POST /api/v1/a2a/sessions/{session_id}/messages`
- `GET /api/v1/a2a/sessions/{session_id}`

### 3. Tenant Validation

Ensure session lookups and message submissions enforce tenant ownership.

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/protocols/a2a.py` | A2A session manager |
| `backend/src/agentic_rag_backend/api/routes/a2a.py` | A2A endpoints |
| `backend/tests/api/routes/test_a2a.py` | A2A API tests |

### Modified Files

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export A2A router |
| `backend/src/agentic_rag_backend/main.py` | Register A2A routes |

## Testing Requirements

### Unit Tests

| Test | Location |
|------|----------|
| Create session, add message, tenant isolation | `backend/tests/api/routes/test_a2a.py` |

### Manual Verification Steps

1. `curl -X POST http://localhost:8000/api/v1/a2a/sessions -d '{"tenant_id": "demo"}'`
2. `curl -X POST http://localhost:8000/api/v1/a2a/sessions/{id}/messages -d '{"tenant_id": "demo", "sender": "agent", "content": "hello"}'`
3. `curl http://localhost:8000/api/v1/a2a/sessions/{id}?tenant_id=demo`

## Dependencies

- No new dependencies required (reuse FastAPI + Pydantic)

## Development Log

### Implementation Summary

- Implemented in-memory A2A session manager with tenant-scoped sessions and messages.
- Added A2A endpoints for session creation, message posting, and transcript retrieval.
- Added tests covering session lifecycle, tenant isolation, and rate limiting.

### Files Changed

| File | Change |
|------|--------|
| `backend/src/agentic_rag_backend/protocols/a2a.py` | New A2A session manager |
| `backend/src/agentic_rag_backend/api/routes/a2a.py` | New A2A endpoints |
| `backend/src/agentic_rag_backend/api/routes/__init__.py` | Export A2A router |
| `backend/src/agentic_rag_backend/main.py` | Register A2A routes |
| `backend/src/agentic_rag_backend/protocols/__init__.py` | Export A2A manager |
| `backend/tests/api/routes/test_a2a.py` | Added A2A endpoint tests |

### Tests

- `backend/tests/api/routes/test_a2a.py`

## Senior Developer Review

**Reviewer:** Code Review Agent
**Review Round:** 1

### Review Outcome: APPROVE

### Notes

- Session manager enforces tenant ownership before message insertion.
- Rate limiting mirrors existing API conventions.
- Tests cover happy path, tenant mismatch, and rate limiting errors.
