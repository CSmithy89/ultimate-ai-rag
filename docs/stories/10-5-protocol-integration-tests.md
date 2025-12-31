# Story 10.5: AG-UI and Protocol Integration Tests

Status: done

## Story

As a QA Engineer,  
I want integration tests for MCP, A2A, and AG-UI protocols,  
So that protocol compliance is validated end-to-end.

## Acceptance Criteria

1. Given MCP tool is registered, when tool is invoked, then correct response is returned.
2. Given A2A session is created, when messages are exchanged, then state is maintained.
3. Given AG-UI stream is started, when events are emitted, then frontend receives them.
4. Given rate limiting is configured, when limits are exceeded, then proper error with Retry-After is returned.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: Addressed - tenant-scoped protocol tests
- [x] Rate limiting / abuse protection: Addressed - rate limit test coverage
- [ ] Input validation / schema enforcement: N/A - integration tests
- [x] Tests (unit/integration): Addressed - protocol integration suite
- [x] Error handling + logging: Addressed - explicit rate limit error checks
- [ ] Documentation updates: N/A - internal test work

## Tasks / Subtasks

- [x] Create `backend/tests/integration/test_mcp_e2e.py` (AC: 1)
- [x] Create `backend/tests/integration/test_a2a_e2e.py` (AC: 2)
- [x] Create `backend/tests/integration/test_agui_streaming.py` (AC: 3)
- [x] Test rate limiting for protocol endpoints (AC: 4)

## Technical Notes

Protocol tests run against in-memory orchestrator/A2A components with integration fixtures.

## Definition of Done

- [x] Protocol integration tests added
- [x] Rate limiting behavior validated

## Dev Notes

- Added integration tests for MCP tool invocation and A2A session lifecycle.
- AG-UI bridge streaming test validates event sequence.
- Rate-limit Retry-After header validated via helper.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added MCP, A2A, and AG-UI integration test files.
- Added rate limit assertion for protocol endpoints.

### File List

- backend/tests/integration/test_mcp_e2e.py
- backend/tests/integration/test_a2a_e2e.py
- backend/tests/integration/test_agui_streaming.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- Protocol tests cover MCP, A2A, and AG-UI flows with clear assertions.
- Rate limit behavior verified via Retry-After header.
