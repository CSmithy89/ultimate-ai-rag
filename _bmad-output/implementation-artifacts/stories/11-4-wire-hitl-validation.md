# Story 11.4: Wire HITL validation endpoint

Status: done

## Story

As a developer,  
I want the HITL validation endpoint connected to the real AG-UI bridge,  
So that human-in-the-loop source validation works in production.

## Acceptance Criteria

1. Given HITL validation is triggered, when user reviews sources, then checkpoints are persisted.
2. Given validation completes, when result is submitted, then AG-UI bridge receives the decision.
3. Given timeout occurs, when fallback runs, then default behavior is applied.
4. Given validation history exists, when queried, then checkpoints are retrievable.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: Addressed - tenant-scoped checkpoints + header checks
- [ ] Rate limiting / abuse protection: N/A - existing copilot rate limiting
- [x] Input validation / schema enforcement: Addressed - UUID validation and tenant checks
- [x] Tests (unit/integration): Addressed - tests added and run
- [x] Error handling + logging: Addressed - timeout fallback + persistence logging
- [x] Documentation updates: Addressed - persistence runbook updated

## Tasks / Subtasks

- [x] Review current HITL validation stub implementation
- [x] Wire to real AG-UI bridge event flow
- [x] Add checkpoint persistence (PostgreSQL or Redis)
- [x] Implement timeout handling and fallback
- [x] Add validation result endpoints
- [x] Write integration tests for HITL flow

## Technical Notes

Prefer Redis-based persistence to avoid blocking the AG-UI stream. Ensure checkpoints are tenant-scoped and expire after a retention window.

## Definition of Done

- [x] HITL checkpoints persisted and retrievable
- [x] AG-UI stream emits validation events and honors responses
- [x] Timeout fallback applied
- [x] Tests run and documented

## Dev Notes

Connected AG-UI streaming to HITL checkpoints with Redis-backed persistence, added checkpoint query endpoints, and updated AG-UI bridge to emit validation events and wait for responses with timeout fallback. Added protocol and API tests to cover HITL flow and executed the suite locally.

## Dev Agent Record

### Agent Model Used

gpt-4o

### Debug Log References

### Completion Notes List

- Added Redis-backed checkpoint persistence and list/query helpers.
- Wired AG-UI bridge to emit HITL events and honor validation responses.
- Added HITL checkpoint query endpoints and updated tests.
### File List

- backend/src/agentic_rag_backend/protocols/ag_ui_bridge.py
- backend/src/agentic_rag_backend/api/routes/copilot.py
- backend/src/agentic_rag_backend/api/routes/ag_ui.py
- backend/src/agentic_rag_backend/main.py
- backend/tests/protocols/test_hitl_manager.py
- backend/tests/protocols/test_ag_ui_bridge.py
- backend/tests/api/routes/test_copilot.py
- _bmad-output/implementation-artifacts/stories/11-4-wire-hitl-validation.md
- _bmad-output/implementation-artifacts/stories/11-4-wire-hitl-validation.context.xml
- _bmad-output/implementation-artifacts/sprint-status.yaml
## Senior Developer Review

Outcome: APPROVE

Notes:
- HITL flow now persists checkpoints and emits AG-UI validation events.
- Validation history endpoints added; tests added and run.
