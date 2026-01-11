# Story 2.2: Multi-Step Query Planning

Status: done

## Story

As a user,
I want the agent to autonomously plan multi-step strategies for complex queries,
so that sophisticated questions are broken down and answered systematically.

## Acceptance Criteria

1. Given a user submits a complex query requiring multiple steps, when the orchestrator agent processes the query, then it generates a visible execution plan.
2. Each step is logged as a "thought".
3. Steps execute in logical sequence.
4. The plan adapts if intermediate steps reveal new information.

## Tasks / Subtasks

- [x] Implement plan generation for complex queries (AC: 1)
  - [x] Derive steps from query characteristics
- [x] Record plan steps as thoughts (AC: 2)
  - [x] Emit a thought entry per step
- [x] Execute steps sequentially (AC: 3)
  - [x] Mark steps completed in order
- [x] Allow plan refinement based on signals (AC: 4)
  - [x] Add adaptation rules for conditional queries

## Dev Notes

- Keep plan output lightweight and deterministic to stay under latency target.
- Store plan and thought data in the orchestrator output; persistence will arrive in Story 2.4.

### Project Structure Notes

- Orchestrator logic lives in `backend/src/agentic_rag_backend/agents/orchestrator.py`.
- API schemas live in `backend/src/agentic_rag_backend/schemas.py`.

### References

- Epic 2 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-2.2`
- Architecture overview: `_bmad-output/architecture.md#Cross-Cutting-Concerns-Identified`

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added plan generation with adaptive steps based on query signals.
- Logged each plan step as a thought and marked completion in sequence.
- Extended API response to include plan and thought list.

### File List

- backend/src/agentic_rag_backend/agents/orchestrator.py
- backend/src/agentic_rag_backend/schemas.py
- backend/src/agentic_rag_backend/main.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- Planning flow is deterministic and keeps execution lightweight.
- Response schema makes plan visibility explicit for later UI integration.
