# Story 2.1: Orchestrator Agent Foundation

Status: done

## Story

As a user,
I want an orchestrator agent that can receive and process my queries,
so that I get intelligent responses from the system.

## Acceptance Criteria

1. Given the backend is running, when a user submits a query via the API, then the orchestrator agent receives the query.
2. The agent returns a response within the NFR1 target (<10s).
3. The agent uses Agno's built-in patterns for execution.

## Tasks / Subtasks

- [x] Add query API contract and endpoint wiring (AC: 1)
  - [x] Define request/response schema
  - [x] Add `POST /query` route to FastAPI
- [x] Implement orchestrator agent wrapper (AC: 1, 3)
  - [x] Provide `run_query` entrypoint
  - [x] Use Agno agent interface when available
- [x] Return response within performance budget (AC: 2)
  - [x] Keep orchestration logic lightweight

## Dev Notes

- Use the backend structure created in Epic 1 (`backend/src/agentic_rag_backend`).
- Keep orchestrator logic isolated in a dedicated module for reuse in later stories.
- Avoid retrieval execution in this story; focus on orchestration skeleton.

### Project Structure Notes

- Backend path: `backend/src/agentic_rag_backend/`.
- New agent modules should live alongside `main.py` or in `backend/agents/`.

### References

- Epic 2 definition: `_bmad-output/project-planning-artifacts/epics.md#Epic-2`
- Architecture overview: `_bmad-output/architecture.md#Core-Architectural-Decisions`

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added orchestrator agent wrapper with optional Agno execution.
- Introduced request/response schema for the query endpoint.
- Wired `POST /query` into FastAPI to invoke the orchestrator.

### File List

- backend/src/agentic_rag_backend/main.py
- backend/src/agentic_rag_backend/orchestrator.py
- backend/src/agentic_rag_backend/schemas.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- Clean, minimal API wiring with a clear orchestrator boundary.
- Optional Agno dependency handled safely for local dev.
