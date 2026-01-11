# Story 2.4: Persistent Trajectory Logging

Status: done

## Story

As a developer,
I want every agent interaction to maintain a persistent thought trace,
so that I can debug and understand the agent's reasoning process.

## Acceptance Criteria

1. Given an agent is processing a query, when it makes decisions, calls tools, or generates responses, then each thought is logged using `agent.log_thought()`.
2. Each action is logged using `agent.log_action()`.
3. Each observation is logged using `agent.log_observation()`.
4. Trajectories are persisted to the database.
5. Trajectories survive container restarts (NFR8).

## Tasks / Subtasks

- [x] Implement persistent trajectory storage (AC: 4, 5)
  - [x] Add database schema for trajectories and events
  - [x] Add logger helper to write events
- [x] Wire logging into orchestrator execution (AC: 1-3)
  - [x] Emit thought logs for each plan step
  - [x] Emit action log for retrieval selection
  - [x] Emit observation log for response generation
- [x] Expose trajectory id to callers (AC: 4)
  - [x] Return trajectory id in API response

## Dev Notes

- Use Postgres via `DATABASE_URL` for persistence.
- Keep the event schema append-only to avoid breaking future analytics.

### Project Structure Notes

- Logging helpers should live in a dedicated module under `backend/src/agentic_rag_backend/`.
- API schema should expose trajectory identifiers for later UI and debugging tools.

### References

- Epic 2 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-2.4`
- Architecture overview: `_bmad-output/architecture.md#Cross-Cutting-Concerns-Identified`

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added Postgres-backed trajectory schema and logger helpers.
- Wired thought, action, and observation logging into the orchestrator.
- Exposed trajectory id in the query response for debugging.

### File List

- backend/src/agentic_rag_backend/trajectory.py
- backend/src/agentic_rag_backend/agents/orchestrator.py
- backend/src/agentic_rag_backend/main.py
- backend/src/agentic_rag_backend/schemas.py
- backend/pyproject.toml

## Senior Developer Review

Outcome: APPROVE

Notes:
- Schema creation and logging helpers are simple and append-only.
- Orchestrator hooks cover thought, action, and observation logging paths.
