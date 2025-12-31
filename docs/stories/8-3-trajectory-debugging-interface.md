# Story 8.3: Trajectory Debugging Interface

Status: done

## Story

As a developer,
I want to review the reasoning trajectory of past queries,
so that I can debug agent behavior and identify issues.

## Acceptance Criteria

1. Given an agent has processed queries with trajectory logging, when a developer opens the trajectory viewer, then they see a list of past agent sessions.
2. Given the trajectory list, when a developer selects a session, then they can drill into individual trajectories.
3. Given a trajectory is selected, when the timeline renders, then they see thoughts, actions, and observations in sequence.
4. Given a trajectory event is listed, when the developer expands it, then they can view tool calls and their results.
5. Given events include timestamps, when the timeline renders, then timing information for each step is visible.
6. Given the trajectory list, when filters are applied, then the list can be filtered by error status or agent type.

## Tasks / Subtasks

- [ ] Add trajectory listing endpoints (AC: 1, 6)
  - [ ] Add `/api/v1/ops/trajectories` with status + agent filters
  - [ ] Include session metadata, event counts, and last event timestamp

- [ ] Add trajectory detail endpoint (AC: 2, 3, 5)
  - [ ] Add `/api/v1/ops/trajectories/{id}` returning events in sequence
  - [ ] Include duration and timestamps per event

- [ ] Build trajectory viewer UI (AC: 1-6)
  - [ ] Add `frontend/app/ops/trajectories/page.tsx`
  - [ ] List trajectories with status labels
  - [ ] Render event timeline with content + timestamps
  - [ ] Add filters for status and agent type

## Dev Notes

- Trajectories are stored in Postgres tables `trajectories` and `trajectory_events`.
- Add `agent_type` to trajectories for filtering and set to `orchestrator`.
- Error status can be inferred from event content containing "error".

### Project Structure Notes

- Ops endpoints live under `backend/src/agentic_rag_backend/api/routes/ops.py`.
- Use the existing ops hooks in `frontend/hooks/use-ops-dashboard.ts`.

### References

- Epic 8 Tech Spec: `docs/epics/epic-8-tech-spec.md`
- Trajectory logger: `backend/src/agentic_rag_backend/trajectory.py`
- Ops API: `backend/src/agentic_rag_backend/api/routes/ops.py`

## Dev Agent Record

### Agent Model Used
GPT-5 (Codex CLI)

### Debug Log References
None.

### Completion Notes List
1. Added ops endpoints to list trajectories and fetch event timelines.
2. Added agent_type to trajectories and logged orchestrator runs accordingly.
3. Built trajectory viewer UI with filters and timeline rendering.

### File List
- `backend/src/agentic_rag_backend/api/routes/ops.py`
- `backend/src/agentic_rag_backend/trajectory.py`
- `backend/src/agentic_rag_backend/agents/orchestrator.py`
- `backend/alembic/versions/20251231_000002_add_trajectory_agent_type.py`
- `backend/tests/test_orchestrator.py`
- `frontend/lib/api.ts`
- `frontend/hooks/use-ops-dashboard.ts`
- `frontend/types/ops.ts`
- `frontend/app/ops/trajectories/page.tsx`

## Senior Developer Review

Outcome: APPROVE

Notes:
- Ops endpoints provide list/detail views with filtering and timing metadata.
- Agent type support is persisted via migration and logged by orchestrator.
- UI delivers a usable timeline and filter controls without blocking the ops dashboard.
