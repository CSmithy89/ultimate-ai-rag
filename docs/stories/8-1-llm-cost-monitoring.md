# Story 8.1: LLM Cost Monitoring

Status: done

## Story

As an ops engineer,
I want to monitor real-time LLM interaction costs,
so that I can track spending and identify optimization opportunities.

## Acceptance Criteria

1. Given the system is processing LLM requests, when an ops engineer views the cost dashboard, then they see real-time token usage per request.
2. Given costs are calculated, when a request completes, then costs are derived from model pricing (input + output tokens).
3. Given usage is stored, when a tenant is selected, then usage is aggregated by tenant (NFR3).
4. Given historical data exists, when the ops dashboard loads, then historical cost trends are displayed.
5. Given alerts are configured, when spending crosses thresholds, then alerts are surfaced in the dashboard.

## Tasks / Subtasks

- [ ] Persist LLM usage events (AC: 1, 2, 3)
  - [ ] Add `llm_usage_events` table in Postgres with tenant_id/model/tokens/costs
  - [ ] Record usage events on orchestrator completion (per request)
  - [ ] Store pricing metadata and routing context in event payload

- [ ] Build cost aggregation endpoints (AC: 1, 3, 4, 5)
  - [ ] Add ops router endpoints for summary + event list
  - [ ] Aggregate usage by tenant and time window (day/week/month)
  - [ ] Add alert threshold endpoints + evaluation logic

- [ ] Build Ops dashboard UI (AC: 1, 4, 5)
  - [ ] Add `frontend/app/ops/page.tsx` dashboard view
  - [ ] Add trend chart for daily costs + per-model breakdown
  - [ ] Add alert threshold configuration + alert status list

## Dev Notes

- Use Postgres for cost storage and aggregation (no new infra).
- Token estimates should use tiktoken; costs derived from per-model pricing map.
- Tenant filtering is mandatory on all cost queries.
- Keep pricing configurable via env for rapid tuning.

### Project Structure Notes

- Backend ops endpoints should live under `backend/src/agentic_rag_backend/api/routes/ops.py`.
- Cost tracking helpers should be isolated under `backend/src/agentic_rag_backend/ops/`.
- Frontend data fetching should use TanStack Query, consistent with `frontend/hooks`.

### References

- Epic 8 Tech Spec: `docs/epics/epic-8-tech-spec.md`
- Epic Definition: `_bmad-output/project-planning-artifacts/epics.md#epic-8-operations--observability`
- Trajectory logging patterns: `backend/src/agentic_rag_backend/trajectory.py`

## Dev Agent Record

### Agent Model Used
GPT-5 (Codex CLI)

### Debug Log References
None.

### Completion Notes List
1. Added cost tracking tables, pricing logic, and alert thresholds in backend ops module.
2. Wired cost tracking into orchestrator runs and exposed ops endpoints.
3. Built ops dashboard UI with summary, trend, and alerts configuration.

### File List
- `backend/src/agentic_rag_backend/ops/cost_tracker.py`
- `backend/src/agentic_rag_backend/ops/__init__.py`
- `backend/src/agentic_rag_backend/db/postgres.py`
- `backend/src/agentic_rag_backend/config.py`
- `backend/src/agentic_rag_backend/main.py`
- `backend/src/agentic_rag_backend/agents/orchestrator.py`
- `backend/src/agentic_rag_backend/api/routes/ops.py`
- `backend/src/agentic_rag_backend/api/routes/__init__.py`
- `.env.example`
- `frontend/types/ops.ts`
- `frontend/lib/api.ts`
- `frontend/hooks/use-ops-dashboard.ts`
- `frontend/app/ops/page.tsx`

## Senior Developer Review

Outcome: APPROVE

Notes:
- Cost tracking is isolated, resilient, and does not block query execution.
- Ops endpoints enforce tenant filtering and return structured summaries.
- Dashboard surfaces summary, trend, and alert configuration cleanly.
