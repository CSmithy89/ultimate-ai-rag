# Story 8.2: Intelligent Model Routing

Status: done

## Story

As an ops engineer,
I want the system to route queries to different LLM models based on complexity,
so that simple queries use cheaper models while complex ones get premium models.

## Acceptance Criteria

1. Given a query is submitted to the system, when the routing logic evaluates the query, then it classifies query complexity (simple, medium, complex).
2. Given a query is classified as simple, when routing occurs, then it uses a cost-effective model (e.g., GPT-4o-mini).
3. Given a query is classified as complex, when routing occurs, then it uses a premium model (e.g., GPT-4o, Claude).
4. Given routing configuration exists, when ops changes settings, then routing decisions follow configurable thresholds and model mappings.
5. Given routing occurs, when a request completes, then the routing decision is logged in the trajectory.
6. Given cost tracking is enabled, when routing uses cheaper models, then cost savings are tracked and reported.

## Tasks / Subtasks

- [ ] Implement model routing engine (AC: 1-4)
  - [ ] Add `backend/src/agentic_rag_backend/ops/model_router.py`
  - [ ] Define complexity scoring + thresholds with env-configurable settings
  - [ ] Map complexity -> model id via config
  - [ ] Return routing decision metadata (complexity, score, reason)

- [ ] Integrate routing into orchestrator (AC: 2, 3, 5)
  - [ ] Select model per request based on router decision
  - [ ] Log routing decision in trajectory events
  - [ ] Cache model agents when available

- [ ] Track cost savings (AC: 6)
  - [ ] Record baseline cost (premium model) vs selected model
  - [ ] Store savings in usage events and summarize in ops endpoints
  - [ ] Surface total savings in ops dashboard

## Dev Notes

- Route model selection must be deterministic and explainable.
- Default premium baseline should be the complex model (gpt-4o).
- Ensure tenant_id filtering remains intact on cost reports.
- Avoid rebuilding the Agno agent per request; cache agents by model_id when possible.

### Project Structure Notes

- Ops routing logic should live under `backend/src/agentic_rag_backend/ops/`.
- Model routing settings belong in `backend/src/agentic_rag_backend/config.py`.
- Ops dashboard should highlight savings without overwhelming existing summary UI.

### References

- Epic 8 Tech Spec: `docs/epics/epic-8-tech-spec.md`
- Epic Definition: `_bmad-output/project-planning-artifacts/epics.md#story-82-intelligent-model-routing`
- Cost tracking: `backend/src/agentic_rag_backend/ops/cost_tracker.py`

## Dev Agent Record

### Agent Model Used
GPT-5 (Codex CLI)

### Debug Log References
None.

### Completion Notes List
1. Added deterministic model router with configurable thresholds and model mappings.
2. Routed orchestrator requests per complexity and logged routing decisions in trajectories.
3. Tracked baseline costs and savings in usage events and surfaced savings in ops summary.

### File List
- `backend/src/agentic_rag_backend/ops/model_router.py`
- `backend/src/agentic_rag_backend/ops/__init__.py`
- `backend/src/agentic_rag_backend/config.py`
- `backend/src/agentic_rag_backend/main.py`
- `backend/src/agentic_rag_backend/db/postgres.py`
- `backend/src/agentic_rag_backend/ops/cost_tracker.py`
- `backend/src/agentic_rag_backend/agents/orchestrator.py`
- `backend/src/agentic_rag_backend/api/routes/ops.py`
- `.env.example`
- `frontend/types/ops.ts`
- `frontend/app/ops/page.tsx`

## Senior Developer Review

Outcome: APPROVE

Notes:
- Routing logic is deterministic, configurable, and logged per request.
- Orchestrator caches agents per model and avoids per-request instantiation cost.
- Savings are persisted and surfaced in ops summary.
