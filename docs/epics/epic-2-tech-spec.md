# Epic 2 Tech Spec: Agentic Query & Reasoning

Date: 2025-12-28

## Overview
Epic 2 introduces an orchestrator agent that accepts user queries, plans multi-step
execution, selects the most appropriate retrieval method, and persists a full
trajectory log of decisions and actions. This epic lays the reasoning foundation
for later retrieval and UI work while meeting response-time and concurrency targets.

## Goals
- Provide an orchestrator agent accessible through the backend API.
- Generate and expose a multi-step execution plan for complex queries.
- Select retrieval mode (vector, graph, hybrid) and log the decision.
- Persist a trajectory log (thoughts, actions, observations) in the database.
- Meet NFR1 response target (<10s) for typical queries.
- Support NFR6 concurrency target (50+ concurrent agent runs).

## Non-goals
- Implement vector or graph retrieval logic (Epic 3).
- Implement ingestion pipeline or indexing (Epic 4).
- Build UI surfaces for plans or traces (Epic 5).
- Production-grade observability dashboards (Epic 7).

## Architecture Summary
### Backend API
- Add a `POST /query` endpoint in `backend/src/agentic_rag_backend/main.py`.
- Request payload includes query text and optional metadata (tenant, user, session).
- Response includes final answer plus a summarized execution plan.

### Orchestrator Agent
- New module: `backend/agents/orchestrator.py` (or `backend/src/agentic_rag_backend/agents/`).
- Uses Agno agent patterns to:
  - Parse the query and generate a step plan.
  - Execute steps in order with adaptive branching.
  - Emit trajectory logs for each thought, action, observation.

### Retrieval Router (Decision Only)
- New module: `backend/src/agentic_rag_backend/retrieval_router.py`.
- Classifies query intent into semantic, relational, or hybrid modes.
- Returns a retrieval strategy enum and logs the decision.
- Actual retrieval calls are stubbed or delegated (Epic 3 will replace stubs).

### Trajectory Logging
- New module: `backend/src/agentic_rag_backend/trajectory.py`.
- Implements `log_thought`, `log_action`, `log_observation` helpers.
- Persists to PostgreSQL using `DATABASE_URL`.
- Tables include:
  - `trajectories` (id, session_id, created_at)
  - `trajectory_events` (id, trajectory_id, type, content, created_at)
- Logs are append-only and survive container restarts (NFR8).

## Key Decisions
- Use Agno agent patterns to align with Epic 1 scaffolding.
- Keep retrieval selection logic in a dedicated router module for testability.
- Persist trajectories in Postgres to avoid additional infra dependencies.
- Return plan metadata from API to enable later UI integration.

## Configuration
Reuses Epic 1 environment variables:
- `DATABASE_URL` for trajectory persistence.
- Optional `AGENT_MAX_STEPS` to cap plan length (default 8).

## Story Breakdown
### Story 2.1: Orchestrator Agent Foundation
- Build orchestrator agent module and wire it into the API.
- Ensure response returned within NFR1 target.
- Use Agno patterns for execution flow.

### Story 2.2: Multi-Step Query Planning
- Implement plan generation and step logging.
- Return plan in API response (summary or structured list).
- Support adaptive steps when new info is uncovered.

### Story 2.3: Dynamic Retrieval Method Selection
- Implement query classification to select vector/graph/hybrid.
- Log selection decision into trajectory.
- Provide stub handlers for retrieval calls (Epic 3 to fill in).

### Story 2.4: Persistent Trajectory Logging
- Add database schema and logging helpers.
- Log thoughts, actions, observations throughout execution.
- Ensure logs persist across restarts.

## Risks and Mitigations
- **Plan inflation**: cap steps via `AGENT_MAX_STEPS`.
- **Performance regression**: keep routing lightweight; avoid heavy retrieval in Epic 2.
- **Schema drift**: keep trajectory schema minimal and append-only.

## Validation
- Unit tests for retrieval router classification.
- API smoke test for `POST /query` with plan output.
- DB persistence check ensures trajectory events saved and retrievable.
