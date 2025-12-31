# Epic 8 Tech Spec: Operations & Observability

**Version:** 1.0  
**Created:** 2025-12-31  
**Status:** Ready for Implementation

---

## Overview

Epic 8 delivers operational visibility into LLM usage, cost optimization via intelligent model routing, and a trajectory debugging interface for agent behavior inspection. It also upgrades trajectory storage to encrypt sensitive traces at rest.

### Business Value

- Ops engineers can track LLM spend in near real-time and spot cost spikes.
- Routing rules send low-complexity queries to cheaper models for measurable savings.
- Developers can inspect historical trajectories for debugging and audit trails.
- Sensitive reasoning traces are protected via AES-256 encryption at rest.

### Functional Requirements Covered

| FR | Description | Story |
|----|-------------|-------|
| FR23 | LLM cost monitoring | 8-1 |
| FR24 | Intelligent model routing | 8-2 |
| FR25 | Trajectory debugging | 8-3 |

### NFRs Addressed

| NFR | Requirement | Implementation |
|-----|-------------|----------------|
| NFR3 | Multi-tenant isolation | Tenant-filtered cost + trajectory queries |
| NFR4 | Encrypted traces | AES-256-GCM for trajectory events |

---

## Architecture Decisions

### 1. Cost Tracking via Postgres Usage Events

**Decision:** Persist LLM usage in a `llm_usage_events` table, recording prompt/completion token counts, model ID, computed costs, and routing metadata.

**Rationale:** Postgres already exists in the stack, enabling reliable aggregation for dashboards and alerts without introducing new infra.

### 2. Model Routing Heuristics + Config Overrides

**Decision:** Classify query complexity (simple/medium/complex) using deterministic heuristics, then map to model IDs via environment configuration.

**Rationale:** Keeps routing transparent, testable, and easy to tune without ML dependencies.

**Limitations:** Keyword-based heuristics can misclassify non-English or math-heavy prompts. Capture routing decisions
in usage events now; consider manual overrides and feedback loops in a future iteration.

### 3. Trajectory Debug APIs

**Decision:** Add ops endpoints to list trajectories and return event timelines with timestamps and metadata, filtered by tenant, agent type, and status.

**Rationale:** Enables debugging without exposing full database access and aligns with existing API patterns.

### 4. Application-Layer Trace Encryption

**Decision:** Encrypt trajectory event content using AES-256-GCM at the application layer before persistence.

**Rationale:** Works across Postgres deployments without requiring extensions or KMS integration, while keeping keys external via env vars.

---

## Component Changes

### New Modules

| Module | Purpose |
|--------|---------|
| `backend/src/agentic_rag_backend/ops/cost_tracker.py` | LLM usage tracking + aggregation |
| `backend/src/agentic_rag_backend/ops/model_router.py` | Query complexity scoring + model selection |
| `backend/src/agentic_rag_backend/ops/trace_crypto.py` | AES-256-GCM encryption helpers |
| `backend/src/agentic_rag_backend/api/routes/ops.py` | Ops endpoints (costs, routing, trajectories) |
| `frontend/app/ops/page.tsx` | Operations dashboard UI |
| `frontend/hooks/use-ops-dashboard.ts` | Cost + trajectory queries (TanStack Query) |

### Modified Modules

| Module | Change |
|--------|--------|
| `backend/src/agentic_rag_backend/agents/orchestrator.py` | Route model, record usage, log routing |
| `backend/src/agentic_rag_backend/trajectory.py` | Encrypt/decrypt trace content |
| `backend/src/agentic_rag_backend/config.py` | Add ops settings + encryption key |
| `backend/src/agentic_rag_backend/main.py` | Register ops router + tracker instances |
| `.env.example` | Document ops settings |

---

## API Contracts

### Cost Monitoring

- `GET /api/v1/ops/costs/summary?tenant_id=...&window=day|week|month`
- `GET /api/v1/ops/costs/events?tenant_id=...&limit=...`
- `POST /api/v1/ops/costs/alerts` (configure thresholds)

### Model Routing

- `GET /api/v1/ops/routing/config`
- `POST /api/v1/ops/routing/config` (update config overrides)

### Trajectories

- `GET /api/v1/ops/trajectories?tenant_id=...&status=error|ok&agent_type=...`
- `GET /api/v1/ops/trajectories/{trajectory_id}?tenant_id=...`

---

## Story Breakdown

1. **8-1 LLM Cost Monitoring**
   - Persist usage events with pricing, model, and tenant metadata
   - Build cost dashboard + alert threshold configuration

2. **8-2 Intelligent Model Routing**
   - Implement complexity scoring and model mapping
   - Log routing decisions in trajectories and usage events

3. **8-3 Trajectory Debugging Interface**
   - Add list/detail endpoints for trajectories with filters
   - UI for timeline inspection, tool call results, timing info

4. **8-4 Encrypted Trace Storage**
   - Encrypt trajectory event content with AES-256-GCM
   - Support secure key management via env config

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Token estimation differs from provider usage | Medium | Use tiktoken, document estimates, warn on unknown model IDs, allow pricing overrides |
| Routing heuristic misclassifies queries | Medium | Configurable thresholds + logging for audit + document limitations |
| Encryption key loss | High | Require env key, warn on dev auto-generation, document rotation strategy |
| Increased DB load from cost aggregation | Low | Indexes on tenant_id/created_at + windowed queries |

## Deployment Notes

- For large Postgres tables, create new indexes concurrently (or during a maintenance window) to avoid long write locks.
  Applicable to trajectory and cost tables introduced in this epic.

---

## Testing Strategy

- Unit tests for complexity scoring and pricing calculations
- API tests for cost summary + trajectory retrieval
- Encryption round-trip tests for trajectory events

---

## Out of Scope

- Automated anomaly detection on cost spikes
- Full BI dashboards or external observability export
- KMS-backed key rotation (future epic)
