# Epic 10 Tech Spec: Testing Infrastructure

**Version:** 1.0  
**Created:** 2025-12-31  
**Status:** Ready for Implementation

---

## Overview

Epic 10 establishes integration and benchmark testing infrastructure across the stack.
It adds real service-backed integration tests, protocol E2E coverage, and NFR benchmarks
for ingestion and query latency. It also resolves skipped tests and enforces clearer
documentation of known limitations.

### Business Value

- Catch integration regressions before production.
- Validate NFR2 (50-page PDF < 5 min) and NFR3 (<2s query latency).
- Reduce production incidents from untested system boundaries.
- Build confidence for refactors and migrations.

### Functional Requirements Covered

| FR | Description | Story |
|----|-------------|-------|
| FR40 | Integration test framework | 10-1 |
| FR41 | Hybrid retrieval integration tests | 10-2 |
| FR42 | Ingestion pipeline integration tests | 10-3 |
| FR43 | Graphiti integration tests | 10-4 |
| FR44 | Protocol integration tests | 10-5 |
| FR45 | Skipped test resolution | 10-6 |
| FR46 | NFR benchmark validation | 10-7 |

### NFRs Addressed

| NFR | Requirement | Implementation |
|-----|-------------|----------------|
| NFR2 | 50-page PDF ingestion < 5 min | Benchmark suite |
| NFR3 | Query latency < 2s | Benchmark suite |
| NFR5 | Scalability (target load) | Benchmark suite |

---

## Architecture Decisions

### 1. Integration Tests Use Real Services via Docker Compose

**Decision:** Run integration tests against real Postgres, Neo4j, and Redis services
provisioned by Docker Compose in CI and local dev.

**Rationale:** Ensures coverage of real persistence and query behavior that unit tests miss.

### 2. Tenant-Isolated Fixtures

**Decision:** Integration tests must use unique tenant_id values with cleanup routines
to avoid data collisions and ensure repeatability.

**Rationale:** Maintains data isolation, mirrors production multi-tenancy, and reduces flakiness.

### 3. Benchmarks in Dedicated Test Suite

**Decision:** Add `backend/tests/benchmarks/` with explicit benchmark targets and
CI reporting for performance thresholds.

**Rationale:** Keeps performance testing separate from functional tests while enabling
automated regression detection.

---

## Component Changes

### New/Updated Tests

| Path | Purpose |
|------|---------|
| `backend/tests/integration/` | Real service integration tests |
| `backend/tests/benchmarks/` | NFR performance benchmarks |
| `backend/tests/integration/test_hybrid_retrieval.py` | Hybrid retrieval E2E |
| `backend/tests/integration/test_ingestion_pipeline.py` | Ingestion E2E |
| `backend/tests/integration/test_graphiti_e2e.py` | Graphiti temporal E2E |
| `backend/tests/integration/test_mcp_e2e.py` | MCP protocol E2E |
| `backend/tests/integration/test_a2a_e2e.py` | A2A protocol E2E |
| `backend/tests/integration/test_agui_streaming.py` | AG-UI streaming E2E |

### CI Changes

| Path | Change |
|------|--------|
| `.github/workflows/ci-backend.yml` | Add integration/benchmark stages |

---

## Story Breakdown

1. **10-1 Integration Test Framework Setup**
   - Add integration fixtures for real services + cleanup.

2. **10-2 Hybrid Retrieval Integration Tests**
   - Validate vector + graph retrieval and evidence payloads.

3. **10-3 Ingestion Pipeline Integration Tests**
   - Validate URL, PDF, entity extraction, and deduplication.

4. **10-4 Graphiti Integration Tests**
   - Validate episode ingestion and temporal queries.

5. **10-5 Protocol Integration Tests**
   - Validate MCP, A2A, and AG-UI flows end-to-end.

6. **10-6 Skipped Test Resolution**
   - Fix or document all skipped tests with reasons.

7. **10-7 NFR Benchmark Validation**
   - Implement ingestion and query performance benchmarks.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Flaky integration tests | High | Isolated tenants, deterministic fixtures |
| Long CI runtime | Medium | Separate integration/benchmark jobs |
| Service startup time | Medium | Prewarm containers, cache layers |

---

## Deployment Notes

No production changes. CI requires Docker services for integration tests.

---

## Testing Strategy

- Unit tests remain unchanged.
- Integration tests run against real services with cleanup fixtures.
- Benchmark tests run on a schedule or dedicated CI stage.

---

## Out of Scope

- Full load testing infrastructure with external tooling
- Distributed tracing performance dashboards
