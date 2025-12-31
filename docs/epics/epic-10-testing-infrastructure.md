# Epic 10: Testing Infrastructure

**Status:** Ready for Development
**Priority:** Critical
**Estimated Effort:** 1-2 Sprints
**Created:** 2025-12-31
**Source:** Tech Debt Audit (Epics 1-8)

---

## Overview

This epic addresses the critical testing debt accumulated across Epics 3-8. Integration tests have been "deferred" in 6 consecutive epics, creating significant risk of hidden bugs in production. This epic establishes proper end-to-end testing infrastructure.

### Business Value

- Catch integration bugs before production
- Validate NFR2 (50-page PDF < 5 min) and NFR3 (<2s query latency)
- Enable confident refactoring
- Reduce production incident risk

### Success Criteria

1. Integration test coverage >= 80% for critical paths
2. All skipped tests fixed or documented
3. E2E tests running in CI pipeline
4. NFR benchmarks validated and passing

---

## Stories

### Story 10.1: Integration Test Framework Setup

**As a** Developer,
**I want** an integration test framework with real database connections,
**So that** I can write end-to-end tests that verify actual behavior.

**Acceptance Criteria:**
1. Given the test framework exists, when integration tests run, then they connect to real PostgreSQL, Neo4j, and Redis
2. Given test isolation is needed, when tests run, then each test uses isolated tenant_id
3. Given tests complete, when cleanup runs, then test data is removed
4. Given CI pipeline runs, when integration tests execute, then they complete in < 5 minutes

**Tasks:**
- [ ] Create `backend/tests/integration/` directory structure
- [ ] Add pytest fixtures for real database connections (using Docker Compose test services)
- [ ] Create tenant isolation fixtures for test data cleanup
- [ ] Configure CI to run integration tests separately from unit tests
- [ ] Add integration test documentation

**Story Points:** 5

---

### Story 10.2: Hybrid Retrieval Integration Tests

**As a** QA Engineer,
**I want** integration tests for the hybrid retrieval pipeline,
**So that** vector + graph search is validated end-to-end.

**Acceptance Criteria:**
1. Given documents are indexed, when hybrid search runs, then both vector and graph results are returned
2. Given entity relationships exist, when graph traversal runs, then correct paths are found
3. Given evidence is requested, when retrieval completes, then citations include source chunks and graph nodes
4. Given multi-tenant data exists, when search runs, then only tenant-scoped results are returned

**Tasks:**
- [ ] Create `backend/tests/integration/test_hybrid_retrieval.py`
- [ ] Add fixtures for indexing test documents with known entities
- [ ] Test vector similarity search with real embeddings
- [ ] Test graph traversal with real Neo4j queries
- [ ] Test combined evidence payload structure
- [ ] Test tenant isolation in retrieval

**Story Points:** 5

---

### Story 10.3: Ingestion Pipeline Integration Tests

**As a** QA Engineer,
**I want** integration tests for the document ingestion pipeline,
**So that** URL crawling, PDF parsing, and entity extraction are validated end-to-end.

**Acceptance Criteria:**
1. Given a URL is submitted, when ingestion completes, then document is stored in PostgreSQL
2. Given a PDF is uploaded, when parsing completes, then chunks are created with embeddings
3. Given entities are extracted, when graph building completes, then nodes exist in Neo4j
4. Given duplicate content is submitted, when deduplication runs, then no duplicates are created

**Tasks:**
- [ ] Create `backend/tests/integration/test_ingestion_pipeline.py`
- [ ] Create PDF test fixtures (sample_simple.pdf, sample_tables.pdf, sample_complex.pdf)
- [ ] Test URL ingestion with mock HTTP server
- [ ] Test PDF parsing and chunking
- [ ] Test entity extraction and graph building
- [ ] Test content deduplication (SHA-256 hash)
- [ ] Validate NFR2: 50-page PDF < 5 min

**Story Points:** 8

---

### Story 10.4: Graphiti Integration Tests

**As a** QA Engineer,
**I want** integration tests for Graphiti temporal knowledge graph,
**So that** episode ingestion and temporal queries are validated.

**Acceptance Criteria:**
1. Given a document is ingested, when episode is created, then entities are extracted with custom types
2. Given temporal data exists, when point-in-time query runs, then correct historical state is returned
3. Given knowledge has changed, when change query runs, then modifications are detected
4. Given Graphiti client connects, when operations run, then coverage >= 85% for db/graphiti.py

**Tasks:**
- [ ] Create `backend/tests/integration/test_graphiti_integration.py`
- [ ] Test episode creation with real Graphiti client
- [ ] Test custom entity type classification (TechnicalConcept, CodePattern, etc.)
- [ ] Test temporal query filtering (valid_at, created_at)
- [ ] Test entity deduplication via Graphiti
- [ ] Improve db/graphiti.py coverage to >= 85%

**Story Points:** 5

---

### Story 10.5: AG-UI and Protocol Integration Tests

**As a** QA Engineer,
**I want** integration tests for MCP, A2A, and AG-UI protocols,
**So that** protocol compliance is validated end-to-end.

**Acceptance Criteria:**
1. Given MCP tool is registered, when tool is invoked, then correct response is returned
2. Given A2A session is created, when messages are exchanged, then state is maintained
3. Given AG-UI stream is started, when events are emitted, then frontend receives them
4. Given rate limiting is configured, when limits are exceeded, then proper error with Retry-After is returned

**Tasks:**
- [ ] Create `backend/tests/integration/test_mcp_e2e.py`
- [ ] Create `backend/tests/integration/test_a2a_e2e.py`
- [ ] Create `backend/tests/integration/test_agui_streaming.py`
- [ ] Test full MCP tool discovery and invocation flow
- [ ] Test A2A session lifecycle (create, message, close)
- [ ] Test AG-UI event streaming with mock frontend
- [ ] Test rate limiting across all protocol endpoints

**Story Points:** 8

---

### Story 10.6: Skipped Test Resolution

**As a** QA Engineer,
**I want** all skipped tests fixed or documented,
**So that** test coverage accurately reflects code quality.

**Acceptance Criteria:**
1. Given a test is skipped, when reviewed, then it either passes or has documented reason
2. Given skip decorator is used, when reason is checked, then descriptive message exists
3. Given all skips are resolved, when test suite runs, then skipped count is documented

**Tasks:**
- [ ] Audit all `@pytest.mark.skip` decorators in test suite
- [ ] Fix tests that can be fixed (3 from Epic 5, others from earlier)
- [ ] Add reason strings to all remaining skips
- [ ] Document known limitations in test README
- [ ] Create follow-up issues for tests requiring infrastructure changes

**Story Points:** 3

---

### Story 10.7: NFR Benchmark Validation

**As a** QA Engineer,
**I want** automated benchmarks for NFR requirements,
**So that** performance requirements are continuously validated.

**Acceptance Criteria:**
1. Given NFR2 (ingestion speed), when 50-page PDF is processed, then completion is < 5 min
2. Given NFR3 (query latency), when query is executed, then response is < 2s
3. Given NFR5 (scalability), when load test runs, then system handles target load
4. Given benchmarks run, when results are recorded, then trends are visible

**Tasks:**
- [ ] Create `backend/tests/benchmarks/` directory
- [ ] Add ingestion speed benchmark (NFR2)
- [ ] Add query latency benchmark (NFR3)
- [ ] Configure benchmark reporting in CI
- [ ] Set up baseline metrics and alerting thresholds

**Story Points:** 5

---

## Dependencies

- Epic 9 (Story 9.3 pre-review checklist) recommended first
- Docker Compose test services must be available

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Test environment flakiness | High | Use isolated containers, proper cleanup |
| Long test execution time | Medium | Parallelize tests, use test marks for selective runs |
| Neo4j/Graphiti connection issues | Medium | Connection pooling, retry logic |

---

## Definition of Done

- [ ] All 7 stories completed and reviewed
- [ ] Integration tests passing in CI
- [ ] Coverage targets met (80%+ for critical paths, 85%+ for db/graphiti.py)
- [ ] All skipped tests resolved or documented
- [ ] NFR benchmarks automated and passing
- [ ] Test documentation updated

---

## References

- Tech Debt Audit: `_bmad-output/implementation-artifacts/tech-debt-audit-2025-12-31.md`
- Testing debt items: E3-03, E4-01, E4-07, E5-05, E5-07, E6-E2E, E7-04, E8-04
