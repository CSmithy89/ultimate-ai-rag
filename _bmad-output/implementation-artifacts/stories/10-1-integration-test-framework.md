# Story 10.1: Integration Test Framework Setup

Status: done

## Story

As a developer,  
I want an integration test framework with real database connections,  
So that I can write end-to-end tests that verify actual behavior.

## Acceptance Criteria

1. Given the test framework exists, when integration tests run, then they connect to real PostgreSQL, Neo4j, and Redis.
2. Given test isolation is needed, when tests run, then each test uses isolated tenant_id.
3. Given tests complete, when cleanup runs, then test data is removed.
4. Given CI pipeline runs, when integration tests execute, then they complete in < 5 minutes.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: Addressed - integration fixtures create unique tenant IDs
- [ ] Rate limiting / abuse protection: N/A - test framework setup
- [ ] Input validation / schema enforcement: N/A - test framework setup
- [x] Tests (unit/integration): Addressed - integration suite added
- [x] Error handling + logging: Addressed - explicit skip reasons for missing services
- [x] Documentation updates: Addressed - integration test docs

## Tasks / Subtasks

- [x] Create `backend/tests/integration/` directory structure (AC: 1)
- [x] Add pytest fixtures for real database connections (AC: 1)
- [x] Create tenant isolation fixtures for test data cleanup (AC: 2, 3)
- [x] Configure CI to run integration tests separately (AC: 4)
- [x] Add integration test documentation (AC: 4)

## Technical Notes

Integration tests will be gated by `INTEGRATION_TESTS=1` and require
`DATABASE_URL`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, and `REDIS_URL`.

## Definition of Done

- [x] Integration fixtures connect to real services
- [x] Tenant isolation and cleanup fixtures present
- [x] CI runs integration tests in a separate job
- [x] Integration test documentation added

## Dev Notes

- Added integration fixtures for Postgres, Neo4j, and Redis with tenant cleanup.
- CI now runs integration tests in a separate job with real services.
- Added documentation for local integration test setup.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Created integration conftest with environment gating and cleanup helpers.
- Added CI job with service containers and integration test run.
- Documented integration test setup in docs/testing.

### File List

- backend/tests/integration/conftest.py
- .github/workflows/ci-backend.yml
- docs/testing/integration-tests.md

## Senior Developer Review

Outcome: APPROVE

Notes:
- Integration gating and cleanup are explicit and safe for CI.
- Separate CI job keeps runtime predictable for unit tests.
