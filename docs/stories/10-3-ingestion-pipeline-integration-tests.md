# Story 10.3: Ingestion Pipeline Integration Tests

Status: done

## Story

As a QA Engineer,  
I want integration tests for the document ingestion pipeline,  
So that URL crawling, PDF parsing, and entity extraction are validated end-to-end.

## Acceptance Criteria

1. Given a URL is submitted, when ingestion completes, then document is stored in PostgreSQL.
2. Given a PDF is uploaded, when parsing completes, then chunks are created with embeddings.
3. Given entities are extracted, when graph building completes, then nodes exist in Neo4j.
4. Given duplicate content is submitted, when deduplication runs, then no duplicates are created.

## Standards Coverage

- [x] Multi-tenancy / tenant isolation: Addressed - tenant-scoped ingestion fixtures
- [ ] Rate limiting / abuse protection: N/A - internal integration tests
- [ ] Input validation / schema enforcement: N/A - internal integration tests
- [x] Tests (unit/integration): Addressed - ingestion integration suite
- [x] Error handling + logging: Addressed - integration fixtures provide skip reasons
- [ ] Documentation updates: N/A - internal test work

## Tasks / Subtasks

- [x] Create `backend/tests/integration/test_ingestion_pipeline.py` (AC: 1-4)
- [x] Create PDF test fixtures (sample_simple.pdf, sample_tables.pdf, sample_complex.pdf) (AC: 2)
- [x] Test URL ingestion with mock HTTP server (AC: 1)
- [x] Test PDF parsing and chunking (AC: 2)
- [x] Test entity extraction and graph building (AC: 3)
- [x] Test content deduplication (AC: 4)

## Technical Notes

Use Docling parser and CrawlerService with a local HTTP server for URL ingestion.

## Definition of Done

- [x] Ingestion pipeline integration tests added
- [x] PDF fixtures available
- [x] Deduplication verified

## Dev Notes

- Added integration tests for URL ingestion, PDF parsing, graph writes, and deduplication.
- Added small PDF fixtures for parser coverage.
- Tests use real Postgres/Neo4j services with integration gating.

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Created ingestion integration test suite using crawler and parser modules.
- Added PDF fixtures under backend/tests/fixtures.
- Verified deduplication via content_hash uniqueness.

### File List

- backend/tests/integration/test_ingestion_pipeline.py
- backend/tests/fixtures/sample_simple.pdf
- backend/tests/fixtures/sample_tables.pdf
- backend/tests/fixtures/sample_complex.pdf

## Senior Developer Review

Outcome: APPROVE

Notes:
- Tests cover URL ingestion, PDF parsing, graph writes, and deduplication with real services.
- Fixtures are minimal and keep runtime within integration constraints.
