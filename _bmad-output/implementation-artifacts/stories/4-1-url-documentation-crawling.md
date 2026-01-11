# Story 4.1: URL Documentation Crawling

Status: done

## Story

As a data engineer,
I want to trigger autonomous crawling of documentation websites,
so that I can ingest external knowledge sources without manual downloads.

## Acceptance Criteria

1. Given a valid documentation URL is provided, when the user triggers crawling via the ingestion API (`POST /api/v1/ingest/url`), then Crawl4AI crawls the documentation site and returns a job_id with status "queued".
2. Given a crawl is in progress, when the crawler encounters a robots.txt file, then it respects the directives and rate limits (default: 1 req/sec).
3. Given a crawl job is running, when the crawler discovers linked pages, then it extracts content from all linked pages up to the configured max_depth (default: 3).
4. Given content is extracted from pages, when the crawl completes, then the extracted content is queued for the parsing pipeline via Redis Streams.
5. Given a crawl job exists, when the user queries the job status endpoint (`GET /api/v1/ingest/jobs/{job_id}`), then progress statistics and crawl metrics are returned.

## Tasks / Subtasks

- [x] Create Pydantic models for ingestion requests and responses (AC: 1, 5)
  - [x] Add `backend/src/agentic_rag_backend/models/ingest.py` with CrawlRequest, CrawlResponse, JobStatus models
  - [x] Add `backend/src/agentic_rag_backend/models/documents.py` with UnifiedDocument model
- [x] Create ingestion API endpoints (AC: 1, 5)
  - [x] Add `backend/src/agentic_rag_backend/api/routes/ingest.py` with POST /ingest/url and GET /ingest/jobs/{job_id}
  - [x] Register router in main.py
  - [x] Implement tenant_id validation and multi-tenancy support
- [x] Create Crawl4AI wrapper service (AC: 2, 3, 4)
  - [x] Add `backend/src/agentic_rag_backend/indexing/crawler.py` with async Crawl4AI integration
  - [x] Implement robots.txt compliance checking
  - [x] Implement configurable rate limiting (default 1 req/sec)
  - [x] Implement depth-limited link extraction and crawling
  - [x] Extract and normalize content (HTML to markdown conversion)
- [x] Create async crawl worker (AC: 3, 4)
  - [x] Add `backend/src/agentic_rag_backend/indexing/workers/crawl_worker.py`
  - [x] Implement Redis Streams consumer for crawl.jobs
  - [x] Implement progress tracking and job status updates
  - [x] Queue extracted content to parse.jobs stream
- [x] Create database models and migrations (AC: 1, 5)
  - [x] Add documents table migration (id, tenant_id, source_type, source_url, content_hash, status, metadata)
  - [x] Add ingestion_jobs table migration (id, tenant_id, document_id, job_type, status, progress, error_message)
  - [x] Create indexes for tenant_id and status columns
- [x] Add Redis client for job queue (AC: 4)
  - [x] Add `backend/src/agentic_rag_backend/db/redis.py` if not exists
  - [x] Implement Redis Streams producer/consumer patterns
- [x] Write unit tests (AC: 1-5)
  - [x] Add `backend/tests/indexing/test_crawler.py` for crawler functions
  - [x] Add `backend/tests/api/test_ingest.py` for API endpoints
  - [x] Mock Crawl4AI and Redis for isolated testing
- [x] Add Crawl4AI dependency to pyproject.toml

## Dev Notes

### Technical Implementation Details

**API Endpoint Design:**
- `POST /api/v1/ingest/url` - Start URL crawl job
  - Request: `{ "url": "https://...", "tenant_id": "uuid", "max_depth": 3, "options": {...} }`
  - Response: `{ "data": { "job_id": "uuid", "status": "queued" }, "meta": {...} }`
- `GET /api/v1/ingest/jobs/{job_id}` - Check job status
  - Response: `{ "data": { "job_id": "uuid", "status": "running", "progress": { "pages_crawled": 10, "total_pages": 50 } }, "meta": {...} }`

**Crawl4AI Integration:**
```python
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

async def crawl_url(url: str, max_depth: int = 3) -> AsyncGenerator[CrawlResult, None]:
    config = CrawlerRunConfig(
        follow_links=True,
        max_depth=max_depth,
        respect_robots_txt=True,
        rate_limit=1.0  # 1 request per second
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url, config=config)
        yield result
```

**Job Queue Flow:**
```
API Request -> Redis Stream (crawl.jobs) -> Crawler Worker -> Redis Stream (parse.jobs)
```

**Output Format:**
Raw HTML/Markdown content with metadata including:
- Source URL
- Crawl timestamp
- Page title
- Content hash (SHA-256 for deduplication)

### Multi-Tenancy Requirements

Every database query MUST include `tenant_id` filtering as per architecture specifications:
- Documents and jobs tables must have tenant_id column
- All queries must filter by tenant_id
- Use tenant middleware for request context

### Error Handling

Use RFC 7807 Problem Details format for API errors:
```json
{
  "type": "https://api.example.com/errors/invalid-url",
  "title": "Invalid URL",
  "status": 400,
  "detail": "The provided URL is not accessible",
  "instance": "/api/v1/ingest/url"
}
```

### Dependencies

Add to `pyproject.toml`:
```toml
"crawl4ai>=0.3.0",
"redis>=5.0.0",
```

### Configuration

Environment variables needed:
```bash
CRAWL4AI_RATE_LIMIT=1.0    # Requests per second
REDIS_URL=redis://localhost:6379
```

## References

- Tech Spec: `_bmad-output/epics/epic-4-tech-spec.md#31-story-41-url-documentation-crawling`
- Architecture: `_bmad-output/architecture.md#data-architecture`
- Epic Definition: `_bmad-output/project-planning-artifacts/epics.md#story-41-url-documentation-crawling`
- Database Schema: `_bmad-output/epics/epic-4-tech-spec.md#4-database-schema`
- API Endpoints: `_bmad-output/epics/epic-4-tech-spec.md#5-api-endpoints`

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A - Implementation completed without significant debugging issues.

### Completion Notes List

1. **Pydantic Models Created:**
   - `models/ingest.py`: CrawlRequest, CrawlResponse, JobStatus, JobProgress, CrawlOptions
   - `models/documents.py`: UnifiedDocument, CrawledPage, DocumentMetadata, SourceType, DocumentStatus

2. **API Endpoints Implemented:**
   - POST `/api/v1/ingest/url` - Creates crawl job, queues to Redis, returns job_id
   - GET `/api/v1/ingest/jobs/{job_id}` - Returns job status with progress metrics
   - GET `/api/v1/ingest/jobs` - Lists jobs with pagination and status filtering

3. **Crawler Service Features:**
   - robots.txt compliance via RobotsTxtChecker class
   - Rate limiting with configurable requests/second
   - Depth-limited BFS crawling
   - HTML to Markdown conversion
   - Content hash generation for deduplication
   - Same-domain link extraction

4. **Database Integration:**
   - PostgreSQL async client with asyncpg
   - Auto-creates tables on startup (documents, ingestion_jobs)
   - Proper indexes for tenant_id and status columns
   - Multi-tenancy enforced on all queries

5. **Redis Integration:**
   - Redis Streams for job queue (crawl.jobs, parse.jobs)
   - Consumer groups for horizontal worker scaling
   - Message acknowledgment after processing

6. **Error Handling:**
   - RFC 7807 Problem Details format
   - Custom exception classes (InvalidUrlError, JobNotFoundError, etc.)
   - Global exception handler registered

7. **Tests Created:**
   - 20+ unit tests for crawler functions
   - 10+ API endpoint tests
   - Comprehensive mocking for Redis and PostgreSQL

### File List

**New Files Created:**
- `backend/src/agentic_rag_backend/models/__init__.py`
- `backend/src/agentic_rag_backend/models/ingest.py`
- `backend/src/agentic_rag_backend/models/documents.py`
- `backend/src/agentic_rag_backend/api/__init__.py`
- `backend/src/agentic_rag_backend/api/routes/__init__.py`
- `backend/src/agentic_rag_backend/api/routes/ingest.py`
- `backend/src/agentic_rag_backend/indexing/__init__.py`
- `backend/src/agentic_rag_backend/indexing/crawler.py`
- `backend/src/agentic_rag_backend/indexing/workers/__init__.py`
- `backend/src/agentic_rag_backend/indexing/workers/crawl_worker.py`
- `backend/src/agentic_rag_backend/db/__init__.py`
- `backend/src/agentic_rag_backend/db/redis.py`
- `backend/src/agentic_rag_backend/db/postgres.py`
- `backend/src/agentic_rag_backend/core/__init__.py`
- `backend/src/agentic_rag_backend/core/errors.py`
- `backend/tests/__init__.py`
- `backend/tests/conftest.py`
- `backend/tests/api/__init__.py`
- `backend/tests/api/test_ingest.py`
- `backend/tests/indexing/__init__.py`
- `backend/tests/indexing/test_crawler.py`

**Modified Files:**
- `backend/pyproject.toml` - Added dependencies
- `backend/src/agentic_rag_backend/main.py` - Registered router and exception handler
- `backend/src/agentic_rag_backend/config.py` - Added crawl rate limit config

## Senior Developer Review

**Reviewer:** Claude Opus 4.5 (Senior Developer Agent)
**Review Date:** 2025-12-28
**Outcome:** APPROVE

### Summary

This implementation of Story 4.1 (URL Documentation Crawling) is well-structured, follows the established architecture patterns, and meets all acceptance criteria. The code demonstrates solid engineering practices with proper error handling, multi-tenancy enforcement, and comprehensive test coverage.

### Acceptance Criteria Verification

| AC | Requirement | Status | Notes |
|----|-------------|--------|-------|
| AC1 | POST /api/v1/ingest/url returns job_id with "queued" status | PASS | Endpoint implemented correctly in `api/routes/ingest.py` |
| AC2 | Respects robots.txt and rate limits (default 1 req/sec) | PASS | `RobotsTxtChecker` class and rate limiting in `CrawlerService` |
| AC3 | Depth-limited crawling (default: 3) | PASS | BFS crawling with configurable `max_depth` parameter |
| AC4 | Extracted content queued to parse.jobs via Redis Streams | PASS | `_queue_for_parsing` in `crawl_worker.py` publishes to `parse.jobs` |
| AC5 | GET /api/v1/ingest/jobs/{job_id} returns progress metrics | PASS | Endpoint returns `JobProgress` with pages_crawled, pages_discovered, etc. |

### Architecture Compliance

| Requirement | Status | Notes |
|-------------|--------|-------|
| snake_case functions | PASS | All functions follow Python naming convention |
| PascalCase classes | PASS | `CrawlerService`, `RobotsTxtChecker`, `PostgresClient`, etc. |
| SCREAMING_SNAKE constants | PASS | `DEFAULT_RATE_LIMIT`, `CRAWL_JOBS_STREAM`, etc. |
| Pydantic validation | PASS | Request/response models with proper Field constraints |
| RFC 7807 errors | PASS | `AppError.to_problem_detail()` produces compliant format |
| Multi-tenancy (tenant_id filtering) | PASS | All DB queries include `WHERE tenant_id = $N` |
| API response wrapper | PASS | `success_response()` returns `{data, meta}` format |

### Code Quality Assessment

**Strengths:**

1. **Clean separation of concerns:** Models, database clients, crawler service, and API routes are properly separated into distinct modules.

2. **Comprehensive error handling:** Custom exception classes (`InvalidUrlError`, `JobNotFoundError`, `CrawlError`, etc.) with RFC 7807 compliant responses.

3. **Async-first design:** All I/O operations use async/await patterns with proper connection pooling for PostgreSQL and Redis.

4. **Idempotent ingestion:** Content hash (SHA-256) used for deduplication with `ON CONFLICT DO UPDATE` in document creation.

5. **Configurable rate limiting:** Both per-request delay and per-second limits are configurable through `CrawlOptions`.

6. **Graceful shutdown:** Worker implements signal handlers for SIGTERM/SIGINT.

7. **Good test coverage:** 20+ unit tests for crawler functions and 15+ API tests with comprehensive mocking.

### Issues Found

**Severity: Low (No blocking issues)**

1. **Deprecation warning (Low):** `datetime.utcnow()` is used in several places (models/documents.py, indexing/crawler.py). This is deprecated in Python 3.12+. Should migrate to `datetime.now(timezone.utc)`.

2. **Config not used in crawler (Low):** The `crawl4ai_rate_limit` from settings is defined but the `CrawlerService` defaults to `DEFAULT_RATE_LIMIT` rather than reading from config. The rate limit is correctly passed through `CrawlOptions`, so this is functional but could be more consistent.

3. **Missing CrawledPage export (Low):** `CrawledPage` model is defined in `models/documents.py` but not exported in `models/__init__.py`. This doesn't break functionality since it's imported directly where needed.

4. **Simplified HTML-to-Markdown (Low):** The `html_to_markdown()` function uses regex-based conversion. The code includes a comment acknowledging this should use a library like `markdownify` or `html2text` in production. Acceptable for MVP.

### Security Review

| Concern | Status | Notes |
|---------|--------|-------|
| Input validation | PASS | Pydantic HttpUrl validates URL format; `is_valid_url()` double-checks |
| SQL injection | PASS | Parameterized queries with `$N` placeholders throughout |
| XSS in markdown | N/A | Content stored as markdown for later processing |
| Tenant isolation | PASS | All queries filter by tenant_id |
| Rate limiting | PASS | Prevents aggressive crawling of target sites |
| robots.txt | PASS | Respects disallow directives |

### Test Coverage Analysis

**API Tests (`test_ingest.py`):**
- Create crawl job success/failure
- URL validation
- Tenant ID enforcement
- Max depth validation
- Job status retrieval
- Job listing with pagination/filtering
- Health check

**Crawler Tests (`test_crawler.py`):**
- URL validation and normalization
- Domain comparison
- Content hashing
- HTML to Markdown conversion
- Title extraction
- Link extraction
- robots.txt compliance
- Depth-limited crawling

### Recommendations for Future Work

1. **Integration tests:** Add integration tests with real Redis/PostgreSQL instances using testcontainers.

2. **Worker instrumentation:** Add OpenTelemetry tracing for distributed debugging across API -> Redis -> Worker flow.

3. **Dead letter queue:** Implement DLQ for failed crawl jobs that exceed retry limits.

4. **URL deduplication:** Consider checking content_hash before crawling to skip already-ingested pages.

5. **HTML parser upgrade:** Replace regex-based HTML parsing with BeautifulSoup or lxml for robustness.

### Files Reviewed

**New Files (21 files):**
- `backend/src/agentic_rag_backend/models/__init__.py`
- `backend/src/agentic_rag_backend/models/ingest.py`
- `backend/src/agentic_rag_backend/models/documents.py`
- `backend/src/agentic_rag_backend/api/__init__.py`
- `backend/src/agentic_rag_backend/api/routes/__init__.py`
- `backend/src/agentic_rag_backend/api/routes/ingest.py`
- `backend/src/agentic_rag_backend/indexing/__init__.py`
- `backend/src/agentic_rag_backend/indexing/crawler.py`
- `backend/src/agentic_rag_backend/indexing/workers/__init__.py`
- `backend/src/agentic_rag_backend/indexing/workers/crawl_worker.py`
- `backend/src/agentic_rag_backend/db/__init__.py`
- `backend/src/agentic_rag_backend/db/redis.py`
- `backend/src/agentic_rag_backend/db/postgres.py`
- `backend/src/agentic_rag_backend/core/__init__.py`
- `backend/src/agentic_rag_backend/core/errors.py`
- `backend/tests/__init__.py`
- `backend/tests/conftest.py`
- `backend/tests/api/__init__.py`
- `backend/tests/api/test_ingest.py`
- `backend/tests/indexing/__init__.py`
- `backend/tests/indexing/test_crawler.py`

**Modified Files (3 files):**
- `backend/pyproject.toml` - Dependencies added correctly
- `backend/src/agentic_rag_backend/main.py` - Router and error handler registered
- `backend/src/agentic_rag_backend/config.py` - crawl4ai_rate_limit config added

### Conclusion

The implementation is production-ready with only minor improvements recommended. All acceptance criteria are met, multi-tenancy is properly enforced, and the code follows established architecture patterns. The test coverage is adequate for the MVP scope.

**Approved for merge.**
