# Story 19-I9: Implement rate-limiting enforcement

Status: done

## Story

As a developer running crawls,
I want the crawler to enforce rate limits,
so that requests do not exceed configured throughput.

## Acceptance Criteria

1. Rate limit is enforced for crawl-many batches.
2. Rate limit is configurable via crawl options.
3. Default behavior remains unchanged if rate limit is not set.
4. Logs indicate when rate limiting is active.

## Tasks / Subtasks

- [x] Add rate_limit parameter to crawl-many.
- [x] Apply batch delay based on rate_limit.
- [x] Wire crawl options into crawl-many calls.

## Dev Notes

- Update `backend/src/agentic_rag_backend/indexing/crawler.py`.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I9)
- `backend/src/agentic_rag_backend/indexing/crawler.py`

## Dev Agent Record

### Agent Model Used

GPT-5 (Codex CLI)

### Completion Notes List

- Added rate_limit support for crawl-many batching with delay enforcement.
- Wired crawl options into crawl-many to apply rate limits.

### File List

- `backend/src/agentic_rag_backend/indexing/crawler.py`

## Senior Developer Review

Outcome: APPROVE

- Batch delay enforces configured rate limit without breaking existing callers.
- Logging shows when rate limiting is active.
