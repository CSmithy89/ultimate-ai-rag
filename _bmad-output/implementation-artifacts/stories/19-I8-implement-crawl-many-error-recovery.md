# Story 19-I8: Implement crawl-many error recovery

Status: done

## Story

As a developer running batch crawls,
I want crawl-many to return partial results when some URLs fail,
so that a single bad URL doesn't abort the entire batch.

## Acceptance Criteria

1. Failures include URL and error context in logs.
2. Behavior is configurable: fail-fast or continue.
3. Success/failure counts are reported on completion.
4. Default behavior remains fail-fast.

## Tasks / Subtasks

- [ ] Add `on_error` option to crawl-many.
- [ ] Track failures and success counts.
- [ ] Update logging with failure context and summary.

## Dev Notes

- Update `backend/src/agentic_rag_backend/indexing/crawler.py`.
- Keep generator behavior for existing callers.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I8)
- `backend/src/agentic_rag_backend/indexing/crawler.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

 - Added on_error handling for crawl_many with fail-fast/continue modes.
 - Logged per-URL failures and summary counts for partial results.

### File List

 - `backend/src/agentic_rag_backend/indexing/crawler.py`

## Senior Developer Review

Outcome: APPROVE

- Failure handling is configurable and preserves default fail-fast behavior.
- Summary logging includes success/failure counts with partial indicator.
