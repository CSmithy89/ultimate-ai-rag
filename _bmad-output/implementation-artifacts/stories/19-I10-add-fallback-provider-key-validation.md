# Story 19-I10: Add fallback provider key validation

Status: done

## Story

As a developer enabling crawl fallbacks,
I want API credentials validated at startup,
so that misconfigured fallbacks fail fast in production.

## Acceptance Criteria

1. Missing APIFY/BRIGHTDATA credentials are detected when fallback is enabled.
2. Production raises a clear error for missing credentials.
3. Development warns but continues.

## Tasks / Subtasks

- [x] Validate fallback credentials during settings load.
- [x] Add tests for prod vs dev behavior.

## Dev Notes

- Update `backend/src/agentic_rag_backend/config.py`.
- Reuse `crawler_strict_validation` to control prod fail-fast.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I10)
- `backend/src/agentic_rag_backend/config.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

- Added fallback credential validation with strict/prod fail-fast behavior.
- Added tests for dev warning and prod failure paths.

### File List

- `backend/src/agentic_rag_backend/config.py`
- `backend/tests/test_crawl_profiles.py`

## Senior Developer Review

Outcome: APPROVE

- Validation uses strict/prod behavior and remains lenient in dev.
- Tests cover both missing credential scenarios.
