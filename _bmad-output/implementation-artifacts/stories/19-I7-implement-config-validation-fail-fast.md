# Story 19-I7: Implement config validation fail-fast

Status: done

## Story

As a developer deploying to production,
I want invalid crawl configuration to fail fast,
so that misconfigurations are caught at startup instead of silently ignored.

## Acceptance Criteria

1. Invalid crawl profile raises `ValueError` in production.
2. Development mode falls back with a warning.
3. Behavior is controlled by `CRAWLER_STRICT_VALIDATION` (default true in prod, false in dev).
4. Validation happens during settings load.

## Tasks / Subtasks

- [x] Add strict validation flag and apply to crawl profile validation.
- [x] Update `.env.example`.
- [x] Add tests for strict vs non-strict behavior.

## Dev Notes

- Update `backend/src/agentic_rag_backend/config.py`.
- Prefer reusing `get_bool_env` helper for env parsing.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I7)
- `backend/src/agentic_rag_backend/config.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

- Added strict validation flag with prod fail-fast behavior.
- Documented the new env setting and added tests for strict vs dev fallback.

### File List

- `backend/src/agentic_rag_backend/config.py`
- `backend/tests/test_crawl_profiles.py`
- `.env.example`

## Senior Developer Review

Outcome: APPROVE

- Production fails fast with clear error messaging; dev keeps fallback.
- Tests cover strict/non-strict behavior paths.
