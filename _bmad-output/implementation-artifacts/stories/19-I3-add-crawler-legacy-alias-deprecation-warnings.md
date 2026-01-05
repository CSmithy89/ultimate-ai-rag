# Story 19-I3: Add crawler legacy alias deprecation warnings

Status: done

## Story

As a developer maintaining crawler integrations,
I want legacy alias functions to emit deprecation warnings,
so that users migrate before removals.

## Acceptance Criteria

1. `extract_links()` and `extract_title()` emit `DeprecationWarning` with migration guidance.
2. Deprecation warnings are logged at WARNING level.
3. Documentation lists deprecated functions and the replacement names.
4. Tests verify the warnings are emitted.

## Tasks / Subtasks

- [ ] Add canonical `get_links()` and `get_title()` helpers.
- [ ] Replace legacy aliases with warning wrappers.
- [ ] Add tests for warnings and return values.
- [ ] Document deprecations and migration path.

## Dev Notes

- Update `backend/src/agentic_rag_backend/indexing/crawler.py`.
- Add tests under `backend/tests/`.
- Keep behavior identical to existing alias behavior.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I3)
- `backend/src/agentic_rag_backend/indexing/crawler.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

 - Added canonical get_links/get_title helpers and deprecated aliases with warnings.
 - Logged deprecation usage and documented migration path.
 - Added tests for warning emission and return values.

### File List

 - `backend/src/agentic_rag_backend/indexing/crawler.py`
 - `backend/tests/test_crawler_deprecations.py`
 - `docs/guides/crawler-deprecations.md`

## Senior Developer Review

Outcome: APPROVE

- Deprecation warnings and log entries provide clear migration guidance.
- Canonical helper functions maintain existing behavior.
- Tests cover warning emission and return values.
