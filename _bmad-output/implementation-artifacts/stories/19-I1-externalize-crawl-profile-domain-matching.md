# Story 19-I1: Externalize crawl profile domain matching

Status: done

## Story

As a developer configuring crawling behavior,
I want crawl profile domain matching rules stored in a config file,
so that updates do not require code changes.

## Acceptance Criteria

1. Domain matching rules (exact, suffix, prefix) load from a JSON or YAML config file at runtime.
2. A default config file ships with the current rule set.
3. Invalid or missing config logs a warning and falls back to defaults.
4. Tests continue to verify domain-based profile selection behavior.

## Tasks / Subtasks

- [ ] Add a versioned config file for domain profile rules.
- [ ] Load domain matching rules from the config file with safe fallback.
- [ ] Update crawl profile tests to cover the externalized config usage.
- [ ] Document the config location and update instructions in dev notes.

## Dev Notes

- Update `backend/src/agentic_rag_backend/indexing/crawl_profiles.py` to load rule data.
- Store the config alongside the crawl profile module for packaging.
- Keep default rule values identical to current behavior.

### References

- ` _bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I1)
- `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

 - Added JSON-backed domain profile rules with safe fallback handling.
 - Included the config file in wheel packaging for distribution.
 - Kept domain matching behavior consistent with prior defaults.

### File List

 - `backend/src/agentic_rag_backend/indexing/crawl_profile_domains.json`
 - `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`
 - `backend/pyproject.toml`

## Senior Developer Review

Outcome: APPROVE

- Domain rule config loads with validation and safe fallback.
- Default rules preserved; behavior unchanged for existing tests.
- Package data includes JSON config to ship defaults.
