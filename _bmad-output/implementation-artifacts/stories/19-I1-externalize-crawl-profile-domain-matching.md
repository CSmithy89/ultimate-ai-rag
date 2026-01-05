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
- Store the default config in `config/crawl-profiles.yaml` (override via env var).
- Keep default rule values identical to current behavior.

### References

- ` _bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I1)
- `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

- Added YAML-backed domain profile rules with safe fallback handling.
- Added default `config/crawl-profiles.yaml` and PyYAML dependency.
- Documented how to customize mappings and override config path.
 - Kept domain matching behavior consistent with prior defaults.

### File List

 - `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`
 - `backend/pyproject.toml`
 - `config/crawl-profiles.yaml`
 - `docs/guides/crawl-profile-mapping.md`
 - `.env.example`

## Senior Developer Review

Outcome: APPROVE

- Domain rule config loads from YAML with validation and safe fallback.
- Default rules preserved; behavior unchanged for existing tests.
- Restart requirement documented with config override option.
