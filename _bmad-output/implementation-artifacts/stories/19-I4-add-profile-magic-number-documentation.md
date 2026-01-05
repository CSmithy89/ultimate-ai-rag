# Story 19-I4: Add profile magic number documentation

Status: done

## Story

As a developer configuring crawl profiles,
I want the default rate limits and concurrency values documented,
so that I understand why those numbers were chosen and how to tune them.

## Acceptance Criteria

1. Document rationale for default `rate_limit` and `max_concurrent` values.
2. Document rationale for default wait timeouts where applicable.
3. Documentation references where these defaults live in code.

## Tasks / Subtasks

- [ ] Add documentation for profile defaults and tuning guidance.
- [ ] Add inline comments where defaults are defined.

## Dev Notes

- Defaults live in `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`.
- Use a guide in `docs/guides/` for longer-form explanation.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Group I: 19-I4)
- `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`

## Dev Agent Record

### Agent Model Used

GPT-5

### Completion Notes List

 - Documented default profile values and rationale in a dedicated guide.
 - Added inline rationale comments next to crawl profile defaults.

### File List

 - `backend/src/agentic_rag_backend/indexing/crawl_profiles.py`
 - `docs/guides/crawl-profile-defaults.md`

## Senior Developer Review

Outcome: APPROVE

- Rationale documented in both code and guide.
- Defaults remain unchanged with clear tuning guidance.
