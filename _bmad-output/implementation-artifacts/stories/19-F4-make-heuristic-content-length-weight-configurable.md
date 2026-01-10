# Story 19-F4: Make Heuristic Content Length Weight Configurable

Status: done

## Story

As a platform engineer,
I want to allow tuning of length-based scoring heuristic,
so that we can maintain quality and operational confidence.

## Acceptance Criteria

1. Length weight is configurable via environment variable
2. Documentation explains the heuristic rationale
3. Tests cover weight values: 0 (disabled), 0.5 (default), 1.0 (max)
4. Logging shows heuristic contribution to final score

## Tasks / Subtasks

- [x] Length weight is configurable via environment variable
- [x] Documentation explains the heuristic rationale
- [x] Tests cover weight values: 0 (disabled), 0.5 (default), 1.0 (max)
- [x] Logging shows heuristic contribution to final score

## Dev Notes

- Implemented per `epic-19-tech-spec.md` for this story.
- Retroactively reconstructed from commit history for traceability.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Story 19-F4)

## Dev Agent Record

### Agent Model Used

Reconstructed (commit history audit)

### Completion Notes List

- Story file reconstructed from epic tech spec and commit history.
- Primary implementation commit: 05e28ff.

### File List

- `backend/src/agentic_rag_backend/config.py`
- `backend/src/agentic_rag_backend/retrieval/grader.py`
- `backend/tests/test_grader.py`
- `docs/guides/advanced-retrieval-configuration.md`

## Senior Developer Review

Outcome: APPROVE

- Retroactive documentation based on merged implementation and tests.