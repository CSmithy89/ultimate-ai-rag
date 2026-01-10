# Story 19-F3: Make CrossEncoderGrader Model Selectable

Status: done

## Story

As a platform engineer,
I want to allow configuration of grader model without code changes,
so that we can maintain quality and operational confidence.

## Acceptance Criteria

1. Grader model is configurable via `GRADER_MODEL` environment variable
2. At least 3 models are tested and documented
3. Model loading is lazy (on first grader use, not startup)
4. Documentation in `docs/guides/advanced-retrieval-configuration.md` lists available models with accuracy/speed tradeoffs
5. Fallback to default if configured model unavailable

## Tasks / Subtasks

- [x] Grader model is configurable via `GRADER_MODEL` environment variable
- [x] At least 3 models are tested and documented
- [x] Model loading is lazy (on first grader use, not startup)
- [x] Documentation in `docs/guides/advanced-retrieval-configuration.md` lists available models with accuracy/speed tradeoffs
- [x] Fallback to default if configured model unavailable

## Dev Notes

- Implemented per `epic-19-tech-spec.md` for this story.
- Retroactively reconstructed from commit history for traceability.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Story 19-F3)

## Dev Agent Record

### Agent Model Used

Reconstructed (commit history audit)

### Completion Notes List

- Story file reconstructed from epic tech spec and commit history.
- Primary implementation commit: e64de88.

### File List

- `backend/src/agentic_rag_backend/config.py`
- `backend/src/agentic_rag_backend/retrieval/__init__.py`
- `backend/src/agentic_rag_backend/retrieval/grader.py`
- `backend/tests/test_grader.py`
- `docs/guides/advanced-retrieval-configuration.md`

## Senior Developer Review

Outcome: APPROVE

- Retroactive documentation based on merged implementation and tests.