# Story 19-G2: Make Context Generation Prompt Configurable

Status: done

## Story

As a platform engineer,
I want to allow customization of contextual retrieval prompt,
so that we can maintain quality and operational confidence.

## Acceptance Criteria

1. Contextual retrieval prompt is loaded from configurable file path
2. Template supports `{document}` and `{chunk}` placeholders
3. Domain-specific prompts can be used (e.g., legal, medical, technical)
4. Documentation provides prompt engineering examples
5. Invalid template gracefully falls back to default

## Tasks / Subtasks

- [x] Contextual retrieval prompt is loaded from configurable file path
- [x] Template supports `{document}` and `{chunk}` placeholders
- [x] Domain-specific prompts can be used (e.g., legal, medical, technical)
- [x] Documentation provides prompt engineering examples
- [x] Invalid template gracefully falls back to default

## Dev Notes

- Implemented per `epic-19-tech-spec.md` for this story.
- Retroactively reconstructed from commit history for traceability.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Story 19-G2)

## Dev Agent Record

### Agent Model Used

Reconstructed (commit history audit)

### Completion Notes List

- Story file reconstructed from epic tech spec and commit history.
- Primary implementation commit: 38d1e3a.

### File List

- `backend/prompts/contextual_retrieval_example.txt`
- `backend/src/agentic_rag_backend/config.py`
- `backend/src/agentic_rag_backend/indexing/contextual.py`
- `backend/src/agentic_rag_backend/main.py`
- `backend/src/agentic_rag_backend/observability/metrics.py`
- `backend/src/agentic_rag_backend/retrieval/cache.py`
- `backend/src/agentic_rag_backend/retrieval/grader.py`
- `backend/src/agentic_rag_backend/retrieval/normalization.py`
- `backend/src/agentic_rag_backend/retrieval/reranking.py`
- `backend/tests/test_group_g_caching_tuning.py`
- `docs/guides/advanced-retrieval-configuration.md`

## Senior Developer Review

Outcome: APPROVE

- Retroactive documentation based on merged implementation and tests.