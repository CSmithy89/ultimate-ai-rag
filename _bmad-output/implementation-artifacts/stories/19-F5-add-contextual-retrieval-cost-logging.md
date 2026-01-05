# Story 19-F5: Add Contextual Retrieval Cost Logging

Status: done

## Story

As a platform engineer,
I want to track LLM costs for contextual retrieval enrichment,
so that we can maintain quality and operational confidence.

## Acceptance Criteria

1. Token usage is logged for each contextual enrichment call
2. Cost estimates computed using model pricing (configurable)
3. Aggregated cost metrics available via Prometheus (if enabled)
4. Dashboard shows contextual retrieval costs over time
5. Cache hit rate is tracked for prompt caching efficiency

## Tasks / Subtasks

- [x] Token usage is logged for each contextual enrichment call
- [x] Cost estimates computed using model pricing (configurable)
- [x] Aggregated cost metrics available via Prometheus (if enabled)
- [x] Dashboard shows contextual retrieval costs over time
- [x] Cache hit rate is tracked for prompt caching efficiency

## Dev Notes

- Implemented per `epic-19-tech-spec.md` for this story.
- Retroactively reconstructed from commit history for traceability.

### References

- `_bmad-output/implementation-artifacts/epic-19-tech-spec.md` (Story 19-F5)

## Dev Agent Record

### Agent Model Used

Reconstructed (commit history audit)

### Completion Notes List

- Story file reconstructed from epic tech spec and commit history.
- Primary implementation commit: dc93d21.

### File List

- `backend/src/agentic_rag_backend/indexing/__init__.py`
- `backend/src/agentic_rag_backend/indexing/contextual.py`
- `backend/src/agentic_rag_backend/observability/__init__.py`
- `backend/src/agentic_rag_backend/observability/metrics.py`
- `backend/tests/test_contextual.py`

## Senior Developer Review

Outcome: APPROVE

- Retroactive documentation based on merged implementation and tests.