# Story 3.3: Hybrid Answer Synthesis

Status: done

## Story

As a user,
I want the system to combine vector and graph results into a coherent answer,
so that I get comprehensive responses leveraging both retrieval methods.

## Acceptance Criteria

1. Given both vector search and graph traversal have returned results, when the retriever agent synthesizes the answer, then it merges results from both sources.
2. It ranks combined results by relevance.
3. It generates a unified response using the LLM.
4. It includes citations from both vector chunks and graph entities.

## Tasks / Subtasks

- [x] Aggregate vector + graph evidence (AC: 1-2)
- [x] Generate unified answer prompt for LLM (AC: 3)
- [x] Ensure citations reference both evidence types (AC: 4)

## Dev Notes

- Use orchestrator to build a hybrid prompt when strategy is HYBRID.
- Provide graph path explanations in evidence for citations.

### Project Structure Notes

- Add synthesis helper under `backend/src/agentic_rag_backend/retrieval/`.
- Keep prompt construction deterministic for testability.

### References

- Epic 3 tech spec: `_bmad-output/epics/epic-3-tech-spec.md#53-story-33-hybrid-answer-synthesis`
- Epic 3 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-3.3`

## Dev Agent Record

### Agent Model Used

N/A

### Debug Log References

### Completion Notes List

- Added hybrid synthesis helper to rank and format vector + graph evidence.
- Orchestrator now delegates prompt building to the hybrid synthesis module.

### File List

- backend/src/agentic_rag_backend/retrieval/hybrid_synthesis.py
- backend/src/agentic_rag_backend/agents/orchestrator.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- Hybrid prompt includes both vector and graph citations.
- Evidence ranking is deterministic and bounded.
