# Story 2.3: Dynamic Retrieval Method Selection

Status: done

## Story

As a user,
I want the agent to automatically choose the best retrieval method,
so that my queries use vector search, graph traversal, or both as appropriate.

## Acceptance Criteria

1. Given the agent is processing a query, when it needs to retrieve information, then it analyzes the query type (semantic vs. relational).
2. It selects Vector RAG for semantic similarity queries.
3. It selects GraphRAG for relationship-based queries.
4. It selects hybrid for complex multi-hop queries.
5. It logs the selection decision in the trajectory.

## Tasks / Subtasks

- [x] Implement retrieval router (AC: 1-4)
  - [x] Classify semantic vs relational intent
  - [x] Return vector/graph/hybrid strategy
- [x] Log selection decision (AC: 5)
  - [x] Emit a thought entry for selection

## Dev Notes

- Retrieval execution is still stubbed in Epic 2; focus on selection logic.
- Keep classification rules explicit and deterministic for testability.

### Project Structure Notes

- Add routing module under `backend/src/agentic_rag_backend/`.
- Orchestrator should call the router before retrieval steps.

### References

- Epic 2 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-2.3`
- Architecture overview: `_bmad-output/architecture.md#Cross-Cutting-Concerns-Identified`

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added retrieval router with explicit semantic/relational/hybrid rules.
- Emitted selection decision as an orchestrator thought.
- Extended API response with retrieval strategy metadata.

### File List

- backend/src/agentic_rag_backend/retrieval_router.py
- backend/src/agentic_rag_backend/agents/orchestrator.py
- backend/src/agentic_rag_backend/schemas.py
- backend/src/agentic_rag_backend/main.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- Clear classification rules and explicit strategy enum improve testability.
- Strategy exposure in response is useful for downstream UI and logging.
