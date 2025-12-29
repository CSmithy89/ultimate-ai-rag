# Story 3.4: Graph-Based Explainability

Status: done

## Story

As a user,
I want to see how the system arrived at its answer using graph connections,
so that I can verify the reasoning and trust the response.

## Acceptance Criteria

1. Given an answer was generated using graph traversal, when the response is returned, then it includes the specific nodes referenced.
2. It shows the relationship edges that connected them.
3. It provides a human-readable explanation of the path.
4. It allows the user to explore the subgraph visually.

## Tasks / Subtasks

- [x] Include graph nodes and edges in API response (AC: 1-2)
- [x] Generate human-readable path explanation (AC: 3)
- [x] Ensure evidence contains IDs for UI exploration (AC: 4)

## Dev Notes

- Use GraphEvidence in query response to expose nodes/edges/paths.
- Explanation should be derived from traversal paths.

### Project Structure Notes

- Update orchestrator evidence mapping if needed.

### References

- Epic 3 tech spec: `docs/epics/epic-3-tech-spec.md#54-story-34-graph-based-explainability`
- Epic 3 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-3.4`

## Dev Agent Record

### Agent Model Used

N/A

### Debug Log References

### Completion Notes List

- Graph evidence already exposes nodes/edges/paths with IDs for UI use.
- Added fallback explanation when no traversal paths are found.

### File List

- backend/src/agentic_rag_backend/schemas.py
- backend/src/agentic_rag_backend/agents/orchestrator.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- GraphEvidence includes nodes, edges, paths, and explanation for UI use.
