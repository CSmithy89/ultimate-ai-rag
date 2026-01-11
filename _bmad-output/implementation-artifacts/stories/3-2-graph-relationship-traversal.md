# Story 3.2: Graph Relationship Traversal

Status: done

## Story

As a user,
I want to query relationships between entities in the knowledge graph,
so that I can discover connections that semantic search alone would miss.

## Acceptance Criteria

1. Given entities and relationships exist in Neo4j, when a user submits a relationship-based query, then the system identifies relevant starting entities.
2. It traverses relationships using Cypher queries.
3. It returns connected entities and relationship paths.
4. It respects tenant isolation via property filters (NFR3).

## Tasks / Subtasks

- [x] Implement graph traversal module (AC: 1-3)
  - [x] Extract query terms to seed traversal
  - [x] Query Neo4j for starting entities
  - [x] Traverse bounded paths with allowed relationship types
- [x] Enforce tenant isolation in traversal queries (AC: 4)
- [x] Return graph evidence in query response (AC: 3)

## Dev Notes

- Use Neo4j client with tenant_id filtering on nodes and relationships.
- Keep traversal bounded (max hops + max paths).

### Project Structure Notes

- Add graph traversal module under `backend/src/agentic_rag_backend/retrieval/`.
- Extend Neo4j client with search + traversal helpers.

### References

- Epic 3 tech spec: `_bmad-output/epics/epic-3-tech-spec.md#52-story-32-graph-relationship-traversal`
- Epic 3 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-3.2`
- Architecture overview: `_bmad-output/architecture.md#Hybrid-Retrieval`

## Dev Agent Record

### Agent Model Used

N/A

### Debug Log References

### Completion Notes List

- Added Neo4j search + traversal helpers with tenant filtering.
- Implemented graph traversal service and path parsing.
- Orchestrator now returns graph evidence with nodes, edges, and paths.

### File List

- backend/src/agentic_rag_backend/db/neo4j.py
- backend/src/agentic_rag_backend/retrieval/graph_traversal.py
- backend/src/agentic_rag_backend/retrieval/__init__.py
- backend/src/agentic_rag_backend/agents/orchestrator.py
- backend/src/agentic_rag_backend/main.py

## Senior Developer Review

Outcome: APPROVE

Notes:
- Tenant filtering is enforced in entity search and traversal.
- Path parsing captures nodes, edges, and traversal sequences.
