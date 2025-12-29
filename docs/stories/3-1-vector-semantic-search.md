# Story 3.1: Vector Semantic Search

Status: done

## Story

As a user,
I want to search for information using semantic similarity,
so that I can find relevant content even when exact keywords don't match.

## Acceptance Criteria

1. Given documents have been indexed with embeddings in pgvector, when a user submits a semantic query, then the system generates an embedding for the query.
2. It performs cosine similarity search against stored vectors.
3. It returns the top-k most relevant chunks.
4. Each result includes similarity score and source reference.

## Tasks / Subtasks

- [x] Implement vector retrieval module (AC: 1-3)
  - [x] Generate query embeddings via EmbeddingGenerator
  - [x] Call pgvector similarity search
  - [x] Cap results by limit/threshold
- [x] Return vector evidence in query response (AC: 4)

## Dev Notes

- Retrieval execution is wired through the orchestrator.
- Use `PostgresClient.search_similar_chunks` for pgvector queries.

### Project Structure Notes

- Add retrieval module under `backend/src/agentic_rag_backend/retrieval/`.
- Extend API schemas to expose vector citations.

### References

- Epic 3 tech spec: `docs/epics/epic-3-tech-spec.md#51-story-31-vector-semantic-search`
- Epic 3 definition: `_bmad-output/project-planning-artifacts/epics.md#Story-3.1`
- Architecture overview: `_bmad-output/architecture.md#Hybrid-Retrieval`

## Dev Agent Record

### Agent Model Used

gpt-5

### Debug Log References

### Completion Notes List

- Added vector search service that queries pgvector with embeddings.
- Extended query response schema with vector evidence citations.
- Orchestrator now uses vector evidence to build answer prompts.

### File List

- backend/src/agentic_rag_backend/retrieval/__init__.py
- backend/src/agentic_rag_backend/retrieval/types.py
- backend/src/agentic_rag_backend/retrieval/vector_search.py
- backend/src/agentic_rag_backend/agents/orchestrator.py
- backend/src/agentic_rag_backend/schemas.py
- backend/src/agentic_rag_backend/main.py
## Senior Developer Review

Outcome: APPROVE

Notes:
- Vector search uses existing pgvector queries and embeds once per request.
- Evidence schema keeps API compatibility while exposing citation data.
