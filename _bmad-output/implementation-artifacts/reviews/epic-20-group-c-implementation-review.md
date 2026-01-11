# Epic 20 Group C - Implementation Readiness Review

Scope: Story 20-C1 (Graph-based rerankers), 20-C2 (Dual-level retrieval), 20-C3 (Parent-child chunk hierarchy).

Goal: Capture critical gaps and the minimum work needed to bring each story to an implementation-ready state that satisfies the acceptance criteria.

## Status Update (2026-01-07)

- Story 20-C1: Graph reranker is integrated into the unified retrieval pipeline; distance and episode queries are batched and honor max distance; ID matching accepts `id` and `uuid`.
- Story 20-C2: Dual-level weights now affect scores; synthesis temperature is configurable; community search is primary when available; routing path uses dual-level for HYBRID queries.
- Story 20-C3: Hierarchical chunks are stored in Postgres; ingestion writes hierarchy; small-to-big is integrated into the pipeline; parent assembly trims overlaps and uses merged token counts.

Remaining verification:
- Full test suite still pending (targeted retrieval tests pass).

## Story 20-C1: Graph-Based Rerankers

### Must-fix gaps
- Pipeline integration is missing; graph rerankers are never invoked by any retrieval path (resolved).
- Entity ID mismatch risk (uuid vs id) (resolved).
- Max-distance config is ignored (resolved).
- Config is not documented in `.env.example` (resolved).

### Implementation tasks
- Wire graph reranking into the main retrieval pipeline (after cross-encoder rerank); define whether MCP tools and REST APIs both use it.
- Normalize entity IDs: either store Graphiti UUIDs in Neo4j `Entity.id`, or map UUID -> id before distance queries.
- Use `max_distance` in shortestPath upper bound and add batch distance querying to avoid per-result shortestPath calls.
- Document `GRAPH_RERANKER_*` in `.env.example` with defaults.

### Tests to add
- Integration test with Graphiti + Neo4j verifying non-zero episode and distance scores.
- Perf test to enforce the <200ms SLA with representative result sizes.

## Story 20-C2: Dual-Level Retrieval

### Must-fix gaps
- Feature flag does not affect standard retrieval (resolved via query routing).
- Weights do not influence ranking or scoring (resolved).
- `DUAL_LEVEL_SYNTHESIS_TEMPERATURE` not implemented (resolved).
- Community detector only fallback (resolved; now primary when available).
- Config not documented in `.env.example` (resolved).

### Implementation tasks
- Decide and document the canonical entry point: either integrate dual-level into main retrieval or explicitly mark it as a separate endpoint with clear usage guidance.
- Apply weights to result ranking or combined scoring, or revise acceptance criteria to match the current behavior.
- Add `dual_level_synthesis_temperature` to settings and wire it into `_synthesize`.
- Use `community_detector` as the primary high-level retrieval mechanism (with Neo4j text match as fallback) or update the story to reflect current behavior.
- Document `DUAL_LEVEL_*` in `.env.example`.

### Tests to add
- Integration test that validates dual-level behavior through the chosen main retrieval entry point.
- Tests that verify weight effects on ranking or combined scores.
- Config test for synthesis temperature wiring.

## Story 20-C3: Parent-Child Chunk Hierarchy

### Must-fix gaps
- Hierarchical chunker and small-to-big wired into ingestion + retrieval (resolved).
- Concrete chunk store implementation exists (resolved).
- Parent chunk assembly duplicates overlap (resolved).
- Chunk-size constraints enforced via merged content tokens (resolved).
- Config documented in `.env.example` (resolved).

### Implementation tasks
- Add a chunk store implementation (DB schema + CRUD + vector search) and wire it into ingestion and retrieval pipelines.
- Gate ingestion with `HIERARCHICAL_CHUNKS_ENABLED` and use `HIERARCHICAL_CHUNK_LEVELS`, `HIERARCHICAL_OVERLAP_RATIO`, `HIERARCHICAL_EMBEDDING_LEVEL`.
- Remove overlap duplication when building parent chunks (e.g., trim child overlap regions).
- Enforce target size when assembling parents (compute actual token count for combined content).
- Integrate small-to-big retrieval into a retrieval entry point (API or MCP) and document usage.
- Document `HIERARCHICAL_*` and `SMALL_TO_BIG_*` in `.env.example`.

### Tests to add
- Integration tests covering ingestion + retrieval with real chunk storage.
- Tests asserting overlap handling and parent size constraints.
- Perf tests for chunking latency (<500ms) and small-to-big overhead (<100ms).

## Open Questions
- Should dual-level and small-to-big be first-class in the primary retrieval endpoints, or remain opt-in via dedicated routes/tools?
- What is the authoritative entity ID format between Graphiti and Neo4j (uuid vs id)?
- Which storage backend should host hierarchical chunks (Postgres, Neo4j, or vector DB)?
