# Advanced RAG Architecture Decision

**Date:** 2026-01-07  
**Scope:** Epic 20 integration and long-term production RAG architecture

## Decision Summary

1. **Single canonical retrieval pipeline**  
   `/query` (Orchestrator) is the source of truth. MCP tools and specialized endpoints become thin wrappers that call the same pipeline.

2. **Graphiti as the primary graph substrate**  
   Align Neo4j `Entity.id` with Graphiti `uuid` (store both if needed for backward compatibility). Avoid ID drift.

3. **Hierarchical chunks stored in Postgres**  
   Use pgvector-backed tables for hierarchical chunks and parent/child relations. Keep retrieval fast and consistent with existing vector search.

4. **Dual-level weights affect ranking**  
   Weights must influence score ordering, not just confidence.

5. **Performance guardrails**  
   Batch graph distance queries, enforce candidate caps, cache hot queries, and route GLOBAL queries to LazyRAG for speed.

## Rationale

- **Consistency:** A single pipeline prevents feature drift and ensures Epic 20 benefits apply to all entrypoints.
- **Performance:** pgvector + Graphiti is the fastest reliable hybrid in this stack.
- **Accuracy:** Cross-encoder + graph signals provide complementary ranking improvements.
- **Operational simplicity:** Centralized flags and one retrieval path reduce configuration risk.

## Architecture (Target)

```
Query
  -> QueryRouter (GLOBAL / LOCAL / HYBRID)
  -> Candidate Generation
     - LOCAL: pgvector + Graphiti entity search
     - GLOBAL: LazyRAG + communities
     - HYBRID: dual-level
  -> Context Assembly
     - small-to-big parent chunks
     - graph traversal evidence
  -> Scoring
     - cross-encoder reranker
     - graph reranker
  -> Grader fallback (CRAG)
  -> Synthesis
```

## Implementation Phases

**Phase 1: Unified pipeline wiring**
- Introduce RetrievalPipeline service.
- Orchestrator uses the pipeline.
- MCP tools call the pipeline.

**Phase 2: Hierarchical chunk storage + ingestion**
- Add hierarchical chunks table + CRUD.
- Wire hierarchical chunker into ingestion.
- Enable small-to-big retrieval.

**Phase 3: Epic 20 feature integrations**
- Graph reranker after cross-encoder.
- Dual-level as a routing mode.
- LazyRAG for global queries.

Status: ✅ Phase 3 implemented in the retrieval pipeline and Orchestrator.

**Phase 4: Verification**
- Integration tests for routing + dual-level + graph reranking.
- Performance benchmarks for SLAs.

Status: 🔄 Targeted retrieval tests executed; full suite pending due to runtime.

## Risks and Mitigations

- **ID drift between Graphiti and Neo4j**  
  Mitigation: store Graphiti UUID in Neo4j `Entity.id`, or maintain a mapping table.

- **Latency creep from graph operations**  
  Mitigation: batch shortest-path queries, cap candidate sizes, and use caching.

- **Feature regressions during unification**  
  Mitigation: keep wrappers and dedicated endpoints as thin adapters over the pipeline.
