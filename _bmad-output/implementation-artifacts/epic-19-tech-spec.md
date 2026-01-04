# Epic 19 Tech Spec: Advanced Retrieval Intelligence

**Date:** 2026-01-04
**Status:** Backlog
**Epic Owner:** Product and Engineering
**Origin:** Party Mode Competitive Analysis + Epic 12 Retro Carry-Forward

---

## Overview

Epic 19 consolidates advanced retrieval capabilities identified through competitive analysis and Epic 12 retrospective action items. This epic positions the platform to compete with Mem0, Zep, Microsoft GraphRAG, LightRAG, Cognee, and RAGFlow.

### Strategic Context

Competitive analysis (2026-01-04) revealed gaps against:
- **Mem0:** Memory scopes (user/session/agent), 91% lower latency
- **Zep/Graphiti:** Bi-temporal model, graph-based rerankers
- **MS GraphRAG:** Community detection, LazyRAG (99% cost reduction)
- **LightRAG:** Dual-level retrieval (low/high), multi-modal
- **Cognee:** Ontology support, self-improving feedback
- **RAGFlow:** Deep document understanding, parent-child chunks

### Goals

- Close competitive gaps in memory, graph intelligence, and retrieval quality
- Complete Epic 12 code review carry-forward items
- Establish retrieval quality benchmarks and monitoring
- Position platform as best-in-class hybrid RAG solution

---

## Story Groups

### Group A: Memory Platform (Compete with Mem0)

#### Story 19-A1: Implement Memory Scopes

**Objective:** Add user/session/agent memory differentiation.

**Scope:**
- User memory: Persists across all conversations with a specific person
- Session memory: Tracks context within a single conversation
- Agent memory: Stores information specific to an AI agent instance

**Configuration:**
```bash
MEMORY_SCOPES_ENABLED=true|false  # Default: false
MEMORY_DEFAULT_SCOPE=user|session|agent  # Default: session
MEMORY_USER_TTL_DAYS=365  # User memory retention
MEMORY_SESSION_TTL_HOURS=24  # Session memory retention
```

**Acceptance Criteria:**
- Memory API accepts scope parameter (user/session/agent)
- Retrieval filters by scope appropriately
- Cross-session personalization works for user scope
- Memory isolation is enforced between scopes

#### Story 19-A2: Implement Memory Consolidation

**Objective:** Reduce redundancy in stored memories over time.

**Acceptance Criteria:**
- Duplicate memories are detected and merged
- Conflicting memories are resolved (newer wins or flagged)
- Memory consolidation runs as background job
- Storage usage decreases after consolidation

---

### Group B: Graph Intelligence (Compete with MS GraphRAG)

#### Story 19-B1: Implement Community Detection

**Objective:** Detect graph communities for hierarchical summarization.

**Scope:**
- Louvain or Leiden algorithm for community detection
- Hierarchical community structure (multi-level)
- Pre-computed community summaries

**Configuration:**
```bash
COMMUNITY_DETECTION_ENABLED=true|false  # Default: false
COMMUNITY_ALGORITHM=louvain|leiden  # Default: louvain
COMMUNITY_RESOLUTION=1.0  # Higher = more communities
COMMUNITY_SUMMARY_MODEL=claude-3-haiku  # Cost-effective model
```

**Acceptance Criteria:**
- Communities are detected from entity/relationship graph
- Hierarchical levels are computed (document → section → entity)
- Community summaries are generated and cached
- "Big picture" queries route through community summaries

#### Story 19-B2: Implement LazyRAG Pattern

**Objective:** Defer summarization to query time for cost reduction.

**Acceptance Criteria:**
- Indexing stores raw graph without summaries
- Query-time summarization generates on-demand
- Caching prevents redundant summarization
- 90%+ indexing cost reduction measured

#### Story 19-B3: Implement Global vs Local Query Routing

**Objective:** Route queries to appropriate retrieval strategy.

**Acceptance Criteria:**
- Query classifier detects global vs local intent
- Global queries use community summaries
- Local queries use entity-focused retrieval
- Routing decision logged in trajectory

---

### Group C: Retrieval Excellence (Differentiation)

#### Story 19-C1: Implement Graph-Based Rerankers

**Objective:** Add Zep-style graph-aware reranking.

**Scope:**
- Episode-mentions reranker (frequency-based)
- Node-distance reranker (graph proximity from centroid)
- Hybrid scoring with existing cross-encoder

**Configuration:**
```bash
RERANKER_GRAPH_ENABLED=true|false  # Default: false
RERANKER_GRAPH_WEIGHT=0.3  # Weight vs semantic reranker
RERANKER_GRAPH_STRATEGY=mentions|distance|hybrid  # Default: hybrid
```

**Acceptance Criteria:**
- Graph-based scores are computed for retrieval results
- Scores are blended with cross-encoder scores
- Reranking logs include graph metrics
- A/B comparison shows improvement on relationship queries

#### Story 19-C2: Implement Dual-Level Retrieval

**Objective:** Add LightRAG-style low/high level retrieval.

**Scope:**
- Low-level: Specific entities, facts, relationships
- High-level: Themes, topics, summaries
- Query routing based on intent

**Acceptance Criteria:**
- Retrieval API accepts level parameter (low/high/auto)
- Auto-routing classifies query intent
- Results include level metadata
- Both levels can be combined for comprehensive answers

#### Story 19-C3: Implement Parent-Child Chunk Hierarchy

**Objective:** Link chunks to parent sections for context expansion.

**Scope:**
- Add parent_chunk_id to ChunkData model
- Add hierarchy_level field (0=doc, 1=section, 2=chunk)
- Implement "small-to-big" retrieval pattern

**Configuration:**
```bash
HIERARCHICAL_CHUNKS_ENABLED=true|false  # Default: false
HIERARCHICAL_EXPAND_LEVELS=1  # Levels to expand on retrieval
```

**Acceptance Criteria:**
- Chunks store parent references
- Retrieval can expand to parent context
- UI shows chunk-in-section context
- Re-indexing script updates existing chunks

#### Story 19-C4: Implement Retrieval Quality Benchmarks

**Objective:** Create evaluation framework for retrieval quality.

**Scope:**
- Evaluation dataset with labeled query-document pairs
- MRR@K (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Precision@K and Recall@K
- A/B comparison framework

**Acceptance Criteria:**
- Benchmark CLI command exists
- Metrics are computed and reported
- Baseline scores are established
- CI can run benchmarks on PRs

#### Story 19-C5: Implement Prometheus Metrics for Retrieval

**Objective:** Export retrieval quality metrics for monitoring.

**Metrics:**
```
retrieval_precision gauge
retrieval_recall gauge
reranking_improvement histogram
grader_score histogram
fallback_triggered_total counter
retrieval_latency_seconds histogram
```

**Acceptance Criteria:**
- Prometheus metrics endpoint exists
- All retrieval operations emit metrics
- Grafana dashboard templates provided
- Alerts configured for quality degradation

---

### Group D: Document Intelligence (RAGFlow Approach)

#### Story 19-D1: Enhance Table/Layout Extraction

**Objective:** Improve deep document understanding.

**Acceptance Criteria:**
- Tables extracted with structure preserved
- Layout hierarchy detected (headings, sections)
- Visual elements annotated
- Extraction quality metrics logged

#### Story 19-D2: Implement Multi-Modal Ingestion

**Objective:** Support images and rich document formats.

**Scope:**
- Image captioning for embedded images
- Office document support (docx, pptx, xlsx)
- Formula extraction from PDFs

**Acceptance Criteria:**
- Images in PDFs are captioned
- Office documents are parsed correctly
- Formulas are preserved in text form
- Multi-modal content is searchable

---

### Group E: Advanced Features (Cognee-Inspired)

#### Story 19-E1: Implement Ontology Support

**Objective:** Add domain ontology integration.

**Acceptance Criteria:**
- Ontology schema can be defined
- Entities are classified per ontology
- Inference rules are applied (is-a, part-of)
- Queries leverage ontology for expansion

#### Story 19-E2: Implement Self-Improving Feedback Loop

**Objective:** Learn from user feedback to improve retrieval.

**Acceptance Criteria:**
- Users can rate retrieval quality
- Feedback is stored with query context
- Model fine-tuning or prompt adjustment based on feedback
- Improvement metrics are tracked over time

---

### Group F: Epic 12 Code Review Carry-Forward

#### Story 19-F1: Add Full Retrieval Pipeline Integration Test

**Priority:** HIGH
**Origin:** Epic 12 Code Review (2026-01-04)

**Objective:** Test complete pipeline: rerank → grade → fallback.

**Acceptance Criteria:**
- Integration test covers full retrieval flow
- Test runs with all features enabled
- Edge cases covered (empty results, fallback trigger)
- Test is included in CI pipeline

#### Story 19-F2: Add Multi-Tenancy Enforcement Tests

**Priority:** HIGH
**Origin:** Epic 12 Code Review (2026-01-04)

**Objective:** Ensure tenant_id isolation in all retrieval paths.

**Acceptance Criteria:**
- Tests assert tenant_id is passed to all queries
- Cross-tenant data leakage is tested and fails
- All retrieval methods have tenant tests
- Security audit passes

#### Story 19-F3: Make CrossEncoderGrader Model Selectable

**Priority:** MEDIUM
**Origin:** Epic 12 Code Review (2026-01-04)

**Configuration:**
```bash
GRADER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2  # Default
```

**Acceptance Criteria:**
- Grader model is configurable via environment
- Multiple models are supported
- Model loading is lazy (on first use)
- Documentation lists available models

#### Story 19-F4: Make Heuristic Content Length Weight Configurable

**Priority:** MEDIUM
**Origin:** Epic 12 Code Review (2026-01-04)

**Configuration:**
```bash
GRADER_HEURISTIC_LENGTH_WEIGHT=0.5  # 0-1, how much length matters
GRADER_HEURISTIC_NORMALIZATION=1000  # Character normalization
```

**Acceptance Criteria:**
- Length weight is configurable
- Documentation explains the heuristic
- Tests cover different weight values

#### Story 19-F5: Add Contextual Retrieval Cost Logging

**Priority:** LOW
**Origin:** Epic 12 Code Review (2026-01-04)

**Acceptance Criteria:**
- Token usage is logged for contextual enrichment
- Cost estimates are computed per operation
- Aggregated cost metrics are available
- Dashboard shows contextual retrieval costs

---

### Group G: Epic 12 Retro Future Enhancements

#### Story 19-G1: Add Reranking Result Caching

**Origin:** Epic 12 Retro Nice-to-Do #1

**Acceptance Criteria:**
- Reranked results cached by query hash
- Cache TTL is configurable
- Cache hit rate is logged
- Repeated queries return faster

#### Story 19-G2: Make Context Generation Prompt Configurable

**Origin:** Epic 12 Retro Nice-to-Do #2

**Acceptance Criteria:**
- Contextual retrieval prompt is templated
- Template is configurable via settings
- Domain-specific prompts can be used
- Documentation provides examples

#### Story 19-G3: Add Cross-Encoder Model Preloading

**Origin:** Epic 12 Retro Nice-to-Do #3

**Configuration:**
```bash
GRADER_PRELOAD_MODEL=true|false  # Default: false
```

**Acceptance Criteria:**
- Model loads at startup when enabled
- First query latency is reduced
- Memory usage is documented
- Startup time impact is measured

#### Story 19-G4: Support Custom Normalization Strategies

**Origin:** Epic 12 Retro Nice-to-Do #4

**Acceptance Criteria:**
- Grader scoring algorithm is pluggable
- Multiple strategies are available
- Custom strategies can be registered
- Documentation explains each strategy

---

## Technical Notes

### Dependencies

- Graphiti (Epic 5) for graph operations
- pgvector (Epic 3) for vector search
- Reranking (Epic 12) for cross-encoder scoring
- CRAG Grader (Epic 12) for quality assessment

### Architecture Impact

- Memory scopes require schema changes to Graphiti episodes
- Community detection requires new graph algorithms
- Prometheus integration requires new middleware
- Multi-modal requires additional processing pipelines

### Implementation Order Recommendation

1. **Phase 1 (Critical Path):** 19-F1, 19-F2 (testing foundation)
2. **Phase 2 (Quick Wins):** 19-C3, 19-C5 (hierarchy, metrics)
3. **Phase 3 (Memory):** 19-A1, 19-A2 (compete with Mem0)
4. **Phase 4 (Graph):** 19-B1, 19-B3 (compete with GraphRAG)
5. **Phase 5 (Excellence):** 19-C1, 19-C2, 19-C4 (differentiation)
6. **Phase 6 (Polish):** 19-G*, 19-F3-F5 (code review items)
7. **Phase 7 (Advanced):** 19-D*, 19-E*, 19-B2 (optional)

---

## Risks

- Scope is large - consider splitting into Epic 19a/19b
- Community detection algorithms may have performance impact
- Memory scopes require careful tenant isolation
- LazyRAG may increase query latency

**Mitigation:** All features are opt-in via configuration flags.

---

## Success Metrics

- Competitive parity with Mem0 on memory features
- Competitive parity with GraphRAG on community queries
- 20%+ improvement on retrieval quality benchmarks
- 100% code coverage on retrieval pipeline
- Sub-500ms p95 retrieval latency maintained

---

## References

- `_bmad-output/implementation-artifacts/epic-12-retro-2026-01-04.md` - Origin of carry-forward items
- Party Mode Competitive Analysis (2026-01-04)
- [Mem0](https://github.com/mem0ai/mem0) | [Zep/Graphiti](https://github.com/getzep/graphiti)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | [LightRAG](https://github.com/HKUDS/LightRAG)
- [Cognee](https://github.com/topoteretes/cognee) | [RAGFlow](https://github.com/infiniflow/ragflow)
- [Weaviate](https://weaviate.io/) | [Qdrant](https://qdrant.tech/)
