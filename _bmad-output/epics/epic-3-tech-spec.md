# Epic 3 Tech Spec: Hybrid Knowledge Retrieval

**Date:** 2025-12-29
**Status:** Complete
**Epic Owner:** Development Team

---

## 1. Overview

Epic 3 delivers hybrid retrieval by combining vector semantic search (pgvector)
with Neo4j relationship traversal, then synthesizing a single, citeable answer.
This replaces Epic 2's retrieval stubs with real evidence pipelines.

### 1.1 Epic Goals

- Provide semantic search over pgvector chunk embeddings.
- Provide relationship traversal across the Neo4j knowledge graph.
- Combine vector + graph evidence into a single answer.
- Expose explainability artifacts (nodes, edges, paths) in API responses.

### 1.2 Functional Requirements Covered

| FR ID | Description |
|-------|-------------|
| FR11 | Vector semantic search |
| FR12 | Graph relationship traversal |
| FR13 | Hybrid answer synthesis |
| FR14 | Graph-based explainability |

### 1.3 Non-Functional Requirements Addressed

| NFR ID | Target | Epic 3 Impact |
|--------|--------|---------------|
| NFR1 | < 10s end-to-end response | Retrieval runs in-process with capped limits |
| NFR5 | 1M+ nodes/edges | Neo4j traversal uses bounded hops and limits |

### 1.4 Non-Goals

- Ingestion pipeline (Epic 4).
- Temporal knowledge graph (Epic 5).
- Copilot UI surfaces (Epic 6).
- Ops dashboards (Epic 8).

---

## 2. Architecture Decisions

### 2.1 Retrieval Modules

Add a dedicated retrieval package under `backend/src/agentic_rag_backend/retrieval/`:

- `vector_search.py` for pgvector semantic similarity.
- `graph_traversal.py` for Neo4j relationship traversal.
- `hybrid_synthesis.py` for combining evidence + answer prompt.

### 2.2 Evidence-First Response

The `/query` endpoint response includes evidence alongside the answer:

- Vector citations: chunk id, document id, similarity score, source metadata.
- Graph citations: nodes, edges, and traversed paths.
- Human-readable path explanation for trust and debugging.

### 2.3 Bounded Traversal

Graph traversal uses:

- Max hops: 2 (configurable)
- Max paths: 10 (configurable)
- Allowed relationships: MENTIONS, AUTHORED_BY, PART_OF, USES, RELATED_TO
- Tenant isolation enforced by `tenant_id` on every query

### 2.4 Hybrid Synthesis Prompt

Hybrid synthesis uses a structured prompt that:

- Injects vector evidence (top-k chunks).
- Injects graph evidence (paths + node details).
- Requests citations in the answer (vector chunks and graph entities).

---

## 3. Retrieval Flow

```text
User Query
   │
   ▼
Retrieval Router (Epic 2)
   │
   ├── vector → pgvector cosine similarity
   ├── graph  → Neo4j traversal
   └── hybrid → both
   │
   ▼
Evidence Aggregation
   │
   ▼
Answer Synthesis (LLM)
   │
   ▼
API Response + Evidence + Explainability
```

---

## 4. API Updates

### `/query` Response Enhancements

Extend `QueryResponse` with optional evidence:

- `evidence.vector[]`: similarity hits with source references.
- `evidence.graph`: nodes, edges, paths, explanation.

---

## 5. Story Breakdown with Technical Approach

### 5.1 Story 3.1: Vector Semantic Search

**As a** user,
**I want** to search for information using semantic similarity,
**So that** I can find relevant content even when exact keywords don't match.

#### Technical Approach

1. Generate query embedding via `EmbeddingGenerator`.
2. Use `PostgresClient.search_similar_chunks` to fetch top-k chunks.
3. Return results with similarity score and source metadata.

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/retrieval/vector_search.py` | Vector retrieval logic |
| `backend/src/agentic_rag_backend/schemas.py` | Vector evidence schema |

#### Acceptance Criteria

- [ ] Query embedding generated successfully.
- [ ] Cosine similarity search returns top-k results.
- [ ] Results include similarity score and source reference.

---

### 5.2 Story 3.2: Graph Relationship Traversal

**As a** user,
**I want** to query relationships between entities in the knowledge graph,
**So that** I can discover connections that semantic search alone would miss.

#### Technical Approach

1. Extract candidate entity terms from the query.
2. Find matching entities in Neo4j with tenant filtering.
3. Traverse relationships using bounded Cypher paths.
4. Return nodes, edges, and path sequences.

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/retrieval/graph_traversal.py` | Graph traversal logic |
| `backend/src/agentic_rag_backend/db/neo4j.py` | Retrieval queries |

#### Acceptance Criteria

- [ ] Starting entities identified from query terms.
- [ ] Neo4j traversal returns connected entities and paths.
- [ ] Tenant isolation enforced in every query.

---

### 5.3 Story 3.3: Hybrid Answer Synthesis

**As a** user,
**I want** the system to combine vector and graph results into a coherent answer,
**So that** I get comprehensive responses leveraging both retrieval methods.

#### Technical Approach

1. Run vector search and graph traversal in the same request.
2. Merge evidence into a single prompt for the LLM.
3. Produce a unified answer with citations.

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/retrieval/hybrid_synthesis.py` | Evidence aggregation + prompt |
| `backend/src/agentic_rag_backend/agents/orchestrator.py` | Orchestrator integration |

#### Acceptance Criteria

- [ ] Evidence from both sources is combined.
- [ ] Answer is synthesized into a single response.
- [ ] Citations reference vector chunks and graph entities.

---

### 5.4 Story 3.4: Graph-Based Explainability

**As a** user,
**I want** to see how the system arrived at its answer using graph connections,
**So that** I can verify the reasoning and trust the response.

#### Technical Approach

1. Include nodes and edges in query response evidence.
2. Generate a human-readable explanation of graph paths.
3. Provide IDs suitable for frontend graph exploration.

#### Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/src/agentic_rag_backend/schemas.py` | Explainability schema |
| `backend/src/agentic_rag_backend/retrieval/graph_traversal.py` | Path explanation |

#### Acceptance Criteria

- [ ] Response includes referenced nodes and relationship edges.
- [ ] Human-readable path explanation is included.
- [ ] Graph evidence is available for UI exploration.
