# Story 20-H1: Implement Sparse Vector Search (BM42)

Status: done

## Story

As a developer building search applications,
I want to combine dense and sparse vector search using BM42,
so that I get both semantic understanding and keyword-matching precision.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group H: Competitive Features. It implements sparse vector search using BM42 (based on Qdrant's approach), enabling:

- **Sparse Vectors**: Term-weighted vectors that capture keyword importance
- **BM42 Encoding**: Modern sparse embedding using attention-based term weighting
- **Hybrid Search**: Combine dense (semantic) and sparse (lexical) search
- **RRF Fusion**: Reciprocal Rank Fusion for combining result sets

**Competitive Positioning**: Qdrant pioneered BM42 for sparse vectors. This gives us hybrid search capabilities matching enterprise vector databases.

**Dependencies**:
- fastembed library for BM42 encoding
- Existing dense vector search infrastructure
- PostgreSQL for storing sparse vectors

## Acceptance Criteria

1. Given text, when encoded with BM42, then sparse vectors are generated.
2. Given a query, when hybrid search is used, then dense and sparse results are combined.
3. Given configurable weights, when RRF fusion is applied, then results are properly ranked.
4. Given SPARSE_VECTORS_ENABLED=false (default), when the system starts, then sparse features are not active.
5. All search operations enforce tenant isolation via `tenant_id` filtering.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/retrieval/
+-- sparse_vectors.py              # NEW: BM42 encoder and hybrid search
```

### Core Components

1. **SparseVector Dataclass** - Sparse vector representation:
   - indices: list[int] - Non-zero positions
   - values: list[float] - Corresponding weights

2. **BM42Encoder Class** - Sparse vector encoder:
   - Uses fastembed's SparseTextEmbedding
   - `encode()` - Batch encode texts
   - `encode_query()` - Single query encoding

3. **HybridVectorSearch Class** - Combined search:
   - Wraps dense search and sparse encoder
   - `search()` - Hybrid search with RRF fusion
   - `_reciprocal_rank_fusion()` - RRF algorithm
   - `_sparse_search()` - Sparse vector similarity search

4. **SparseVectorAdapter Class** - Feature flag wrapper

### Configuration

```bash
SPARSE_VECTORS_ENABLED=true|false            # Default: false
SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions
HYBRID_DENSE_WEIGHT=0.7                      # Default: 0.7
HYBRID_SPARSE_WEIGHT=0.3                     # Default: 0.3
```

## Tasks / Subtasks

- [x] Add fastembed to pyproject.toml dependencies
- [x] Create SparseVector dataclass
- [x] Implement BM42Encoder class
- [x] Implement HybridVectorSearch class with RRF
- [x] Implement SparseVectorAdapter with feature flag
- [x] Add configuration variables to settings
  - [x] SPARSE_VECTORS_ENABLED
  - [x] SPARSE_MODEL
  - [x] HYBRID_DENSE_WEIGHT
  - [x] HYBRID_SPARSE_WEIGHT
- [x] Export from retrieval/__init__.py
- [x] Write unit tests for SparseVector
- [x] Write unit tests for BM42Encoder
- [x] Write unit tests for HybridVectorSearch
- [x] Write unit tests for SparseVectorAdapter

## Testing Requirements

### Unit Tests
- SparseVector creation and serialization
- BM42Encoder encoding (with mock)
- Hybrid search with RRF fusion
- Feature flag behavior
- Tenant isolation in search

### Integration Tests
- End-to-end hybrid search with real embeddings

## Definition of Done

- [x] All acceptance criteria pass
- [x] All tasks completed
- [x] Unit test coverage >= 80% for sparse vectors module
- [x] Feature flag (SPARSE_VECTORS_ENABLED) works correctly
- [x] Configuration documented
- [x] Code review approved
- [x] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-H1 section)
- BM42 uses attention-based term weighting (better than BM25)
- RRF constant k=60 is standard (per Reciprocal Rank Fusion paper)
- Sparse vectors stored with indices/values format for efficiency
- fastembed downloads models on first use (~100MB)

### References

- `_bmad-output/epics/epic-20-tech-spec.md` (Group H: Competitive Features)
- [Qdrant BM42 Documentation](https://qdrant.tech/documentation/concepts/vectors/#sparse-vectors)
- [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)

---

## Dev Agent Record

### File List

| File | Action | Description |
|------|--------|-------------|
| `backend/pyproject.toml` | MODIFIED | Add fastembed dependency |
| `backend/src/agentic_rag_backend/retrieval/sparse_vectors.py` | NEW | SparseVector, BM42Encoder, HybridVectorSearch, SparseVectorAdapter |
| `backend/src/agentic_rag_backend/retrieval/__init__.py` | MODIFIED | Export sparse vector components |
| `backend/src/agentic_rag_backend/config.py` | MODIFIED | Add sparse vector settings |
| `backend/tests/retrieval/__init__.py` | NEW or EXISTING | Test module init |
| `backend/tests/retrieval/test_sparse_vectors.py` | NEW | Unit tests for sparse vectors |

### Change Log

| Date | Change | Details |
|------|--------|---------|
| 2026-01-06 | Initial implementation | Created sparse_vectors.py with SparseVector, BM42Encoder, HybridVectorSearch, SparseVectorAdapter |
| 2026-01-06 | Configuration | Added SPARSE_VECTORS_ENABLED, SPARSE_MODEL, HYBRID_DENSE_WEIGHT, HYBRID_SPARSE_WEIGHT to config.py |
| 2026-01-06 | Tests | Created test_sparse_vectors.py with 45 unit tests |
| 2026-01-06 | Code review fixes | Updated DoD checkboxes, status, change log
