# Story 5.3: Hybrid Retrieval Integration

Status: done

## Story

As a user,
I want the AI copilot to retrieve knowledge using Graphiti's hybrid search,
so that I get more accurate and contextually relevant answers.

## Acceptance Criteria

1. Given a user query, when the orchestrator retrieves context, then Graphiti's hybrid search (semantic + BM25 + graph traversal) is used.
2. Given search results are returned, when they are ranked, then semantic similarity, keyword relevance, and graph proximity are combined.
3. Given the query mentions specific entities, when graph traversal runs, then related entities and their relationships are included in context.
4. Given a search completes, when latency is measured, then it is under 100ms (excluding LLM calls).
5. Given the retrieval strategy selector chooses "graph", when retrieval runs, then Graphiti's graph-based search is prioritized.

## Standards Coverage

- [ ] Multi-tenancy / tenant isolation: Planned - tenant-scoped retrieval queries
- [ ] Rate limiting / abuse protection: N/A - internal retrieval pipeline
- [ ] Input validation / schema enforcement: Planned - validate retrieval inputs
- [ ] Tests (unit/integration): Planned - add retrieval tests
- [ ] Error handling + logging: Planned - consistent retrieval errors and logs
- [ ] Documentation updates: Planned - document retrieval strategy

## Tasks / Subtasks

- [ ] Create Graphiti retrieval service (AC: 1, 2, 4)
  - [ ] Add `backend/src/agentic_rag_backend/retrieval/graphiti_retrieval.py`
  - [ ] Implement `hybrid_search()` function using Graphiti SDK
  - [ ] Configure search parameters (semantic weight, BM25 weight, graph depth)
  - [ ] Add latency tracking and logging

- [ ] Integrate with orchestrator agent (AC: 1, 3)
  - [ ] Update `backend/src/agentic_rag_backend/agents/orchestrator.py`
  - [ ] Replace/augment existing retrieval with Graphiti search
  - [ ] Pass retrieved entities and relationships to LLM context
  - [ ] Maintain trajectory logging for retrieval steps

- [ ] Update retrieval router for Graphiti (AC: 5)
  - [ ] Modify `backend/src/agentic_rag_backend/retrieval_router.py`
  - [ ] Add Graphiti-specific retrieval strategy
  - [ ] Route based on query characteristics

- [ ] Implement graph traversal for entity context (AC: 3)
  - [ ] Add `get_entity_neighborhood()` function
  - [ ] Retrieve N-hop neighbors for mentioned entities
  - [ ] Include relationship types and temporal context
  - [ ] Limit traversal depth for latency

- [ ] Add search quality benchmarking (AC: 2, 4)
  - [ ] Create benchmark dataset with known answers
  - [ ] Compare Graphiti vs legacy retrieval quality
  - [ ] Measure latency p50, p95, p99
  - [ ] Document results in test report

- [ ] Write unit and integration tests (AC: 1-5)
  - [ ] Add `backend/tests/retrieval/test_graphiti_retrieval.py`
  - [ ] Test hybrid search configuration
  - [ ] Test graph traversal depth limiting
  - [ ] Test latency requirements

## Technical Notes

### Hybrid Search Pattern

```python
async def hybrid_search(
    graphiti: Graphiti,
    query: str,
    tenant_id: str,
    limit: int = 10,
) -> SearchResults:
    results = await graphiti.search(
        query=query,
        group_ids=[tenant_id],
        num_results=limit,
    )
    return results
```

### Graph Traversal

```python
async def get_entity_neighborhood(
    graphiti: Graphiti,
    entity_name: str,
    max_depth: int = 2,
) -> list[EntityNode]:
    # Graphiti handles graph traversal internally
    # Results include related entities via edges
    pass
```

## Definition of Done

- [ ] Graphiti hybrid search integrated with orchestrator
- [ ] Search results combine semantic + BM25 + graph scores
- [ ] Graph traversal includes entity neighborhoods
- [ ] Search latency < 100ms (measured)
- [ ] Retrieval quality >= baseline (benchmarked)
- [ ] All tests passing
- [ ] Code reviewed and merged
