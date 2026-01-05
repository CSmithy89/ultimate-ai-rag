# Story 20-C3: Implement Parent-Child Chunk Hierarchy

Status: done

## Story

As a developer building AI-powered applications,
I want parent-child chunk hierarchy with small-to-big retrieval,
so that queries match on precise small chunks but return larger parent chunks for complete context in LLM responses.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group C: Retrieval Excellence Features. It implements the small-to-big retrieval pattern that combines precise matching with rich context.

**Competitive Positioning**: This feature competes with advanced chunking strategies used in production RAG systems. The small-to-big pattern is a proven technique for improving retrieval quality while maintaining precise matching capability.

**Why This Matters**:
- **Precision vs. Context Trade-off:** Small chunks match queries precisely but lack context; large chunks provide context but match imprecisely. This feature gets both.
- **LLM Context Quality:** LLMs perform better with coherent, complete context rather than fragmented snippets
- **Reduced Hallucination:** Providing larger parent chunks reduces the chance of LLM hallucinating missing context
- **Flexibility:** Different retrieval use cases can choose which level to return (from sentence-level to section-level)

**Dependencies**:
- Epic 19 (Quality Foundation) - COMPLETED
- Epic 5 (Graphiti) - Provides graph storage for chunk relationships
- Story 20-C1 (Graph-Based Rerankers) - Can rerank parent chunks after retrieval
- Story 20-C2 (Dual-Level Retrieval) - Can combine small-to-big with dual-level
- Existing vector search infrastructure (Epic 3)
- Existing indexing pipeline (Epic 4)

**Enables**:
- Better answer quality for complex questions
- More complete context for synthesis
- Foundation for future document intelligence features (Group D)

## Acceptance Criteria

1. Given a document, when chunked hierarchically, then 4 levels are created (Level 0: 256 tokens, Level 1: 512 tokens, Level 2: 1024 tokens, Level 3: 2048 tokens).
2. Given a hierarchical chunk, when inspected, then child chunks reference their parent_id and parent chunks reference child_ids.
3. Given a query, when small-to-big retrieval runs, then small chunks (Level 0) are used for matching but parent chunks at SMALL_TO_BIG_RETURN_LEVEL are returned.
4. Given multiple matching small chunks with the same parent, when retrieval runs, then deduplication prevents returning overlapping/duplicate parents.
5. Given HIERARCHICAL_CHUNKS_ENABLED=true, when documents are ingested, then hierarchical chunks are created and stored.
6. Given HIERARCHICAL_CHUNKS_ENABLED=false (default), when documents are ingested, then standard single-level chunking is used.
7. Given configurable HIERARCHICAL_CHUNK_LEVELS, when set to custom values, then chunk sizes are adjusted accordingly.
8. Given configurable HIERARCHICAL_OVERLAP_RATIO, when chunks are created, then overlap between chunks matches the configured ratio.
9. Given configurable SMALL_TO_BIG_RETURN_LEVEL, when retrieval completes, then chunks at the specified level are returned.
10. All hierarchical chunk operations enforce tenant isolation via `tenant_id` filtering.
11. Hierarchical chunking adds <500ms total latency over standard chunking for typical documents (<10 pages).
12. Small-to-big retrieval adds <100ms latency over standard vector search.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/
+-- indexing/
|   +-- hierarchical_chunker.py       # NEW: Hierarchical chunk creation
|   +-- __init__.py                   # Update exports
+-- retrieval/
|   +-- small_to_big.py               # NEW: Small-to-big retriever
|   +-- __init__.py                   # Update exports
+-- db/
|   +-- chunk_store.py                # NEW or extend: Chunk storage with relationships
```

### Core Components

1. **HierarchicalChunk Dataclass** - Chunk with parent/child relationships:
   - `id`: Unique identifier
   - `content`: Chunk text content
   - `level`: Hierarchy level (0 = smallest, higher = larger)
   - `parent_id`: Reference to parent chunk (or None for top level)
   - `child_ids`: List of child chunk IDs
   - `metadata`: Additional metadata (document source, position, etc.)
   - `embedding`: Vector embedding (for Level 0 chunks only)

2. **HierarchicalChunker Class** - Creates hierarchical chunk trees from documents:
   - `__init__()` with configurable level_sizes and overlap_ratio
   - `chunk_document()` main method to create all levels
   - `_create_level_chunks()` creates chunks at a specific level
   - `_combine_to_level()` combines lower-level chunks into higher-level parents
   - `_tokenize()` helper for token-based splitting

3. **SmallToBigRetriever Class** - Retrieves small chunks but returns parent context:
   - `__init__()` with vector_search, chunk_store, and return_level config
   - `retrieve()` main method implementing small-to-big pattern
   - `_get_ancestor_at_level()` traverses up the hierarchy to find parent
   - `_deduplicate_by_parent()` removes overlapping parents

4. **ChunkStore Class** - Storage for hierarchical chunks (extend existing or create new):
   - `store_chunk()` stores a chunk with relationships
   - `get()` retrieves chunk by ID
   - `get_children()` retrieves child chunks
   - `get_parent()` retrieves parent chunk
   - `search_at_level()` searches chunks at a specific level

### Database Schema

```sql
-- PostgreSQL: Hierarchical chunks table
CREATE TABLE hierarchical_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    document_id UUID NOT NULL,
    content TEXT NOT NULL,
    level INTEGER NOT NULL DEFAULT 0,
    parent_id UUID,
    child_ids UUID[] DEFAULT '{}',
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- Only for Level 0
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT fk_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    CONSTRAINT fk_parent FOREIGN KEY (parent_id) REFERENCES hierarchical_chunks(id),
    CONSTRAINT chk_level CHECK (level >= 0 AND level <= 10)
);

-- Indexes for efficient hierarchy traversal
CREATE INDEX idx_chunks_tenant ON hierarchical_chunks(tenant_id);
CREATE INDEX idx_chunks_parent ON hierarchical_chunks(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX idx_chunks_level ON hierarchical_chunks(tenant_id, level);
CREATE INDEX idx_chunks_document ON hierarchical_chunks(tenant_id, document_id);

-- Vector index for Level 0 chunks only
CREATE INDEX idx_chunks_embedding ON hierarchical_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE level = 0 AND embedding IS NOT NULL;
```

```cypher
// Neo4j: Chunk hierarchy relationships
(:Chunk {
    id: String,
    content: String,
    level: Integer,
    tenantId: String,
    documentId: String,
    tokenCount: Integer
})

// Parent-child relationships
(:Chunk)-[:HAS_CHILD]->(:Chunk)
(:Chunk)-[:HAS_PARENT]->(:Chunk)

// Document relationship
(:Document)-[:HAS_CHUNK]->(:Chunk)
```

### Integration with Ingestion Pipeline

The hierarchical chunker integrates into the existing ingestion pipeline:

```
Document -> Parser -> HierarchicalChunker -> Level 0 Embeddings -> Storage
                                          -> Level 1-3 Storage (no embeddings)
```

When `HIERARCHICAL_CHUNKS_ENABLED=true`:
1. Parser extracts text from document
2. HierarchicalChunker creates all chunk levels
3. Level 0 chunks get embeddings generated
4. All chunks stored with parent/child relationships

When `HIERARCHICAL_CHUNKS_ENABLED=false`:
1. Standard single-level chunking (existing behavior)

### Configuration

```bash
# Epic 20 - Parent-Child Chunks
HIERARCHICAL_CHUNKS_ENABLED=true|false       # Default: false
HIERARCHICAL_CHUNK_LEVELS=256,512,1024,2048  # Token sizes per level (comma-separated)
HIERARCHICAL_OVERLAP_RATIO=0.1               # Overlap between chunks (0.0-0.5)
SMALL_TO_BIG_RETURN_LEVEL=2                  # Level to return (0-3, default: 2 = 1024 tokens)
HIERARCHICAL_EMBEDDING_LEVEL=0               # Which level gets embeddings (default: 0 = smallest)
```

### API Integration

When hierarchical chunks are enabled, retrieval responses include hierarchy info:

```json
{
  "data": {
    "query": "What is the company's revenue?",
    "results": [
      {
        "id": "chunk_2_5",
        "content": "The company's 2024 annual revenue reached $5.2 billion, representing a 15% increase from...",
        "level": 2,
        "matched_child_ids": ["chunk_0_18", "chunk_0_19"],
        "token_count": 1024,
        "document_id": "doc_123",
        "score": 0.87
      }
    ],
    "retrieval_mode": "small_to_big",
    "matched_at_level": 0,
    "returned_at_level": 2
  },
  "meta": {"requestId": "...", "timestamp": "..."}
}
```

## Tasks / Subtasks

### Indexing Module

- [ ] Create `hierarchical_chunker.py` module (`backend/src/agentic_rag_backend/indexing/hierarchical_chunker.py`)
- [ ] Implement HierarchicalChunk dataclass
  - [ ] Fields: id, content, level, parent_id, child_ids, metadata, embedding, token_count
  - [ ] Method: `to_dict()` for serialization
  - [ ] Method: `from_dict()` for deserialization
- [ ] Implement HierarchicalChunker class
  - [ ] `__init__()` with level_sizes and overlap_ratio configuration
  - [ ] `chunk_document()` main orchestration method
  - [ ] `_create_level_chunks()` for initial chunking at level 0
  - [ ] `_combine_to_level()` for creating parent chunks from children
  - [ ] `_tokenize()` helper using tiktoken for accurate token counting
  - [ ] `_generate_chunk_id()` for unique, deterministic IDs
- [ ] Add proper parent/child relationship linking in chunk creation
- [ ] Update `indexing/__init__.py` exports

### Storage Module

- [ ] Create or extend chunk store (`backend/src/agentic_rag_backend/db/chunk_store.py`)
- [ ] Implement ChunkStore class (or extend existing)
  - [ ] `store_chunk()` with relationship handling
  - [ ] `store_chunk_batch()` for efficient bulk storage
  - [ ] `get()` retrieve chunk by ID
  - [ ] `get_children()` retrieve child chunks by parent_id
  - [ ] `get_parent()` retrieve parent chunk
  - [ ] `get_at_level()` retrieve chunks at specific level for document
- [ ] Create PostgreSQL migration for hierarchical_chunks table
- [ ] Add Neo4j Cypher queries for chunk relationships
- [ ] Ensure tenant isolation in all queries

### Retrieval Module

- [ ] Create `small_to_big.py` module (`backend/src/agentic_rag_backend/retrieval/small_to_big.py`)
- [ ] Implement SmallToBigRetriever class
  - [ ] `__init__()` with vector_search, chunk_store, return_level
  - [ ] `retrieve()` main method implementing small-to-big pattern
  - [ ] `_search_small_chunks()` search at Level 0
  - [ ] `_get_ancestor_at_level()` traverse hierarchy to target level
  - [ ] `_deduplicate_by_parent()` remove overlapping parents
  - [ ] `_enrich_results()` add matched child info to returned chunks
- [ ] Implement SmallToBigAdapter for configuration and feature flag handling
- [ ] Update `retrieval/__init__.py` exports

### Configuration Module

- [ ] Add configuration variables to settings (`backend/src/agentic_rag_backend/core/config.py`)
  - [ ] HIERARCHICAL_CHUNKS_ENABLED (bool, default: false)
  - [ ] HIERARCHICAL_CHUNK_LEVELS (list[int], default: [256, 512, 1024, 2048])
  - [ ] HIERARCHICAL_OVERLAP_RATIO (float, default: 0.1)
  - [ ] SMALL_TO_BIG_RETURN_LEVEL (int, default: 2)
  - [ ] HIERARCHICAL_EMBEDDING_LEVEL (int, default: 0)
- [ ] Add validation for level sizes (must be increasing)
- [ ] Add validation for overlap ratio (0.0-0.5)
- [ ] Add validation for return_level (must be within level count)

### Pipeline Integration

- [ ] Integrate HierarchicalChunker into document ingestion pipeline
- [ ] Add feature flag check for hierarchical vs. standard chunking
- [ ] Update embedding generation to only embed Level 0 chunks
- [ ] Store chunk hierarchy relationships during ingestion

### API Integration

- [ ] Add small-to-big option to retrieval endpoint
- [ ] Update retrieval response schema for hierarchy info
- [ ] Add `/api/v1/chunks/{id}/hierarchy` endpoint for hierarchy traversal
- [ ] Add status endpoint for hierarchical chunking configuration

### Testing

- [ ] Write unit tests for HierarchicalChunk dataclass
- [ ] Write unit tests for HierarchicalChunker
  - [ ] Test chunk creation at each level
  - [ ] Test parent-child relationship linking
  - [ ] Test overlap handling
  - [ ] Test edge cases (very short/long documents)
- [ ] Write unit tests for SmallToBigRetriever
  - [ ] Test retrieval with mock vector search and chunk store
  - [ ] Test deduplication logic
  - [ ] Test ancestor traversal
- [ ] Write unit tests for ChunkStore
- [ ] Write integration tests with real storage
- [ ] Write performance tests for chunking latency
- [ ] Write performance tests for retrieval latency
- [ ] Test tenant isolation in all operations

### Documentation

- [ ] Update .env.example with hierarchical chunking configuration
- [ ] Add docstrings to all public classes and methods
- [ ] Document hierarchy traversal patterns in code comments

## Testing Requirements

### Unit Tests

- HierarchicalChunk dataclass serialization/deserialization
- HierarchicalChunker initialization with various level configurations
- Chunk creation at Level 0 with correct token counts
- Parent chunk creation from child combination
- Parent-child reference linking correctness
- Overlap handling at chunk boundaries
- Edge cases: empty document, single sentence, very long document
- SmallToBigRetriever initialization
- Small chunk search with mock vector search
- Ancestor traversal to target level
- Deduplication of overlapping parents
- ChunkStore CRUD operations
- Feature flag toggle behavior

### Integration Tests

- End-to-end document ingestion with hierarchical chunking
- Retrieval using small-to-big pattern with real storage
- Hierarchy traversal via chunk relationships
- Multi-document ingestion with proper isolation
- Tenant isolation: Cross-tenant chunk access returns empty

### Performance Tests

- Hierarchical chunking latency < 500ms for typical documents (<10 pages)
- Small-to-big retrieval latency < 100ms over standard search
- Bulk document ingestion with hierarchical chunking
- Memory usage during chunking for large documents
- Storage overhead comparison (hierarchical vs. standard)

### Security Tests

- Tenant isolation in PostgreSQL queries (parameterized tenant_id)
- Tenant isolation in Neo4j queries
- Input validation for configuration values
- Chunk ID validation to prevent traversal attacks

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] All tasks completed
- [ ] Unit test coverage >= 80% for hierarchical_chunker and small_to_big modules
- [ ] Integration tests pass with real storage backends
- [ ] Performance targets met:
  - [ ] Chunking latency < 500ms for typical documents
  - [ ] Retrieval latency < 100ms over baseline
- [ ] Configuration documented in .env.example
- [ ] Feature flag (HIERARCHICAL_CHUNKS_ENABLED) works correctly
- [ ] Parent-child relationships correctly stored and traversable
- [ ] Deduplication prevents duplicate parent returns
- [ ] Multi-tenancy enforced in all operations
- [ ] Code review approved
- [ ] No regressions in existing ingestion or retrieval tests
- [ ] Database migrations tested (PostgreSQL and Neo4j)

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-C3 section)
- Use tiktoken for accurate token counting (matches OpenAI tokenization)
- Only Level 0 chunks need embeddings (saves storage and computation)
- Consider batch embedding generation for Level 0 chunks
- Use asyncio.gather for parallel chunk storage when possible
- Chunk IDs should be deterministic for idempotent re-ingestion
- Consider caching frequently-accessed parent chunks

### Small-to-Big Pattern Explanation

The small-to-big retrieval pattern solves a fundamental RAG trade-off:

1. **Small chunks** (256 tokens) are ideal for precise semantic matching
   - Query: "What was Q4 2024 revenue?"
   - Match: Exact sentence with the number

2. **Large chunks** (1024+ tokens) are ideal for LLM context
   - Include surrounding sentences for context
   - Avoid mid-sentence cutoffs
   - Provide complete thought units

3. **Small-to-Big** combines both:
   - Search on small chunks for precision
   - Return parent chunks for context
   - Best of both worlds

### Level Size Guidelines

| Level | Tokens | Use Case |
|-------|--------|----------|
| 0 | 256 | Sentence-level matching (embeddings here) |
| 1 | 512 | Paragraph-level context |
| 2 | 1024 | Section-level context (default return level) |
| 3 | 2048 | Sub-document context |

### Overlap Strategy

- 10% overlap (default) prevents context loss at chunk boundaries
- Higher overlap = more storage but better boundary handling
- Overlap applies at each level independently

### Performance Considerations

- Only Level 0 chunks get embeddings (storage optimization)
- Batch chunk storage for efficiency
- Use connection pooling for database operations
- Consider Redis caching for hot parent chunks
- Async traversal for hierarchy lookup

### References

- `_bmad-output/epics/epic-20-tech-spec.md` (Group C: Retrieval Excellence)
- `backend/src/agentic_rag_backend/retrieval/dual_level.py` (20-C2 for integration pattern)
- `backend/src/agentic_rag_backend/indexing/` (existing ingestion pipeline)
- `backend/src/agentic_rag_backend/db/pgvector.py` (vector storage)
- [LlamaIndex Small-to-Big Retrieval](https://docs.llamaindex.ai/en/stable/examples/retrievers/recursive_retriever_nodes.html)
- [RAG Best Practices: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/)
