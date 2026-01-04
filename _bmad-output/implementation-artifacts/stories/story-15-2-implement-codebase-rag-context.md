# Story 15-2: Implement Codebase RAG Context

**Status:** done
**Epic:** 15 - Codebase Intelligence
**Priority:** High
**Complexity:** High (5-7 days estimated)

---

## User Story

As a **developer using the RAG platform**,
I want **the codebase indexed as a first-class knowledge source**,
So that **I can ask natural language questions about functions, classes, and data flow and receive grounded code context**.

---

## Background

The platform already supports RAG over documents and knowledge graph content. However, developers need the same retrieval capabilities over their codebase. This story introduces code indexing, chunking, embedding, and retrieval for code symbols, while capturing key relationships (calls/imports/defined-in) in the knowledge graph.

---

## Acceptance Criteria

- [ ] Given a code repository, when indexing runs, then symbols are extracted and embedded
- [ ] Symbol relationships are captured in the knowledge graph (e.g., function A calls function B)
- [ ] Queries about the codebase return relevant code context
- [ ] Indexing respects .gitignore and configurable exclusion patterns
- [ ] Incremental indexing is supported for changed files
- [ ] Indexing completes within 5 minutes for a 100K LOC repository
- [ ] Search results include file paths, line numbers, and code snippets

---

## Technical Details

### API Endpoints

```
POST /api/v1/codebase/index
POST /api/v1/codebase/search
```

### Configuration

```
CODEBASE_RAG_ENABLED=true|false
CODEBASE_LANGUAGES=python,typescript,javascript
CODEBASE_EXCLUDE_PATTERNS=["**/node_modules/**","**/__pycache__/**"]
CODEBASE_MAX_CHUNK_SIZE=1000
CODEBASE_INCLUDE_CLASS_CONTEXT=true
CODEBASE_INCREMENTAL_INDEXING=true
CODEBASE_INDEX_CACHE_TTL_SECONDS=86400
```

---

## Implementation Tasks

### Phase 1: Indexing Pipeline

- [ ] Implement file scanner with .gitignore + exclude patterns
- [ ] Reuse AST symbol extraction from Story 15-1
- [ ] Create code chunker (symbol-level chunks + class context)
- [ ] Generate embeddings and store chunks in pgvector
- [ ] Cache symbol tables in Redis (reuse Story 15-1 cache utilities)

### Phase 2: Graph Relationships

- [ ] Build relationships: CALLS, IMPORTS, DEFINED_IN
- [ ] Create entities for symbols/modules/files in Neo4j
- [ ] Store relationships with confidence metadata

### Phase 3: Retrieval

- [ ] Add code search service (vector search + optional graph enrichment)
- [ ] Return file paths, line ranges, and snippets in results

### Phase 4: API + Config

- [ ] Add /codebase/index and /codebase/search endpoints
- [ ] Add CODEBASE_* RAG config to Settings and .env.example

---

## Testing Requirements

### Unit Tests

- [ ] File scanner respects .gitignore and exclude patterns
- [ ] Chunker includes class context for method chunks
- [ ] Graph builder extracts CALLS and IMPORTS relationships

### Integration Tests

- [ ] Index a small repo and ensure chunks are stored in pgvector
- [ ] Search returns code snippets with file paths and line numbers

---

## Definition of Done

- [ ] Codebase indexing creates embeddings + stores chunks
- [ ] Graph relationships stored in Neo4j for code entities
- [ ] Search endpoint returns relevant code context
- [ ] Incremental indexing supported via cached file mtimes
- [ ] Config variables documented in .env.example
- [ ] Tests added for scanner/chunker/relationships

---

## Implementation Notes (2026-01-04)

### Modules Added

- `backend/src/agentic_rag_backend/codebase/indexing/`
  - `scanner.py`: File scanning with .gitignore + exclude patterns
  - `chunker.py`: Symbol-level chunking with optional class context
  - `graph_builder.py`: CALLS/IMPORTS relationship extraction
  - `indexer.py`: Orchestrates scanning, embedding, storage, graph writes
- `backend/src/agentic_rag_backend/codebase/retrieval/code_search.py`
  - Vector search wrapper with optional graph relationship enrichment

### API + Config

- Added `POST /api/v1/codebase/index` and `POST /api/v1/codebase/search`
- Added CODEBASE_RAG_* settings to `config.py` and `.env.example`

### Storage + Graph

- Code chunks stored in existing pgvector `chunks` table with metadata:
  `source_type=codebase`, `file_path`, `symbol_name`, `line_start`, `line_end`
- Neo4j relationship types extended to include `CALLS`, `IMPORTS`, `DEFINED_IN`

### Tests

- Added unit tests for scanner, chunker, and graph builder

---

## Senior Developer Review (2026-01-04)

### Review Summary

Implementation covers indexing, chunking, embedding, and retrieval paths with Neo4j relationship storage. A few correctness gaps were found around incremental indexing and call-graph extraction; fixes applied below.

### Issues Found

1. **[HIGH] Incremental Indexing Overwrote Cached Symbol Table**
   - File: `/backend/src/agentic_rag_backend/codebase/indexing/indexer.py`
   - Problem: Incremental indexing rebuilt a partial symbol table from changed files and cached it, discarding existing symbols.
   - Fix: Load cached symbol table from Redis, remove changed files, then merge new symbols; added `SymbolTable.remove_file()` helper.

2. **[MEDIUM] Vector Search Limit Ignored Request Limit**
   - File: `/backend/src/agentic_rag_backend/codebase/retrieval/code_search.py`
   - Problem: Vector search used a fixed default limit, causing truncated results for larger queries.
   - Fix: Adjusted vector search limit to `max(limit*2, default)`.

3. **[MEDIUM] Call Graph Extraction Flagged Function Definitions as Calls**
   - File: `/backend/src/agentic_rag_backend/codebase/indexing/graph_builder.py`
   - Problem: Regex captured function names in definitions, creating self-call relationships.
   - Fix: Skip matches on lines starting with `def`, `class`, or `function`.

4. **[LOW] Files Without Symbols Produced No Chunks**
   - File: `/backend/src/agentic_rag_backend/codebase/indexing/indexer.py`
   - Problem: Files with parse failures or no symbols were excluded from embeddings.
   - Fix: Added fallback module-level chunking for non-empty files.

### Outcome

**APPROVE** - All issues have been addressed. Indexing now preserves symbol tables during incremental updates, returns adequate search results, and produces cleaner call relationships.
