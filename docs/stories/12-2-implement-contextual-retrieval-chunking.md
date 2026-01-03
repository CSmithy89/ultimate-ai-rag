# Story 12.2: Implement Contextual Retrieval Chunking

Status: review

## Story

As a **developer**,
I want **contextual enrichment of chunks with document title and summary before embedding**,
So that **retrieval accuracy improves by 35-67% through preserved document context**.

## Acceptance Criteria

1. Given a document with title and summary, when chunks are embedded, then each chunk prepends the title and summary context string.
2. Given `CONTEXTUAL_RETRIEVAL_ENABLED=true`, when ingestion runs, then contextual enrichment is applied.
3. Given `CONTEXTUAL_RETRIEVAL_ENABLED` is not set or `false`, when ingestion runs, then standard chunking is used (opt-in by default).
4. Given contextual enrichment is enabled, when processing chunks, then prompt caching reduces costs by ~90% (Anthropic).
5. Given existing content in the database, when reindexing is triggered, then a migration script re-embeds with contextual enrichment.
6. Given `CONTEXTUAL_MODEL=<model>`, when context is generated, then the specified cost-effective model is used (default: claude-3-haiku).
7. Given contextual enrichment runs, when a chunk is processed, then the context generation latency is logged.

## Standards Coverage

- [ ] Multi-tenancy / tenant isolation: N/A - contextual enrichment is per-document
- [ ] Rate limiting / abuse protection: N/A - uses existing ingestion limits
- [ ] Input validation / schema enforcement: Must validate CONTEXTUAL_MODEL values
- [ ] Tests (unit/integration): Must add contextual enrichment tests
- [ ] Error handling + logging: Must handle model API failures gracefully
- [ ] Documentation updates: Must update configuration guide

## Tasks / Subtasks

- [x] Add contextual retrieval configuration to config.py (AC: 2, 3, 6)
  - [x] Add CONTEXTUAL_RETRIEVAL_ENABLED, CONTEXTUAL_MODEL, CONTEXTUAL_PROMPT_CACHING
  - [x] Add CONTEXTUAL_REINDEX_BATCH_SIZE for migration
  - [x] Default to disabled
- [x] Implement contextual chunk enricher (AC: 1, 4, 7)
  - [x] Create ContextualChunkEnricher class
  - [x] Implement context generation with LLM (title + summary prepend)
  - [x] Add prompt caching support (Anthropic cache_control)
  - [x] Log context generation latency
- [x] Integrate contextual enrichment into ingestion pipeline (AC: 1, 2, 3)
  - [x] Add enrichment step before embedding in index_worker
  - [x] Pass enriched chunk content to embedding generator
  - [x] Preserve original chunk for storage, use enriched for embedding
- [x] Create reindexing migration script (AC: 5)
  - [x] Create scripts/reindex_contextual.py
  - [x] Implement batch processing with configurable size
  - [x] Add progress logging and dry-run capability
- [x] Add tests for contextual retrieval (AC: 1-7)
  - [x] Unit tests for ContextualChunkEnricher (22 tests)
  - [x] Unit tests for prompt caching behavior
  - [x] Unit tests for factory function create_contextual_enricher
- [x] Update configuration documentation (AC: 3)
  - [x] Create docs/guides/advanced-retrieval-configuration.md

## Technical Notes

- **Contextual Retrieval (Anthropic):** Prepending document context to chunks improves retrieval by 35-67%
- **Prompt Caching:** Cache the document context (title/summary), only generate chunk-specific context
- **Cost-Effective Models:** Use claude-3-haiku or gpt-4o-mini to minimize per-chunk cost
- **Context Template:**
  ```
  Document: {title}
  Section: {section_heading}
  Context: {generated_context_about_chunk}

  {original_chunk_content}
  ```
- **Prompt Caching Strategy:** Cache the document preamble + few-shot examples, only vary the chunk content

## Definition of Done

- [x] Acceptance criteria met
- [x] Standards coverage updated
- [x] Tests run and documented (22 tests passing)
- [x] Configuration guide updated

## Dev Notes

### Implementation Decisions

1. **Lazy LLM Client Initialization**: The `ContextualChunkEnricher` lazily initializes the LLM client on first use, avoiding unnecessary API connections.

2. **Graceful Degradation**: If context generation fails, the enricher returns an empty context string rather than failing the entire chunk - embeddings proceed without context.

3. **Document Content Truncation**: Full document content is truncated to 6000 characters to fit within LLM context windows while providing sufficient context.

4. **Prompt Caching Strategy**: Uses Anthropic's `cache_control: ephemeral` on the document section to cache the document content across chunk enrichments within the same document.

5. **Original vs Enriched Content**: Original chunk content is stored in the database; enriched content (with context prepended) is used only for embedding generation.

### Testing Notes

- All 22 unit tests pass
- Tests use `sys.modules` patching for lazy imports
- Tests cover Anthropic, OpenAI, and error handling scenarios

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

N/A

### Completion Notes List

- ContextualChunkEnricher supports both Anthropic (claude) and OpenAI (gpt) models
- Prompt caching reduces costs by ~90% for Anthropic models
- Integration with index_worker preserves backward compatibility
- Reindexing script supports dry-run and batch processing

### File List

**New Files:**
- `backend/src/agentic_rag_backend/indexing/contextual.py` - Core contextual enrichment module
- `backend/src/agentic_rag_backend/scripts/__init__.py` - Scripts package init
- `backend/src/agentic_rag_backend/scripts/reindex_contextual.py` - Migration script
- `backend/tests/test_contextual.py` - Unit tests (22 tests)
- `docs/guides/advanced-retrieval-configuration.md` - Configuration guide

**Modified Files:**
- `backend/src/agentic_rag_backend/config.py` - Added contextual retrieval settings
- `backend/src/agentic_rag_backend/indexing/__init__.py` - Export contextual module
- `backend/src/agentic_rag_backend/indexing/workers/index_worker.py` - Integration with enricher
