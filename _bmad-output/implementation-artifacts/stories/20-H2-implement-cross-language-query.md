# Story 20-H2: Implement Cross-Language Query

Status: done

## Story

As a developer building multilingual applications,
I want to support queries in languages different from the indexed content,
so that users can search in their native language regardless of document language.

## Context

This story is part of Epic 20: Advanced Retrieval Intelligence, specifically Group H: Competitive Features. It implements cross-language query support enabling:

- **Multilingual Embeddings**: Use multilingual models that map all languages to same vector space
- **Query Translation**: Optionally translate queries to match document language
- **Language Detection**: Automatically detect query language for routing

**Competitive Positioning**: Enterprise RAG systems support multilingual search. This enables global deployment with users in different regions.

**Dependencies**:
- sentence-transformers (already installed) for multilingual embeddings
- Optional: translation API for query translation

## Acceptance Criteria

1. Given a query in any language, when multilingual embedding is used, then similar documents are found regardless of language.
2. Given CROSS_LANGUAGE_TRANSLATION=true, when a query is submitted, then it is translated before search.
3. Given CROSS_LANGUAGE_ENABLED=false (default), when the system starts, then cross-language features are not active.
4. Given a query, when language detection is used, then the query language is correctly identified.

## Technical Approach

### Module Structure

```
backend/src/agentic_rag_backend/retrieval/
+-- cross_language.py              # NEW: Cross-language query support
```

### Core Components

1. **LanguageDetector** - Detect query language:
   - Uses simple heuristics or langdetect library
   - Returns ISO 639-1 language code

2. **CrossLanguageEmbedding** - Multilingual embedding:
   - Uses multilingual-e5-base or similar model
   - Maps all languages to shared vector space

3. **QueryTranslator** - Optional query translation:
   - Uses LLM for translation (existing provider)
   - Caches translations to reduce API calls

4. **CrossLanguageAdapter** - Feature flag wrapper:
   - Wraps existing embedding provider
   - Applies translation if enabled

### Configuration

```bash
CROSS_LANGUAGE_ENABLED=true|false            # Default: false
CROSS_LANGUAGE_EMBEDDING=multilingual-e5-base # Model name
CROSS_LANGUAGE_TRANSLATION=true|false        # Default: false
```

## Tasks / Subtasks

- [x] Create cross_language.py module
- [x] Implement LanguageDetector class
- [x] Implement CrossLanguageEmbedding class
- [x] Implement QueryTranslator class (optional translation)
- [x] Implement CrossLanguageAdapter with feature flag
- [x] Add configuration variables to settings
  - [x] CROSS_LANGUAGE_ENABLED
  - [x] CROSS_LANGUAGE_EMBEDDING
  - [x] CROSS_LANGUAGE_TRANSLATION
- [x] Export from retrieval/__init__.py
- [x] Write unit tests for all components

## Testing Requirements

### Unit Tests
- Language detection accuracy
- Multilingual embedding generation
- Query translation (with mock)
- Feature flag behavior
- Adapter fallback when disabled

## Definition of Done

- [x] All acceptance criteria pass
- [x] All tasks completed
- [x] Unit test coverage >= 80%
- [x] Feature flag (CROSS_LANGUAGE_ENABLED) works correctly
- [x] Configuration documented
- [x] Code review approved
- [x] No regressions in existing tests

## Dev Notes

- Reference tech spec: `_bmad-output/epics/epic-20-tech-spec.md` (Story 20-H2 section)
- multilingual-e5-base from HuggingFace is ~300MB
- Language detection can use simple character patterns for common languages
- Translation via existing LLM provider to avoid new dependencies

---

## Dev Agent Record

### File List

| File | Action | Description |
|------|--------|-------------|
| `backend/src/agentic_rag_backend/retrieval/cross_language.py` | NEW | LanguageDetector, CrossLanguageEmbedding, QueryTranslator, CrossLanguageAdapter |
| `backend/src/agentic_rag_backend/retrieval/__init__.py` | MODIFIED | Export cross-language components |
| `backend/src/agentic_rag_backend/config.py` | MODIFIED | Add cross-language settings |
| `backend/tests/retrieval/test_cross_language.py` | NEW | Unit tests |

### Change Log

| Date | Change | Details |
|------|--------|---------|
| 2026-01-06 | Initial implementation | Created story file |
| 2026-01-06 | Full implementation | Created cross_language.py with LanguageDetector (Unicode patterns, Latin markers), CrossLanguageEmbedding (sentence-transformers), QueryTranslator (LLM caching), CrossLanguageAdapter (feature flag wrapper). Added 3 config vars. 38 unit tests passing. |
