# Epic 15 Tech Spec: Codebase Intelligence

**Date:** 2025-12-31
**Updated:** 2026-01-03 (Party Mode Analysis - RENAMED & REFOCUSED)
**Status:** Backlog
**Epic Owner:** Product and Engineering

---

## Overview

Epic 15 focuses on codebase intelligence features that differentiate the platform for developers. This is a key differentiator for a developer-focused RAG platform.

### Key Decision (2026-01-03)

**REMOVED multimodal video/image processing. FOCUSED on codebase intelligence.**

**Rationale:**
- YouTube transcript API covers 90%+ of video RAG use cases (already in Epic 13)
- Full video processing (CLIP + Whisper) has high complexity/cost, low ROI
- Codebase hallucination detection is a unique differentiator for developer platform

**Decision Document:** `docs/roadmap-decisions-2026-01-03.md`

### Goals

- Detect LLM responses that reference non-existent code symbols (hallucination detection).
- Enable codebase as a knowledge source for RAG (codebase context).
- Position the platform as developer-first with unique code intelligence features.

### Scope

**In scope**
- AST-based codebase hallucination detection for LLM responses.
- Codebase RAG context (index repository as knowledge).
- Symbol extraction and validation.

**Out of scope (REMOVED)**
- ~~Full video ingestion with CLIP + Whisper~~
- ~~Image ingestion with CLIP embeddings~~

**Rationale for removal:** YouTube transcript ingestion (Epic 13) covers 90%+ of video RAG use cases at a fraction of the complexity and cost.

---

## Stories

### Story 15-1: Implement Codebase Hallucination Detector

**Objective:** Detect LLM responses that reference non-existent code symbols, files, or API endpoints.

**Why This Matters:**
LLMs frequently hallucinate when generating code:
- Non-existent function names
- Incorrect file paths
- Made-up API endpoints
- Wrong class/method signatures

For a developer platform, catching these hallucinations is critical for trust.

**Detection Capabilities:**

| Element | Detection Method |
|---------|------------------|
| Functions/Methods | AST parsing + symbol table |
| Classes | AST parsing + symbol table |
| File paths | Filesystem validation |
| API endpoints | OpenAPI spec matching |
| Import statements | Module existence check |

**Configuration:**
```bash
HALLUCINATION_DETECTOR_ENABLED=true|false  # Default: false
HALLUCINATION_DETECTOR_MODE=warn|block  # warn: annotate, block: reject
HALLUCINATION_DETECTOR_LANGUAGES=python,typescript,javascript  # Supported languages
```

**Acceptance Criteria**
- Given an LLM response that references code elements, when validation runs, then AST and symbol search detect unknown classes, functions, or files.
- The detector reports a warning with a list of missing symbols.
- Detection can be configured to block or annotate responses.
- **Supports Python, TypeScript, and JavaScript initially.**
- **Detection is opt-in and configurable.**

### Story 15-2: Implement Codebase RAG Context

**Objective:** Index a code repository as a knowledge source for RAG queries.

**Use Cases:**
- "How does authentication work in this codebase?"
- "What functions call the UserService?"
- "Explain the data flow from API to database"

**Indexing Strategy:**

```
┌─────────────────────────────────────────────────────────────┐
│                   CODEBASE INDEXING                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Repository                                                   │
│      │                                                        │
│      ▼                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ AST Parser   │ -> │ Symbol       │ -> │ Embedding    │   │
│  │ (per file)   │    │ Extractor    │    │ Generator    │   │
│  └──────────────┘    └──────────────┘    └──────────────┘   │
│                                                   │          │
│                                                   ▼          │
│                                          ┌──────────────┐   │
│                                          │ Vector Store │   │
│                                          │ + Graph      │   │
│                                          └──────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Indexed Elements:**
- Function/method signatures with docstrings
- Class definitions with relationships
- File-level documentation
- Import/dependency graphs
- Call graphs (which functions call which)

**Acceptance Criteria**
- Given a code repository, when indexing runs, then symbols are extracted and embedded.
- Symbol relationships are captured in the knowledge graph (e.g., "function A calls function B").
- Queries about the codebase return relevant code context.
- **Indexing respects .gitignore and configurable exclusion patterns.**
- **Incremental indexing is supported for changed files.**

---

## Technical Notes

- **Language Support:** Start with Python, TypeScript, JavaScript. Use Tree-sitter for parsing.
- **AST Parsing:** Use language-specific parsers (tree-sitter, ast module for Python).
- **Symbol Tables:** Build in-memory symbol tables for validation.
- **Caching:** Cache parsed ASTs for performance.

## Dependencies

- Ingestion pipeline (Epic 4, Epic 13).
- Vector store (Epic 3).
- Graphiti for symbol relationships (Epic 5).

## Risks

- AST parsing complexity varies by language.
- Large codebases may require significant indexing time.
- **Mitigation:** Start with most common languages, support incremental indexing.

## Success Metrics

- Hallucination detector catches >= 80% of invalid symbol references on evaluation set.
- Codebase indexing completes within 5 minutes for a 100K LOC repository.
- Codebase queries return relevant context with >= 70% precision.

## References

- `docs/roadmap-decisions-2026-01-03.md` - Decision rationale (multimodal removal)
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
