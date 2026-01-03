# Roadmap Decisions & Analysis

**Date:** 2026-01-03
**Session:** Party Mode Deep Analysis
**Participants:** BMad Master, Winston (Architect), Mary (Analyst), Amelia (Developer), John (Product Manager), Murat (Test Architect), Bob (Scrum Master), Sally (UX Designer), Dr. Quinn (Problem Solver)

---

## Executive Summary

This document captures the strategic decisions made during the 2026-01-03 roadmap analysis session. The analysis evaluated `docs/recommendations_2025.md` against `sprint-status.yaml` and the epic breakdown, validating technologies with external research.

**Overall Assessment:** The roadmap is sound (Grade: A-). Key refinements documented below.

---

## Vision Clarification

### Platform Purpose

**This is a Developer Platform, not an application.**

> *"An advanced RAG system generator where developers choose their stack (database, LLM, agent framework) and get a fully configured, extensible foundation."*

Comparable to:
- Create React App for React projects
- Rails new for Ruby applications
- Django startproject for Python web apps

### User Journey

```
$ rag-install

? Select LLM Provider: OpenRouter (Multi-provider)
? Select Database: PostgreSQL + pgvector
? Select Agent Framework: PydanticAI
? Enable Advanced Features: [x] Reranking [x] CRAG

✓ Generating configuration...
✓ Running docker compose...

🚀 Your RAG system is ready!
```

---

## Key Decisions

### 1. Reranker + Graphiti Relationship

**Decision:** Complementary, not redundant.

| Stage | Component | Role |
|-------|-----------|------|
| Stage 1 | Graphiti Hybrid | Fast, broad retrieval (semantic + BM25 + graph), ~50 candidates |
| Stage 2 | Cross-Encoder Reranker | Precise scoring of query-document pairs, top 10 |

**Configuration:** Reranking is opt-in via `RERANKER_ENABLED` flag.

**Documentation:** `docs/guides/advanced-retrieval-configuration.md`

---

### 2. Contextual Retrieval Optimization

**Decision:** Keep feature, but optimize for cost.

**Optimizations:**
1. Use prompt caching (Anthropic) - 90% cost reduction
2. Use cost-effective models (Claude Haiku, GPT-4o-mini)
3. Batch processing during ingestion, not query time
4. Make opt-in via `CONTEXTUAL_RETRIEVAL_ENABLED` flag

**Documentation:** `docs/guides/advanced-retrieval-configuration.md`

---

### 3. MCP Server Architecture

**Decision:** Wrap Graphiti MCP, don't duplicate.

**Rationale:**
- Graphiti already has a tested, maintained MCP server
- DRY principle - don't rebuild what exists
- Add RAG-specific tools as extensions

**Extended Tools:**
- `vector_search` - pgvector semantic search
- `hybrid_retrieve` - Combined vector + graph
- `ingest_url` - Crawl4AI/Apify web ingestion
- `ingest_pdf` - Docling PDF processing
- `ingest_youtube` - YouTube transcript extraction
- `query_with_reranking` - Cross-encoder reranked results
- `explain_answer` - Trajectory/explainability

**Documentation:** `docs/guides/mcp-wrapper-architecture.md`

---

### 4. Epic 15 Refocus: Codebase Intelligence

**Decision:** Remove multimodal video/image processing.

**Rationale:**
- YouTube transcript API covers 90%+ of video RAG use cases
- Full video processing (CLIP + Whisper) has high complexity/cost, low ROI
- Codebase hallucination detection is a unique differentiator for developer platform

**Removed Stories:**
- ~~15-1: Implement Full Video Ingestion (CLIP + Whisper)~~
- ~~15-2: Implement Image Ingestion~~

**New Stories:**
- 15-1: Implement Codebase Hallucination Detector (AST-based symbol validation)
- 15-2: Implement Codebase RAG Context (Index repository as knowledge)

**Epic Renamed:** "Codebase Intelligence" (was "Multimodal & Codebase")

---

### 5. Framework Agnosticism Purpose

**Decision:** Framework adapters are developer extension points.

**Clarification:**
- This is NOT runtime framework switching
- This enables developers to BUILD ON their preferred framework
- The CLI asks "Which framework?" and configures the project for that framework

**Framework Audience:**

| Framework | Target Developer |
|-----------|-----------------|
| Agno | Default, general-purpose |
| PydanticAI | Type-safety enthusiasts |
| CrewAI | Multi-agent workflow builders |
| LangGraph | Stateful workflow engineers |
| Anthropic Agent SDK | Claude-native developers |

**Story 16-4 Update:** Include Agent Skills integration (open standard adopted by Microsoft/VS Code).

---

### 6. CLI Timing: Intentionally Last

**Decision:** Epic 17 remains the final epic.

**Rationale:**
1. Must know ALL options before offering them in CLI
2. All frameworks must be implemented before offering choice
3. All features must be documented before CLI can link to docs
4. All combinations must be tested for the validator
5. CLI is the "capstone UX" - the packaging of everything else

---

## Updated Sprint Status

See: `_bmad-output/implementation-artifacts/sprint-status.yaml`

Key changes:
- Added decision comments to each epic
- Added documentation links
- Renamed Epic 15 to "Codebase Intelligence"
- Added configuration flag notes to stories
- Clarified Epic 16 purpose

---

## Research Validation

All major technologies were validated with external research:

| Technology | Source | Validation |
|------------|--------|------------|
| Cross-Encoder Reranking | [Databricks, Pinecone](https://www.pinecone.io/learn/series/rag/rerankers/) | +48% retrieval improvement |
| Contextual Retrieval | [Anthropic](https://www.anthropic.com/news/contextual-retrieval) | 35-67% error reduction |
| Corrective RAG | [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/) | Production-ready pattern |
| Graphiti MCP | [DeepWiki](https://deepwiki.com/getzep/graphiti) | Built-in MCP server confirmed |
| OpenRouter | [Documentation](https://openrouter.ai/docs/api/reference/embeddings) | Embeddings + 300+ models |
| PydanticAI | [Context7](https://context7.com/pydantic/pydantic-ai) | A2A + MCP support native |
| CrewAI | [GitHub](https://github.com/crewaiinc/crewai) | Hierarchical process mode |
| LangGraph | [Documentation](https://langchain-ai.github.io/langgraph) | Checkpointing + persistence |
| Anthropic Agent SDK | [Blog](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) | Agent Skills open standard |
| youtube-transcript-api | [PyPI](https://pypi.org/project/youtube-transcript-api/) | Active development (v1.2.3, Oct 2025) |

---

## Documentation Created

| Document | Path | Purpose |
|----------|------|---------|
| Advanced Retrieval Configuration | `docs/guides/advanced-retrieval-configuration.md` | Config flags for Epic 12 |
| MCP Wrapper Architecture | `docs/guides/mcp-wrapper-architecture.md` | Architecture for Epic 14 |
| Roadmap Decisions | `docs/roadmap-decisions-2026-01-03.md` | This document |

---

## Next Steps

1. **Epic 12:** Implement with configuration flags per `advanced-retrieval-configuration.md`
2. **Epic 14:** Implement MCP wrapper per `mcp-wrapper-architecture.md`
3. **Epic 15:** Execute with renamed/refocused stories
4. **Epic 16:** Add Agent Skills to Story 16-4
5. **Epic 17:** Execute as capstone after all features complete
6. **Epic 18:** Update documentation stories to reference new guides

---

## References

- `docs/recommendations_2025.md` - Original recommendations
- `_bmad-output/implementation-artifacts/sprint-status.yaml` - Updated sprint status
- `_bmad-output/project-planning-artifacts/epics.md` - Epic definitions
- `docs/guides/advanced-retrieval-configuration.md` - Epic 12 configuration
- `docs/guides/mcp-wrapper-architecture.md` - Epic 14 architecture
