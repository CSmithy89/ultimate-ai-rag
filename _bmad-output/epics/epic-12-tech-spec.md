# Epic 12 Tech Spec: Advanced Retrieval (Archon Upgrade)

**Date:** 2025-12-31
**Updated:** 2026-01-03 (Party Mode Analysis)
**Status:** Complete
**Epic Owner:** Product and Engineering

---

## Overview

Epic 12 upgrades retrieval quality to SOTA by adding cross-encoder reranking, contextualized embeddings, and a corrective grader agent. The goal is higher precision in top results, more faithful answers, and graceful fallback when initial retrieval is weak.

### Key Decision (2026-01-03)

**Reranker + Graphiti are COMPLEMENTARY, not redundant:**
- **Graphiti (Stage 1):** Fast, broad retrieval (~200ms, 50+ candidates)
- **Reranker (Stage 2):** Precise scoring of query-document pairs (~50-200ms, top 10)

**All features are OPT-IN via configuration flags** to support different deployment profiles.

**Full Configuration Guide:** `docs/guides/advanced-retrieval-configuration.md`

### Goals

- Increase top-10 relevance with cross-encoder reranking.
- Improve chunk meaning via contextual embeddings (title + summary prepends).
- Reduce low-quality answers using a corrective RAG grader and fallback strategy.

### Scope

**In scope**
- Cross-encoder reranking integrated into the retrieval pipeline.
- Contextual retrieval chunk enrichment in ingestion and reindexing.
- Corrective grader agent with fallback retrieval path.
- **Configuration system for all features (opt-in by default).**

**Out of scope**
- New ingestion sources (handled in Epic 13).
- Codebase intelligence (Epic 15).

---

## Stories

### Story 12-1: Implement Cross-Encoder Reranking

**Objective:** Add a reranking stage after initial retrieval to boost precision.

**Configuration Flags:**
```bash
RERANKER_ENABLED=true|false  # Default: false
RERANKER_PROVIDER=cohere|flashrank  # Default: flashrank
RERANKER_TOP_K=10  # Default: 10
```

**Provider Selection:**
- **Cohere:** High accuracy, API-based, 100+ languages, 32K context (Rerank 4)
- **FlashRank:** CPU-optimized, local, no API cost, good for cost-sensitive deployments

**Acceptance Criteria**
- Given a query returns K candidates, when reranking is enabled, then a cross-encoder scores query-document pairs and returns the top N.
- The reranker is configurable (Cohere Rerank API or local FlashRank) via environment configuration.
- Latency impact is measured and logged per request.
- Retrieval traces include pre-rerank and post-rerank result lists.
- **Feature is disabled by default and opt-in.**

### Story 12-2: Implement Contextual Retrieval Chunking

**Objective:** Enrich each chunk with document title and summary before embedding.

**Configuration Flags:**
```bash
CONTEXTUAL_RETRIEVAL_ENABLED=true|false  # Default: false
CONTEXTUAL_MODEL=claude-3-haiku-20240307  # Cost-effective model
CONTEXTUAL_PROMPT_CACHING=true  # 90% cost reduction (Anthropic)
```

**Cost Optimization (REQUIRED):**
1. Use prompt caching (Anthropic) - 90% cost reduction
2. Use cost-effective models (Claude Haiku, GPT-4o-mini)
3. Batch processing during ingestion, not query time

**Acceptance Criteria**
- Given a document with title and summary, when chunks are embedded, then each chunk prepends the title and summary context string.
- A migration path exists for re-embedding existing content (configurable batch job).
- Retrieval quality metrics are captured before and after reindexing.
- **Prompt caching is implemented to reduce costs by 90%.**
- **Feature is disabled by default and opt-in.**

### Story 12-3: Implement Corrective RAG Grader Agent

**Objective:** Add a grader agent that evaluates retrieval relevance and triggers fallback retrieval when needed.

**Configuration Flags:**
```bash
GRADER_ENABLED=true|false  # Default: false
GRADER_THRESHOLD=0.5  # Score below triggers fallback
GRADER_FALLBACK_ENABLED=true|false  # Default: true
GRADER_FALLBACK_STRATEGY=web_search|expanded_query|alternate_index
```

**Grader Implementation:**
- Use lightweight model (T5-large or similar) for grading, NOT full LLM
- Score range: 0.0-1.0

**Acceptance Criteria**
- Given a set of retrieved results, when the grader score is below threshold, then the system triggers fallback retrieval (expanded search, alternate index, or web if configured).
- Grader decisions are logged in the trajectory with scores and thresholds.
- The grader threshold is configurable per environment.
- **Grader uses lightweight model, not full LLM calls.**
- **Feature is disabled by default and opt-in.**

---

## Technical Notes

- **Pipeline flow:** retrieve K (Graphiti) -> rerank (cross-encoder) -> select top N -> CRAG grade -> LLM synthesis.
- **All features opt-in:** Default configuration has all features disabled for baseline performance.
- **Metrics:** store relevance, latency, and cost to validate improvements.

## Configuration Summary

| Feature | Flag | Default | Cost Impact |
|---------|------|---------|-------------|
| Reranking | `RERANKER_ENABLED` | false | +$0.001/query (Cohere) or free (FlashRank) |
| Contextual | `CONTEXTUAL_RETRIEVAL_ENABLED` | false | +$0.01/chunk ingestion (with caching) |
| CRAG | `GRADER_ENABLED` | false | Minimal (lightweight model) |

## Dependencies

- Existing hybrid retrieval pipeline (Epic 3).
- Graphiti integration (Epic 5).
- Configuration system (Epic 11 updates).

## Risks

- Reranking latency could exceed response time targets if not tuned.
- Re-embedding may require significant compute.
- **Mitigation:** All features opt-in, latency monitoring built-in.

## Success Metrics

- +20% improvement in top-10 relevance on evaluation set.
- No more than +1.5s median latency increase with reranking enabled.
- **Cost increase < 10% for contextual retrieval with prompt caching.**

## References

- `docs/guides/advanced-retrieval-configuration.md` - **Full configuration guide**
- `docs/roadmap-decisions-2026-01-03.md` - Decision rationale
- `_bmad-output/prd.md`
- `_bmad-output/architecture.md`
- `_bmad-output/project-planning-artifacts/epics.md`
- `docs/recommendations_2025.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
