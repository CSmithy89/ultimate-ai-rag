# Advanced Retrieval Configuration Guide

**Date:** 2026-01-03
**Version:** 1.0
**Related Epic:** Epic 12 - Advanced Retrieval (Archon Upgrade)

---

## Overview

This guide documents the configuration options for the Advanced Retrieval features introduced in Epic 12. All features are **opt-in** and designed to complement the existing Graphiti hybrid retrieval pipeline.

### Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Query ──► STAGE 1: Graphiti Hybrid ──► STAGE 2: Reranker ──► LLM│
│            (semantic + BM25 + graph)     (cross-encoder)         │
│            ~50 candidates                 top 10                  │
│                      │                         │                  │
│                      ▼                         ▼                  │
│               CRAG Grader ◄───── If score < threshold ──────────►│
│                      │                                            │
│                      ▼                                            │
│               Fallback Retrieval (web search, expanded query)    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key Decision:** Graphiti and the Reranker are complementary, not redundant.
- **Graphiti (Stage 1):** Fast, broad retrieval (~200ms, 50+ candidates)
- **Reranker (Stage 2):** Precise scoring of query-document pairs (~50-200ms, top 10)

---

## Configuration Reference

### Environment Variables

Add these to your `.env` file to enable/configure advanced retrieval:

```bash
# ============================================
# CROSS-ENCODER RERANKING (Story 12-1)
# ============================================

# Enable/disable reranking (default: false)
RERANKER_ENABLED=true

# Reranker provider selection
# Options: cohere | flashrank
# - cohere: High accuracy, API-based, 100+ languages, 32K context (Rerank v3.5)
# - flashrank: CPU-optimized, local, no API cost, good for cost-sensitive deployments
RERANKER_PROVIDER=flashrank

# Number of top results after reranking (default: 10)
RERANKER_TOP_K=10

# Model override (optional, uses provider defaults)
# Cohere default: rerank-v3.5
# FlashRank default: ms-marco-MiniLM-L-12-v2
RERANKER_MODEL=rerank-v3.5

# Cohere API key (required only if RERANKER_PROVIDER=cohere)
COHERE_API_KEY=your-cohere-api-key

# ============================================
# CONTEXTUAL RETRIEVAL (Story 12-2)
# ============================================

# Enable/disable contextual chunk enrichment (default: false)
CONTEXTUAL_RETRIEVAL_ENABLED=true

# Model for generating chunk context (use cost-effective model)
# Recommended: claude-3-haiku, gpt-4o-mini
CONTEXTUAL_MODEL=claude-3-haiku-20240307

# Enable prompt caching for cost reduction (Anthropic only)
# Reduces contextual retrieval costs by ~90%
CONTEXTUAL_PROMPT_CACHING=true

# Batch size for re-embedding existing content
CONTEXTUAL_REINDEX_BATCH_SIZE=100

# ============================================
# CORRECTIVE RAG / GRADER (Story 12-3)
# ============================================

# Enable/disable CRAG grader (default: false)
GRADER_ENABLED=true

# Relevance threshold (0.0-1.0) - below triggers fallback
GRADER_THRESHOLD=0.5

# Enable fallback retrieval when grader score is low
GRADER_FALLBACK_ENABLED=true

# Fallback strategy when grader triggers
# Options: web_search | expanded_query | alternate_index
GRADER_FALLBACK_STRATEGY=web_search

# Tavily API key for web search fallback
TAVILY_API_KEY=your-tavily-api-key
```

---

## Feature Details

### 1. Cross-Encoder Reranking

Cross-encoders score the **query + document pair together**, providing higher precision than bi-encoder (embedding-only) scoring.

#### When to Enable

| Scenario | Recommendation |
|----------|----------------|
| High-accuracy requirements | Enable with Cohere |
| Cost-sensitive / CPU-only | Enable with FlashRank |
| Low-latency priority (<200ms) | Disable or use FlashRank |
| Simple queries | Optional |

#### Performance Benchmarks

| Configuration | Latency | NDCG@10 Improvement |
|---------------|---------|---------------------|
| Graphiti only | ~200ms | Baseline |
| + FlashRank | ~250ms | +15-20% |
| + Cohere Rerank | ~400ms | +25-30% |

*Based on internal testing. Actual results vary by dataset.*

#### Code Integration Point

```python
# backend/src/agentic_rag_backend/retrieval/reranking.py

from agentic_rag_backend.retrieval import (
    create_reranker_client,
    get_reranker_adapter,
    RerankerProviderType,
)
from agentic_rag_backend.config import get_settings

settings = get_settings()

# Create reranker if enabled
reranker = None
if settings.reranker_enabled:
    adapter = get_reranker_adapter(settings)
    reranker = create_reranker_client(adapter)

# Use in retrieval pipeline
if reranker:
    reranked = await reranker.rerank(
        query=query,
        hits=vector_hits,
        top_k=settings.reranker_top_k,
    )
    # reranked contains RerankedHit objects with:
    # - hit: the original VectorHit
    # - rerank_score: cross-encoder score (0.0-1.0)
    # - original_rank: position before reranking
```

---

### 2. Contextual Retrieval

Contextual retrieval prepends document title and summary to each chunk before embedding, improving retrieval accuracy by 35-67%.

#### How It Works

```
BEFORE (Standard Chunking):
┌─────────────────────────────┐
│ "The price is $50/month"    │  ← Loses context
└─────────────────────────────┘

AFTER (Contextual Chunking):
┌─────────────────────────────────────────────────────────┐
│ "Document: Pricing Guide                                │
│  Section: Enterprise Plan                               │
│  Context: This section describes pricing tiers...       │
│                                                         │
│  The price is $50/month"                               │  ← Context preserved
└─────────────────────────────────────────────────────────┘
```

#### Cost Optimization

Without optimization, contextual retrieval is expensive (LLM call per chunk). We mitigate this with:

1. **Prompt Caching (Anthropic):** Cache the document context, only generate chunk-specific additions
2. **Cost-Effective Models:** Use Claude Haiku or GPT-4o-mini instead of larger models
3. **Batch Processing:** Generate context during ingestion, not at query time

#### Migration Path

To re-embed existing content with contextual retrieval:

```bash
# Run the reindexing job
uv run python -m agentic_rag_backend.scripts.reindex_contextual \
  --batch-size 100 \
  --tenant-id your-tenant
```

---

### 3. Corrective RAG (CRAG)

CRAG evaluates retrieval quality and triggers fallback strategies when results are insufficient.

#### Grader Logic

```
Retrieved Documents
       │
       ▼
┌──────────────────┐
│   GRADER AGENT   │  ← Lightweight model (T5-large or similar)
│  Score: 0.0-1.0  │
└────────┬─────────┘
         │
    ┌────┴────┐
    │ Score   │
    │ < 0.5?  │
    └────┬────┘
    Yes  │  No
    ▼    ▼
┌────────┐  ┌────────────┐
│Fallback│  │  Proceed   │
│Strategy│  │with results│
└────────┘  └────────────┘
```

#### Fallback Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `web_search` | Query Tavily for current web data | Knowledge gaps, current events |
| `expanded_query` | Reformulate query and retry | Ambiguous queries |
| `alternate_index` | Search different knowledge base | Multi-domain systems |

---

## Recommended Configurations

### Production (Balanced)

```bash
RERANKER_ENABLED=true
RERANKER_PROVIDER=flashrank
RERANKER_TOP_K=10

CONTEXTUAL_RETRIEVAL_ENABLED=true
CONTEXTUAL_MODEL=claude-3-haiku-20240307
CONTEXTUAL_PROMPT_CACHING=true

GRADER_ENABLED=true
GRADER_THRESHOLD=0.5
GRADER_FALLBACK_ENABLED=true
GRADER_FALLBACK_STRATEGY=web_search
```

### High Accuracy (Enterprise)

```bash
RERANKER_ENABLED=true
RERANKER_PROVIDER=cohere
COHERE_RERANK_MODEL=rerank-v4-pro
RERANKER_TOP_K=10

CONTEXTUAL_RETRIEVAL_ENABLED=true
CONTEXTUAL_MODEL=claude-3-haiku-20240307
CONTEXTUAL_PROMPT_CACHING=true

GRADER_ENABLED=true
GRADER_THRESHOLD=0.6
GRADER_FALLBACK_ENABLED=true
GRADER_FALLBACK_STRATEGY=web_search
```

### Low Latency / Cost Optimized

```bash
RERANKER_ENABLED=false

CONTEXTUAL_RETRIEVAL_ENABLED=false

GRADER_ENABLED=false
```

---

## Metrics & Observability

All advanced retrieval features log to the trajectory store:

```python
# Logged metrics per request
{
    "retrieval_stage_1_count": 50,
    "retrieval_stage_1_latency_ms": 180,
    "reranker_enabled": true,
    "reranker_provider": "flashrank",
    "reranker_latency_ms": 45,
    "reranker_top_k": 10,
    "grader_score": 0.72,
    "grader_fallback_triggered": false,
    "contextual_retrieval_enabled": true
}
```

Dashboard queries available in Epic 18 documentation guide.

---

## References

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Cohere Rerank Documentation](https://docs.cohere.com/docs/rerank)
- [FlashRank GitHub](https://github.com/AnswerDotAI/rerankers)
- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph CRAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
- `_bmad-output/implementation-artifacts/epic-12-tech-spec.md`
- `_bmad-output/implementation-artifacts/sprint-status.yaml`
