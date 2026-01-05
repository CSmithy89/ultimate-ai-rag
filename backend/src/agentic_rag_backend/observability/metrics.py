"""Prometheus metric definitions for retrieval operations.

This module defines all Prometheus metrics for the Agentic RAG retrieval pipeline,
including counters, histograms, and gauges for monitoring retrieval quality and
performance.

All metrics include a tenant_id label for multi-tenant analysis.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    REGISTRY,
)
import structlog

logger = structlog.get_logger(__name__)

# Default registry (can be overridden for testing)
_registry: CollectorRegistry = REGISTRY


def get_metrics_registry() -> CollectorRegistry:
    """Get the current metrics registry.

    Returns:
        The CollectorRegistry used for all metrics
    """
    return _registry


# =============================================================================
# Counter Metrics
# =============================================================================

RETRIEVAL_REQUESTS_TOTAL = Counter(
    "retrieval_requests_total",
    "Total number of retrieval requests",
    labelnames=["strategy", "tenant_id"],
    registry=_registry,
)
"""Counter for total retrieval requests.

Labels:
    strategy: vector|graph|hybrid
    tenant_id: Tenant identifier for multi-tenancy
"""

RETRIEVAL_FALLBACK_TRIGGERED_TOTAL = Counter(
    "retrieval_fallback_triggered_total",
    "Total number of times retrieval fallback was triggered",
    labelnames=["reason", "tenant_id"],
    registry=_registry,
)
"""Counter for retrieval fallback triggers.

Labels:
    reason: low_score|empty_results|timeout
    tenant_id: Tenant identifier for multi-tenancy
"""

GRADER_EVALUATIONS_TOTAL = Counter(
    "grader_evaluations_total",
    "Total number of grader evaluations",
    labelnames=["result", "tenant_id"],
    registry=_registry,
)
"""Counter for grader evaluations.

Labels:
    result: pass|fail|fallback
    tenant_id: Tenant identifier for multi-tenancy
"""

# =============================================================================
# Histogram Metrics
# =============================================================================

# Latency buckets in seconds: 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

RETRIEVAL_LATENCY_SECONDS = Histogram(
    "retrieval_latency_seconds",
    "Retrieval operation latency in seconds",
    labelnames=["strategy", "phase", "tenant_id"],
    buckets=LATENCY_BUCKETS,
    registry=_registry,
)
"""Histogram for retrieval latency.

Labels:
    strategy: vector|graph|hybrid
    phase: embed|search|rerank|grade
    tenant_id: Tenant identifier for multi-tenancy
"""

# Improvement ratio buckets: 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x, 3x, 5x
IMPROVEMENT_BUCKETS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0)

RERANKING_IMPROVEMENT_RATIO = Histogram(
    "reranking_improvement_ratio",
    "Ratio of post-rerank score to pre-rerank score",
    labelnames=["tenant_id"],
    buckets=IMPROVEMENT_BUCKETS,
    registry=_registry,
)
"""Histogram for reranking improvement.

This measures how much reranking improved the top result's relevance score
compared to the original retrieval score.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
"""

# Score buckets: 0-1 range in 0.1 increments
SCORE_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

GRADER_SCORE = Histogram(
    "grader_score",
    "Grader relevance scores",
    labelnames=["model", "tenant_id"],
    buckets=SCORE_BUCKETS,
    registry=_registry,
)
"""Histogram for grader scores.

Labels:
    model: The grader model used (e.g., heuristic, cross-encoder)
    tenant_id: Tenant identifier for multi-tenancy
"""

# =============================================================================
# Gauge Metrics
# =============================================================================

RETRIEVAL_PRECISION = Gauge(
    "retrieval_precision",
    "Precision of retrieval results at k",
    labelnames=["strategy", "k", "tenant_id"],
    registry=_registry,
)
"""Gauge for retrieval precision@k.

Labels:
    strategy: vector|graph|hybrid
    k: The k value for precision@k
    tenant_id: Tenant identifier for multi-tenancy
"""

RETRIEVAL_RECALL = Gauge(
    "retrieval_recall",
    "Recall of retrieval results at k",
    labelnames=["strategy", "k", "tenant_id"],
    registry=_registry,
)
"""Gauge for retrieval recall@k.

Labels:
    strategy: vector|graph|hybrid
    k: The k value for recall@k
    tenant_id: Tenant identifier for multi-tenancy
"""

ACTIVE_RETRIEVAL_OPERATIONS = Gauge(
    "active_retrieval_operations",
    "Number of currently active retrieval operations",
    labelnames=["tenant_id"],
    registry=_registry,
)
"""Gauge for active retrieval operations.

Labels:
    tenant_id: Tenant identifier for multi-tenancy
"""


# =============================================================================
# Contextual Retrieval Metrics (Story 19-F5)
# =============================================================================

CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL = Counter(
    "contextual_enrichment_tokens_total",
    "Total tokens used for contextual enrichment",
    labelnames=["type", "model", "tenant_id"],
    registry=_registry,
)
"""Counter for contextual enrichment token usage.

Labels:
    type: Token type (input|output)
    model: Model used for enrichment (e.g., claude-3-haiku)
    tenant_id: Tenant identifier for multi-tenancy
"""

CONTEXTUAL_ENRICHMENT_COST_USD_TOTAL = Counter(
    "contextual_enrichment_cost_usd_total",
    "Total estimated cost in USD for contextual enrichment",
    labelnames=["model", "tenant_id"],
    registry=_registry,
)
"""Counter for contextual enrichment cost.

Labels:
    model: Model used for enrichment
    tenant_id: Tenant identifier for multi-tenancy
"""

CONTEXTUAL_ENRICHMENT_CACHE_HITS_TOTAL = Counter(
    "contextual_enrichment_cache_hits_total",
    "Total cache hits for prompt caching in contextual enrichment",
    labelnames=["model", "tenant_id"],
    registry=_registry,
)
"""Counter for prompt cache hits.

Labels:
    model: Model used for enrichment
    tenant_id: Tenant identifier for multi-tenancy
"""

CONTEXTUAL_ENRICHMENT_CACHE_MISSES_TOTAL = Counter(
    "contextual_enrichment_cache_misses_total",
    "Total cache misses for prompt caching in contextual enrichment",
    labelnames=["model", "tenant_id"],
    registry=_registry,
)
"""Counter for prompt cache misses.

Labels:
    model: Model used for enrichment
    tenant_id: Tenant identifier for multi-tenancy
"""

CONTEXTUAL_ENRICHMENT_CHUNKS_TOTAL = Counter(
    "contextual_enrichment_chunks_total",
    "Total chunks enriched with contextual retrieval",
    labelnames=["model", "tenant_id"],
    registry=_registry,
)
"""Counter for enriched chunks.

Labels:
    model: Model used for enrichment
    tenant_id: Tenant identifier for multi-tenancy
"""

CONTEXTUAL_ENRICHMENT_LATENCY_SECONDS = Histogram(
    "contextual_enrichment_latency_seconds",
    "Latency for contextual enrichment operations in seconds",
    labelnames=["model", "tenant_id"],
    buckets=LATENCY_BUCKETS,
    registry=_registry,
)
"""Histogram for contextual enrichment latency.

Labels:
    model: Model used for enrichment
    tenant_id: Tenant identifier for multi-tenancy
"""


# =============================================================================
# Helper Functions
# =============================================================================


def record_retrieval_request(
    strategy: str,
    tenant_id: str,
) -> None:
    """Record a retrieval request.

    Args:
        strategy: Retrieval strategy used (vector|graph|hybrid)
        tenant_id: Tenant identifier
    """
    RETRIEVAL_REQUESTS_TOTAL.labels(
        strategy=strategy,
        tenant_id=tenant_id,
    ).inc()


def record_retrieval_fallback(
    reason: str,
    tenant_id: str,
) -> None:
    """Record a retrieval fallback trigger.

    Args:
        reason: Reason for fallback (low_score|empty_results|timeout)
        tenant_id: Tenant identifier
    """
    RETRIEVAL_FALLBACK_TRIGGERED_TOTAL.labels(
        reason=reason,
        tenant_id=tenant_id,
    ).inc()


def record_grader_evaluation(
    result: str,
    tenant_id: str,
) -> None:
    """Record a grader evaluation.

    Args:
        result: Evaluation result (pass|fail|fallback)
        tenant_id: Tenant identifier
    """
    GRADER_EVALUATIONS_TOTAL.labels(
        result=result,
        tenant_id=tenant_id,
    ).inc()


def record_retrieval_latency(
    strategy: str,
    phase: str,
    tenant_id: str,
    duration_seconds: float,
) -> None:
    """Record retrieval latency.

    Args:
        strategy: Retrieval strategy used (vector|graph|hybrid)
        phase: Operation phase (embed|search|rerank|grade)
        tenant_id: Tenant identifier
        duration_seconds: Duration in seconds
    """
    RETRIEVAL_LATENCY_SECONDS.labels(
        strategy=strategy,
        phase=phase,
        tenant_id=tenant_id,
    ).observe(duration_seconds)


def record_reranking_improvement(
    tenant_id: str,
    pre_score: float,
    post_score: float,
) -> None:
    """Record reranking improvement ratio.

    Args:
        tenant_id: Tenant identifier
        pre_score: Score before reranking
        post_score: Score after reranking
    """
    if pre_score > 0:
        ratio = post_score / pre_score
        RERANKING_IMPROVEMENT_RATIO.labels(
            tenant_id=tenant_id,
        ).observe(ratio)


def record_grader_score(
    model: str,
    tenant_id: str,
    score: float,
) -> None:
    """Record a grader score.

    Args:
        model: Grader model name
        tenant_id: Tenant identifier
        score: Grader score (0.0-1.0)
    """
    GRADER_SCORE.labels(
        model=model,
        tenant_id=tenant_id,
    ).observe(score)


def set_retrieval_precision(
    strategy: str,
    k: int,
    tenant_id: str,
    precision: float,
) -> None:
    """Set retrieval precision@k.

    Args:
        strategy: Retrieval strategy used (vector|graph|hybrid)
        k: The k value for precision@k
        tenant_id: Tenant identifier
        precision: Precision value (0.0-1.0)
    """
    RETRIEVAL_PRECISION.labels(
        strategy=strategy,
        k=str(k),
        tenant_id=tenant_id,
    ).set(precision)


def set_retrieval_recall(
    strategy: str,
    k: int,
    tenant_id: str,
    recall: float,
) -> None:
    """Set retrieval recall@k.

    Args:
        strategy: Retrieval strategy used (vector|graph|hybrid)
        k: The k value for recall@k
        tenant_id: Tenant identifier
        recall: Recall value (0.0-1.0)
    """
    RETRIEVAL_RECALL.labels(
        strategy=strategy,
        k=str(k),
        tenant_id=tenant_id,
    ).set(recall)


@contextmanager
def track_active_retrieval(tenant_id: str) -> Generator[None, None, None]:
    """Context manager to track active retrieval operations.

    Args:
        tenant_id: Tenant identifier

    Yields:
        None
    """
    ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id).inc()
    try:
        yield
    finally:
        ACTIVE_RETRIEVAL_OPERATIONS.labels(tenant_id=tenant_id).dec()


# =============================================================================
# Contextual Retrieval Helper Functions
# =============================================================================


def record_contextual_enrichment(
    model: str,
    tenant_id: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    cache_hits: int,
    cache_misses: int,
    chunks_enriched: int,
    latency_seconds: float,
) -> None:
    """Record contextual enrichment metrics.

    Args:
        model: Model used for enrichment (e.g., claude-3-haiku-20240307)
        tenant_id: Tenant identifier
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated
        cost_usd: Estimated cost in USD
        cache_hits: Number of cache hits (Anthropic prompt caching)
        cache_misses: Number of cache misses
        chunks_enriched: Number of chunks enriched
        latency_seconds: Total latency in seconds
    """
    # Token usage
    CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL.labels(
        type="input",
        model=model,
        tenant_id=tenant_id,
    ).inc(input_tokens)

    CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL.labels(
        type="output",
        model=model,
        tenant_id=tenant_id,
    ).inc(output_tokens)

    # Cost
    CONTEXTUAL_ENRICHMENT_COST_USD_TOTAL.labels(
        model=model,
        tenant_id=tenant_id,
    ).inc(cost_usd)

    # Cache metrics
    CONTEXTUAL_ENRICHMENT_CACHE_HITS_TOTAL.labels(
        model=model,
        tenant_id=tenant_id,
    ).inc(cache_hits)

    CONTEXTUAL_ENRICHMENT_CACHE_MISSES_TOTAL.labels(
        model=model,
        tenant_id=tenant_id,
    ).inc(cache_misses)

    # Chunks
    CONTEXTUAL_ENRICHMENT_CHUNKS_TOTAL.labels(
        model=model,
        tenant_id=tenant_id,
    ).inc(chunks_enriched)

    # Latency
    CONTEXTUAL_ENRICHMENT_LATENCY_SECONDS.labels(
        model=model,
        tenant_id=tenant_id,
    ).observe(latency_seconds)
