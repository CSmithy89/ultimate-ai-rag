"""Observability package for Prometheus metrics and monitoring.

This package provides:
- Prometheus metric definitions for retrieval operations
- Middleware for exposing /metrics endpoint
- Decorators for timing and counting operations
"""

from .metrics import (
    # Counters
    RETRIEVAL_REQUESTS_TOTAL,
    RETRIEVAL_FALLBACK_TRIGGERED_TOTAL,
    GRADER_EVALUATIONS_TOTAL,
    # Histograms
    RETRIEVAL_LATENCY_SECONDS,
    RERANKING_IMPROVEMENT_RATIO,
    GRADER_SCORE,
    # Gauges
    RETRIEVAL_PRECISION,
    RETRIEVAL_RECALL,
    ACTIVE_RETRIEVAL_OPERATIONS,
    # Contextual Retrieval Metrics (Story 19-F5)
    CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL,
    CONTEXTUAL_ENRICHMENT_COST_USD_TOTAL,
    CONTEXTUAL_ENRICHMENT_CACHE_HITS_TOTAL,
    CONTEXTUAL_ENRICHMENT_CACHE_MISSES_TOTAL,
    CONTEXTUAL_ENRICHMENT_CHUNKS_TOTAL,
    CONTEXTUAL_ENRICHMENT_LATENCY_SECONDS,
    # Telemetry Metrics (Story 21-B1)
    FRONTEND_TELEMETRY_EVENTS_TOTAL,
    # Helper functions
    record_retrieval_request,
    record_retrieval_fallback,
    record_grader_evaluation,
    record_retrieval_latency,
    record_reranking_improvement,
    record_grader_score,
    set_retrieval_precision,
    set_retrieval_recall,
    track_active_retrieval,
    record_contextual_enrichment,
    record_frontend_telemetry,
    get_metrics_registry,
)
from .decorators import (
    track_retrieval_operation,
    measure_latency,
)
from .middleware import (
    create_metrics_endpoint,
    MetricsConfig,
)

__all__ = [
    # Counters
    "RETRIEVAL_REQUESTS_TOTAL",
    "RETRIEVAL_FALLBACK_TRIGGERED_TOTAL",
    "GRADER_EVALUATIONS_TOTAL",
    # Histograms
    "RETRIEVAL_LATENCY_SECONDS",
    "RERANKING_IMPROVEMENT_RATIO",
    "GRADER_SCORE",
    # Gauges
    "RETRIEVAL_PRECISION",
    "RETRIEVAL_RECALL",
    "ACTIVE_RETRIEVAL_OPERATIONS",
    # Contextual Retrieval Metrics (Story 19-F5)
    "CONTEXTUAL_ENRICHMENT_TOKENS_TOTAL",
    "CONTEXTUAL_ENRICHMENT_COST_USD_TOTAL",
    "CONTEXTUAL_ENRICHMENT_CACHE_HITS_TOTAL",
    "CONTEXTUAL_ENRICHMENT_CACHE_MISSES_TOTAL",
    "CONTEXTUAL_ENRICHMENT_CHUNKS_TOTAL",
    "CONTEXTUAL_ENRICHMENT_LATENCY_SECONDS",
    # Telemetry Metrics (Story 21-B1)
    "FRONTEND_TELEMETRY_EVENTS_TOTAL",
    # Helper functions
    "record_retrieval_request",
    "record_retrieval_fallback",
    "record_grader_evaluation",
    "record_retrieval_latency",
    "record_reranking_improvement",
    "record_grader_score",
    "set_retrieval_precision",
    "set_retrieval_recall",
    "track_active_retrieval",
    "record_contextual_enrichment",
    "record_frontend_telemetry",
    "get_metrics_registry",
    # Decorators
    "track_retrieval_operation",
    "measure_latency",
    # Middleware
    "create_metrics_endpoint",
    "MetricsConfig",
]
