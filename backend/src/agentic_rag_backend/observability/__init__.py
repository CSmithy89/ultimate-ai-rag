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
    "get_metrics_registry",
    # Decorators
    "track_retrieval_operation",
    "measure_latency",
    # Middleware
    "create_metrics_endpoint",
    "MetricsConfig",
]
