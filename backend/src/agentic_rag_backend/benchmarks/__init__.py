"""Retrieval quality benchmark tools.

This module provides infrastructure for evaluating retrieval quality
using standard Information Retrieval metrics.
"""

from .metrics import (
    RetrievalMetrics,
    compute_all_metrics,
    dcg_at_k,
    idcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    ndcg_at_k_single,
    precision_at_k,
    precision_at_k_single,
    recall_at_k,
    recall_at_k_single,
    reciprocal_rank,
)

from .datasets import (
    Document,
    EvaluationDataset,
    Query,
    create_sample_dataset,
    load_dataset,
    save_dataset,
)

from .runner import (
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkRunner,
    RetrievalStrategy,
    StrategyResult,
)

__all__ = [
    # Metrics
    "RetrievalMetrics",
    "compute_all_metrics",
    "dcg_at_k",
    "idcg_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "ndcg_at_k_single",
    "precision_at_k",
    "precision_at_k_single",
    "recall_at_k",
    "recall_at_k_single",
    "reciprocal_rank",
    # Datasets
    "Document",
    "EvaluationDataset",
    "Query",
    "create_sample_dataset",
    "load_dataset",
    "save_dataset",
    # Runner
    "BenchmarkConfig",
    "BenchmarkReport",
    "BenchmarkRunner",
    "RetrievalStrategy",
    "StrategyResult",
]
