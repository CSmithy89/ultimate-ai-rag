"""Information Retrieval metrics for retrieval quality benchmarks.

This module implements standard IR metrics for evaluating retrieval systems:
- MRR@K (Mean Reciprocal Rank at K)
- NDCG@K (Normalized Discounted Cumulative Gain at K)
- Precision@K
- Recall@K

All metrics support graded relevance (not just binary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    """Container for computed retrieval quality metrics.

    Attributes:
        mrr: Mean Reciprocal Rank at K
        ndcg: Normalized Discounted Cumulative Gain at K
        precision: Precision at K
        recall: Recall at K
        k: The K value used for computation
        num_queries: Number of queries evaluated
    """

    mrr: float
    ndcg: float
    precision: float
    recall: float
    k: int
    num_queries: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            f"mrr@{self.k}": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg, 4),
            f"precision@{self.k}": round(self.precision, 4),
            f"recall@{self.k}": round(self.recall, 4),
            "k": self.k,
            "num_queries": self.num_queries,
        }


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate reciprocal rank for a single query.

    Reciprocal Rank is 1/rank of the first relevant document.
    Returns 0 if no relevant document is found in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs for this query
        k: Number of top results to consider

    Returns:
        Reciprocal rank (1/rank of first relevant doc, or 0)
    """
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def mrr_at_k(
    all_retrieved_ids: Sequence[Sequence[str]],
    all_relevant_ids: Sequence[set[str]],
    k: int,
) -> float:
    """Calculate Mean Reciprocal Rank at K across all queries.

    MRR is the average of reciprocal ranks across queries.
    A higher MRR indicates the system ranks relevant documents higher.

    Args:
        all_retrieved_ids: List of retrieved ID sequences (one per query)
        all_relevant_ids: List of relevant ID sets (one per query)
        k: Number of top results to consider

    Returns:
        Mean Reciprocal Rank at K (0.0 to 1.0)

    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(all_retrieved_ids) != len(all_relevant_ids):
        raise ValueError(
            f"Mismatched lengths: {len(all_retrieved_ids)} queries vs "
            f"{len(all_relevant_ids)} relevance sets"
        )
    if len(all_retrieved_ids) == 0:
        raise ValueError("No queries provided")

    total_rr = sum(
        reciprocal_rank(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids)
    )
    return total_rr / len(all_retrieved_ids)


def dcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at K.

    DCG weights relevance by log2(rank + 1) to penalize relevant
    documents appearing lower in the ranking.

    Uses the formula: DCG = sum(rel_i / log2(i + 1)) for i in 1..k

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevance_grades: Mapping of doc_id -> relevance grade (0.0-1.0)
        k: Number of top results to consider

    Returns:
        DCG score (0.0 to sum of all relevance grades)
    """
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        relevance = relevance_grades.get(doc_id, 0.0)
        # log2(1) = 0, so we use log2(rank + 1) to avoid division by zero at rank 1
        dcg += relevance / math.log2(rank + 1)
    return dcg


def idcg_at_k(
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """Calculate Ideal DCG at K (best possible DCG).

    IDCG is calculated by sorting all relevant documents by their
    relevance grade and computing DCG on that ideal ranking.

    Args:
        relevance_grades: Mapping of doc_id -> relevance grade (0.0-1.0)
        k: Number of top results to consider

    Returns:
        Ideal DCG score
    """
    # Sort by relevance grade descending
    sorted_grades = sorted(relevance_grades.values(), reverse=True)
    idcg = 0.0
    for rank, relevance in enumerate(sorted_grades[:k], start=1):
        idcg += relevance / math.log2(rank + 1)
    return idcg


def ndcg_at_k_single(
    retrieved_ids: Sequence[str],
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """Calculate Normalized DCG at K for a single query.

    NDCG = DCG / IDCG, normalized to 0-1 range.
    Returns 0 if IDCG is 0 (no relevant documents).

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevance_grades: Mapping of doc_id -> relevance grade (0.0-1.0)
        k: Number of top results to consider

    Returns:
        NDCG score (0.0 to 1.0)
    """
    idcg = idcg_at_k(relevance_grades, k)
    if idcg == 0:
        return 0.0
    dcg = dcg_at_k(retrieved_ids, relevance_grades, k)
    return dcg / idcg


def ndcg_at_k(
    all_retrieved_ids: Sequence[Sequence[str]],
    all_relevance_grades: Sequence[dict[str, float]],
    k: int,
) -> float:
    """Calculate average NDCG at K across all queries.

    Args:
        all_retrieved_ids: List of retrieved ID sequences (one per query)
        all_relevance_grades: List of relevance grade dicts (one per query)
        k: Number of top results to consider

    Returns:
        Average NDCG at K (0.0 to 1.0)

    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(all_retrieved_ids) != len(all_relevance_grades):
        raise ValueError(
            f"Mismatched lengths: {len(all_retrieved_ids)} queries vs "
            f"{len(all_relevance_grades)} relevance grade dicts"
        )
    if len(all_retrieved_ids) == 0:
        raise ValueError("No queries provided")

    total_ndcg = sum(
        ndcg_at_k_single(retrieved, grades, k)
        for retrieved, grades in zip(all_retrieved_ids, all_relevance_grades)
    )
    return total_ndcg / len(all_retrieved_ids)


def precision_at_k_single(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision at K for a single query.

    Precision@K = (# relevant in top-k) / k

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision at K (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_in_top_k = len(top_k & relevant_ids)
    return relevant_in_top_k / k


def precision_at_k(
    all_retrieved_ids: Sequence[Sequence[str]],
    all_relevant_ids: Sequence[set[str]],
    k: int,
) -> float:
    """Calculate average Precision at K across all queries.

    Args:
        all_retrieved_ids: List of retrieved ID sequences (one per query)
        all_relevant_ids: List of relevant ID sets (one per query)
        k: Number of top results to consider

    Returns:
        Average Precision at K (0.0 to 1.0)

    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(all_retrieved_ids) != len(all_relevant_ids):
        raise ValueError(
            f"Mismatched lengths: {len(all_retrieved_ids)} queries vs "
            f"{len(all_relevant_ids)} relevance sets"
        )
    if len(all_retrieved_ids) == 0:
        raise ValueError("No queries provided")

    total_precision = sum(
        precision_at_k_single(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids)
    )
    return total_precision / len(all_retrieved_ids)


def recall_at_k_single(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall at K for a single query.

    Recall@K = (# relevant in top-k) / (# total relevant)
    Returns 0 if there are no relevant documents.

    Args:
        retrieved_ids: Ordered list of retrieved document IDs
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall at K (0.0 to 1.0)
    """
    if len(relevant_ids) == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant_in_top_k = len(top_k & relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def recall_at_k(
    all_retrieved_ids: Sequence[Sequence[str]],
    all_relevant_ids: Sequence[set[str]],
    k: int,
) -> float:
    """Calculate average Recall at K across all queries.

    Args:
        all_retrieved_ids: List of retrieved ID sequences (one per query)
        all_relevant_ids: List of relevant ID sets (one per query)
        k: Number of top results to consider

    Returns:
        Average Recall at K (0.0 to 1.0)

    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(all_retrieved_ids) != len(all_relevant_ids):
        raise ValueError(
            f"Mismatched lengths: {len(all_retrieved_ids)} queries vs "
            f"{len(all_relevant_ids)} relevance sets"
        )
    if len(all_retrieved_ids) == 0:
        raise ValueError("No queries provided")

    total_recall = sum(
        recall_at_k_single(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved_ids, all_relevant_ids)
    )
    return total_recall / len(all_retrieved_ids)


def compute_all_metrics(
    all_retrieved_ids: Sequence[Sequence[str]],
    all_relevance_grades: Sequence[dict[str, float]],
    k: int,
) -> RetrievalMetrics:
    """Compute all IR metrics at K.

    This is the main entry point for computing all metrics together.
    Relevance grades are used for NDCG; binary relevance (grade > 0)
    is derived for MRR, Precision, and Recall.

    Args:
        all_retrieved_ids: List of retrieved ID sequences (one per query)
        all_relevance_grades: List of relevance grade dicts (one per query)
            Each dict maps doc_id -> relevance grade (0.0-1.0)
        k: Number of top results to consider

    Returns:
        RetrievalMetrics containing all computed metrics

    Raises:
        ValueError: If inputs have mismatched lengths or are empty
    """
    if len(all_retrieved_ids) != len(all_relevance_grades):
        raise ValueError(
            f"Mismatched lengths: {len(all_retrieved_ids)} queries vs "
            f"{len(all_relevance_grades)} relevance grade dicts"
        )
    if len(all_retrieved_ids) == 0:
        raise ValueError("No queries provided")

    # Convert graded relevance to binary for MRR/Precision/Recall
    all_relevant_ids = [
        {doc_id for doc_id, grade in grades.items() if grade > 0}
        for grades in all_relevance_grades
    ]

    return RetrievalMetrics(
        mrr=mrr_at_k(all_retrieved_ids, all_relevant_ids, k),
        ndcg=ndcg_at_k(all_retrieved_ids, all_relevance_grades, k),
        precision=precision_at_k(all_retrieved_ids, all_relevant_ids, k),
        recall=recall_at_k(all_retrieved_ids, all_relevant_ids, k),
        k=k,
        num_queries=len(all_retrieved_ids),
    )
